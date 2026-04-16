import os, re, unicodedata

os.environ['KERAS_BACKEND']        = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'

import numpy as np
import keras
import sentencepiece as spm
import ftfy

# ── Constants — must match values used during training ─────────────────────────
SEQ_LENGTH         = 256
CONTEXT_TURNS      = 2
REPETITION_PENALTY = 1.3

# ── Load tokenizer and model ───────────────────────────────────────────────────
print("Loading tokenizer...")
with open('chatbot_sp_v1.model', 'rb') as f:
    sp_model_bytes = f.read()
sp = spm.SentencePieceProcessor()
sp.load_from_serialized_proto(sp_model_bytes)
print(f"  Vocabulary size: {sp.get_piece_size()}")

print("Loading model (this may take a moment)...")
model = keras.models.load_model('chatbot_best_v1.keras')
print("  Ready.\n")

# ── Helper functions ───────────────────────────────────────────────────────────

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "ain't": "is not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "here's": "here is", "who's": "who is",
    "what's": "what is", "let's": "let us", "you're": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "we'd": "we would", "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would", "he'd": "he would",
    "she'd": "she would", "he'll": "he will", "she'll": "she will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mustn't": "must not", "needn't": "need not",
}

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def expand_contractions(s):
    for contraction, expansion in CONTRACTIONS.items():
        s = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, s)
    return s

def normalize_string(s):
    s = ftfy.fix_text(s)
    s = unicode_to_ascii(s.lower().strip())
    s = expand_contractions(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def chat(user_message, history, temperature=0.8, top_p=0.9, max_new_tokens=80):
    clean_message = normalize_string(user_message)

    turns = []
    for h, b in history[-(CONTEXT_TURNS - 1):]:
        turns.append(f"Human: {h}")
        turns.append(f"Bot: {b}")
    turns.append(f"Human: {clean_message}")
    turns.append("Bot:")

    prompt        = "\n".join(turns)
    prompt_ids    = sp.encode_as_ids(prompt)
    generated_ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = generated_ids[-SEQ_LENGTH:]
        pad_len = SEQ_LENGTH - len(context)
        padded  = [0] * pad_len + context
        x       = np.array([padded], dtype=np.int32)

        logits = model(x, training=False)[0, -1, :].numpy()
        logits = logits / max(temperature, 1e-8)
        probs  = np.exp(logits - np.max(logits))
        probs  = probs / probs.sum()

        sorted_idx  = np.argsort(probs)[::-1]
        cumulative  = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumulative > top_p
        cutoff_mask[0] = False
        probs[sorted_idx[cutoff_mask]] = 0.0
        probs = probs / probs.sum()

        for token_id in set(generated_ids[len(prompt_ids):]):
            probs[token_id] /= REPETITION_PENALTY
        probs = probs / probs.sum()

        next_id = int(np.random.choice(len(probs), p=probs))
        generated_ids.append(next_id)

        reply_so_far = sp.decode_ids(generated_ids[len(prompt_ids):])
        if "Human:" in reply_so_far:
            return reply_so_far.split("Human:")[0].strip()

    return sp.decode_ids(generated_ids[len(prompt_ids):]).strip()

# ── Main chat loop ─────────────────────────────────────────────────────────────

print("Chatbot ready. Type 'quit' or 'exit' to stop, 'reset' to clear history.\n")

history = []

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye.")
        break

    if user_input.lower() == "reset":
        history = []
        print("(conversation history cleared)\n")
        continue

    reply = chat(user_input, history)
    print(f"Bot: {reply}\n")
    history.append((normalize_string(user_input), reply))