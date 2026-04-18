import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
import sentencepiece as spm


# ============================================================================
# SETTINGS
# ============================================================================

CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = Path("willow_tokenizer.model")

MAX_SEQ_LEN = 256
MAX_CONTEXT_TURNS = 4

GEN_MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40
SEED = 42

mixed_precision.set_global_policy("float32")


# ============================================================================
# CUSTOM LAYERS
# Must match the saved model
# ============================================================================

@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, maxlen, d_model, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.d_model = d_model
        self.token_emb = keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.pos_emb = keras.layers.Embedding(maxlen, d_model, name="position_embedding")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "d_model": self.d_model,
        })
        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="mha",
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="gelu"),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(d_model),
            ],
            name="ffn",
        )
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln1")
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln2")
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False, padding_mask=None):
        x_dtype = x.dtype
        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=x_dtype),
            -1,
            0,
        )

        if padding_mask is not None:
            pad = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype=x_dtype)
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :] * pad
        else:
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        attn_output = self.att(
            query=x,
            value=x,
            key=x,
            attention_mask=causal_mask,
            training=training,
        )
        attn_output = self.drop1(attn_output, training=training)
        attn_output = tf.cast(attn_output, x_dtype)

        x = tf.cast(x, x_dtype)
        x = self.ln1(x + attn_output)

        ffn_output = self.ffn(x, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, x_dtype)

        x = tf.cast(x, x_dtype)
        return self.ln2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_lr, warmup_steps, total_steps, min_lr=1e-6, **kwargs):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.target_lr * (step / tf.maximum(warmup_steps, 1.0))
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress))
        decay_lr = self.min_lr + (self.target_lr - self.min_lr) * cosine_decay

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


# ============================================================================
# TOKENIZER
# ============================================================================

SPECIAL_USER = "<user>"
SPECIAL_ASSISTANT = "<assistant>"
SPECIAL_SEP = "<sep>"

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

PAD_ID = sp.pad_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

USER_ID = sp.piece_to_id(SPECIAL_USER)
ASSISTANT_ID = sp.piece_to_id(SPECIAL_ASSISTANT)
SEP_ID = sp.piece_to_id(SPECIAL_SEP)


def validate_special_tokens():
    missing = []
    for token_name, token_id in [
        (SPECIAL_USER, USER_ID),
        (SPECIAL_ASSISTANT, ASSISTANT_ID),
        (SPECIAL_SEP, SEP_ID),
    ]:
        if token_id < 0:
            missing.append(token_name)

    if missing:
        raise ValueError(f"Tokenizer is missing special tokens: {missing}")


def encode_turn(role_token: str, text: str) -> list[int]:
    role_id = sp.piece_to_id(role_token)
    return [role_id] + sp.encode(text, out_type=int) + [SEP_ID]


# ============================================================================
# MODEL LOADING
# ============================================================================

def list_models():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {CHECKPOINT_DIR}")

    models = sorted([p for p in CHECKPOINT_DIR.iterdir() if p.suffix == ".keras"])

    if not models:
        raise FileNotFoundError(f"No .keras models found in: {CHECKPOINT_DIR}")

    print("\nAvailable models:\n")
    for i, model_path in enumerate(models):
        print(f"[{i}] {model_path.name}")

    return models


def choose_model(models):
    while True:
        raw = input("\nChoose model index: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print("Invalid selection.")


def load_chat_model(model_path: Path):
    print(f"\nLoading model: {model_path}\n")
    model = keras.models.load_model(str(model_path), compile=False)
    return model


# ============================================================================
# GENERATION
# ============================================================================

def sample_next_token(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float64)

    banned_ids = {PAD_ID, USER_ID, ASSISTANT_ID}
    for banned_id in banned_ids:
        if 0 <= banned_id < len(logits):
            logits[banned_id] = -1e10

    if temperature is None or temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_indices]
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / np.sum(probs)
        return int(np.random.choice(top_indices, p=probs))

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(len(logits)), p=probs))


def build_prompt_ids(conversation_turns: list[str]) -> list[int]:
    prompt_ids = [BOS_ID]

    for i, turn in enumerate(conversation_turns):
        role_token = SPECIAL_USER if i % 2 == 0 else SPECIAL_ASSISTANT
        prompt_ids.extend(encode_turn(role_token, turn))

    prompt_ids.append(ASSISTANT_ID)
    return prompt_ids


def generate_reply(model, conversation_turns: list[str]) -> str:
    prompt_ids = build_prompt_ids(conversation_turns)
    generated = prompt_ids[:]
    new_tokens = []

    for _ in range(GEN_MAX_NEW_TOKENS):
        x = generated[-MAX_SEQ_LEN:]

        if len(x) < MAX_SEQ_LEN:
            x = x + [PAD_ID] * (MAX_SEQ_LEN - len(x))

        x_tensor = np.array([x], dtype=np.int32)
        logits = model({"input_ids": x_tensor}, training=False)[0].numpy()

        last_real_idx = max(i for i, tok in enumerate(x) if tok != PAD_ID)
        next_logits = logits[last_real_idx]

        next_id = sample_next_token(
            next_logits,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        if next_id in (EOS_ID, SEP_ID):
            break

        generated.append(next_id)
        new_tokens.append(next_id)

    return sp.decode(new_tokens).strip()


# ============================================================================
# CHAT LOOP
# ============================================================================

def chat_loop(model):
    print("Chat started.")
    print("Commands:")
    print("  /reset  -> clear conversation history")
    print("  /quit   -> exit")
    print()

    conversation_turns = []

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"/quit", "quit", "exit"}:
            print("Bye.")
            break

        if user_text.lower() == "/reset":
            conversation_turns.clear()
            print("History cleared.\n")
            continue

        turns_for_model = conversation_turns + [user_text]
        reply = generate_reply(model, turns_for_model)

        print(f"Willow: {reply}\n")

        conversation_turns.append(user_text)
        conversation_turns.append(reply)

        if len(conversation_turns) > MAX_CONTEXT_TURNS:
            conversation_turns = conversation_turns[-MAX_CONTEXT_TURNS:]


# ============================================================================
# MAIN
# ============================================================================

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    validate_special_tokens()

    print("Tokenizer loaded.")
    print(f"Vocab size: {sp.vocab_size()}")
    print(f"PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")
    print(f"<user>={USER_ID}, <assistant>={ASSISTANT_ID}, <sep>={SEP_ID}")

    models = list_models()
    model_path = choose_model(models)
    model = load_chat_model(model_path)

    chat_loop(model)


if __name__ == "__main__":
    main()