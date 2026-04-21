#!/usr/bin/env python3

import os
import re
import tempfile
import unicodedata
import weakref
from collections import Counter
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
import sentencepiece as spm

try:
    import ftfy
except ImportError:
    ftfy = None


# ============================================================================
# SETTINGS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_PATH = BASE_DIR / "willow_tokenizer.model"

CHATBOT_NAME = "Optimus"
SEED = 42

CONFIG = {
    "max_context_turns": 6,
    "max_seq_len": 384,
    "d_model": 512,
    "num_heads": 8,
    "ff_dim": 2048,
    "num_layers": 8,
    "dropout": 0.15,
    "preview_temperature": 0.5,
    "preview_top_k": 40,
    "preview_max_new_tokens": 80,
}

MAX_CONTEXT_TURNS = CONFIG["max_context_turns"]
GEN_MAX_NEW_TOKENS = CONFIG["preview_max_new_tokens"]
TEMPERATURE = CONFIG["preview_temperature"]
TOP_K = CONFIG["preview_top_k"]
REPETITION_PENALTY = 1.15
FREQUENCY_PENALTY = 0.08
RECENT_TOKEN_PENALTY = 0.12
RECENT_TOKEN_WINDOW = 12
NO_REPEAT_NGRAM_SIZE = 3

mixed_precision.set_global_policy("float32")


# ============================================================================
# CUSTOM LAYERS
# Must match the saved model
# ============================================================================

@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.d_model = d_model
        self.token_emb = keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.pos_emb = keras.layers.Embedding(maxlen, d_model, name="position_embedding")

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        seq_len = input_shape[-1] if input_shape[-1] is not None else self.maxlen
        self.pos_emb.build((seq_len,))
        super().build(input_shape)

    def call(self, input_ids):
        positions = tf.range(start=0, limit=tf.shape(input_ids)[-1], delta=1)
        return self.token_emb(input_ids) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "d_model": self.d_model,
        })
        return config


@keras.saving.register_keras_serializable()
class TiedLMHead(keras.layers.Layer):
    def __init__(self, vocab_size: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self._tied_embeddings_ref = None
        self._untied_embeddings = None

    def build(self, input_shape):
        self._untied_embeddings = self.add_weight(
            name="untied_embeddings",
            shape=(self.vocab_size, self.d_model),
            initializer="zeros",
            trainable=False,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def tie_to(self, embedding_variable: tf.Variable) -> None:
        self._tied_embeddings_ref = weakref.ref(embedding_variable)

    def call(self, x):
        tied_var = self._tied_embeddings_ref() if self._tied_embeddings_ref is not None else None
        weight_tensor = tied_var if tied_var is not None else self._untied_embeddings
        weight_tensor = tf.cast(weight_tensor, tf.float32)
        return tf.matmul(tf.cast(x, tf.float32), weight_tensor, transpose_b=True) + tf.cast(self.bias, tf.float32)

    def save_own_variables(self, store):
        store["bias"] = self.bias.numpy()
        store["untied_embeddings"] = self._untied_embeddings.numpy()

    def load_own_variables(self, store):
        saw_bias = False
        saw_matrix = False
        for key in store.keys():
            value = store[key]
            shape = tuple(value.shape)
            if key == "untied_embeddings" or shape == (self.vocab_size, self.d_model):
                self._untied_embeddings.assign(value)
                saw_matrix = True
            elif key == "bias" or shape == (self.vocab_size,):
                self.bias.assign(value)
                saw_bias = True

        if not saw_bias:
            pass
        if not saw_matrix:
            pass

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.vocab_size)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
        })
        return config


def retie_weights(model: keras.Model) -> None:
    embed_weights = model.get_layer("embed").token_emb.embeddings
    model.get_layer("lm_head").tie_to(embed_weights)


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
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
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        self.att.build(query_shape=input_shape, value_shape=input_shape, key_shape=input_shape)
        self.ffn.build(input_shape)
        self.drop1.build(input_shape)
        self.drop2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False, padding_mask=None):
        x_dtype = x.dtype
        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=x_dtype), -1, 0)
        if padding_mask is not None:
            pad = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype=x_dtype)
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :] * pad
        else:
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        normed = self.ln1(x)
        attn_output = self.att(
            query=normed,
            value=normed,
            key=normed,
            attention_mask=causal_mask,
            training=training,
        )
        attn_output = self.drop1(attn_output, training=training)
        attn_output = tf.cast(attn_output, x_dtype)
        x = x + attn_output

        normed = self.ln2(x)
        ffn_output = self.ffn(normed, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, x_dtype)
        return x + ffn_output

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
    def __init__(self, target_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        super().__init__()
        self.target_lr = float(target_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.min_lr = self.target_lr * self.min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.target_lr * (step / tf.maximum(1.0, warmup_steps))
        progress = (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * tf.minimum(progress, 1.0))
        )
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


def build_chatbot_model(vocab_size: int, max_seq_len: int) -> keras.Model:
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    padding_mask = keras.ops.cast(keras.ops.not_equal(input_ids, PAD_ID), "float32")

    embed_layer = TokenAndPositionEmbedding(vocab_size, max_seq_len, CONFIG["d_model"], name="embed")
    x = embed_layer(input_ids)
    x = keras.layers.Dropout(CONFIG["dropout"])(x)

    for block_index in range(CONFIG["num_layers"]):
        x = TransformerBlock(
            d_model=CONFIG["d_model"],
            num_heads=CONFIG["num_heads"],
            ff_dim=CONFIG["ff_dim"],
            dropout=CONFIG["dropout"],
            name=f"transformer_block_{block_index}",
        )(x, padding_mask=padding_mask)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")(x)

    lm_head = TiedLMHead(vocab_size, CONFIG["d_model"], dtype="float32", name="lm_head")
    lm_head.tie_to(embed_layer.token_emb.embeddings)
    logits = lm_head(x)

    return keras.Model(inputs=input_ids, outputs=logits, name=CHATBOT_NAME.lower())


CUSTOM_OBJECTS = {
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "TransformerBlock": TransformerBlock,
    "WarmupCosineDecay": WarmupCosineDecay,
    "TiedLMHead": TiedLMHead,
}


# ============================================================================
# TOKENIZER
# ============================================================================

SPECIAL_USER = "<user>"
SPECIAL_ASSISTANT = "<assistant>"
SPECIAL_SEP = "<sep>"

_whitespace_re = re.compile(r"\s+")
_non_text_re = re.compile(r"[^a-zA-Z0-9.!?,;:()\[\]{}\-\s'\"/]+")

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer model not found: {TOKENIZER_PATH}")

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

PAD_ID = sp.pad_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
USER_ID = sp.piece_to_id(SPECIAL_USER)
ASSISTANT_ID = sp.piece_to_id(SPECIAL_ASSISTANT)
SEP_ID = sp.piece_to_id(SPECIAL_SEP)
VOCAB_SIZE = sp.get_piece_size()


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


def normalize_text(text: str) -> str:
    text = str(text)
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = _non_text_re.sub(" ", text)
    text = _whitespace_re.sub(" ", text).strip()
    return text


def encode_turn(role_token: str, text: str) -> list[int]:
    return [sp.piece_to_id(role_token)] + sp.encode(text, out_type=int) + [SEP_ID]


# ============================================================================
# MODEL LOADING
# ============================================================================

def list_models():
    models = []

    models.extend(sorted(BASE_DIR.glob("*.keras")))
    if CHECKPOINT_DIR.exists():
        models.extend(sorted(CHECKPOINT_DIR.glob("*.keras")))

    unique_models = []
    seen = set()
    for model_path in models:
        resolved = model_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_models.append(resolved)

    if not unique_models:
        raise FileNotFoundError(f"No .keras models found in: {BASE_DIR}")

    print("\nAvailable models:\n")
    for index, model_path in enumerate(unique_models):
        try:
            label = model_path.relative_to(BASE_DIR)
        except ValueError:
            label = model_path
        print(f"[{index}] {label}")

    return unique_models


def choose_model(models):
    while True:
        raw = input("\nChoose model index: ").strip()
        try:
            index = int(raw)
            if 0 <= index < len(models):
                return models[index]
        except ValueError:
            pass
        print("Invalid selection.")


def load_chat_model(model_path: Path):
    print(f"\nLoading model: {model_path}\n")

    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    except Exception as error:
        print(f"Full load failed ({type(error).__name__}): {error}")
        print("Falling back to build_chatbot_model(...) + load_weights(...)")
        model = build_chatbot_model(VOCAB_SIZE, CONFIG["max_seq_len"])
        model.load_weights(str(model_path))

    retie_weights(model)
    return model


# ============================================================================
# GENERATION
# ============================================================================

def apply_repetition_controls(logits: np.ndarray, reply_tokens: list[int]) -> np.ndarray:
    adjusted = logits.astype(np.float64).copy()

    banned_ids = {PAD_ID, USER_ID, ASSISTANT_ID}
    for banned_id in banned_ids:
        if 0 <= banned_id < len(adjusted):
            adjusted[banned_id] = -1e10

    if not reply_tokens:
        return adjusted

    if REPETITION_PENALTY and REPETITION_PENALTY > 1.0:
        for token_id in set(reply_tokens):
            if 0 <= token_id < len(adjusted):
                if adjusted[token_id] >= 0:
                    adjusted[token_id] /= REPETITION_PENALTY
                else:
                    adjusted[token_id] *= REPETITION_PENALTY

    if FREQUENCY_PENALTY and FREQUENCY_PENALTY > 0:
        for token_id, count in Counter(reply_tokens).items():
            if 0 <= token_id < len(adjusted):
                adjusted[token_id] -= FREQUENCY_PENALTY * count

    if RECENT_TOKEN_PENALTY and RECENT_TOKEN_PENALTY > 0 and RECENT_TOKEN_WINDOW > 0:
        for token_id in reply_tokens[-RECENT_TOKEN_WINDOW:]:
            if 0 <= token_id < len(adjusted):
                adjusted[token_id] -= RECENT_TOKEN_PENALTY

    blocked_tokens = get_blocked_ngram_tokens(reply_tokens, NO_REPEAT_NGRAM_SIZE)
    for token_id in blocked_tokens:
        if 0 <= token_id < len(adjusted):
            adjusted[token_id] = -1e10

    return adjusted


def get_blocked_ngram_tokens(reply_tokens: list[int], ngram_size: int) -> set[int]:
    if ngram_size is None or ngram_size < 2:
        return set()

    prefix_len = ngram_size - 1
    if len(reply_tokens) < prefix_len:
        return set()

    prefix = tuple(reply_tokens[-prefix_len:])
    blocked = set()

    for start in range(len(reply_tokens) - ngram_size + 1):
        ngram = reply_tokens[start:start + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(ngram[-1])

    return blocked


def sample_next_token(
    logits: np.ndarray,
    reply_tokens: list[int],
    temperature=TEMPERATURE,
    top_k=TOP_K,
):
    logits = apply_repetition_controls(logits, reply_tokens)

    if temperature is None or temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), len(logits))
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_indices]
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / np.sum(probs)
        return int(np.random.choice(top_indices, p=probs))

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(len(logits)), p=probs))


def generate_reply(
    model,
    conversation_turns: list[str],
    max_new_tokens=GEN_MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
):
    prompt_ids = [BOS_ID]
    for index, turn in enumerate(conversation_turns):
        role_token = SPECIAL_USER if index % 2 == 0 else SPECIAL_ASSISTANT
        prompt_ids.extend(encode_turn(role_token, normalize_text(turn)))

    prompt_ids.append(ASSISTANT_ID)
    generated = prompt_ids[:]
    new_tokens = []

    for _ in range(max_new_tokens):
        x = generated[-CONFIG["max_seq_len"]:]
        x_tensor = np.array([x], dtype=np.int32)

        logits = model(x_tensor, training=False)[0].numpy()
        next_logits = logits[-1]
        next_id = sample_next_token(
            next_logits,
            reply_tokens=new_tokens,
            temperature=temperature,
            top_k=top_k,
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
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

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

        print(f"{CHATBOT_NAME}: {reply}\n")

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
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")
    print(f"<user>={USER_ID}, <assistant>={ASSISTANT_ID}, <sep>={SEP_ID}")

    models = list_models()
    model_path = choose_model(models)
    model = load_chat_model(model_path)

    chat_loop(model)


if __name__ == "__main__":
    main()
