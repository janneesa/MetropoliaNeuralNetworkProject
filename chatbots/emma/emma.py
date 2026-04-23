#!/usr/bin/env python3

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import sentencepiece as spm
import tensorflow as tf
import keras
from keras import mixed_precision

try:
    import ftfy
except ImportError:
    ftfy = None


# ============================================================================
# CONST VALUES - DO NOT CHANGE
# These must stay aligned with Emma's training artifacts.
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_PATH = BASE_DIR / "emma_tokenizer.model"

CHATBOT_NAME = "Emma"
CHATBOT_SLUG = CHATBOT_NAME.lower()

SPECIAL_USER = "<user>"
SPECIAL_ASSISTANT = "<assistant>"
SPECIAL_SEP = "<sep>"

MODEL_VOCAB_SIZE = 16000
MODEL_MAX_SEQ_LEN = 512
MODEL_D_MODEL = 512
MODEL_NUM_HEADS = 8
MODEL_FF_DIM = 2048
MODEL_NUM_LAYERS = 10
MODEL_DROPOUT = 0.1

_whitespace_re = re.compile(r"\s+")
_non_text_re = re.compile(r"[^a-zA-Z0-9.!?,;:()\[\]{}\-\s'\"/]+")


# ============================================================================
# USER-TUNABLE VALUES - SAFE TO CHANGE
# These only affect inference-time chat behavior.
# They do not change the trained weights inside the saved model.
# ============================================================================

# Number of stored conversation turns kept in session history.
# Higher values let the model see more back-and-forth, but older turns are dropped sooner when lower.
DEFAULT_MAX_CONTEXT_TURNS = 6

# Hard cap for how many new tokens one reply may generate before stopping.
# Higher values allow longer answers but also increase latency and rambling risk.
DEFAULT_MAX_NEW_TOKENS = 80

# Sampling temperature applied to next-token logits.
# Lower values make replies more conservative and likely; higher values make them more varied.
# With Emma. I noticed that 0.5 is quite good with the best_emma.keras, but feel free to experiment
DEFAULT_TEMPERATURE = 0.5

# Restricts sampling to the top K candidate tokens after temperature is applied.
# Lower values keep replies tighter and safer; higher values allow broader phrasing.
DEFAULT_TOP_K = 40

# Penalizes tokens that have already appeared anywhere in the current reply.
# This helps reduce loops and repeated wording without changing the model weights.
DEFAULT_REPETITION_PENALTY = 1.15

# Subtracts a small penalty for each time a token has already appeared in the current reply.
# This pushes the model away from overusing the same words over and over.
DEFAULT_FREQUENCY_PENALTY = 0.08

# Extra penalty for very recently used tokens.
# This specifically targets short-range repetition like stuttering or repeated fragments.
DEFAULT_RECENT_TOKEN_PENALTY = 0.12

# How many most-recent generated tokens are considered by the recent-token penalty.
# Larger windows fight repetition more aggressively, but can also suppress legitimate reuse.
DEFAULT_RECENT_TOKEN_WINDOW = 14

# Blocks the model from completing an n-gram it has already produced in the same reply.
# Example: with 3, the generator avoids repeating the same 3-token phrase pattern.
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3

# Sampling seed for reproducible replies when randomness is enabled.
# Keeping this fixed makes repeated test runs easier to compare.
DEFAULT_SEED = 42

# CLI-only default for whether terminal output streams as chunks or prints once at the end.
CLI_STREAM_RESPONSES = True


@dataclass(frozen=True)
class GenerationConfig:
    """Inference controls for chatting with the already trained model.

    These values shape prompting, token sampling, and anti-repetition behavior.
    They do not retrain the model or modify checkpoint weights.
    """

    max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    recent_token_penalty: float = DEFAULT_RECENT_TOKEN_PENALTY
    recent_token_window: int = DEFAULT_RECENT_TOKEN_WINDOW
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    seed: int | None = DEFAULT_SEED


logging.getLogger("tensorflow").setLevel(logging.ERROR)
mixed_precision.set_global_policy("float32")


# ============================================================================
# CUSTOM LAYERS
# Must match the saved model.
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
        self.pos_emb.build(input_shape)
        super().build(input_shape)

    def call(self, input_ids):
        positions = tf.range(start=0, limit=tf.shape(input_ids)[-1], delta=1)
        return self.token_emb(input_ids) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "maxlen": self.maxlen,
                "d_model": self.d_model,
            }
        )
        return config


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
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False, padding_mask=None):
        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=x.dtype), -1, 0)
        if padding_mask is not None:
            pad = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype=x.dtype)
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :] * pad
        else:
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        residual = x
        attn_output = self.att(
            query=x,
            value=x,
            key=x,
            attention_mask=causal_mask,
            training=training,
        )
        attn_output = self.drop1(attn_output, training=training)
        attn_output = tf.cast(attn_output, residual.dtype)
        x = self.ln1(residual + attn_output)

        residual = x
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, residual.dtype)
        return self.ln2(residual + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
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
            1.0 + tf.cos(math.pi * tf.minimum(progress, 1.0))
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
    input_ids = keras.Input(shape=(max_seq_len,), dtype="int32", name="input_ids")
    padding_mask = keras.ops.cast(keras.ops.not_equal(input_ids, PAD_ID), "float32")

    x = TokenAndPositionEmbedding(vocab_size, max_seq_len, MODEL_D_MODEL, name="embed")(input_ids)
    x = keras.layers.Dropout(MODEL_DROPOUT)(x)

    for block_index in range(MODEL_NUM_LAYERS):
        x = TransformerBlock(
            d_model=MODEL_D_MODEL,
            num_heads=MODEL_NUM_HEADS,
            ff_dim=MODEL_FF_DIM,
            dropout=MODEL_DROPOUT,
            name=f"transformer_block_{block_index}",
        )(x, padding_mask=padding_mask)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")(x)
    logits = keras.layers.Dense(vocab_size, dtype="float32", name="lm_head")(x)
    return keras.Model(inputs=input_ids, outputs=logits, name=CHATBOT_SLUG)


CUSTOM_OBJECTS = {
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "TransformerBlock": TransformerBlock,
    "WarmupCosineDecay": WarmupCosineDecay,
}


# ============================================================================
# TOKENIZER
# ============================================================================

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


def validate_tokenizer() -> None:
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

    if VOCAB_SIZE != MODEL_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocab size mismatch. Expected {MODEL_VOCAB_SIZE}, got {VOCAB_SIZE}.")


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

def list_models() -> list[Path]:
    models = []
    models.extend(BASE_DIR.glob("*.keras"))
    if CHECKPOINT_DIR.exists():
        models.extend(CHECKPOINT_DIR.glob("*.keras"))

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

    def sort_key(model_path: Path):
        if model_path.name == f"best_{CHATBOT_SLUG}.keras":
            priority = 0
        elif model_path.name == f"final_{CHATBOT_SLUG}.keras":
            priority = 1
        elif model_path.parent.resolve() == CHECKPOINT_DIR.resolve():
            priority = 2
        else:
            priority = 3
        return (priority, str(model_path))

    unique_models.sort(key=sort_key)
    return unique_models


def print_model_choices(models: list[Path]) -> None:
    print("\nAvailable models:\n")
    for index, model_path in enumerate(models):
        try:
            label = model_path.relative_to(BASE_DIR)
        except ValueError:
            label = model_path
        print(f"[{index}] {label}")


def choose_model(models: list[Path]) -> Path:
    print_model_choices(models)
    default_index = 0

    while True:
        raw = input(f"\nChoose model index [{default_index}]: ").strip()
        if not raw:
            return models[default_index]

        try:
            index = int(raw)
            if 0 <= index < len(models):
                return models[index]
        except ValueError:
            pass

        print("Invalid selection.")


def resolve_model_path(model_path: str | Path | None = None, prompt_for_model: bool = False) -> Path:
    if model_path is not None:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            cwd_candidate = candidate.resolve()
            base_dir_candidate = (BASE_DIR / candidate).resolve()
            candidate = cwd_candidate if cwd_candidate.exists() else base_dir_candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Model not found: {candidate}")
        return candidate

    models = list_models()
    if prompt_for_model:
        return choose_model(models)
    return models[0]


def load_chat_model(model_path: Path, verbose: bool = False) -> keras.Model:
    if verbose:
        print(f"\nLoading model: {model_path}\n")

    try:
        return keras.models.load_model(
            str(model_path),
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    except Exception as error:
        if verbose:
            print(f"Full load failed ({type(error).__name__}): {error}")
            print("Falling back to build_chatbot_model(...) + load_weights(...)")
        model = build_chatbot_model(VOCAB_SIZE, MODEL_MAX_SEQ_LEN)
        model.load_weights(str(model_path))
        return model


# ============================================================================
# GENERATION
# ============================================================================

def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)


def trim_conversation_history(conversation_turns: list[str], max_context_turns: int) -> list[str]:
    if max_context_turns <= 0 or len(conversation_turns) <= max_context_turns:
        return conversation_turns
    return conversation_turns[-max_context_turns:]


def build_prompt_ids(conversation_turns: list[str]) -> list[int]:
    prompt_ids = [BOS_ID]
    for index, turn in enumerate(conversation_turns):
        role_token = SPECIAL_USER if index % 2 == 0 else SPECIAL_ASSISTANT
        prompt_ids.extend(encode_turn(role_token, normalize_text(turn)))
    prompt_ids.append(ASSISTANT_ID)
    return prompt_ids


def get_blocked_ngram_tokens(reply_tokens: list[int], ngram_size: int) -> set[int]:
    if ngram_size < 2:
        return set()

    prefix_len = ngram_size - 1
    if len(reply_tokens) < prefix_len:
        return set()

    prefix = tuple(reply_tokens[-prefix_len:])
    blocked = set()

    for start in range(len(reply_tokens) - ngram_size + 1):
        ngram = reply_tokens[start : start + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(ngram[-1])

    return blocked


def apply_repetition_controls(
    logits: np.ndarray,
    reply_tokens: list[int],
    generation_config: GenerationConfig,
) -> np.ndarray:
    # These controls reshape logits only for the current generation step.
    # They influence token choice during chatting, but do not alter the model itself.
    adjusted = logits.astype(np.float64).copy()

    banned_ids = {PAD_ID, USER_ID, ASSISTANT_ID}
    for banned_id in banned_ids:
        if 0 <= banned_id < len(adjusted):
            adjusted[banned_id] = -1e10

    if not reply_tokens:
        return adjusted

    if generation_config.repetition_penalty > 1.0:
        for token_id in set(reply_tokens):
            if 0 <= token_id < len(adjusted):
                if adjusted[token_id] >= 0:
                    adjusted[token_id] /= generation_config.repetition_penalty
                else:
                    adjusted[token_id] *= generation_config.repetition_penalty

    if generation_config.frequency_penalty > 0:
        for token_id, count in Counter(reply_tokens).items():
            if 0 <= token_id < len(adjusted):
                adjusted[token_id] -= generation_config.frequency_penalty * count

    if generation_config.recent_token_penalty > 0 and generation_config.recent_token_window > 0:
        for token_id in reply_tokens[-generation_config.recent_token_window :]:
            if 0 <= token_id < len(adjusted):
                adjusted[token_id] -= generation_config.recent_token_penalty

    blocked_tokens = get_blocked_ngram_tokens(reply_tokens, generation_config.no_repeat_ngram_size)
    for token_id in blocked_tokens:
        if 0 <= token_id < len(adjusted):
            adjusted[token_id] = -1e10

    return adjusted


def sample_next_token(
    logits: np.ndarray,
    reply_tokens: list[int],
    generation_config: GenerationConfig,
) -> int:
    logits = apply_repetition_controls(logits, reply_tokens, generation_config)

    if generation_config.temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / generation_config.temperature

    if generation_config.top_k > 0:
        top_k = min(int(generation_config.top_k), len(logits))
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_indices]
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / np.sum(probs)
        return int(np.random.choice(top_indices, p=probs))

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(len(logits)), p=probs))


def iter_generated_token_ids(
    model: keras.Model,
    conversation_turns: list[str],
    generation_config: GenerationConfig,
) -> Iterator[int]:
    prompt_ids = build_prompt_ids(conversation_turns)
    generated = prompt_ids[:]
    reply_tokens = []

    for _ in range(generation_config.max_new_tokens):
        x = generated[-MODEL_MAX_SEQ_LEN:]
        if len(x) < MODEL_MAX_SEQ_LEN:
            x = x + [PAD_ID] * (MODEL_MAX_SEQ_LEN - len(x))

        x_tensor = np.array([x], dtype=np.int32)
        logits = model(x_tensor, training=False)[0].numpy()
        last_real_idx = max(i for i, token_id in enumerate(x) if token_id != PAD_ID)
        next_logits = logits[last_real_idx]
        next_id = sample_next_token(next_logits, reply_tokens, generation_config)

        if next_id in (EOS_ID, SEP_ID):
            break

        generated.append(next_id)
        reply_tokens.append(next_id)
        yield next_id


def decode_reply_tokens(reply_token_ids: list[int], strip: bool = True) -> str:
    text = sp.decode(reply_token_ids)
    return text.strip() if strip else text


def compute_stream_delta(previous_text: str, current_text: str) -> str:
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]

    common_prefix = os.path.commonprefix([previous_text, current_text])
    return current_text[len(common_prefix) :]


def generate_reply(
    model: keras.Model,
    conversation_turns: list[str],
    generation_config: GenerationConfig,
) -> str:
    reply_token_ids = list(iter_generated_token_ids(model, conversation_turns, generation_config))
    return decode_reply_tokens(reply_token_ids)


def stream_reply(
    model: keras.Model,
    conversation_turns: list[str],
    generation_config: GenerationConfig,
) -> Iterator[str]:
    reply_token_ids = []
    streamed_text = ""

    for token_id in iter_generated_token_ids(model, conversation_turns, generation_config):
        reply_token_ids.append(token_id)
        current_text = decode_reply_tokens(reply_token_ids, strip=False)
        delta = compute_stream_delta(streamed_text, current_text)
        if delta:
            yield delta
        streamed_text = current_text

    return decode_reply_tokens(reply_token_ids)


# ============================================================================
# SERVICE / SESSION
# ============================================================================

class EmmaChatService:
    def __init__(
        self,
        model: keras.Model,
        model_path: Path,
        default_generation_config: GenerationConfig | None = None,
    ):
        self.model = model
        self.model_path = model_path
        self.default_generation_config = default_generation_config or GenerationConfig()

    @classmethod
    def build(
        cls,
        model_path: str | Path | None = None,
        prompt_for_model: bool = False,
        default_generation_config: GenerationConfig | None = None,
        verbose: bool = False,
    ) -> "EmmaChatService":
        validate_tokenizer()

        generation_config = default_generation_config or GenerationConfig()
        seed_everything(generation_config.seed)

        resolved_model_path = resolve_model_path(model_path=model_path, prompt_for_model=prompt_for_model)
        model = load_chat_model(resolved_model_path, verbose=verbose)
        return cls(model=model, model_path=resolved_model_path, default_generation_config=generation_config)

    def create_session(
        self,
        generation_config: GenerationConfig | None = None,
        conversation_turns: list[str] | None = None,
    ) -> "EmmaChatSession":
        return EmmaChatSession(
            model=self.model,
            generation_config=generation_config or self.default_generation_config,
            conversation_turns=list(conversation_turns or []),
        )


class EmmaChatSession:
    def __init__(
        self,
        model: keras.Model,
        generation_config: GenerationConfig | None = None,
        conversation_turns: list[str] | None = None,
    ):
        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        self.conversation_turns = list(conversation_turns or [])

    def reset(self) -> None:
        self.conversation_turns.clear()

    def chat(self, message: str, stream: bool = False) -> str | Iterator[str]:
        user_message = str(message).strip()
        if not user_message:
            raise ValueError("Message must not be empty.")

        if stream:
            return self._chat_stream(user_message)
        return self._chat_once(user_message)

    def _chat_once(self, user_message: str) -> str:
        seed_everything(self.generation_config.seed)
        turns_for_model = self.conversation_turns + [user_message]
        reply = generate_reply(self.model, turns_for_model, self.generation_config)
        self._append_turns(user_message, reply)
        return reply

    def _chat_stream(self, user_message: str) -> Iterator[str]:
        seed_everything(self.generation_config.seed)
        turns_for_model = self.conversation_turns + [user_message]
        stream = stream_reply(self.model, turns_for_model, self.generation_config)

        while True:
            try:
                chunk = next(stream)
            except StopIteration as stop:
                reply = stop.value or ""
                self._append_turns(user_message, reply)
                return
            else:
                yield chunk

    def _append_turns(self, user_message: str, reply: str) -> None:
        self.conversation_turns.extend([user_message, reply])
        self.conversation_turns = trim_conversation_history(
            self.conversation_turns,
            self.generation_config.max_context_turns,
        )


# ============================================================================
# CLI
# ============================================================================

def run_cli_chat(session: EmmaChatSession, stream: bool = True) -> None:
    print("Chat started.")
    print("Commands:")
    print("  /reset  -> clear conversation history")
    print("  /quit   -> exit")
    print()

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/quit", "quit", "exit"}:
            print("Bye.")
            break

        if lowered == "/reset":
            session.reset()
            print("History cleared.\n")
            continue

        if stream:
            print(f"{CHATBOT_NAME}: ", end="", flush=True)
            for chunk in session.chat(user_text, stream=True):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            reply = session.chat(user_text, stream=False)
            print(f"{CHATBOT_NAME}: {reply}\n")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    service = EmmaChatService.build(prompt_for_model=True, verbose=True)
    session = service.create_session()
    run_cli_chat(session, stream=CLI_STREAM_RESPONSES)


if __name__ == "__main__":
    main()
