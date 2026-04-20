#!/usr/bin/env python3
"""Trace Willow's prompt-to-token pipeline on a few sample prompts."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


def import_willow():
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    original_cwd = Path.cwd()
    os.chdir(BASE_DIR)
    try:
        import willow as willow_module
    finally:
        os.chdir(original_cwd)

    willow_module.CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    willow_module.TOKENIZER_PATH = BASE_DIR / "willow_tokenizer.model"
    return willow_module


willow = import_willow()
np = willow.np
keras = willow.keras
tf = willow.tf


DEFAULT_PROMPTS = [
    "Tell me a short joke about computers.",
    "What is a calm way to handle a stressful day?",
    "Describe a forest after rain in two sentences.",
]

DEFAULT_COMPARISON_TEMPERATURES = [0.3, 1.4]
DEFAULT_DEMO_SEED = 42 if willow.SEED is None else willow.SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Willow checkpoint and trace how it responds to sample prompts.",
    )
    parser.add_argument(
        "--model",
        help="Checkpoint filename or path. Defaults to the newest .keras checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to inspect. Repeat to inspect multiple prompts.",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=len(DEFAULT_PROMPTS),
        help="How many built-in prompts to run when --prompt is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_DEMO_SEED,
        help="Base seed for reproducible sampling.",
    )
    parser.add_argument(
        "--base-temperature",
        type=float,
        default=willow.TEMPERATURE,
        help="Temperature used for the actual demo response.",
    )
    parser.add_argument(
        "--compare-temperature",
        dest="compare_temperatures",
        type=float,
        action="append",
        help=(
            "Extra temperatures to compare at each step. "
            "Repeat to add more values. Defaults to 0.3 and 1.4."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=willow.TOP_K,
        help="Top-k cutoff used during sampling.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="How many candidate tokens to print in each step table.",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=6,
        help="How many generated tokens to inspect in detail before switching to quiet mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum reply length to generate for each prompt.",
    )
    return parser.parse_args()


def banner(title: str, fill: str = "=") -> None:
    print()
    print(fill * 90)
    print(title)
    print(fill * 90)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def visible_text(text: str) -> str:
    if not text:
        return "''"
    return text.encode("unicode_escape").decode("ascii")


def shorten(text: str, width: int) -> str:
    return textwrap.shorten(text, width=width, placeholder="...")


def piece_label(token_id: int) -> str:
    token_id = int(token_id)
    special_map = {
        willow.PAD_ID: "<pad>",
        willow.BOS_ID: "<bos>",
        willow.EOS_ID: "<eos>",
        willow.USER_ID: willow.SPECIAL_USER,
        willow.ASSISTANT_ID: willow.SPECIAL_ASSISTANT,
        willow.SEP_ID: willow.SPECIAL_SEP,
    }
    if token_id in special_map:
        return special_map[token_id]
    return willow.sp.id_to_piece(token_id)


def decode_piece(token_id: int) -> str:
    token_id = int(token_id)
    if token_id in {
        willow.PAD_ID,
        willow.BOS_ID,
        willow.EOS_ID,
        willow.USER_ID,
        willow.ASSISTANT_ID,
        willow.SEP_ID,
    }:
        return piece_label(token_id)
    return willow.sp.decode([token_id])


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    divider_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(divider_line)
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def get_prompt_list(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt
    prompt_count = max(1, min(args.prompt_count, len(DEFAULT_PROMPTS)))
    return DEFAULT_PROMPTS[:prompt_count]


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        requested = Path(model_arg)
        if requested.is_absolute():
            return requested
        if requested.exists():
            return requested.resolve()
        candidate = willow.CHECKPOINT_DIR / requested.name
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Model not found: {model_arg}")

    models = sorted(willow.CHECKPOINT_DIR.glob("*.keras"))
    if not models:
        raise FileNotFoundError(f"No .keras checkpoints found in {willow.CHECKPOINT_DIR}")
    return models[-1].resolve()


def build_probe_model(model) -> tuple[keras.Model, list[tuple[str, str]]]:
    transformer_names = [layer.name for layer in model.layers if layer.name.startswith("transformer_block_")]

    probe_names = ["embed"]
    if transformer_names:
        probe_names.append(transformer_names[0])
        middle_name = transformer_names[len(transformer_names) // 2]
        if middle_name not in probe_names:
            probe_names.append(middle_name)
        if transformer_names[-1] not in probe_names:
            probe_names.append(transformer_names[-1])
    probe_names.append("final_layer_norm")

    labels = {
        "embed": "Token + position embedding",
        "final_layer_norm": "Final layer norm",
    }
    for name in transformer_names:
        block_number = name.rsplit("_", maxsplit=1)[-1]
        labels[name] = f"Transformer block {block_number}"

    probe_outputs = [model.get_layer(name).output for name in probe_names]
    probe_model = keras.Model(inputs=model.inputs, outputs=probe_outputs, name="willow_probe")
    probe_info = [(name, labels[name]) for name in probe_names]
    return probe_model, probe_info


def build_input_window(generated_ids: list[int]) -> tuple[list[int], int, int]:
    window = generated_ids[-willow.MAX_SEQ_LEN:]
    pad_count = max(0, willow.MAX_SEQ_LEN - len(window))
    if pad_count:
        window = window + [willow.PAD_ID] * pad_count
    last_real_idx = max(i for i, token_id in enumerate(window) if token_id != willow.PAD_ID)
    return window, last_real_idx, pad_count


def filter_logits(raw_logits: np.ndarray, reply_ids: list[int]) -> np.ndarray:
    return willow.apply_repetition_controls(raw_logits, reply_ids)


def get_candidate_ids(filtered_logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k is None or top_k <= 0 or top_k >= len(filtered_logits):
        candidate_ids = np.arange(len(filtered_logits))
    else:
        candidate_ids = np.argpartition(filtered_logits, -top_k)[-top_k:]

    candidate_logits = filtered_logits[candidate_ids]
    order = np.argsort(candidate_logits)[::-1]
    return candidate_ids[order]


def compute_probs(filtered_logits: np.ndarray, candidate_ids: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        probs = np.zeros(len(candidate_ids), dtype=np.float64)
        probs[int(np.argmax(filtered_logits[candidate_ids]))] = 1.0
        return probs

    scaled = filtered_logits[candidate_ids] / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    return probs / np.sum(probs)


def pick_from_probs(candidate_ids: np.ndarray, probs: np.ndarray, draw: float) -> tuple[int, float]:
    cumulative = np.cumsum(probs)
    choice_index = int(np.searchsorted(cumulative, draw, side="right"))
    choice_index = min(choice_index, len(candidate_ids) - 1)
    return int(candidate_ids[choice_index]), float(probs[choice_index])


def describe_prompt(prompt: str, prompt_ids: list[int]) -> None:
    section("Part 1 - Prompt Encoding")
    print(f"User prompt: {prompt}")
    print()
    print(
        "Serialized prompt layout: "
        "<bos> + <user> + user text pieces + <sep> + <assistant>"
    )
    print(f"Prompt token count before padding: {len(prompt_ids)}")
    print(f"Prompt token ids: {prompt_ids}")
    print()

    rows = []
    for position, token_id in enumerate(prompt_ids):
        rows.append([
            str(position),
            str(token_id),
            piece_label(token_id),
            visible_text(decode_piece(token_id)),
        ])

    print_table(
        headers=["pos", "id", "piece", "decoded"],
        rows=rows,
    )


def print_architecture_summary(model, probe_info: list[tuple[str, str]]) -> None:
    section("Part 0 - Model Overview")
    transformer_count = sum(1 for layer in model.layers if layer.name.startswith("transformer_block_"))
    embed_layer = model.get_layer("embed")
    print(f"Checkpoint: {model.name}")
    print(f"Input tensor: {model.inputs[0].shape} {model.inputs[0].dtype}")
    print(f"Output tensor: {model.outputs[0].shape} {model.outputs[0].dtype}")
    print(f"Vocabulary size: {willow.sp.vocab_size()}")
    print(f"Hidden width (d_model): {embed_layer.d_model}")
    print(f"Transformer blocks: {transformer_count}")
    print(f"Parameters: {model.count_params():,}")
    print()
    print("High-level pipeline:")
    print("1. Token ids enter as a fixed-length 256-token window.")
    print("2. The embedding layer turns ids into 512-wide vectors and adds learned positions.")
    print("3. Eight causal transformer blocks update each position without seeing future tokens.")
    print("4. Final layer norm stabilizes the hidden state.")
    print("5. The LM head projects each position to 16,000 vocabulary logits.")
    print()
    print("Probe stages used in the trace:")
    for _, label in probe_info:
        print(f"- {label}")


def format_stage_vector(values: np.ndarray, limit: int = 6) -> str:
    slice_values = ", ".join(f"{value:+.3f}" for value in values[:limit])
    return f"[{slice_values}]"


def stage_rows(
    stage_outputs: Iterable[np.ndarray],
    probe_info: list[tuple[str, str]],
    last_real_idx: int,
) -> list[list[str]]:
    rows = []
    for (_, label), stage_output in zip(probe_info, stage_outputs):
        array = np.asarray(stage_output)
        active_slice = array[0, : last_real_idx + 1].astype(np.float64, copy=False)
        finite_values = active_slice[np.isfinite(active_slice)]
        if finite_values.size:
            mean_text = f"{finite_values.mean():+.4f}"
            std_text = f"{finite_values.std():.4f}"
        else:
            mean_text = "n/a"
            std_text = "n/a"
        last_token_vector = array[0, last_real_idx]
        rows.append([
            label,
            str(tuple(array.shape)),
            mean_text,
            std_text,
            format_stage_vector(last_token_vector),
        ])
    return rows


def preview_if_chosen(current_reply_ids: list[int], candidate_id: int) -> str:
    preview_text = willow.sp.decode(current_reply_ids + [candidate_id]).strip()
    if not preview_text:
        preview_text = piece_label(candidate_id)
    return shorten(visible_text(preview_text), 40)


def print_candidate_table(
    current_reply_ids: list[int],
    candidate_ids: np.ndarray,
    filtered_logits: np.ndarray,
    temperatures: list[float],
    top_n: int,
) -> None:
    top_candidates = candidate_ids[:top_n]
    probability_maps = {
        temperature: compute_probs(filtered_logits, candidate_ids, temperature)
        for temperature in temperatures
    }

    headers = ["rank", "id", "piece", "raw_logit"]
    headers.extend(f"p@{temperature:.2f}" for temperature in temperatures)
    headers.append("reply_if_chosen")

    rows = []
    for rank, candidate_id in enumerate(top_candidates, start=1):
        candidate_index = int(np.where(candidate_ids == candidate_id)[0][0])
        row = [
            str(rank),
            str(int(candidate_id)),
            piece_label(int(candidate_id)),
            f"{filtered_logits[int(candidate_id)]:+.4f}",
        ]
        for temperature in temperatures:
            row.append(f"{probability_maps[temperature][candidate_index]:.4f}")
        row.append(preview_if_chosen(current_reply_ids, int(candidate_id)))
        rows.append(row)

    print_table(headers=headers, rows=rows)


def print_sampling_summary(
    candidate_ids: np.ndarray,
    filtered_logits: np.ndarray,
    temperatures: list[float],
    draw: float,
    base_temperature: float,
    current_reply_ids: list[int],
) -> int:
    section("Part 4 - Sampling Decision")
    print("Sampling pool details:")
    print(f"- Shared random draw: {draw:.6f}")
    print(f"- Top-k pool size: {len(candidate_ids)}")
    print("- Banned from sampling: <pad>, <user>, <assistant>")
    print(
        "- Repetition controls: "
        f"penalty={willow.REPETITION_PENALTY}, "
        f"frequency={willow.FREQUENCY_PENALTY}, "
        f"recent={willow.RECENT_TOKEN_PENALTY}x{willow.RECENT_TOKEN_WINDOW}, "
        f"no-repeat-{willow.NO_REPEAT_NGRAM_SIZE}-gram"
    )
    print("- Stop tokens are still allowed: <sep>, <eos>")
    print()

    greedy_probs = compute_probs(filtered_logits, candidate_ids, 0.0)
    greedy_token, _ = pick_from_probs(candidate_ids, greedy_probs, 0.0)
    print(
        "Greedy choice (temperature <= 0): "
        f"{piece_label(greedy_token)} -> {preview_if_chosen(current_reply_ids, greedy_token)}"
    )

    chosen_token = greedy_token
    for temperature in temperatures:
        probs = compute_probs(filtered_logits, candidate_ids, temperature)
        sampled_token, sampled_prob = pick_from_probs(candidate_ids, probs, draw)
        marker = " <-- actual demo choice" if abs(temperature - base_temperature) < 1e-12 else ""
        print(
            f"Temperature {temperature:.2f}: "
            f"{piece_label(sampled_token)} "
            f"(p={sampled_prob:.4f}) -> {preview_if_chosen(current_reply_ids, sampled_token)}"
            f"{marker}"
        )
        if abs(temperature - base_temperature) < 1e-12:
            chosen_token = sampled_token

    return chosen_token


def trace_prompt(
    model,
    probe_model,
    probe_info: list[tuple[str, str]],
    prompt: str,
    args: argparse.Namespace,
    prompt_index: int,
) -> None:
    banner(f"Prompt {prompt_index + 1}: {prompt}")
    prompt_ids = willow.build_prompt_ids([prompt])
    describe_prompt(prompt, prompt_ids)

    generated_ids = prompt_ids[:]
    reply_ids: list[int] = []
    temperatures = build_temperature_list(args.base_temperature, args.compare_temperatures)
    rng = np.random.default_rng(args.seed + prompt_index)
    quiet_tokens = 0

    for step in range(1, args.max_new_tokens + 1):
        window, last_real_idx, pad_count = build_input_window(generated_ids)
        input_tensor = np.array([window], dtype=np.int32)
        stage_outputs = probe_model({"input_ids": input_tensor}, training=False)
        logits = model({"input_ids": input_tensor}, training=False)[0].numpy()
        next_logits = logits[last_real_idx]
        filtered_logits = filter_logits(next_logits, reply_ids)
        candidate_ids = get_candidate_ids(filtered_logits, args.top_k)
        draw = float(rng.random())
        chosen_token = int(
            pick_from_probs(
                candidate_ids,
                compute_probs(filtered_logits, candidate_ids, args.base_temperature),
                draw,
            )[0]
        )

        if step <= args.trace_steps:
            section(f"Part 2 - Decode Step {step}")
            print(f"Current decoded reply: {visible_text(willow.sp.decode(reply_ids).strip())}")
            print(f"Context tokens used: {last_real_idx + 1}/{willow.MAX_SEQ_LEN}")
            print(f"Padding added: {pad_count}")
            print(
                "Active prediction position: "
                f"{last_real_idx} ({piece_label(window[last_real_idx])})"
            )
            print(
                "Causal mask meaning: this position can only attend to earlier real tokens, "
                "never to future ones."
            )
            print()

            section("Part 3 - Forward Pass Snapshot")
            print_table(
                headers=["stage", "shape", "mean", "std", "last_token[:6]"],
                rows=stage_rows(stage_outputs, probe_info, last_real_idx),
            )
            print()
            print(f"Logits tensor shape: {tuple(logits.shape)}")
            print(
                f"Next-token logits vector shape: {tuple(next_logits.shape)} "
                f"at position {last_real_idx}"
            )
            print()
            print("Top visible candidates after repetition controls and banned-token filtering:")
            print_candidate_table(
                current_reply_ids=reply_ids,
                candidate_ids=candidate_ids,
                filtered_logits=filtered_logits,
                temperatures=temperatures,
                top_n=args.top_n,
            )
            print()
            chosen_token = print_sampling_summary(
                candidate_ids=candidate_ids,
                filtered_logits=filtered_logits,
                temperatures=temperatures,
                draw=draw,
                base_temperature=args.base_temperature,
                current_reply_ids=reply_ids,
            )
        else:
            quiet_tokens += 1

        if chosen_token in (willow.EOS_ID, willow.SEP_ID):
            stop_label = piece_label(chosen_token)
            if step <= args.trace_steps:
                print()
                print(f"Stop token reached: {stop_label}")
            else:
                print()
                print(
                    f"Quiet decoding stopped after {quiet_tokens} extra token(s) "
                    f"because the model emitted {stop_label}."
                )
            break

        generated_ids.append(chosen_token)
        reply_ids.append(chosen_token)

    if quiet_tokens:
        print()
        print(f"Quiet decoding generated {quiet_tokens} additional token(s) after the trace limit.")

    section("Final Reply")
    final_reply = willow.sp.decode(reply_ids).strip()
    print(final_reply or "<empty reply>")


def build_temperature_list(base_temperature: float, compare_temperatures: list[float] | None) -> list[float]:
    temperatures: list[float] = []
    for value in [base_temperature, *(compare_temperatures or DEFAULT_COMPARISON_TEMPERATURES)]:
        if value <= 0:
            continue
        if value not in temperatures:
            temperatures.append(value)
    if not temperatures:
        temperatures.append(1.0)
    return temperatures


def main() -> None:
    args = parse_args()
    willow.validate_special_tokens()
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    model_path = resolve_model_path(args.model)
    banner("Willow Pipeline Demo")
    print(f"Base directory: {BASE_DIR}")
    print(f"Tokenizer: {willow.TOKENIZER_PATH}")
    print(f"Checkpoint: {model_path}")
    print(f"Seed: {args.seed if args.seed is not None else 'unseeded'}")
    print(f"Base temperature: {args.base_temperature}")
    print(f"Comparison temperatures: {build_temperature_list(args.base_temperature, args.compare_temperatures)}")
    print(f"Top-k: {args.top_k}")
    print(f"Trace steps per prompt: {args.trace_steps}")
    print(f"Max new tokens per prompt: {args.max_new_tokens}")
    print()
    print(
        "Special tokens: "
        f"PAD={willow.PAD_ID}, BOS={willow.BOS_ID}, EOS={willow.EOS_ID}, "
        f"{willow.SPECIAL_USER}={willow.USER_ID}, "
        f"{willow.SPECIAL_ASSISTANT}={willow.ASSISTANT_ID}, "
        f"{willow.SPECIAL_SEP}={willow.SEP_ID}"
    )

    model = willow.load_chat_model(model_path)
    probe_model, probe_info = build_probe_model(model)
    print_architecture_summary(model, probe_info)

    for prompt_index, prompt in enumerate(get_prompt_list(args)):
        trace_prompt(
            model=model,
            probe_model=probe_model,
            probe_info=probe_info,
            prompt=prompt,
            args=args,
            prompt_index=prompt_index,
        )


if __name__ == "__main__":
    main()
