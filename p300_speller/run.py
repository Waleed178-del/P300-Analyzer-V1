#!/usr/bin/env python3
"""run.py — command-line entry point for the P300 Speller.

Usage::

    python run.py train  [--config CONFIG] [--sequences N] [--words W [W ...]]
    python run.py spell  [--config CONFIG] [--sequences N] [--chars N]
                         [--simulate-intent TEXT]
    python run.py selftest [--config CONFIG]

Examples::

    # Calibrate on the configured words, then free-spell interactively.
    python run.py train
    python run.py spell

    # Fast headless end-to-end check on the synthetic Simulator.
    python run.py selftest

The script keeps almost no logic of its own: it parses arguments, loads the
configuration, applies a few command-line overrides, and delegates to
:class:`main_pipeline.P300Pipeline`.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``src/`` importable regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from main_pipeline import P300Pipeline, load_config  # noqa: E402

DEFAULT_CONFIG = os.path.join(_HERE, "configs", "config.yaml")


def _resolve_paths(config: dict) -> dict:
    """Make relative model/output paths absolute against the project root."""
    config["classifier"]["model_path"] = os.path.join(
        _HERE, config["classifier"]["model_path"]
    )
    out = config["session"].get("output_text_path")
    if out:
        config["session"]["output_text_path"] = os.path.join(_HERE, out)
    return config


def cmd_train(args: argparse.Namespace) -> int:
    """Run calibration and persist a trained model."""
    config = _resolve_paths(load_config(args.config))
    pipeline = P300Pipeline(config)
    report = pipeline.train(words=args.words, n_sequences=args.sequences)
    print("[train] complete:", report)
    print(f"[train] model saved to {config['classifier']['model_path']}")
    return 0


def cmd_spell(args: argparse.Namespace) -> int:
    """Free-spell text with a previously trained model."""
    config = _resolve_paths(load_config(args.config))
    if not os.path.exists(config["classifier"]["model_path"]):
        print(
            "[spell] no trained model found; run `python run.py train` first.",
            file=sys.stderr,
        )
        return 2
    pipeline = P300Pipeline(config)
    text = pipeline.spell(
        n_characters=args.chars,
        n_sequences=args.sequences,
        simulated_intent=args.simulate_intent,
    )
    print(f"[spell] decoded text: {text!r}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Headless end-to-end smoke test on the Simulator.

    Forces simulator + headless mode, trains on a short word with few sequences,
    then spells a known intent and reports decoding accuracy. Useful for CI and
    for validating the install without hardware or a display.
    """
    config = _resolve_paths(load_config(args.config))
    config["acquisition"]["use_simulator"] = True
    config["speller"]["headless"] = True
    config["speller"]["inter_character_pause_s"] = 0.1
    fast_sequences = 8

    pipeline = P300Pipeline(config)
    print("[selftest] training on 'CAT' ...")
    report = pipeline.train(words=["CAT"], n_sequences=fast_sequences)
    print("[selftest] training:", report)

    intent = "CAT"
    print(f"[selftest] spelling intent {intent!r} ...")
    text = pipeline.spell(
        n_characters=len(intent),
        n_sequences=fast_sequences,
        simulated_intent=intent,
    )
    correct = sum(a == b for a, b in zip(text, intent))
    print(f"[selftest] decoded {text!r} ({correct}/{len(intent)} correct)")
    return 0 if correct == len(intent) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description="P300 Speller — communication prosthesis"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="path to config.yaml"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="calibrate on known words")
    p_train.add_argument("--sequences", type=int, default=None,
                         help="override sequences per character")
    p_train.add_argument("--words", nargs="+", default=None,
                         help="override calibration words")
    p_train.set_defaults(func=cmd_train)

    p_spell = sub.add_parser("spell", help="free-text spelling")
    p_spell.add_argument("--sequences", type=int, default=None,
                         help="override sequences per character")
    p_spell.add_argument("--chars", type=int, default=None,
                         help="stop after N characters")
    p_spell.add_argument("--simulate-intent", default=None,
                         help="(simulator only) text a synthetic user intends")
    p_spell.set_defaults(func=cmd_spell)

    p_self = sub.add_parser("selftest", help="headless end-to-end smoke test")
    p_self.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
