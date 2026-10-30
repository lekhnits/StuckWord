"""Command-line entry point: ``stuckword <command>``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np

from stuckword import __version__
from stuckword.channels import forward_select_channels, rank_channels
from stuckword.config import EpochConfig
from stuckword.datasets import DATASETS, DEFAULT_DATA_DIR, download_openneuro
from stuckword.model import build_classifier, evaluate_classifier
from stuckword.preprocessing import binary_labels, make_recall_epochs, preprocess_raw

logger = logging.getLogger("stuckword")


def _report(epochs: mne.BaseEpochs, groups: np.ndarray, args: argparse.Namespace) -> None:
    """Classifier, channel ranking and channel selection for recorded sessions."""
    data, y, sfreq = epochs.get_data(copy=False), binary_labels(epochs), epochs.info["sfreq"]
    ch_names = epochs.ch_names
    print(f"{len(y)} epochs: {int(y.sum())} tip-of-the-tongue, {int((y == 0).sum())} knew")

    result = evaluate_classifier(
        build_classifier(sfreq, ch_names=ch_names), data, y, groups,
        n_permutations=args.permutations,
    )  # fmt: skip
    print("All channels ->", result.summary())

    for band in ("theta", "alpha"):
        print(f"\nTop channels by {band} AUC:")
        print(rank_channels(data, y, sfreq, ch_names, band=band).head(6).to_string(index=False))

    selection = forward_select_channels(
        data, y, sfreq, ch_names, groups, max_channels=args.max_channels
    )
    print("\nGreedy channel subset (optimistic - confirm on a new session):")
    print(selection.to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        selection.to_csv(args.out / "channel_selection.csv", index=False)
        print(f"\nSaved results to {args.out}")


def cmd_datasets(_: argparse.Namespace) -> None:
    for info in DATASETS.values():
        print(f"{info.key:<18} {info.name}\n{'':<18} {info.modality}\n{'':<18} -> {info.use_for}")
        print(f"{'':<18} {info.url}\n")


def cmd_download(args: argparse.Namespace) -> None:
    path = download_openneuro(args.key, args.target, include=args.include)
    print(f"Downloaded to {path}")


def cmd_sanity_eegbci(args: argparse.Namespace) -> None:
    from stuckword.sanity import eegbci_alpha_check, is_alpha_reactive

    ratios = eegbci_alpha_check(args.subject)
    print(ratios.round(2).to_string())
    print("Alpha reactive:", is_alpha_reactive(ratios))


def cmd_analyze(args: argparse.Namespace) -> None:
    from stuckword.acquisition.xdf import load_xdf_session

    window = EpochConfig(lock=args.lock, tmin=args.tmin, tmax=args.tmax)
    all_epochs, groups = [], []
    for session, path in enumerate(args.files):
        epochs = make_recall_epochs(preprocess_raw(load_xdf_session(path)), window)
        all_epochs.append(epochs)
        groups.append(np.full(len(epochs), session))
    _report(mne.concatenate_epochs(all_epochs, verbose="error"), np.concatenate(groups), args)


def cmd_flashcards(args: argparse.Namespace) -> None:
    from stuckword.acquisition.flashcards import (
        FlashcardApp,
        LslMarkerSink,
        PrintMarkerSink,
        load_word_list,
        shuffled,
    )

    cards = shuffled(load_word_list(args.words), seed=args.seed)[: args.n]
    sink = PrintMarkerSink() if args.dry_run else LslMarkerSink()
    FlashcardApp(cards, sink, args.log).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stuckword", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="List relevant public datasets").set_defaults(func=cmd_datasets)

    command = sub.add_parser("download", help="Download an OpenNeuro dataset")
    command.add_argument(
        "key", choices=[key for key, info in DATASETS.items() if info.openneuro_id]
    )
    command.add_argument("--target", type=Path, default=DEFAULT_DATA_DIR)
    command.add_argument("--include", nargs="*", help='Glob filters, e.g. "sub-01/*"')
    command.set_defaults(func=cmd_download)

    command = sub.add_parser("sanity-eegbci", help="Eyes-closed alpha check on public data")
    command.add_argument("--subject", type=int, default=1)
    command.set_defaults(func=cmd_sanity_eegbci)

    def add_report_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--permutations", type=int, default=0, help="Permutation test size")
        command.add_argument("--max-channels", type=int, default=4)
        command.add_argument("--out", type=Path, help="Directory for result tables")

    command = sub.add_parser("analyze", help="Analyse recorded XDF sessions (one file per session)")
    command.add_argument("files", type=Path, nargs="+")
    command.add_argument("--lock", choices=["retrieved", "cue"], default="retrieved")
    command.add_argument("--tmin", type=float, default=-2.0)
    command.add_argument("--tmax", type=float, default=0.0)
    add_report_args(command)
    command.set_defaults(func=cmd_analyze)

    command = sub.add_parser("flashcards", help="Run the recall task (streams LSL markers)")
    command.add_argument("--words", type=Path, required=True, help="CSV with prompt,answer columns")
    command.add_argument("--log", type=Path, required=True, help="Per-trial CSV log")
    command.add_argument("-n", type=int, default=50, help="Trials in this block")
    command.add_argument("--seed", type=int)
    command.add_argument("--dry-run", action="store_true", help="Print markers instead of LSL")
    command.set_defaults(func=cmd_flashcards)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    mne.set_log_level("INFO" if args.verbose else "WARNING")
    args.func(args)


if __name__ == "__main__":
    main()
