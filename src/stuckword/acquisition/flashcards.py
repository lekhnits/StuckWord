"""Flashcard recall task that emits LSL markers (days 3-4) and optionally gives live hints (6-7).

Trial structure (one deviation from the original plan, to keep speech out of the analysis window):

1. Fixation cross (1 s).
2. Prompt word appears                         -> ``cue`` marker.
3. Press SPACE the moment the word comes to mind,
   or when giving up (timeout after 15 s)      -> ``retrieved`` marker.
4. Say the translation aloud, then the answer is shown.
5. Rate: 1 = knew it straight away, 2 = tip of the tongue, 3 = no idea  -> ``label`` marker.

Pressing SPACE *before* speaking gives a clean end-of-retrieval timestamp, so the window
``[-2 s, 0]`` before it contains effortful search but no articulation EMG.
"""

from __future__ import annotations

import csv
import random
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

from stuckword.markers import KEY_TO_LABEL, MarkerKind, RecallLabel, format_marker

FIXATION_MS = 1000
RETRIEVAL_TIMEOUT_MS = 15_000
ANSWER_DELAY_MS = 1500
ITI_MS = 800
HINT_POLL_MS = 100


@dataclass(frozen=True)
class Card:
    prompt: str
    answer: str


def load_word_list(path: Path | str) -> list[Card]:
    """CSV with ``prompt`` and ``answer`` columns (UTF-8)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [Card(row["prompt"].strip(), row["answer"].strip()) for row in csv.DictReader(fh)]


class MarkerSink(Protocol):
    def push(self, text: str) -> None: ...
    def clock(self) -> float: ...


class LslMarkerSink:
    """Pushes string markers to an LSL outlet that LabRecorder can record."""

    def __init__(self, name: str = "StuckWordMarkers") -> None:
        import pylsl

        self._pylsl = pylsl
        info = pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, "string", f"{name}-1")
        self._outlet = pylsl.StreamOutlet(info)

    def push(self, text: str) -> None:
        self._outlet.push_sample([text])

    def clock(self) -> float:
        return float(self._pylsl.local_clock())


class PrintMarkerSink:
    """Dry-run sink: prints markers; useful for rehearsing the task without hardware."""

    def push(self, text: str) -> None:
        print(f"[{self.clock():10.3f}] {text}")

    def clock(self) -> float:
        return time.monotonic()


class HintProvider(Protocol):
    def reset(self) -> None: ...
    def should_hint(self) -> bool: ...


@dataclass
class TrialRecord:
    trial: int
    prompt: str
    answer: str
    cue_time: float
    retrieved_time: float | None = None
    timed_out: bool = False
    hinted: bool = False
    label: str = ""


class _Phase(Enum):
    FIXATION = auto()
    RETRIEVAL = auto()
    ANSWER = auto()
    RATING = auto()
    DONE = auto()


class FlashcardApp:
    """Minimal full-screen Tk task. All timing goes through ``sink.clock()``."""

    def __init__(
        self,
        cards: list[Card],
        sink: MarkerSink,
        log_path: Path,
        hint_provider: HintProvider | None = None,
        on_finish: Callable[[list[TrialRecord]], None] | None = None,
    ) -> None:
        self.cards = cards
        self.sink = sink
        self.log_path = log_path
        self.hint_provider = hint_provider
        self.on_finish = on_finish
        self.records: list[TrialRecord] = []
        self._phase = _Phase.FIXATION
        self._trial = -1
        self._timeout_job: str | None = None

        self.root = tk.Tk()
        self.root.title("StuckWord")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self._label = tk.Label(self.root, fg="white", bg="black", font=("Helvetica", 64))
        self._label.pack(expand=True)
        self._hint = tk.Label(self.root, fg="gray70", bg="black", font=("Helvetica", 40))
        self._hint.pack(pady=40)
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Escape>", lambda _event: self._finish())

    def run(self) -> None:
        self.sink.push(format_marker(MarkerKind.SESSION_START, 0))
        self.root.after(500, self._next_trial)
        self.root.mainloop()

    # --- trial flow -----------------------------------------------------------------------
    def _next_trial(self) -> None:
        self._trial += 1
        if self._trial >= len(self.cards):
            self._finish()
            return
        self._phase = _Phase.FIXATION
        self._show("+")
        self.root.after(FIXATION_MS, self._show_cue)

    def _show_cue(self) -> None:
        card = self.cards[self._trial]
        self._phase = _Phase.RETRIEVAL
        self._show(card.prompt)
        self.sink.push(format_marker(MarkerKind.CUE, self._trial, card.prompt))
        self.records.append(TrialRecord(self._trial, card.prompt, card.answer, self.sink.clock()))
        self._timeout_job = self.root.after(RETRIEVAL_TIMEOUT_MS, self._end_retrieval, True)
        if self.hint_provider is not None:
            self.hint_provider.reset()
            self.root.after(HINT_POLL_MS, self._poll_hint)

    def _poll_hint(self) -> None:
        record = self.records[-1]
        if self._phase is not _Phase.RETRIEVAL or record.hinted or self.hint_provider is None:
            return
        if self.hint_provider.should_hint():
            first_letter = record.answer[:1]
            record.hinted = True
            self._hint.config(text=f"{first_letter}…")
            self.sink.push(format_marker(MarkerKind.HINT, self._trial, first_letter))
        else:
            self.root.after(HINT_POLL_MS, self._poll_hint)

    def _end_retrieval(self, timed_out: bool = False) -> None:
        if self._phase is not _Phase.RETRIEVAL:
            return
        if self._timeout_job is not None:
            self.root.after_cancel(self._timeout_job)
            self._timeout_job = None
        record = self.records[-1]
        record.retrieved_time, record.timed_out = self.sink.clock(), timed_out
        payload = "timeout" if timed_out else "space"
        self.sink.push(format_marker(MarkerKind.RETRIEVED, self._trial, payload))
        self._phase = _Phase.ANSWER
        self._show("say it aloud")
        self.root.after(ANSWER_DELAY_MS, self._show_answer)

    def _show_answer(self) -> None:
        card = self.cards[self._trial]
        self._phase = _Phase.RATING
        self._show(f"{card.prompt} → {card.answer}")
        self._hint.config(text="1 knew it   2 tip of the tongue   3 no idea")

    def _rate(self, label: RecallLabel) -> None:
        self.records[-1].label = label.value
        self.sink.push(format_marker(MarkerKind.LABEL, self._trial, label.value))
        self._append_log(self.records[-1])
        self._show("")
        self.root.after(ITI_MS, self._next_trial)

    def _on_key(self, event: tk.Event) -> None:
        if self._phase is _Phase.RETRIEVAL and event.keysym == "space":
            self._end_retrieval()
        elif self._phase is _Phase.RATING and event.char in KEY_TO_LABEL:
            self._rate(KEY_TO_LABEL[event.char])

    # --- helpers --------------------------------------------------------------------------
    def _show(self, text: str) -> None:
        self._label.config(text=text)
        self._hint.config(text="")

    def _append_log(self, record: TrialRecord) -> None:
        """Write each trial immediately, so a crash never loses a session."""
        is_new = not self.log_path.exists()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(TrialRecord)])
            if is_new:
                writer.writeheader()
            writer.writerow(asdict(record))

    def _finish(self) -> None:
        if self._phase is _Phase.DONE:
            return
        self._phase = _Phase.DONE
        self.sink.push(format_marker(MarkerKind.SESSION_END, max(self._trial, 0)))
        self.root.destroy()
        if self.on_finish is not None:
            self.on_finish(self.records)


def shuffled(cards: list[Card], seed: int | None = None) -> list[Card]:
    cards = list(cards)
    random.Random(seed).shuffle(cards)
    return cards
