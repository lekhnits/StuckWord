import pytest

from stuckword.markers import (
    MarkerKind,
    RecallLabel,
    format_marker,
    markers_to_annotations,
    parse_marker,
)


def test_format_parse_roundtrip():
    text = format_marker(MarkerKind.CUE, 7, "perro")
    marker = parse_marker(text)
    assert (marker.kind, marker.trial, marker.payload) == (MarkerKind.CUE, 7, "perro")


def test_payload_with_separator_is_rejected():
    with pytest.raises(ValueError):
        format_marker(MarkerKind.CUE, 0, "a|b")


def test_annotations_are_labelled_after_the_fact_and_relative_to_t0():
    markers = [
        (100.0, format_marker(MarkerKind.SESSION_START, 0)),
        (101.0, format_marker(MarkerKind.CUE, 0, "dog")),
        (103.5, format_marker(MarkerKind.RETRIEVED, 0, "space")),
        (106.0, format_marker(MarkerKind.LABEL, 0, RecallLabel.TIP_OF_TONGUE.value)),
        (110.0, format_marker(MarkerKind.CUE, 1, "cat")),  # aborted: no label
    ]
    annotations = markers_to_annotations(markers, t0=100.0)
    assert list(annotations.description) == ["cue/tot", "retrieved/tot"]
    assert list(annotations.onset) == pytest.approx([1.0, 3.5])
