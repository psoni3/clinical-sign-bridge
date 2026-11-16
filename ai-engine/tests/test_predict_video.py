import json
import sys

import pytest

import predict_video


def test_load_label_mapping_returns_checkpoint_order(tmp_path) -> None:
    mapping_path = tmp_path / "labels.json"
    mapping_path.write_text(
        json.dumps({"SECOND": 1, "FIRST": 0, "THIRD": 2, "FOURTH": 3, "FIFTH": 4}),
        encoding="utf-8",
    )

    labels = predict_video.load_label_mapping(
        mapping_path,
        ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH"],
    )

    assert labels == ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH"]


def test_load_label_mapping_rejects_order_mismatch(tmp_path) -> None:
    mapping_path = tmp_path / "labels.json"
    mapping_path.write_text(
        json.dumps({"SECOND": 1, "FIRST": 0, "THIRD": 2, "FOURTH": 3, "FIFTH": 4}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="order does not match"):
        predict_video.load_label_mapping(
            mapping_path,
            ["SECOND", "FIRST", "THIRD", "FOURTH", "FIFTH"],
        )


def test_main_prints_filename_frame_count_and_ranked_predictions(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    video_path = tmp_path / "input clip.mp4"
    video_path.touch()
    monkeypatch.setattr(
        predict_video,
        "predict",
        lambda _: (12, [("SIGN", 0.75), ("OTHER", 0.25)]),
    )
    monkeypatch.setattr(sys, "argv", ["predict_video.py", str(video_path)])

    assert predict_video.main() == 0

    output = capsys.readouterr().out
    assert "Input: input clip.mp4" in output
    assert "Frames: 12" in output
    assert "1. SIGN: 0.750000" in output
    assert "2. OTHER: 0.250000" in output
