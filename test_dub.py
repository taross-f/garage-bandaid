import numpy as np
import pytest

import camera
import dub
from test_camera import SR, bursts


def test_to_bed_clock_places_song_on_the_recording_clock():
    after = np.stack([bursts(20), bursts(20, seed=1)], axis=1)
    start, slope = 48000 * 30 + 0.37, -15.5e-6
    first, piece = dub.to_bed_clock(after, start, slope)
    # 置いた音を camera.py と同じ向きで song.wav の時間に戻すと、元の after に戻る
    back = camera.warp(piece, start - first, 1 + slope, len(after))
    inner = slice(SR, -SR)
    assert np.max(np.abs(back[inner] - after[inner])) < 1e-3 * np.max(np.abs(after))
    assert first == int(np.ceil(start))
    assert len(piece) == pytest.approx(len(after) * (1 + slope), abs=2)


def test_splice_keeps_bed_outside_and_song_inside():
    bed = np.ones((10 * SR, 2))
    piece = np.full((4 * SR, 2), 0.5)
    out = dub.splice(bed, [(3 * SR, piece)], -6.0, SR)
    gain = 10 ** (-6 / 20)
    assert np.allclose(out[: 3 * SR], gain)
    assert np.allclose(out[7 * SR:], gain)
    fade = int(dub.CROSSFADE_SEC * SR)
    assert np.allclose(out[3 * SR + fade: 7 * SR - fade], 0.5)


def test_splice_rejects_too_short_section():
    with pytest.raises(ValueError):
        dub.splice(np.ones((SR * 2, 2)), [(0, np.ones((SR // 2, 2)))], 0.0, SR)


def test_dub_refuses_to_overwrite_source(tmp_path):
    video = tmp_path / "live.MP4"
    with pytest.raises(SystemExit):
        dub.dub(video, tmp_path / "live.wav", video)
