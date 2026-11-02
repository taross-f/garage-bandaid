import numpy as np

from audio_metrics import clip_runs, integrated_lufs, true_peak_dbfs
import pytest

from mix import apply_board, eq_plugins, hpf, normalize_master, rms_db, true_peak_limit
import prep
from prep import padded_range
from songs import slugify

SR = 48000


def noise_bursts(seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.15, (int(SR * seconds), 1))
    x[SR:SR + 2000] *= 5
    return x


def test_limiter_respects_ceiling_and_keeps_alignment():
    x = noise_bursts()
    out = true_peak_limit(x, SR, -1.0, 2.0, 80.0)
    assert out.shape == x.shape
    assert true_peak_dbfs(out) <= -0.95
    quiet = slice(3 * SR, 5 * SR)
    assert np.allclose(out[quiet], x[quiet], atol=1e-3)


def test_normalize_master_hits_targets():
    cfg = {"target_lufs": -14.0, "true_peak_dbtp": -1.0,
           "limiter_lookahead_ms": 2.0, "limiter_release_ms": 80.0}
    out, _ = normalize_master(noise_bursts(), SR, cfg)
    assert abs(integrated_lufs(out, SR) + 14.0) <= 0.05
    assert true_peak_dbfs(out) <= -1.0


def test_hpf_is_zero_phase():
    impulse = np.zeros((SR, 1))
    impulse[SR // 2] = 1.0
    out = hpf(impulse, SR, 100)
    assert int(np.argmax(np.abs(out))) == SR // 2


def test_low_shelf_cuts_only_below_freq():
    t = np.arange(SR) / SR
    gains = {}
    for freq in (60, 2000):
        tone = 0.1 * np.sin(2 * np.pi * freq * t)[:, None]
        out = apply_board(tone, SR, eq_plugins(
            [{"type": "low_shelf", "freq": 200, "gain_db": -6.0, "q": 0.7}]))
        gains[freq] = rms_db(out) - rms_db(tone)
    assert gains[60] < -4.5
    assert abs(gains[2000]) < 0.2


def test_clip_runs_counts_only_long_runs():
    x = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=float)
    result = clip_runs(x, 1.0, min_run=3)
    assert result["runs"] == 1
    assert result["clipped_samples"] == 3


def test_padded_range_adds_margin_within_file():
    assert padded_range(10.0, 20.0, 100.0) == (7.0, 23.0)
    assert padded_range(1.0, 99.0, 100.0) == (0.0, 100.0)
    assert padded_range(None, None, 100.0) == (0.0, 100.0)
    with pytest.raises(ValueError):
        padded_range(50.0, 40.0, 100.0)


def test_slugify_makes_shell_friendly_name():
    assert slugify("20261031_my song") == "20261031_my_song"
    with pytest.raises(ValueError):
        slugify("  ")


def test_new_song_removes_partial_work_on_failure(tmp_path, monkeypatch):
    import songs
    monkeypatch.setattr(songs, "SONGS_DIR", tmp_path / "songs")
    source = tmp_path / "take.wav"
    source.write_bytes(b"")

    def failing(source, song, start, end, progress):
        song.stems.mkdir(parents=True)
        raise RuntimeError("demucs failed")

    monkeypatch.setattr(prep, "prepare", failing)
    with pytest.raises(RuntimeError):
        prep.new_song(source, "take", None, None)
    assert not (tmp_path / "songs" / "take" / "work").exists()
