# garage-bandaid

English | [日本語](README.ja.md)

First aid for live recordings made from the audience.

garage-bandaid cuts songs out of a live recording and splits each into four stems (vocals, drums, bass, other) with [demucs](https://github.com/facebookresearch/demucs) (htdemucs). Each stem is then re-processed and mixed back together, and the result is brought to -14 LUFS with a true-peak ceiling of -1 dBTP.

A local web UI lets you play several mix versions in sync and switch between them instantly without losing your place.

## Setup

Requirements:

- [uv](https://docs.astral.sh/uv/)
- ffmpeg (used to cut the recording and resample to 48 kHz)

```bash
uv sync
```

This creates `.venv` with Python 3.11 and the locked dependencies. Intel Macs are not supported (torch 2.5.1 has no wheels for them).

Stem separation runs on the CPU and takes about 4–8 minutes for a 4-minute song.

## Using the web UI (recommended)

```bash
uv run studio.py
```

This opens `http://127.0.0.1:8765`. The page is only a front end: all processing runs on your machine, and no audio leaves it.

1. **Add a song**: click "＋ 曲を追加" and pick a recording (wav / flac / aiff) from the project folder. Give a start and end time in seconds, or leave them empty to use the whole file. 3 seconds of padding are added on each side.
2. **Make a version**: change values in the editor and click "レンダリング". Every parameter shows its meaning and a typical range.
3. **Compare**: play the versions you have checked in sync.

   | Key | Action |
   |---|---|
   | `Space` | Play / stop |
   | `0` | before (the original, loudness-matched only) |
   | `1`–`9` | Checked versions |
   | `←` `→` | Seek 5 s |

   Drag on the waveform to loop a section.
4. **Adopt**: write the version's settings back to the song's `mix.yaml` (comments are kept) and re-render `out/`.

To reuse settings across songs, "初期値に" makes a version the default for newly added songs, and "初期値を読み込む" loads that default into the current song's editor.

The UI labels are in Japanese.

## Command line

```bash
# Cut, diagnose and separate (once per song)
uv run prep.py "recordings/live.wav" --name my_song --start 612 --end 845

# Edit songs/my_song/mix.yaml, then re-mix (separation is not repeated)
uv run mix.py my_song

# Re-mix every song
uv run mix.py --all
```

## Output

Everything for a song goes under `songs/<name>/`.

| Path | Contents |
|---|---|
| `out/after.wav` | Final mix, 48 kHz / 24-bit. Same length and start as `work/song.wav`, so it lines up with video |
| `out/before.wav` | The original, loudness-matched only (for comparison) |
| `out/report.json` | Before / after measurements |
| `mix.yaml` | Mix settings, with a comment on every parameter |
| `work/song.wav` | The cut from the original recording |
| `work/stems/` | The four separated stems |
| `work/diagnosis.json` | Clipping ratio, loudness, true peak, band balance |
| `versions/` | Versions rendered from the web UI |

## Processing

1. Per stem: zero-phase high-pass filter → peaking / low-shelf EQ → compressor → gain.
2. A little of the original recording is blended back in (`source_blend_db`) to mask separation artifacts.
3. Master: 30 Hz high-pass → EQ → bus compressor.
4. Loudness normalization to -14 LUFS, then a 4× oversampled true-peak limiter at -1 dBTP.

## Tests

```bash
uv run pytest -q
```
