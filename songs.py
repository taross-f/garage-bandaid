"""曲ごとのフォルダ構成。songs/<name>/ に mix.yaml・work/・out/ を置く。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
SONGS_DIR = ROOT / "songs"
TEMPLATE_YAML = ROOT / "mix.template.yaml"
STEM_NAMES = ("vocals", "drums", "bass", "other")


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w-]+", "_", text).strip("_")
    if not slug:
        raise ValueError(f"cannot make a song name from {text!r}")
    return slug


@dataclass(frozen=True)
class Song:
    name: str

    @property
    def dir(self) -> Path:
        return SONGS_DIR / self.name

    @property
    def config(self) -> Path:
        return self.dir / "mix.yaml"

    @property
    def work(self) -> Path:
        return self.dir / "work"

    @property
    def song_wav(self) -> Path:
        return self.work / "song.wav"

    @property
    def stems(self) -> Path:
        return self.work / "stems"

    @property
    def out(self) -> Path:
        return self.dir / "out"

    @property
    def versions(self) -> Path:
        return self.dir / "versions"


def prepared_songs() -> list[Song]:
    """分離まで済んでいる曲を名前順に返す。diagnosis.json は prep の最後に書かれる。"""
    return [Song(p.parent.parent.name) for p in sorted(SONGS_DIR.glob("*/work/diagnosis.json"))]
