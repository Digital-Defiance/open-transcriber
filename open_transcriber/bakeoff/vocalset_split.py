"""Annotated-VocalSet singer-disjoint split: 12 train / 4 val / 4 held-out audit (CC BY 4.0).

The split is keyed by singer id and chosen deterministically by stable SHA-256 ordering with the project's
split seed text, so it reproduces exactly and never overlaps a singer across splits. No audio is downloaded or
committed; this emits a manifest of singer ids per split that Phase C freezes.

VocalSet is CC BY 4.0: singing may be used for probe train/validation under its terms (recorded in the task
folder's licence notes). The held-out audit singers are never trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Stable seed text used across the bake-off's deterministic selections (report §3.1).
SEED_TEXT = "copyist-ot2-20261020"
#: VocalSet's 20 singers (11 male, 9 female), by the dataset's ids. Singer ids only; no audio.
VOCALSET_SINGERS = tuple(
    [f"male{i}" for i in range(1, 12)] + [f"female{i}" for i in range(1, 10)]
)
SPLIT_SIZES = {"train": 12, "val": 4, "audit": 4}


def _rank(singer: str) -> str:
    """A stable sort key: SHA-256 of the seed text and singer id."""
    return hashlib.sha256(f"{SEED_TEXT}:{singer}".encode("utf-8")).hexdigest()


@dataclass
class VocalSetSplit:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    audit: list[str] = field(default_factory=list)

    def all_singers(self) -> list[str]:
        return self.train + self.val + self.audit

    def to_dict(self) -> dict:
        return {"seed_text": SEED_TEXT, "licence": "CC BY 4.0",
                "train": self.train, "val": self.val, "audit": self.audit}


def build_split(singers: tuple[str, ...] = VOCALSET_SINGERS,
                sizes: dict[str, int] = SPLIT_SIZES) -> VocalSetSplit:
    """Deterministic, singer-disjoint split by stable SHA-256 ordering of the singer ids."""
    total = sum(sizes.values())
    if len(set(singers)) < total:
        raise ValueError(f"need at least {total} distinct singers, have {len(set(singers))}")
    ordered = sorted(set(singers), key=_rank)
    train = ordered[:sizes["train"]]
    val = ordered[sizes["train"]:sizes["train"] + sizes["val"]]
    audit = ordered[sizes["train"] + sizes["val"]:sizes["train"] + sizes["val"] + sizes["audit"]]
    split = VocalSetSplit(train=train, val=val, audit=audit)
    # disjointness guard
    groups = [set(train), set(val), set(audit)]
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            if groups[a] & groups[b]:
                raise ValueError("singer overlap across splits")
    return split


def write_manifest(split: VocalSetSplit, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the split manifest here")
    args = parser.parse_args(argv)
    split = build_split()
    print(f"train ({len(split.train)}): {', '.join(split.train)}")
    print(f"val   ({len(split.val)}): {', '.join(split.val)}")
    print(f"audit ({len(split.audit)}): {', '.join(split.audit)}")
    if args.out:
        print("wrote", write_manifest(split, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
