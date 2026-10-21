"""Frozen development-split manifests for the bake-off, selected once by stable SHA-256 ordering.

The bake-off scores probes only on these development sets; the candidate-sealed TEST partitions (RWC-Pop,
the RWC-other test split, the USDX test items, COTB) are never touched during the bake-off. Each manifest
records its member ids and a content hash so a probe can never be scored against an unfrozen split. The
builder refuses to include any test-partition id (report §3.1, §5.3).

Development sets (report §5.3):
- 30-item RWC-other dev split, stratified 20 G / 6 R / 3 C / 1 J;
- 12 QA-surviving karaoke dev songs (diagnostic only, flagged);
- 4 Annotated-VocalSet validation singers (from ``vocalset_split``);
- 80 unseen sampler scores and 40 unseen SoulX clips (by seed range);
- the 5 YuE2 truth takes (regression only, flagged).

Seed text ``copyist-ot2-20261020``; counts and strata frozen here. Content was rephrased for compliance with
licensing restrictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import manifests, vocalset_split

SEED_TEXT = "copyist-ot2-20261020"
SPLITS_DIR = Path(__file__).with_name("splits")

#: Base seeds for the material scripts' unseen blocks (defined here so dev_splits stays free of music21/
#: copyist imports and the probe-eval path can run in the torch-only env). The material scripts import these.
SAMPLER_BASE_SEED = 90_000_000
SOULX_BASE_SEED = 91_000_000

#: RWC-other dev stratum sizes by collection letter (report §3.1).
RWC_OTHER_DEV_STRATA = {"G": 20, "R": 6, "C": 3, "J": 1}
KARAOKE_DEV_N = 12
#: Unseen development seed ranges (disjoint from the training seed ranges used by the material scripts).
SAMPLER_DEV_N = 80
SOULX_DEV_N = 40
#: Round-trip YuE2 truth takes, regression only (never a judge of quality).
YUE2_TRUTH_TAKES = ("amazing-grace", "major-general", "silent-night", "the-tyger", "when-the-saints")


def _rank(key: str) -> str:
    return hashlib.sha256(f"{SEED_TEXT}:{key}".encode("utf-8")).hexdigest()


def content_hash(members: list[str]) -> str:
    """Stable content hash of a split's member ids (order-independent)."""
    joined = "\n".join(sorted(members))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def rwc_other_dev(data_manifest: dict | None = None) -> list[str]:
    """The 30 RWC-other dev ids: the lowest-ranked ids per collection, by stratum size."""
    m = data_manifest or manifests.load("rwc-other")
    by_letter: dict[str, list[str]] = {}
    for item in m["items"]:
        if item["status"].startswith("excluded"):
            continue
        letter = item["id"][4].upper()          # rwc-g001 -> G
        by_letter.setdefault(letter, []).append(item["id"])
    chosen = []
    for letter, n in RWC_OTHER_DEV_STRATA.items():
        ranked = sorted(by_letter.get(letter, []), key=_rank)
        if len(ranked) < n:
            raise ValueError(f"RWC-other has {len(ranked)} {letter} items, need {n} for the dev split")
        chosen.extend(ranked[:n])
    return sorted(chosen)


def rwc_other_test(data_manifest: dict | None = None) -> list[str]:
    """The RWC-other candidate-sealed test ids (everything not in the dev split)."""
    m = data_manifest or manifests.load("rwc-other")
    dev = set(rwc_other_dev(m))
    return sorted(i["id"] for i in m["items"]
                  if not i["status"].startswith("excluded") and i["id"] not in dev)


def rwc_pop_test(data_manifest: dict | None = None) -> list[str]:
    """RWC-Pop is entirely a sealed test partition."""
    m = data_manifest or manifests.load("rwc-pop")
    return sorted(i["id"] for i in m["items"])


def qa_survivors(data_manifest: dict | None = None, data: Path | None = None) -> list[str]:
    """Karaoke ids that survived chart QA (not dropped). Falls back to all scored ids if QA is absent."""
    from . import qa
    m = data_manifest or manifests.load("usdx")
    scored = [i["id"] for i in m["items"] if i["status"] == "scored"]
    survivors = []
    for iid in scored:
        try:
            verdicts = qa.load_qa(iid, data)
        except (FileNotFoundError, ValueError):
            survivors.append(iid)       # no QA on disk: keep (frozen by id; QA is diagnostic only)
            continue
        if not verdicts.get("dropped"):
            survivors.append(iid)
    return sorted(survivors)


def karaoke_dev(data_manifest: dict | None = None, data: Path | None = None) -> list[str]:
    """The 12 karaoke dev songs (diagnostic only): the lowest-ranked QA survivors."""
    survivors = qa_survivors(data_manifest, data)
    if len(survivors) < KARAOKE_DEV_N:
        raise ValueError(f"only {len(survivors)} QA survivors, need {KARAOKE_DEV_N}")
    return sorted(sorted(survivors, key=_rank)[:KARAOKE_DEV_N])


def _frozen_karaoke_dev(directory: Path = SPLITS_DIR) -> list[str] | None:
    """The committed karaoke-dev member ids, if the frozen manifest exists (avoids importing qa/scipy)."""
    path = Path(directory) / "karaoke-dev.json"
    if path.exists():
        return list(load_split("karaoke-dev", directory)["members"])
    return None


def karaoke_test(data_manifest: dict | None = None, data: Path | None = None) -> list[str]:
    """Karaoke ids not in the dev split (diagnostic secondary + HookTheory overlap), sealed from the bake-off.

    Prefers the committed frozen ``karaoke-dev`` manifest so the sealed set can be computed without importing
    the QA/CREPE tooling (the probe-eval env has no scipy); falls back to rebuilding from QA if unfrozen.
    """
    m = data_manifest or manifests.load("usdx")
    frozen = _frozen_karaoke_dev()
    dev = set(frozen) if frozen is not None else set(karaoke_dev(m, data))
    return sorted(i["id"] for i in m["items"]
                  if not i["status"].startswith("excluded") and i["id"] not in dev)


def sampler_dev_seeds() -> list[str]:
    return [f"sampler-{s}" for s in range(SAMPLER_BASE_SEED, SAMPLER_BASE_SEED + SAMPLER_DEV_N)]


def soulx_dev_seeds() -> list[str]:
    return [f"soulx-{s}" for s in range(SOULX_BASE_SEED, SOULX_BASE_SEED + SOULX_DEV_N)]


def test_partition_ids(data: Path | None = None) -> set[str]:
    """Every id that must never appear in a development split (what the bake-off is sealed from)."""
    ids = set(rwc_pop_test()) | set(rwc_other_test()) | set(karaoke_test(data=data))
    # COTB has no ids yet (curated later); its prefix is reserved and guarded by name.
    return ids


def build_splits(data: Path | None = None) -> dict[str, dict]:
    """Build the frozen dev-split manifests, guarding against any test-partition id leaking in."""
    vs = vocalset_split.build_split()
    splits = {
        "rwc-other-dev": {"kind": "real", "members": rwc_other_dev(),
                          "note": "RWC-other development split, 20 G / 6 R / 3 C / 1 J (CC BY-NC, eval-only)"},
        "karaoke-dev": {"kind": "real-diagnostic", "members": karaoke_dev(data=data),
                        "note": "12 QA-surviving karaoke dev songs, DIAGNOSTIC ONLY (CC BY-NC, eval-only)"},
        "vocalset-val": {"kind": "cc-by-val", "members": [f"vocalset-{s}" for s in vs.val],
                         "note": "4 Annotated-VocalSet validation singers (CC BY 4.0)"},
        "sampler-dev": {"kind": "synthetic", "members": sampler_dev_seeds(),
                        "note": f"{SAMPLER_DEV_N} unseen sampler scores (by seed)"},
        "soulx-dev": {"kind": "synthetic", "members": soulx_dev_seeds(),
                      "note": f"{SOULX_DEV_N} unseen SoulX clips (by seed)"},
        "yue2-truth": {"kind": "regression", "members": sorted(YUE2_TRUTH_TAKES),
                       "note": "5 YuE2 truth takes, REGRESSION ONLY (never a judge of quality)"},
    }
    sealed = test_partition_ids(data)
    for name, split in splits.items():
        leaked = sorted(set(split["members"]) & sealed)
        if leaked:
            raise ValueError(f"{name}: test-partition ids leaked into a dev split: {leaked}")
        # COTB guard by name
        if any(str(m).lower().startswith("cotb") for m in split["members"]):
            raise ValueError(f"{name}: COTB ids must not appear in a dev split")
        split["seed_text"] = SEED_TEXT
        split["count"] = len(split["members"])
        split["content_hash"] = content_hash([str(x) for x in split["members"]])
    return splits


def write_splits(splits: dict[str, dict], directory: Path = SPLITS_DIR) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, split in splits.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"name": name, **split}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def load_split(name: str, directory: Path = SPLITS_DIR) -> dict:
    return json.loads((Path(directory) / f"{name}.json").read_text(encoding="utf-8"))


def split_hashes(directory: Path = SPLITS_DIR) -> dict[str, str]:
    """{split name: content hash} for the committed frozen manifests."""
    return {p.stem: load_split(p.stem, directory)["content_hash"]
            for p in sorted(Path(directory).glob("*.json"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the frozen manifests into splits/")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args(argv)
    splits = build_splits(args.data)
    for name, split in splits.items():
        print(f"{name}: {split['count']} items, hash {split['content_hash'][:12]}")
    if args.write:
        for path in write_splits(splits):
            print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
