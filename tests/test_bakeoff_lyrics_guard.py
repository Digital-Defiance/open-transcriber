"""Tests for the lyrics and NC-file guard, on made-up text and a temporary git repo."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import lyrics_guard as guard

# A made-up phrase standing in for a lyric line. Not from any song.
PHRASE = "copper kettles hum beneath violet ladders tonight"


def make_index():
    index = guard.Index({}, {}, set(), 0)
    index.add("usdx-made-up-song", [PHRASE])
    index.add("rwc-x001", ["la la la la la la la la"])
    index.add("rwc-x002", ["\u6708\u304c\u9759\u304b\u306b\u6d77\u3092\u7167\u3089\u3057\u3066\u3044\u308b"])
    return index


def git_repo(files: dict) -> Path:
    root = Path(tempfile.mkdtemp())
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        subprocess.run(["git", "-C", str(root), "add", "-f", name], check=True)
    return root


class TextTests(unittest.TestCase):
    def test_planted_phrase_is_caught_and_not_echoed(self):
        # Should fail: a made-up 6-word "lyric" planted in a report is caught.
        index = make_index()
        text = "Line one is fine.\nThe report says: Copper kettles HUM beneath violet ladders, tonight.\n"
        findings = guard.scan_text("report.md", text, index)
        self.assertEqual([(f.where, f.reason) for f in findings], [("report.md:2", "usdx-made-up-song")])
        self.assertNotIn("kettles", repr(findings))

    def test_five_words_pass(self):
        index = make_index()
        self.assertEqual(guard.scan_text("x", "copper kettles hum beneath violet", index), [])

    def test_low_information_ngrams_are_ignored(self):
        index = make_index()
        self.assertEqual(guard.scan_text("x", "la la la la la la la la la", index), [])

    def test_credits_are_allowed(self):
        index = make_index()
        index.allow(PHRASE)  # as if it were the song's title
        self.assertEqual(guard.scan_text("x", PHRASE, index), [])

    def test_japanese_run_is_caught(self):
        index = make_index()
        text = "\u6708\u304c\u9759\u304b\u306b\u6d77\u3092\u7167\u3089\u3057\u3066"
        self.assertEqual([f.reason for f in guard.scan_text("x", text, index)], ["rwc-x002"])

    def test_allowed_overlap(self):
        index = make_index()
        key = ("songs/when-the-saints.yaml", "usdx-made-up-song")
        guard.ALLOWED_OVERLAPS[key] = "test"
        try:
            findings = guard.scan_text("repo:songs/when-the-saints.yaml", PHRASE, index,
                                       "songs/when-the-saints.yaml")
            self.assertTrue(findings and all(f.allowed for f in findings))
        finally:
            del guard.ALLOWED_OVERLAPS[key]


class FileRuleTests(unittest.TestCase):
    def test_path_rules(self):
        self.assertIsNotNone(guard.path_rule("out/judge/x.json"))
        self.assertIsNotNone(guard.path_rule(".cache/judge/env/x.py"))
        self.assertIsNotNone(guard.path_rule("voices/hope/a.txt"))
        self.assertIsNotNone(guard.path_rule("songs/lights-on-june/song.yaml"))
        self.assertIsNotNone(guard.path_rule("docs/take.WAV"))
        self.assertIsNotNone(guard.path_rule("refs/rwc-p001.tsv"))
        self.assertIsNone(guard.path_rule("open_transcriber/judge/results/stage0a/x.csv"))
        self.assertIsNone(guard.path_rule("docs/transcribing.md"))

    def test_chart_content_is_detected(self):
        chart = "#TITLE:x\n#BPM:300\n" + "".join(f": {i} 2 5 la\n" for i in range(12)) + "E\n"
        self.assertTrue(guard.is_chart(chart))
        self.assertFalse(guard.is_chart("#BPM:300\n: 0 2 5 la\n"))


class RepoTests(unittest.TestCase):
    def test_staged_wav_and_planted_phrase_are_refused(self):
        # Should fail: a staged .wav and a planted phrase both fail the guard.
        repo = git_repo({
            "notes.md": "Fine text.\n" + PHRASE + "\n",
            "audio/take.wav": b"RIFF\0\0\0\0WAVE",
            "clean.md": "Nothing to see.\n",
        })
        findings = guard.scan_repo(repo, make_index())
        reasons = {f.where.split(":", 1)[1]: f.reason for f in findings}
        self.assertEqual(reasons["audio/take.wav"], ".wav files are never committed")
        self.assertEqual(reasons["notes.md:2"], "usdx-made-up-song")
        self.assertNotIn("clean.md", " ".join(reasons))

    def test_clean_repo_passes(self):
        repo = git_repo({"a.md": "Just words about scoring.\n", "b.py": "print('ok')\n"})
        self.assertEqual(guard.scan_repo(repo, make_index()), [])


if __name__ == "__main__":
    unittest.main()
