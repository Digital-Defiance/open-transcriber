"""Tests for the committed test-set manifests and their helpers."""

import copy
import struct
import tempfile
import unittest
import wave
from collections import Counter
from pathlib import Path

from open_transcriber.bakeoff import manifests


class CommittedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = manifests.load_all()

    def test_counts(self):
        self.assertEqual(len(self.sets["rwc-pop"]["items"]), 100)
        other = Counter(i["collection"] for i in self.sets["rwc-other"]["items"])
        self.assertEqual(other, {"RWC-G": 52, "RWC-R": 15, "RWC-C": 6, "RWC-J": 2})
        usdx = Counter(i["status"] for i in self.sets["usdx"]["items"])
        self.assertEqual(usdx, {"scored": 32, "scored-apart": 4, "excluded": 2, "excluded-0a": 1})

    def test_ids_are_unique_across_sets(self):
        ids = [i["id"] for m in self.sets.values() for i in m["items"]]
        self.assertEqual(len(ids), 214)
        self.assertEqual(len(set(ids)), 214)

    def test_rwc_ids_and_classical_jazz_pieces(self):
        ids = {i["id"] for i in self.sets["rwc-other"]["items"]}
        self.assertTrue({"rwc-c045", "rwc-c050", "rwc-j036", "rwc-j037"} <= ids)
        self.assertEqual([i["id"] for i in self.sets["rwc-pop"]["items"]][:2], ["rwc-p001", "rwc-p002"])

    def test_karaoke_licences_and_hooktheory_split(self):
        items = self.sets["usdx"]["items"]
        audio = Counter(i["licence"]["audio"] for i in items)
        self.assertEqual(audio["CC BY 4.0"], 2)
        self.assertEqual(audio["CC0 1.0"], 1)
        self.assertEqual(audio["CC BY-SA 2.5"], 1)
        self.assertEqual(sum(n for lic, n in audio.items() if "NC" in lic), 35)
        apart = sorted(i["title"] for i in items if i["status"] == "scored-apart")
        self.assertEqual(apart, ["Code Monkey", "I Feel Fantastic", "Re: Your Brains", "Shop Vac"])
        self.assertTrue(all(i["hooktheory"] == (i["status"] == "scored-apart") for i in items))
        self.assertTrue(all("ND" in i["licence"]["audio"] for i in items if i["status"] == "excluded"))

    def test_freestyle_count_in_the_32(self):
        scored = [i for i in self.sets["usdx"]["items"] if i["status"] == "scored"]
        self.assertEqual(sum(i["chart"]["freestyle"] for i in scored), 153)

    def test_nc_items_say_testing_only(self):
        for manifest in self.sets.values():
            for item in manifest["items"]:
                use = item.get("allowed_use", manifest["allowed_use"])
                self.assertIn("testing only", use)


def minimal():
    sha = "a" * 64
    return {
        "format": manifests.FORMAT, "set": "rwc-pop", "title": "t", "source": {}, "licence": "x",
        "allowed_use": "testing only", "items": [
            {"id": "rwc-p001", "title": "t", "status": "scored", "cluster": "rwc-p001",
             "audio": {"path": "a.wav", "sha256": sha}, "annotations": [{"path": "a.mid", "sha256": sha}],
             "reference": None},
        ],
    }


class ValidateTests(unittest.TestCase):
    def test_minimal_manifest_validates(self):
        manifests.validate(minimal())

    def test_missing_sha256_raises(self):
        # Should fail: every listed file needs a hash.
        bad = minimal()
        del bad["items"][0]["audio"]["sha256"]
        with self.assertRaises(ValueError):
            manifests.validate(bad)

    def test_duplicate_id_raises(self):
        bad = minimal()
        bad["items"].append(copy.deepcopy(bad["items"][0]))
        with self.assertRaises(ValueError):
            manifests.validate(bad)

    def test_unknown_status_raises(self):
        bad = minimal()
        bad["items"][0]["status"] = "maybe"
        with self.assertRaises(ValueError):
            manifests.validate(bad)


class HelperTests(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(manifests.usdx_id("Jonathan Coulton - Re- Your Brains"), "usdx-jonathan-coulton-re-your-brains")
        self.assertEqual(manifests.usdx_id("Systemabsturz - Verdächtig"), "usdx-systemabsturz-verdachtig")
        self.assertEqual(manifests.rwc_id("RWC_G019"), "rwc-g019")

    def test_wav_info_pcm_and_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            pcm = Path(tmp) / "pcm.wav"
            with wave.open(str(pcm), "wb") as out:
                out.setnchannels(2)
                out.setsampwidth(2)
                out.setframerate(44100)
                out.writeframes(b"\0\0\0\0" * 44100)
            info = manifests.wav_info(pcm)
            self.assertEqual((info["format"], info["channels"], info["rate"], info["frames"]), ("pcm", 2, 44100, 44100))
            self.assertEqual(info["seconds"], 1.0)
            flt = Path(tmp) / "float.wav"
            frames = 24000
            data = struct.pack("<f", 0.0) * frames
            fmt = struct.pack("<HHIIHH", 3, 1, 48000, 48000 * 4, 4, 32)
            body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
            flt.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
            info = manifests.wav_info(flt)
            self.assertEqual((info["format"], info["channels"], info["rate"], info["seconds"]), ("float", 1, 48000, 0.5))

    def test_licence_urls_keep_only_labels_and_urls(self):
        text = ("Some words that are not kept here at all\nMP3 File:\nFound on http://example.org/song\n"
                "License: http://creativecommons.org/licenses/by-nc/3.0/\nLyrics:\nmore words never kept\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "license.txt"
            path.write_text(text, encoding="utf-8")
            pairs = manifests.licence_urls(path)
        self.assertEqual(pairs, [("Mp3 File", "http://example.org/song"),
                                 ("Mp3 File", "http://creativecommons.org/licenses/by-nc/3.0/")])

    def test_hand_checked_table_matches_the_audit(self):
        statuses = Counter(s for s, *_ in manifests.USDX_SONGS.values())
        self.assertEqual(statuses, {"scored": 32, "scored-apart": 4, "excluded": 2, "excluded-0a": 1})


if __name__ == "__main__":
    unittest.main()
