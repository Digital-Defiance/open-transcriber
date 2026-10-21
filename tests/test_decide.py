"""Table-driven tests for the fixed GO/NO-GO decision function."""

import unittest

from open_transcriber.bakeoff import decide as D


def control(real=50.0, synth=50.0, mt=50.0, seeds=(20261020, 20261021, 20261022)):
    return D.EncoderResult(
        encoder="control",
        real_melody={s: real for s in seeds},
        synthetic_melody={s: synth for s in seeds},
        multitask_mean={s: mt for s in seeds},
    )


def cand(name, real, synth, mt, fits=True, size=0.0, seeds=(20261020, 20261021, 20261022)):
    def spread(v):
        return {s: (v[i] if isinstance(v, (list, tuple)) else v) for i, s in enumerate(seeds)}
    return D.EncoderResult(encoder=name, real_melody=spread(real), synthetic_melody=spread(synth),
                           multitask_mean=spread(mt), fits_mac=fits, size_rank=size)


class DecideTests(unittest.TestCase):
    def test_clear_go(self):
        ctrl = control(real=50, synth=50, mt=50)
        c = cand("musicfm", real=55, synth=57, mt=50)  # +5 real, +7 synth, no mt loss, all 3 seeds
        d = D.decide([c], ctrl)
        self.assertTrue(d.go)
        self.assertEqual(d.winner, "musicfm")

    def test_no_go_when_real_gain_too_small(self):
        ctrl = control(real=50, synth=50, mt=50)
        c = cand("weak", real=52, synth=58, mt=50)  # only +2 real (<3)
        d = D.decide([c], ctrl)
        self.assertFalse(d.go)
        self.assertIsNone(d.winner)

    def test_no_go_when_synthetic_gain_too_small(self):
        ctrl = control(real=50, synth=50, mt=50)
        c = cand("x", real=55, synth=53, mt=50)  # +5 real but only +3 synth (<5)
        self.assertFalse(D.decide([c], ctrl).go)

    def test_needs_two_of_three_seeds(self):
        ctrl = control(real=50, synth=50, mt=50)
        # only one seed clears the real-melody gate
        c = cand("one-seed", real=[55, 51, 51], synth=[57, 57, 57], mt=50)
        self.assertFalse(D.decide([c], ctrl).go)
        c2 = cand("two-seed", real=[55, 55, 51], synth=[57, 57, 57], mt=50)
        self.assertTrue(D.decide([c2], ctrl).go)

    def test_multitask_loss_blocks_go(self):
        ctrl = control(real=50, synth=50, mt=60)   # control has the best multi-task mean
        c = cand("melody-only", real=55, synth=57, mt=57)  # 3-pt multitask loss (>2) every seed
        self.assertFalse(D.decide([c], ctrl).go)

    def test_fits_mac_required(self):
        ctrl = control(real=50, synth=50, mt=50)
        c = cand("too-big", real=60, synth=60, mt=50, fits=False)
        self.assertFalse(D.decide([c], ctrl).go)

    def test_lexicographic_tie_break_melody(self):
        ctrl = control(real=50, synth=50, mt=50)
        a = cand("a", real=55, synth=57, mt=50)
        b = cand("b", real=58, synth=60, mt=50)   # clearly higher real melody
        d = D.decide([a, b], ctrl)
        self.assertTrue(d.go)
        self.assertEqual(d.winner, "b")

    def test_tie_break_multitask_then_size(self):
        ctrl = control(real=50, synth=50, mt=50)
        # near-equal real melody (within 1 pt): decide on multi-task, then size
        a = cand("a", real=55.0, synth=57, mt=52.0, size=2.0)
        b = cand("b", real=55.3, synth=57, mt=55.0, size=5.0)  # higher multitask by >0.5
        self.assertEqual(D.decide([a, b], ctrl).winner, "b")
        # equal real + equal multitask: smaller/faster wins
        a2 = cand("small", real=55.0, synth=57, mt=52.0, size=1.0)
        b2 = cand("big", real=55.2, synth=57, mt=52.2, size=9.0)
        self.assertEqual(D.decide([a2, b2], ctrl).winner, "small")

    def test_whisper_add_and_skip(self):
        ctrl = control(real=50, synth=50, mt=50)
        c = cand("winner", real=55, synth=57, mt=50)
        add = D.decide([c], ctrl, whisper_subset_gain=2.5, whisper_overall_loss=0.2)
        self.assertTrue(add.add_whisper)
        skip = D.decide([c], ctrl, whisper_subset_gain=1.0, whisper_overall_loss=0.2)  # gain < 2
        self.assertFalse(skip.add_whisper)
        skip2 = D.decide([c], ctrl, whisper_subset_gain=3.0, whisper_overall_loss=0.9)  # loss > 0.5
        self.assertFalse(skip2.add_whisper)


if __name__ == "__main__":
    unittest.main()
