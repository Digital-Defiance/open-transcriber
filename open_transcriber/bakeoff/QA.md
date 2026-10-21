# Chart QA for the karaoke songs

UltraStar Deluxe's charts were made by fans for a game that ignores octaves and times notes loosely, so they're checked automatically before they judge anything. The thresholds are in `qa_thresholds.json` (version 1). They were written and committed before any chart was checked or any system was scored, and they are never changed after a system has been scored. Every mask file records the SHA-256 of the thresholds it was made with, and the scorer refuses masks made with any other version.

## What is checked

Each chart note is compared with the song's own singing, using tools none of the tested systems uses:

- **Pitch:** a CREPE pitch track (torchcrepe 0.0.24's MIT weights, run through ONNX Runtime) of the separated vocal. A note passes when the median pitch of its voiced frames, taken from 20% into the note to 10% before its end, is within half a semitone of the chart's pitch class. The charts' octaves are unreliable, so only the pitch class is checked, as the scoring does.
- **Onset:** any of three kinds of evidence within 100 ms of the chart's onset: a spectral-flux peak of the vocal, the start of voicing after a silence, or a pitch step of at least three quarters of a semitone.
- **Verifiable:** a note needs at least 3 voiced frames, covering at least a quarter of its window, for its pitch to be checked at all. Otherwise it's masked as unverifiable.

A note is kept only if both its pitch and its onset pass. Otherwise it's masked, with one reason: unverifiable, pitch or onset, in that order. Freestyle notes have no pitch and are always masked. A song is dropped when more than half its pitched notes are masked.

Masked notes are don't-cares in scoring (see `score.py`): an estimate that matches one is neither a hit nor a false alarm.

## What is never done

No chart is ever shifted, transposed or edited. Each song's median offset between its chart and the onset evidence, and its masked share by reason, are reported as numbers, never used to correct anything.

## Why these numbers

- **Half a semitone** is the scoring's own pitch tolerance (50 cents).
- **100 ms** for onsets: the charts sit on their own grids of about 30 to 85 ms, so a correct note can be one grid step off; 100 ms is the karaoke sets' looser scoring tolerance too.
- **The pitch window** skips the scoop at a note's start and the release at its end, where sung pitch wanders.
- **Voicing** needs CREPE's periodicity of at least 0.5 (smoothed over 3 frames) and a frame no more than 40 dB below the stem's loud frames, so Demucs's quiet leftovers don't count as singing.
- **Dropping songs past 50% masked** keeps a mostly wrong chart from judging anyone.

The spectral flux stamps each frame at its centre; an abrupt onset peaks about 20 ms early (measured on test tones), which the 100 ms tolerance absorbs.

## The bias this could add, and its check

The QA reads the same Demucs vocal that Copyist and raw Basic Pitch transcribe, so its masks could favour systems that hear the stem the same way. SheetSage2 hears the mix. The pre-registered sensitivity analysis scores the karaoke sets with freestyle masks only (no QA masks); the gap between the two bounds that bias, and both are reported.
