# Lyric pool

Short pre-1929 public-domain poem and verse excerpts, used by `scores.py` as a
source of singable syllables for the generated training songs. None of these is
a round-trip benchmark song's text, and none is a candidate "big song" from the
T1 report (section 3). Each file opens with a `# source:` comment giving the
author, year and public-domain basis; `scores.py` strips those comment lines.

| File | Author | Year | Public-domain basis |
|---|---|---|---|
| `rossetti_wind.txt` | Christina Rossetti, "Who Has Seen the Wind?" | 1872 | Published pre-1929 (US public domain) |
| `dickinson_nobody.txt` | Emily Dickinson, "I'm Nobody! Who are you?" | 1891 | First published 1891, pre-1929 (US public domain) |
| `stevenson_swing.txt` | Robert Louis Stevenson, "The Swing" | 1885 | Published pre-1929 (US public domain) |
| `whitman_captain.txt` | Walt Whitman, "O Captain! My Captain!" (opening) | 1865 | Published pre-1929 (US public domain) |
| `field_wynken.txt` | Eugene Field, "Wynken, Blynken, and Nod" (opening) | 1889 | Published pre-1929 (US public domain) |
| `carroll_jabberwocky.txt` | Lewis Carroll, "Jabberwocky" (first stanza) | 1871 | Published pre-1929 (US public domain) |

These are deliberately unrelated to the held-out tunes (Amazing Grace, Silent
Night, When the Saints, The Tyger, Major-General) and to the big-song
candidates (O Holy Night, Battle Hymn, Star-Spangled Banner, We Three Kings,
Alexander's Ragtime Band, Sweet Georgia Brown, Take Me Out to the Ball Game,
Danny Boy, Go Down Moses, Joshua Fit the Battle, Home on the Range). The
held-out filter (`heldout.is_held_out`) is still applied to every generated
song as a backstop.
