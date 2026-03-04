# Videos (one folder per training round)

Place **output videos** here: one folder per training round, each showing results on the **original professor image** (night → day).

- **round1/** — Videos from Round 1 weights (e.g. side-by-side or progression on prof image).
- **round2/** — Videos from Round 2 weights.
- **round3/** — Videos from Round 3 weights.

Use `scripts/visualization/generate_video.py` with the appropriate checkpoint/outputs to generate clips, then save them in the matching round folder.
