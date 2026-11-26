# DanceAI

## Run on the project videos

Run from the project root after `uv sync`:

```bash
uv run python -m danceai.pipeline
uv run python preview_alignment.py
```

The first command extracts six joint angles and replaces each video's
`<name>-measurement.json`. Skip it when the measurements are already current.
The second command compares `data/dance2.mp4` with `data/my-video.MOV`, prints
mean absolute errors in degrees per joint, and saves `data/alignment_preview.mp4`.

```bash
uv run python -m danceai.comparison  # angle errors only
uv run python -m danceai.alignment   # matching source timestamps only
uv run python preview_alignment.py data/dance2.mp4 data/my-video.MOV --output data/comparison.mp4
uv run python -m unittest discover -s tests -v
```

`alignment.py` finds corresponding frame pairs using subsequence DTW.
`comparison.py` averages absolute angle differences over those pairs; its output
is a preliminary pose comparison, not a calibrated dance-quality percentage.
Frames missing any required angle are excluded, and usable frame counts are
printed. This can hide movement during missing intervals. No interpolation or
tempo constraint is applied yet.

The preview preserves image proportions and labels both source timestamps.
It plays one DTW pair per output frame at 30 FPS, without audio, so repeated
frames and changed playback speed are expected. Inspect the correspondence
visually before interpreting the angle errors. New previews replace the output
only after successful encoding and frame-count verification.

## Licensing

- Source code: [Apache License, Version 2.0](LICENSE).
- Original project documentation: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
- Third-party model weights and datasets retain their respective terms.
