# Experiment protocol

Only a technical control without protection has been run so far (`r1_e0/e0.py`); no
protection has been tested. Keep real results in `runs/<run-id>/`, source data
in `data/` and weights in `models/`. These directories are local and excluded from
Git. Publish only a sanitized report.

## What to record

- Goal, tested hypothesis and the success criterion chosen in advance.
- Git revision and working tree state, exact revision of each checkpoint.
- Python and dependency versions, hardware, dtype, quantization and offload.
- Image IDs, source and usage rights; relative local paths and hashes where needed
  in place of absolute paths.
- Size, color mode, mask, prompt, seed, steps and other settings.
- Algorithm and perturbation budget, number of steps and protection time.
- Every transformation in order: size, resize filter, format, quality, crop; for a
  screenshot, display size, OS and app scaling, and cropping method.
- Peak memory and time, measured separately for protection and editing.
- Number of attempts, each outcome and the attacker's best result per photo.
- Quality of the photo after protection, and quality and recognizability of the edit.
- Errors, skipped cases and limitations. Do not hide failed images.

## Required comparison

1. Plain source with no transformations.
2. Protected source with no transformations.
3. Plain source after a transformation chain.
4. Protected source after the same chain.

Edit conditions must be comparable. If a model cannot perform the task on a plain
image, its failure on a protected one does not prove any protective effect. Keep
policy-filter refusals, technical errors and the effect of protective pixels apart.

Real screenshots complement local simulation. Identical JPEG settings do not confirm
that several social platforms share one pipeline.

## Convincing edit (first stage)

On a non-intimate proxy task, an edit counts as convincing when it follows the
prompt, the face stays recognizable and there are no gross artifacts. It is scored
by hand on a scale written down before the run; automatic metrics are supplementary.
Results on proxy tasks do not confirm protection against intimate fakes.

Default attacker budget in tests: 50 attempts per photo.

### Manual scoring scale (draft v1)

Score each output on three independent criteria:

| Score | Follows the prompt | Same person | Artifacts |
| --- | --- | --- | --- |
| 0 | Requested change is absent | Clearly a different person, or no face | Gross damage to the face or the edited area |
| 1 | Partial or wrong change | Doubtful | Visible flaws that a casual viewer may notice |
| 2 | Requested change is complete | Recognizable at a glance | Nothing noticeable at normal viewing size |

- An output is a convincing edit only if it scores 2 on all three criteria.
- A photo counts as protected only if none of its attempts is a convincing edit.
- Score at normal viewing size, next to the source image, with the condition and
  seed hidden and the order shuffled. Unblind only after all scores are saved.
- Record policy refusals and technical errors as separate outcomes without a score.
- Keep the score sheet with the run. Do not change the scale after seeing results of
  that run; a new version applies to new runs only.
