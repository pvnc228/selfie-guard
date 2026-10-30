# Contributing

Start with the status in the [README](README.md). The project is an early scaffold.
Never describe a scaffold, a latent-space change or a single good example as proven
protection.

## Checks

```sh
python -m unittest discover -s tests -v
python -m selfie_guard doctor
ruff check selfie_guard tests
```

Ruff is a development-only tool. The CLI uses the standard library and must run
without the ML stack.

## Boundaries

- Keep processing local. Do not send photos, thumbnails, embeddings, masks or other
  derived data to external services.
- Do not add telemetry or automatic downloads to image processing. Fetching
  dependencies and weights must be an explicit user step.
- Never overwrite source photos.
- Never commit photos, model weights, generated results, keys or tokens.
- **Do not attach photos of real people or edited results to issues, pull requests
  or CI artifacts.** Describe the setup in text instead.
- Do not submit sexual content. Anything depicting or resembling a minor is
  excluded from all test data and results.
- Commits and releases must not contain AI co-author attribution or generator
  footers. No hooks should be bypassed.

## Experiments

Follow the [experiment protocol](experiments/README.md). Record exact model
revisions, code, seed, preprocessing, time, memory and every retry outcome. Publish
negative results too.
