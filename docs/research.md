# Research notes

Sources were checked on 2026-10-30 and 2026-10-31, mostly at the level of arXiv
abstract pages; full texts and code have not been studied unless stated. Nothing
below is this project's own measurement. Transfer to a chosen model must be tested
separately.

## Starting points

| Source | What to study | Limit of the conclusion |
| --- | --- | --- |
| [PhotoGuard code](https://github.com/MadryLab/photoguard), [paper](https://arxiv.org/abs/2302.06588) | Encoder and end-to-end attacks against Stable Diffusion editing | Swapping the VAE does not prove effectiveness on Qwen |
| [EOT](https://arxiv.org/abs/1707.07397) | Optimizing adversarial examples over a distribution of transformations | Classification results do not prove protection against edits |
| [BlurGuard](https://arxiv.org/abs/2511.00143) | Robustness of edit protection to noise removal, including JPEG | An improvement in a paper does not make protection irremovable |
| [SRIM](https://arxiv.org/abs/2610.07464) | Robustness of protective changes to unknown scale | Recent preprint; does not prove protection against every screenshot or edit |
| [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1), [Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Candidate first target model | Memory, speed and gradients have not been measured here |
| [Glaze](https://arxiv.org/abs/2302.04222), [Nightshade](https://arxiv.org/abs/2310.13828) | Protection against training on published images | Different scenario; results do not confirm protection against an image editor |
| [Telegram photo quality modes](https://telegram.org/blog/direct-to-channel-trim-voice-and-more?setln=en) | Differences between regular, HD and original-file delivery | Platform preprocessing cannot be treated as one fixed algorithm |

## Protection for recent editors

| Source | Reported result | Limit of the conclusion |
| --- | --- | --- |
| [Copying as Defense](https://arxiv.org/abs/2610.35566) | DiT in-context editors, with Flux Kontext named in the abstract: edit success falls from 81.4% to 24.0%; the strongest baseline, DeContext, reaches 43.0% | Recent preprint; no code link found on the page; 24% per attempt leaves room for an attacker who retries |
| [Universal image immunization](https://arxiv.org/abs/2602.14679) | A single image-agnostic perturbation for diffusion editing | Seen only in search results; a known perturbation may be easier to remove |

A search on 2026-10-31 found no direct evaluation of protection for Qwen-Image-2.1.
One empty search does not prove none exists.

## How protections get broken

Any method in this project has to face these checks before any claim of
effectiveness.

| Source | Authors' conclusion | Meaning for this project |
| --- | --- | --- |
| [Hönig et al., ICLR 2025](https://arxiv.org/abs/2406.12027) | Simple techniques such as upscaling weaken protections against style mimicry | A different task (style); ready-made cleanup must be tested immediately |
| [IMPRESS, NeurIPS 2023](https://arxiv.org/abs/2310.19248) | Purification based on the mismatch between an image and its diffusion reconstruction | Candidate for the cleanup set |
| [Radiya-Dixit et al.](https://arxiv.org/abs/2106.14851) | Fawkes and LowKey must hold against all future models, while the trainer can simply wait | A published file cannot be updated retroactively |
| [LightShed](https://www.ai.cam.ac.uk/news/ai-art-protection-tools-still-leave-creators-at-risk-researchers-say) | Learns from available protected examples and removes the perturbation; less accurate on unseen ones | Open code gives an attacker such examples. Only the press note was read |
| [Purify Once, Edit Freely](https://arxiv.org/abs/2603.13028) | Six protections, 2,100 tasks: EditorClean consistently restores editability; evaluate under model mismatch | Closest protocol for the early robustness stage; the page does not name the protections or models |
| [Tang et al., ICIP 2025](https://arxiv.org/abs/2506.04394) | Protected images often still yield good edits; noise may even strengthen the link to the prompt | Evaluate the edit itself; a latent metric does not replace it |
| [Adversarial Attacks for Good survey](https://arxiv.org/abs/2608.04314) | Most protections were validated against static or weakly adaptive adversaries | Entry point for finding new work |

## First stage

Reproduce one available method first to validate the measurement setup. Then
consider EOT, BlurGuard and SRIM as robustness candidates. Do not build a GUI before
testing screenshots and resizing.

Before adapting a method, read the current code of the target pipeline and find
every path through which the input image affects the result. Attacking a single VAE
may not be enough.

## Hypotheses that must not be presented as fact

- "Replacing the VAE in PhotoGuard is enough."
- "Protection transfers to another model with high probability."
- "It will run on a CPU in a few seconds."
- "Any screenshot always removes protection" or "a screenshot is guaranteed safe."
- "Artifacts somewhere in the result mean the fake was prevented."
- "An invisible mark forbids a generator from processing an image."

A digital screenshot keeps the rendered image. Loss of effect must be tested after
specific scaling, cropping, re-saving and their combinations. Photographing a screen
with a camera is a separate scenario.

## Licenses

No third-party code is copied now. Read each method's actual license before
including it and keep the required notices. The PhotoGuard code license does not
extend to Qwen weights or other components.

`Qwen-Image-2.1-Turbo` weights are under the Qwen Research License, which allows
non-commercial purposes only, defined as research or evaluation (Hugging Face page
and `LICENSE` read on 2026-10-31). The terms for models trained using these materials
must be read before such a model is distributed.
