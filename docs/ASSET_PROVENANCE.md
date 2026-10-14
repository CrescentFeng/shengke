# Published asset provenance / 公开素材来源

Reviewed: 2026-10-14. The repository contains source, documentation and a small
set of synthetic demonstration recordings. It does not claim original ownership
of the reference speakers, model weights or underlying dataset.

## Synthetic audio

The following files were matched to their adjacent generation reports and to
reference hashes in `data/tts-eval/`. They demonstrate model output rather than
speech actually spoken by the reference speaker. They are not a reusable voice
licence or evidence of commercial clearance.

| Published file | Model | Reference ID | Output SHA-256 |
| --- | --- | --- | --- |
| `artifacts/shengke/release-v1/jobs/31b444f0-4726-4880-8481-ee034c27671e/speech.wav` | cosyvoice2 | `D12_991_10s` | `941875b0d277e00b79a8cfefc61ef773138b02e504a21c7c7811934015e4ef8a` |
| `artifacts/shengke/release-v1/jobs/833d4c8f-4ede-4256-a86d-176db9523773/speech.wav` | gpt-sovits-v2 | `D12_991_10s` | `ee530689c7002caf3725a9bb3b9526f4585707c8105ded89a9a8f96028a07fbb` |
| `artifacts/shengke/release-v1/jobs/89e8ce5e-a870-4d92-95df-305e8980de53/speech.wav` | cosyvoice2 | `D12_892` | `a64f6be622a5667378a67c10b8754bb161d263048d2991f5c0fccdcdd3278cb9` |
| `artifacts/shengke/release-v1/jobs/b4a81d95-f110-46e5-98f9-3c7b85cd8b56/speech.wav` | gpt-sovits-v2 | `D12_991_10s` | `ee530689c7002caf3725a9bb3b9526f4585707c8105ded89a9a8f96028a07fbb` |
| `artifacts/shengke/release-v1/jobs/f070b2b7-54d8-4ce7-85e8-1e4cff8c2b71/speech.wav` | gpt-sovits-v2 | `D12_991_10s` | `ee530689c7002caf3725a9bb3b9526f4585707c8105ded89a9a8f96028a07fbb` |

`apps/shengke-showcase/audio/gpt-sovits-demo.wav` is byte-identical to the GPT
output with SHA-256 `ee530689c7002caf3725a9bb3b9526f4585707c8105ded89a9a8f96028a07fbb`.
The report records pause insertion and tempo processing. Original reference audio
is obtained separately from the sources recorded in [the reference policy](../data/tts-eval/README.md).

THCHS-30's published Apache-2.0 terms, upstream model terms, statutory rights and
any applicable speaker/personality rights remain separate. This publication does
not assert exclusive copyright in machine output or blanket downstream consent.
The original project notice applies only to protectable contributions actually
owned by the author; no restriction here overrides third-party or statutory rights.

## Visual and written material

Release PNGs show the local project interface and static showcase. They are
engineering evidence, not official imagery of model vendors or proof of their
endorsement. The project-authored PDF under `output/pdf/` is a historical
2026-09-20 planning document; its targets are not current achieved metrics.
The current release documents and recorded evidence define completed work.

## Future assets

Record source, author, licence, allowed distribution, changes and hashes before
adding audio, images, datasets or models. Keep private consent documents outside
Git. Do not publish user-uploaded voices, raw databases, private recordings,
credentials or materials whose required permissions are absent. Removing a file
from the current tree does not remove it from public history or old tags.
