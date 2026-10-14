# Third-party notices / 第三方权利说明

The root [LICENSE](LICENSE) applies only to original rights held by Crescent Feng.
It does not relicense the following components or their output-related rights.
Models and external runtimes are obtained separately; installation does not
prove permission for every use or redistribution.

| Component | Role and recorded source | Licence boundary |
| --- | --- | --- |
| CosyVoice2 | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | Pinned source LICENSE is Apache-2.0. The local weight hashes are recorded but the local snapshot's upstream revision remains unverified; do not infer weight provenance from a current model card. |
| GPT-SoVITS v2 | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), commit `a7e38f93f1ad8d263093ca26ce0dea3fb7080ea2` | Pinned code is MIT, copyright 2024 RVC-Boss. [Model repository](https://huggingface.co/lj1995/GPT-SoVITS) declares MIT; BERT/HuBERT components retain their own notices. |
| ESPnet VITS / Transformer | [ESPnet](https://github.com/espnet/espnet), `espnet==202402`; [VITS weights](https://huggingface.co/espnet/kan-bayashi_csmsc_vits), [Transformer weights](https://huggingface.co/espnet/kan-bayashi_csmsc_transformer) | Framework, model weights and CSMSC training data are separate rights layers; inspect each pinned distribution before reuse. Weights are not bundled. |
| Parallel WaveGAN | [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), CSMSC vocoder | Obtain code and weights separately under their own terms; hashes/configuration are in `configs/shengke/csmsc-vocoder.json`. |
| THCHS-30 | [CSLT, Tsinghua University / OpenSLR 18](https://www.openslr.org/18/) | Publisher declares Apache-2.0. Keep attribution, modifications and [licence text](LICENSES/Apache-2.0.txt); see [NOTICE](NOTICE). |
| Reference mirrors / alignments | [FluidInference/THCHS-30-tests](https://huggingface.co/datasets/FluidInference/THCHS-30-tests), [anyspeech/THCHS-30-alignments](https://huggingface.co/datasets/anyspeech/THCHS-30-alignments) | Fixed revisions, source hashes and cropping recipes are in `data/tts-eval/`; they are not original recordings by this project. |
| Shared-platform integrations | Faster-Whisper/Whisper, FunASR, pyannote.audio and WebRTC VAD | Optional code and model dependencies have separate terms and, where required, gated access acceptance. No credentials or pretrained models are included. |
| Runtime and infrastructure | FastAPI, Gradio, PyTorch, FFmpeg, PostgreSQL, MinIO, Docker and Python dependencies | Consult the installed distribution's licences and bundled notices. MinIO and FFmpeg have independent copyleft/build-dependent obligations; the project's notice cannot override them. |

The dependency manifests are `pyproject.toml`, `services/tts-runtime/requirements*.txt`
and the environment locks under `configs/shengke/`. These are acquisition records,
not a complete transitive licence audit or blanket commercial clearance.

Generated demonstration audio is identified as synthetic and traced in
[ASSET_PROVENANCE.md](docs/ASSET_PROVENANCE.md). A dataset's copyright licence
does not itself establish a speaker's separate consent for impersonation,
endorsement or every downstream use. No such consent is asserted here.

Upstream code licence references checked on 2026-10-14:

- [CosyVoice pinned LICENSE](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/LICENSE)
- [GPT-SoVITS pinned LICENSE](https://github.com/RVC-Boss/GPT-SoVITS/blob/a7e38f93f1ad8d263093ca26ce0dea3fb7080ea2/LICENSE)
