# ComfyUI Ming 3.0 TTS

English | [简体中文](README_ZH.md)

Experimental native ComfyUI port of [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts), for the dense 0.5B and Ling V3 MoE Tiny 9B checkpoints. Uses the existing ComfyUI environment; no Transformers downgrade, remote Python code, FlashAttention requirement, or dependency on either reference custom-node pack.

Tiny 9B is the upstream Tiny model with approximately 8.98B parameters in the complete audio bundle. Its language backbone has approximately 7.9B total parameters and 1.3B active per token. The published and local bundle names include `tiny-9b` to distinguish it from 0.5B.

**Status:** real 0.5B BF16 and LLM-INT8 generation tested for English built-in voice, voice design and Chinese cloning on an RTX 5090. Tiny BF16 and LLM-INT8 English cloning have also been tested with FLA. Short Tiny INT8 English podcasts were checked for complete dialogue and speaker changes using two separate reference voices. Tiny uses FLA's chunk KDA kernel for prompts and fused recurrent KDA for single-token decoding when available on CUDA; otherwise it uses a portable FP32 recurrence, which can be substantially slower. Other accelerators, longer podcasts and low-VRAM execution remain unverified.

Ming retains AIMDO dynamic weight loading and prefetch. Allocation-compiler recording is disabled in its block loops after repeated `aimdo memory compile error` failures with comfy-aimdo 0.5.6 on Windows. This does not change ComfyUI's global compiler settings. Isolated worker-thread Whisper-to-clone workflows passed with AIMDO enabled for both 0.5B INT8 and Tiny INT8 after this change.

## Quick start

1. Put this directory, including its `assets` folder, under `ComfyUI/custom_nodes/ComfyUI-Ming-3-tts`.
2. Install the requirements using the Python that starts ComfyUI. From the ComfyUI directory, for a Windows venv:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r custom_nodes\ComfyUI-Ming-3-tts\requirements.txt
   ```

3. Restart ComfyUI, then add **Ming TTS Load Model**, **Ming TTS Generate** and an existing **Preview Audio** or **Save Audio** node.
4. Start with **Ming-3.0-tts-0.5b**, precision `auto`, device `auto`. Enable `download_if_missing` if the selected original or INT8 bundle is not installed. Existing local model folders load directly.
5. Connect Load Model to Generate. Choose task `ip`, voice `US_James`, enter a short English sentence, and queue it.

For cloning, use the workflow below. Tiny is needed for native multi-speaker dialogue. Both models output mono 44.1 kHz audio.

## Example workflows

Drag a JSON file from [example_workflows](example_workflows) into ComfyUI, or open it with **Workflow → Open**. Each example includes Markdown notes explaining the task, controls, model downloads and folder layout. Examples use this custom node and standard ComfyUI nodes.

| Workflow | Use |
| --- | --- |
| [Built-in voice (ip)](example_workflows/01_builtin_voice_ip.json) | Use a named voice such as `US_James`, without a reference. |
| [Voice design](example_workflows/02_voice_design.json) | Describe the voice you want to create. |
| [Clone with a manual transcript](example_workflows/03_voice_clone_manual.json) | Supply a recording and its exact transcript. |
| [Clone with Whisper](example_workflows/04_voice_clone_whisper.json) | Transcribe the reference automatically and generate new speech. |
| [Freeform direction](example_workflows/05_freeform_direction.json) | Give natural-language delivery instructions, with an optional reference. |
| [Basic attributes](example_workflows/06_basic_attributes.json) | Request speaking speed, pitch and volume. |
| [Emotion](example_workflows/07_emotion.json) | Select emotional delivery using a speaker reference. |
| [Chinese dialect](example_workflows/08_chinese_dialect.json) | Request Cantonese or Sichuanese. |
| [Tiny 9B podcast](example_workflows/09_tiny_9b_podcast.json) | Two speakers with separate reference recordings and Whisper transcripts. |
| [Vocal tags](example_workflows/10_vocal_tags.json) | Put events such as `[Sigh]` and `[Laughter]` in speech text. |
| [Cross-language cloning](example_workflows/11_cross_language_clone.json) | Generate Japanese from a reference that may use another supported language. |

Reference examples leave **Load Audio** empty for your own recording. For manual cloning, also fill in its exact transcript. Downloads are off by default; install the selected models or explicitly enable each loader's `download_if_missing`. Speech examples save FLAC with an audio preview under `ComfyUI/output/ming_tts/`. Seed is 42 with fixed control, and flow steps remain 10. Most examples select 0.5B INT8; the podcast selects Tiny 9B INT8. The same single-speaker graphs can use either model family.

## Install and models

The upstream cookbooks are included in [English](docs/instruction_en.md) and [简体中文](docs/instructions.md). They contain the complete voice catalog, vocal tags and control examples from [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts/tree/main/cookbooks). Their Python examples use the upstream API; use this README's node settings and workflows in ComfyUI.

Install `requirements.txt` with the Python used by ComfyUI, then restart ComfyUI. It declares `onnxruntime` for the bundled CampPlus speaker encoder and `tqdm` for terminal progress. ComfyUI supplies torch, torchaudio, transformers, tokenizers, safetensors and comfy-kitchen. Keep the PyTorch/CUDA build supplied for your ComfyUI installation.

Base dependencies are declared in `requirements.txt` and `pyproject.toml`. FLA is an optional `fla` extra in `pyproject.toml`, excluded from the base requirements. Whisper uses the Transformers installation supplied by ComfyUI.

For Tiny on CUDA, optional `fla-core==0.4.2` enables optimized KDA kernels. With compatible PyTorch and Triton already installed, use `python -m pip install --no-deps fla-core==0.4.2`. Kernel checks passed on Windows with PyTorch 2.10.0+cu130 and triton-windows 3.6.0.post26; this does not require replacing either package. Without FLA, or on non-CUDA devices, Tiny uses the portable PyTorch FP32 KDA recurrence. This path is slower and has mathematical/synthetic checks but no full-checkpoint speech comparison. Kernel errors from an installed FLA backend are reported directly.

FLA means Flash Linear Attention. Its GPU kernels accelerate Tiny's Kimi Delta Attention. It is optional; 0.5B does not use KDA. Each Tiny generation prints its actual path:

```text
Ming Tiny KDA: FLA (chunk prefill, fused recurrent decode)
```

or a fallback message with the reason, for example:

```text
Ming Tiny KDA: PyTorch FP32 fallback (FLA not available)
```

### GPU compatibility

| GPU family | Architecture / compute capability | Status |
| --- | --- | --- |
| RTX 30 series | Ampere / 8.6 | Expected compatible; no physical test here |
| RTX 40 series | Ada / 8.9 | Expected compatible; no physical test here |
| RTX 50 series | Blackwell / 12.0 | RTX 5090 tested; other cards unverified |

Architecture references: [NVIDIA CUDA GPU list](https://developer.nvidia.com/cuda/gpus). The port uses ComfyUI/PyTorch attention and Comfy Kitchen operations rather than requiring a 50-series-only weight format. INT8 ConvRot kernel availability depends on the installed ComfyUI/comfy-kitchen and PyTorch build. FLA additionally needs compatible Triton kernels for the GPU.

VRAM capacity matters separately from GPU generation. Short standalone tests allocated approximately 2.5–2.8 GiB for 0.5B, 9.9 GiB for Tiny INT8 and 16.7 GiB for Tiny BF16, excluding other applications and some driver/context overhead. Longer references and output can need more. Small-VRAM configurations may need offloading and have not been validated here. These are measurements, not minimum-VRAM guarantees.

CPU execution is available and slower. MPS, ROCm, DirectML, XPU and other accelerator configurations have not been tested with full checkpoints. Supported quantization/casting paths are selected by ComfyUI.

### Transformers compatibility

The checked version range is **4.57.6–5.20.2**, with the following specific releases tested. Keep **5.3.0** if it already works with the rest of your ComfyUI installation; installing this node does not require upgrading to the newest release.

| Transformers | Hugging Face Hub | Tokenizers | Validation |
| --- | --- | --- | --- |
| 4.57.6 | 0.36.2 | 0.22.2 | Real 0.5B INT8 speech + Whisper; both model tokenizers/task prompts |
| 5.3.0 | 1.3.0 | 0.22.2 | Main tested environment: real dense/Tiny BF16/INT8, Whisper and AIMDO workflows |
| 5.5.0 | 1.5.0 | 0.22.2 | Real 0.5B INT8 speech + Whisper; both model tokenizers/task prompts |
| 5.20.2 | 1.31.0 | 0.23.1 | Same compatibility check; isolated test also used safetensors 0.8.0 |

The alternate versions were installed into isolated test folders without changing the working environment. This is not an exhaustive test of every intermediate release or every mode under each version. Other versions are unverified. Transformers, Hub, Tokenizers and safetensors must satisfy the requirements of the selected Transformers release; for example, the tested 4.x and 5.x stacks use different Hub versions.

The Ming language/audio models run through this repository's native implementation. Transformers supplies tokenizer APIs and the local Whisper model; the port does not require the upstream reference runtime's Transformers 4.46.3 pin.

The four original merged and INT8 ConvRot weight files are hosted at [drbaph/Ming-3.0-tts-ComfyUI](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI). Runtime assets ship with this custom node. New weight downloads go under the running ComfyUI installation's checkpoint directory:

| Model | Download |
| --- | --- |
| 0.5B original merged | [model.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-0.5b/model.safetensors?download=true) |
| 0.5B INT8 ConvRot | [ming-3-tts-0.5b-int8-convrot.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-0.5b-int8-convrot/ming-3-tts-0.5b-int8-convrot.safetensors?download=true) |
| Tiny 9B original merged | [ming-3-tts-tiny-9b.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-tiny-9b/ming-3-tts-tiny-9b.safetensors?download=true) |
| Tiny 9B INT8 ConvRot | [ming-3-tts-tiny-9b-int8-convrot.safetensors](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI/resolve/main/checkpoints/Ming-3.0-tts-tiny-9b-int8-convrot/ming-3-tts-tiny-9b-int8-convrot.safetensors?download=true) |

Put each downloaded file in its matching model folder:

```text
ComfyUI/models/checkpoints/Ming-3.0-tts-0.5b/
ComfyUI/models/checkpoints/Ming-3.0-tts-0.5b-int8-convrot/
ComfyUI/models/checkpoints/Ming-3.0-tts-tiny-9b/
ComfyUI/models/checkpoints/Ming-3.0-tts-tiny-9b-int8-convrot/
```

The Hugging Face repository uses `checkpoints/<model-folder>/<weights>.safetensors`; place its `checkpoints` directory inside `ComfyUI/models`. Each model folder needs only its safetensors weight file. The custom node includes `assets/0.5b` and `assets/tiny-9b` for the matching config/tokenizer files, plus one shared `assets/campplus.onnx` speaker encoder. Only the selected weight file is downloaded; choosing Tiny INT8 does not download the original or any 0.5B weights. Both Load Model and Whisper Transcribe expose **download_if_missing**, default off. Device selectors include `auto`, `cpu` and explicit devices reported by ComfyUI, such as `cuda:0`.

The INT8 filenames are `ming-3-tts-0.5b-int8-convrot.safetensors` and `ming-3-tts-tiny-9b-int8-convrot.safetensors`. The merged original Tiny file is `ming-3-tts-tiny-9b.safetensors`. Each checkpoint contains the LLM, flow head, VAE and bridge together. After restarting ComfyUI, select the matching folder name; saved workflows using the old Tiny folder name need the new `tiny-9b` choice.

### Existing models, shared drives and links

Existing `models/ming3_tts` bundles remain discoverable. Complete local bundles can retain their own config, tokenizer and CampPlus files. For the four published model names, weight-only folders use the packaged assets. Custom folder names need a complete local config/tokenizer bundle. The loader also searches model folders directly under registered `checkpoints` paths, their `ming3_tts` subfolders, explicit `ming_tts` paths, and `ming3_tts` folders in models trees mapped through ComfyUI's standard model categories. Local models are preferred over downloading. Mapped paths are searched without creating folders or downloading into them.

To share a model collection, add a section to `extra_model_paths.yaml`, then restart ComfyUI:

```yaml
shared_ming:
  ming_tts: D:/AI/models/ming3_tts
```

Put the model folders inside that directory. An existing standard mapping also works:

```yaml
shared_models:
  base_path: D:/AI/models
  checkpoints: checkpoints
```

This discovers models under `D:/AI/models/checkpoints/ming3_tts`, directly inside the mapped `checkpoints` directory, or under `D:/AI/models/ming3_tts`. Directory symlinks and Windows junctions to model folders work too; the weights can stay on another drive. For example, inside the local `checkpoints` directory:

```powershell
New-Item -ItemType Junction -Path 'C:\Users\drbaph\Documents\ComfyUI\models\checkpoints\Ming-3.0-tts-tiny-9b-int8-convrot' -Target 'D:\AI\models\Ming-3.0-tts-tiny-9b-int8-convrot'
```

Whisper also discovers linked folders and `whisper`/`audio_encoders` folders in mapped model trees. Its new downloads remain in the running installation's `models/whisper`. New Ming downloads always use the running installation's `models/checkpoints`, even when an extra path is marked as the default.

### Loading and system RAM

INT8 ConvRot weights are already saved in ComfyUI's native quantized format. Loading wraps the saved INT8 data and scales; it does not expand the whole checkpoint to BF16 in RAM. Floating-point weights are cast to the selected compute dtype where needed. Activations and generation caches depend on the current input and cannot be saved as a replacement checkpoint.

The loader uses `comfy.utils.load_torch_file` and passes ComfyUI's detected disk-speed policy to the model patcher. With AIMDO enabled on fast storage, file-backed staging avoids retaining a full pinned CPU copy alongside resident VRAM weights. Slow storage or VRAM pressure can still require RAM caching. The normal ComfyUI fast-disk/high-RAM settings are respected; this node does not change them. Dtype conversions still require new storage. No additional checkpoint conversion is needed.

An isolated Tiny INT8 comparison on the RTX 5090 at 10 flow steps measured process resident memory after generation at 18.67 GiB before this integration and 2.52 GiB afterward. Private resident memory (USS) fell from 10.37 to 2.27 GiB; Windows private committed memory fell from 32.70 to 14.42 GiB. The updated process peaked at 2.53 GiB resident memory across initial generation, repeat generation and offload/reload checks. All three outputs were bit-identical to the baseline. A short 0.5B check also matched exactly, using 1.93 GiB resident / 1.71 GiB USS. These measurements include the standalone Python process and are not total-system or general minimum-RAM guarantees.

This is not RAM-free SSD-to-VRAM loading: Windows file caching, transfer buffers and offloading can still use system memory. Saving another weight format does not remove those requirements. NVIDIA's [GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage) can bypass CPU bounce buffers on supported systems, but is not supported on Windows and is not the I/O path used by this node.

## Nodes

| Node | Purpose |
| --- | --- |
| Ming TTS Load Model | Original or locally converted checkpoint; auto/BF16/FP32 compute; ComfyUI device or CPU. |
| Ming TTS Generate | Select cloning, voice design, built-in voice, freeform direction, basic attributes, emotion or dialect. Inputs change with the task. |
| Ming TTS Podcast Speaker | Name, reference AUDIO and its spoken transcript. |
| Ming TTS Podcast (Tiny 9B) | Native dialogue generation with growing speaker inputs and `Name: speech` lines. |
| Ming TTS Whisper Transcribe | Local Whisper AUDIO-to-transcript conversion for cloning and podcast references. |

Connect existing **Load Audio** nodes for references and existing **Preview Audio** or **Save Audio** nodes to outputs. Outputs use standard `AUDIO`, mono at 44.1 kHz. Cloning accepts Chinese, English, German, Spanish, French, Korean and Japanese. Inline Ming tags such as `[Laughter]` and `[Sigh]` stay in the text. Each generation selects one task; delivery controls are not mixed arbitrarily.

All node settings and reference inputs include hover tooltips, including the difference between speech text and reference transcripts, compute precision and INT8 storage, and the effects of sampling controls.

The built-in voice input accepts the exact upstream voice name listed below. Voice design and built-in voices need no reference. Freeform direction has an optional speaker reference. Attributes, emotion and dialect use CampPlus speaker conditioning without a reference transcript.

Whisper Transcribe discovers complete Hugging Face model folders under `models/whisper` and `models/audio_encoders`, including registered extra paths. It reads safetensors models locally unless `download_if_missing` is enabled. Missing official OpenAI large-v3-turbo, large-v3, medium, small, base and tiny models can then be downloaded under `models/whisper`. The existing `models/whisper/large-v3-turbo` works directly. Connect **Load Audio** to both Whisper and Generate's clone reference. Convert `reference_text` to an input and connect Whisper's `transcript`; for podcast references, connect it to Podcast Speaker's `text`. The transcript is also shown for review. Whisper is ASR (speech recognition), separate from Ming's speech generation. Generate and Podcast default to seed 42.

The existing generation modes cover upstream task families without separate nodes for each: `clone` handles seed/zero-shot and all seven target languages; inline vocal tags handle paralinguistic generation; `voice_design`, `ip`, `freeform`, `basic`, `emotion` and `dialect` expose their corresponding tasks. Podcast covers Tiny's multi-speaker task. Task availability does not imply every model/mode/language combination has been validated; see the status above.

### Built-in voice names (`ip`)

Select task `ip` and paste the identifier into **voice**. The upstream catalog lists these English voices:

| Identifier | Voice | Accent |
| --- | --- | --- |
| `US_Emma` | Female | American |
| `US_James` | Male | American |
| `US_Stella` | Female | American |
| `UK_George` | Male | British |
| `UK_Charlotte` | Female | British |

Chinese examples include `灵小安`, `灵小柔` and `灵楷`. Copy Chinese identifiers exactly without translating them. Some Chinese voices require a complete emotion-specific name, such as `灵未央_高兴` or `灵未央_中性`; use the catalog's invocation name rather than its base name or an invented suffix. This still uses task `ip`, rather than the separate `emotion` task.

See the included [upstream voice catalog](docs/instruction_en.md#built-in-premium-voices) for the 100+ Chinese voices, descriptions and available emotion variants. These are upstream-documented identifiers; the speech checks here used `US_James`. No reference recording or transcript is required for `ip`.

### Voice-cloning workflow

```text
Load Audio ──────────────→ Generate: task.reference_audio
     └→ Whisper Transcribe → Generate: task.reference_text
Load Model ──────────────→ Generate: model
Generate: AUDIO ─────────→ Preview Audio / Save Audio
```

- `text` is the **new sentence to speak**.
- `task.reference_audio` is the recording of the voice to clone.
- `task.reference_text` is the **transcript of that recording**, not the new sentence.
- `task.language` is the target language of the generated speech.

Convert the transcript widget to an input to connect Whisper's STRING output. Check the transcription and correct names or missing words when needed. A manually typed accurate transcript works without Whisper. Use a clear single-speaker reference; longer references increase encoding and prefill work. The node does not impose an artificial reference-duration limit.

Whisper runs locally using the standard ComfyUI patcher, loads on the selected device for transcription, and unloads afterward. It can detect the source language (`auto`) or accept a language name/code. Audio longer than 30 seconds uses Whisper's long-form generation; a 41-second input and CPU transcription have been tested.

### Tasks and controls

| Generate task | Reference input | Controls / use |
| --- | --- | --- |
| `clone` | Audio and its transcript | Chinese, English, German, Spanish, French, Korean or Japanese |
| `voice_design` | None | Describe a voice in `style` |
| `ip` | None | Official voice name in `voice`, such as `US_James` |
| `freeform` | Optional audio | Natural-language delivery `instruction` |
| `basic` | Audio | `speed`: slow/medium/fast; `pitch` and `volume`: low/medium/high |
| `emotion` | Audio | happy, sad, angry, afraid, disgusted, surprised or neutral |
| `dialect` | Audio | Sichuanese or Cantonese |

Basic attributes, emotion and dialect use speaker embeddings rather than the full audio/transcript continuation used by cloning. Each task follows its upstream conditioning contract; controls from different tasks are not arbitrarily combined. For example, choose `basic` with `speed=slow` for explicit speed control.

Inline vocal tags belong in the speech text. The upstream set is `[Breathing]`, `[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`, `[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`, `[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`. They are generation instructions, not post-processing sound effects.

### Tiny podcast workflow

Create one **Ming TTS Podcast Speaker** per voice, supplying its name, reference audio and transcript. Connect these to the growing `speakers` inputs of **Ming TTS Podcast (Tiny 9B)** and connect a Tiny model. Use the exact speaker names, one turn per line:

```text
Alice: Hello, and welcome to our conversation.
Bob: Thanks, Alice. What are we discussing today?
Alice: How to make a clear reference recording.
```

This is one native multi-speaker generation. The script's length is still subject to `max_steps`; increase it for longer conversations. Longer scripts and additional speaker counts remain unverified.

### Sampling settings and duration

| Setting | Default | Meaning |
| --- | ---: | --- |
| `seed` | 42 | Reproducible random sampling; existing workflows retain saved values |
| `cfg` | 2.0 | Model-native guidance strength |
| `flow_steps` | 10 | Flow-matching solver steps for each generated audio chunk |
| `temperature` | 0.0 | Stochastic flow sampling control |
| `sigma` | 0.25 | Flow noise scale; advanced input |
| `max_steps` | 250 | Maximum generated audio chunks, about 20 seconds |
| `trim` | true | Trim trailing silence after generation |

One decoding iteration generates approximately 80 ms of audio. The model can stop before `max_steps`; this is a ceiling, not a requested duration or speaking-speed setting. A cap of 125 is approximately 10 seconds and can cut off a sentence. Increasing the cap allows more speech but does not stretch or slow an already completed sentence.

The terminal shows a persistent `tqdm` decoding bar with elapsed time and iterations per second, while ComfyUI still receives its normal progress updates. On a natural stop the final bar uses the actual iteration count and remains in the terminal. Iterations are audio chunks, not individual inner flow evaluations: 125 chunks at 10 flow steps require 1,250 flow evaluations plus language-model work.

A final line reports audio duration, generation time and real-time factor (RTF). RTF is generation seconds divided by audio seconds; below 1 means faster than real time. Generation timing excludes model loading and includes reference preparation, prefill and audio decoding.

## INT8 ConvRot

Download the ready-to-use INT8 checkpoints from the model links above. They use the shared ComfyUI format: INT8 `weight`, FP32 per-row `weight_scale`, and `comfy_quant` metadata specifying `int8_tensorwise`, `convrot=true` and the group size. The native loader uses `comfy.ops.mixed_precision_ops`; no separate local INT8 linear implementation is used.

The first profile converts 0.5B's LLM attention/MLP projections, or Tiny's expert/shared/dense MLP projections. Audio VAE, flow head, audio-to-LLM aggregator, embeddings, routing/gates, speaker/stop heads and recurrent state retain floating point behavior. Compute precision remains separate from storage precision. Converted speech quality and speed require full-model comparison.

| Model | Original tensor data | Estimated initial INT8 profile |
| --- | ---: | ---: |
| 0.5B | 2.76 GiB | 2.42 GiB |
| Tiny | 16.73 GiB | 10.22 GiB |

These are checkpoint storage estimates from the published headers, excluding small serialization overhead; they are not peak VRAM measurements. Tiny's 1.3B active parameters do not mean only 1.3B parameters need storage.

The real combined 0.5B conversion contains 168 INT8 LLM matrices. All 1,012 other original tensors are retained without quantization. Short CUDA generation tests used about 2.83 GiB peak allocated memory for BF16 and 2.49 GiB for this INT8 profile; these are standalone test-process measurements, not a promise for other workflows. INT8 is a storage/memory option here, with no established speed improvement.

Offline transcription with the existing local Whisper model recovered the same English and Chinese text from the BF16 and INT8 samples. This checks intelligibility on those samples, not perceptual or speaker-similarity equivalence.

Tiny's 30 original shards were combined before conversion, with copied tensor data verified against each source shard's SHA-256. The converted checkpoint has 8,904 INT8 matrices; all 1,381 remaining original tensors are bitwise unchanged. Its file size is 10,975,490,488 bytes.

One warmed Makol English clone comparison on the RTX 5090 used the same reference, text, seed 1234, CFG 2 and 10 flow steps:

| Tiny checkpoint | Audio duration | Generation time | Decode it/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 6.2 s | 29.79 s | 2.84 | 16.74 GiB |
| INT8 ConvRot | 6.1 s | 54.08 s | 1.61 | 9.89 GiB |

Generation time includes reference preparation, prefill and audio decoding; model loading and warm-up are excluded. INT8 reduced memory but was slower in this test.

## Troubleshooting

- **Missing model:** enable `download_if_missing` for a published Ming bundle or official Whisper model, or place the complete local bundle in the documented folder. Local custom Whisper folders need their own config, tokenizer, preprocessor and safetensors weights.
- **Old inputs or an old traceback after an update:** restart the ComfyUI Python process. A browser refresh does not reload custom-node Python code. This version does not create allocation-compiler regions in Ming's block loops.
- **Speech stops before the end of the text:** raise `max_steps` if it reached the cap. If it stopped naturally, a higher cap alone will not change delivery; check the reference/transcript and selected task.
- **Speech is too fast:** `max_steps` does not control pace. `basic` exposes speed control; `clone` follows the model's learned reference-conditioned delivery.
- **Tiny is slow:** check the printed FLA/fallback line, model precision and offloading pressure. INT8 saves storage/VRAM but was slower than BF16 in the measured Tiny test.
- **FLA import or kernel errors:** use the tested optional package and a working Triton build for the GPU, or leave FLA uninstalled to use the PyTorch path. Installed-kernel execution errors are reported rather than silently switching algorithms.
- **Out of memory:** try 0.5B, INT8 storage, a shorter reference/output or CPU execution. Other loaded models and long references also consume memory. Low-VRAM behavior has not been certified across cards.
- **Poor clone or wrong words:** review the reference transcript, use a clean voice reference, and write ambiguous numbers or abbreviations as the words you want spoken. Whisper recognition does not measure speaker similarity or naturalness.
