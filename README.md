# ComfyUI Ming 3.0 TTS

English | [简体中文](README_ZH.md)

<img width="1516" height="892" alt="Screenshot 2026-10-20 021200" src="https://github.com/user-attachments/assets/260b619d-a5f0-489a-8380-e53bab1b66c6" />

Native ComfyUI port of [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts), supporting 0.5B and Tiny 9B speech generation.

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

For Tiny on CUDA, optional `fla-core==0.4.2` enables faster KDA kernels. With compatible PyTorch and Triton installed, use `python -m pip install --no-deps fla-core==0.4.2`. The checked Windows setup uses PyTorch 2.10.0+cu130 and triton-windows 3.6.0.post26.

FLA means Flash Linear Attention. It is optional and only used by Tiny; without it, Tiny uses a slower PyTorch fallback. The terminal reports which backend is active:

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

See [NVIDIA's GPU list](https://developer.nvidia.com/cuda/gpus). CUDA and INT8 support depend on the installed PyTorch and comfy-kitchen versions; FLA also requires compatible Triton kernels.

Approximate CUDA allocation in short runs: 0.5B **2.5–2.8 GiB**, Tiny INT8 **9.9 GiB**, Tiny BF16 **16.7 GiB**. Actual memory use varies with references, output length and other loaded models.

CPU execution is available and slower. MPS, ROCm, DirectML, XPU and other accelerator configurations have not been tested with full checkpoints.

### Transformers compatibility

The checked version range is **4.57.6–5.20.2**, with the following specific releases tested. Keep **5.3.0** if it already works with the rest of your ComfyUI installation; installing this node does not require upgrading to the newest release.

| Transformers | Hugging Face Hub | Tokenizers |
| --- | --- | --- |
| 4.57.6 | 0.36.2 | 0.22.2 |
| 5.3.0 | 1.3.0 | 0.22.2 |
| 5.5.0 | 1.5.0 | 0.22.2 |
| 5.20.2 | 1.31.0 | 0.23.1 |

Download the models from [drbaph/Ming-3.0-tts-ComfyUI](https://huggingface.co/drbaph/Ming-3.0-tts-ComfyUI):

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

### Existing models, shared drives and links

Models can also be loaded from registered extra paths, symlinks and Windows junctions. Existing `models/ming3_tts` folders remain supported.

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

Directory symlinks and Windows junctions let you keep weights on another drive. From the ComfyUI directory:

```powershell
New-Item -ItemType Junction -Path '.\models\checkpoints\Ming-3.0-tts-tiny-9b-int8-convrot' -Target 'D:\AI\models\Ming-3.0-tts-tiny-9b-int8-convrot'
```

## Nodes

| Node | Purpose |
| --- | --- |
| Ming TTS Load Model | Original or locally converted checkpoint; auto/BF16/FP32 compute; ComfyUI device or CPU. |
| Ming TTS Generate | Select cloning, voice design, built-in voice, freeform direction, basic attributes, emotion or dialect. Inputs change with the task. |
| Ming TTS Podcast Speaker | Name, reference AUDIO and its spoken transcript. |
| Ming TTS Podcast (Tiny 9B) | Native dialogue generation with growing speaker inputs and `Name: speech` lines. |
| Ming TTS Whisper Transcribe | Local Whisper AUDIO-to-transcript conversion for cloning and podcast references. |

Connect existing **Load Audio** nodes for references and existing **Preview Audio** or **Save Audio** nodes to outputs. Outputs use standard `AUDIO`, mono at 44.1 kHz. Cloning accepts Chinese, English, German, Spanish, French, Korean and Japanese. Inline Ming tags such as `[Laughter]` and `[Sigh]` stay in the text. Each generation selects one task; delivery controls are not mixed arbitrarily.

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

See the included [upstream voice catalog](docs/instruction_en.md#built-in-premium-voices) for the full Chinese voice list and emotion variants.

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

Convert `reference_text` to an input to connect Whisper's transcript, or type an accurate transcript manually. Use a clear single-speaker recording and correct recognition errors before generating.

Whisper looks for models under `models/whisper`, `models/audio_encoders` and registered extra paths. Available downloads: large-v3-turbo, large-v3, medium, small, base and tiny. Set `language=auto` to detect the recording's language, or enter a language name/code.

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

Basic attributes, emotion and dialect need a recording but no transcript. To request slower delivery, choose `basic` with `speed=slow`.

Inline vocal tags belong in the speech text. The upstream set is `[Breathing]`, `[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`, `[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`, `[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`.

### Tiny podcast workflow

Create one **Ming TTS Podcast Speaker** per voice, supplying its name, reference audio and transcript. Connect these to the growing `speakers` inputs of **Ming TTS Podcast (Tiny 9B)** and connect a Tiny model. Use the exact speaker names, one turn per line:

```text
Alice: Hello, and welcome to our conversation.
Bob: Thanks, Alice. What are we discussing today?
Alice: How to make a clear reference recording.
```

Use a Tiny model and increase `max_steps` when a longer conversation reaches the output limit.

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

The terminal shows a persistent progress bar with elapsed time and audio chunks per second.

The final summary reports audio duration, generation time and real-time factor (RTF). RTF below 1 means faster than real time; model loading is excluded.

## INT8 ConvRot

Use the preconverted INT8 downloads above to reduce weight storage and VRAM use. INT8 does not guarantee faster generation.

| Model | Original weights | INT8 weights |
| --- | ---: | ---: |
| 0.5B | 2.76 GiB | 2.42 GiB |
| Tiny 9B | 16.73 GiB | 10.22 GiB |

## Troubleshooting

- **Missing model:** enable `download_if_missing` or place the weights in the matching folder above. Whisper needs a complete local Hugging Face model folder.
- **Old inputs or an old traceback after an update:** restart the ComfyUI Python process. A browser refresh does not reload custom-node Python code.
- **Speech stops before the end of the text:** raise `max_steps` if it reached the cap. If it stopped naturally, a higher cap alone will not change delivery; check the reference/transcript and selected task.
- **Speech is too fast:** `max_steps` does not control pace. `basic` exposes speed control; `clone` follows the model's learned reference-conditioned delivery.
- **Tiny is slow:** check the printed FLA/fallback line, model precision and offloading pressure. INT8 saves memory but does not guarantee faster generation.
- **FLA import or kernel errors:** use the tested optional package and a working Triton build for the GPU, or leave FLA uninstalled to use the PyTorch path. Installed-kernel execution errors are reported rather than silently switching algorithms.
- **Out of memory:** try 0.5B, INT8 storage, a shorter reference/output or CPU execution. Other loaded models and long references also consume memory. Low-VRAM behavior has not been certified across cards.
- **Poor clone or wrong words:** review the reference transcript, use a clean voice reference, and write ambiguous numbers or abbreviations as the words you want spoken.
