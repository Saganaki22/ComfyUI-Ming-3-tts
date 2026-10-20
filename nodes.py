from comfy_api.latest import ComfyExtension, io

from . import loader, prompts, runtime
from .whisper import MingWhisperTranscribe


MingModel = io.Custom("MING_TTS_MODEL")
MingSpeaker = io.Custom("MING_TTS_SPEAKER")
CATEGORY = "audio/Ming TTS"


def sampling_inputs(seed=42):
    return [
        io.Int.Input("seed", default=seed, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Random seed for reference encoding and speech sampling. Keep it fixed to compare settings; default 42."),
        io.Float.Input("cfg", default=2.0, min=0, max=20, step=0.1, tooltip="Guidance strength; 0.5B and Tiny use different native formulas. Default 2. Higher values do not guarantee better speech."),
        io.Int.Input("flow_steps", default=10, min=1, max=100, tooltip="Flow solver evaluations per audio chunk. Default 10. More steps increase generation time and may change quality; this does not set audio duration."),
        io.Float.Input("sigma", default=0.25, min=0, max=10, step=0.01, advanced=True, tooltip="Scale of extra flow noise when temperature is above zero. Default 0.25; has no effect at temperature 0."),
        io.Float.Input("temperature", default=0.0, min=0, max=10, step=0.05, tooltip="Extra stochastic noise during flow sampling. At 0, only the initial latent noise is sampled. Higher values change variation and may reduce clarity."),
        io.Int.Input("max_steps", default=250, min=1, max=10000, tooltip="Maximum audio chunks, about 80 ms each: 125 is about 10 seconds. The model may finish earlier; this does not slow speech or force an exact duration.", advanced=True),
        io.Boolean.Input("trim", default=True, display_name="trim trailing silence", tooltip="Remove trailing quiet audio, retaining a short tail. Does not change speaking speed."),
    ]


class MingLoadModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = loader.model_choices()
        preferred = "Ming-3.0-tts-0.5b-int8-convrot"
        return io.Schema(
            node_id="MingLoadModel", display_name="Ming TTS Load Model", category=CATEGORY,
            inputs=[
                io.Combo.Input("model", options=choices, default=preferred if preferred in choices else choices[0], tooltip="Model folder under models/checkpoints, a registered extra path or a linked folder. Existing models/ming3_tts bundles also work. 0.5B is faster; Tiny supports native dialogue. INT8 folders keep quantized storage."),
                io.Combo.Input("precision", options=["auto", "bf16", "fp32"], tooltip="Compute dtype. Auto selects BF16 when supported, otherwise FP32. BF16 saves memory; this setting does not convert INT8 weights back to an original checkpoint."),
                io.Combo.Input("device", options=loader.device_choices(), tooltip="Execution device. Auto uses ComfyUI's selected device; an explicit device such as cuda:0 is used exactly. ComfyUI manages weight offloading."),
                io.Boolean.Input("download_if_missing", default=False,
                                 tooltip="Download only the selected Ming bundle from drbaph/Ming-3.0-tts-ComfyUI into models/checkpoints when missing. Existing local models are used first."),
            ], outputs=[MingModel.Output()],
        )

    @classmethod
    def execute(cls, model, precision, device, download_if_missing=False):
        return io.NodeOutput(loader.load_model(model, precision, device, download_if_missing))


class MingGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingGenerate", display_name="Ming TTS Generate", category=CATEGORY,
            inputs=[
                MingModel.Input("model", tooltip="Connect Ming TTS Load Model. The loaded checkpoint determines the model family and weight precision."),
                io.String.Input("text", multiline=True, dynamic_prompts=False, tooltip="New speech to generate, not the reference transcript. Supports Ming vocal tags such as [Laughter] and [Sigh]."),
                io.DynamicCombo.Input("task", options=[
                    io.DynamicCombo.Option("clone", [
                        io.Audio.Input("reference_audio", tooltip="Recording of the voice to clone. Supply the exact spoken transcript below. Longer references increase encoding and prefill work."), io.String.Input("reference_text", multiline=True, tooltip="Transcript of reference_audio, not the new speech. Type it manually or convert this widget to an input and connect Whisper's transcript."),
                        io.Combo.Input("language", options=list(prompts.LANGUAGES), tooltip="Language of the new speech. The reference can be in a different supported language."),
                    ]),
                    io.DynamicCombo.Option("voice_design", [io.String.Input("style", multiline=True, tooltip="Describe the voice to create, such as age, accent and vocal character. No reference recording is needed.")]),
                    io.DynamicCombo.Option("ip", [io.String.Input("voice", default="US_James", tooltip="Official built-in voice name, such as US_James. Use a name supported by the checkpoint.")]),
                    io.DynamicCombo.Option("freeform", [
                        io.String.Input("instruction", multiline=True, tooltip="Natural-language direction for voice or delivery. Optional reference audio supplies speaker identity."), io.Audio.Input("reference_audio", optional=True, tooltip="Optional recording for speaker identity. This task uses a speaker embedding and needs no transcript."),
                    ]),
                    io.DynamicCombo.Option("basic", [
                        io.Audio.Input("reference_audio", tooltip="Recording for speaker identity. Basic attributes use a speaker embedding and need no transcript."),
                        io.Combo.Input("speed", options=list(prompts.SPEEDS), tooltip="Requested speaking pace: slow, medium or fast. This changes model conditioning, not playback speed or max_steps."),
                        io.Combo.Input("pitch", options=list(prompts.LEVELS), tooltip="Requested vocal pitch: low, medium or high. Applied as model conditioning."),
                        io.Combo.Input("volume", options=list(prompts.LEVELS), tooltip="Requested vocal loudness: low, medium or high. Applied as model conditioning, not a fixed output-gain multiplier."),
                    ]),
                    io.DynamicCombo.Option("emotion", [
                        io.Audio.Input("reference_audio", tooltip="Recording for speaker identity. Emotion conditioning needs no reference transcript."), io.Combo.Input("emotion", options=list(prompts.EMOTIONS), tooltip="Requested emotional delivery. This task is separate from clone and basic attribute controls."),
                    ]),
                    io.DynamicCombo.Option("dialect", [
                        io.Audio.Input("reference_audio", tooltip="Recording for speaker identity. Dialect conditioning needs no reference transcript."), io.Combo.Input("dialect", options=list(prompts.DIALECTS), tooltip="Requested Chinese dialect: Sichuanese or Cantonese. Enter suitable speech text for the selected dialect."),
                    ]),
                ], tooltip="Select one task and its conditioning inputs. Clone uses audio plus transcript; other tasks use their own voice or delivery controls."),
                *sampling_inputs(),
            ], outputs=[io.Audio.Output()],
            description="Clone a reference voice, design a voice, use a built-in voice, or select one delivery task. Inline [Laughter], [Sigh] and other Ming tags stay in the text.",
        )

    @classmethod
    def execute(cls, model, text, task, **sampling):
        mode = task["task"]
        audio = task.get("reference_audio")
        references = [{"audio": audio, "text": task.get("reference_text", "")}] if audio is not None else []
        return io.NodeOutput(runtime.synthesize(model, text, mode, task, references=references, **sampling))


class MingReferenceSpeaker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingReferenceSpeaker", display_name="Ming TTS Podcast Speaker", category=CATEGORY,
            inputs=[
                io.String.Input("name", default="Alice", tooltip="Unique speaker name. Use the exact same name before ':' in the podcast script."),
                io.Audio.Input("audio", tooltip="Reference recording of this speaker. Its spoken words must match the transcript below."),
                io.String.Input("text", multiline=True, tooltip="Transcript of this speaker's reference recording. Type it or connect Whisper's transcript; this is not the podcast dialogue."),
            ],
            outputs=[MingSpeaker.Output()],
            description="A named reference and its spoken transcript for native Tiny dialogue.",
        )

    @classmethod
    def execute(cls, name, audio, text):
        return io.NodeOutput({"name": name, "audio": audio, "text": text})


class MingPodcast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingPodcast", display_name="Ming TTS Podcast (Tiny 9B)", category=CATEGORY,
            inputs=[
                MingModel.Input("model", tooltip="Connect a Tiny checkpoint from Ming TTS Load Model. The 0.5B checkpoint does not support native podcast dialogue."),
                io.String.Input("text", multiline=True, dynamic_prompts=False, tooltip="One 'Alice: speech' line per turn, matching the connected speaker names."),
                io.Autogrow.Input("speakers", template=io.Autogrow.TemplatePrefix(
                    MingSpeaker.Input("speaker", tooltip="Connect a Ming TTS Podcast Speaker containing its name, recording and transcript."), prefix="speaker", min=2, max=100,
                ), tooltip="Speaker references in conditioning order. Connect at least two and use their exact names in the dialogue script."),
                *sampling_inputs(),
            ], outputs=[io.Audio.Output()],
            description="One native multi-speaker generation with Ming Tiny. Speaker order controls the reference transcript and audio continuation.",
        )

    @classmethod
    def execute(cls, model, text, speakers, **sampling):
        references = [speakers[key] for key in sorted(speakers, key=lambda name: int(name.removeprefix("speaker"))) if speakers[key] is not None]
        if len(references) < 2:
            raise ValueError("Connect at least two podcast speakers.")
        names = [ref["name"] for ref in references]
        if len(names) != len(set(names)):
            raise ValueError("Give each podcast speaker a different name.")
        return io.NodeOutput(runtime.synthesize(model, text, "podcast", {}, references=references, **sampling))


class MingExtension(ComfyExtension):
    async def get_node_list(self):
        return [MingLoadModel, MingGenerate, MingReferenceSpeaker, MingPodcast, MingWhisperTranscribe]
