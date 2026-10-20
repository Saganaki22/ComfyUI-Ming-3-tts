"""Task serialization at the Ming integration boundary."""

import json
import re

from . import dense_prompt, tiny_prompt


LANGUAGES = ("Chinese", "English", "German", "Spanish", "French", "Korean", "Japanese")
EMOTIONS = {"happy": "高兴", "sad": "悲伤", "angry": "愤怒", "afraid": "恐惧", "disgusted": "厌恶", "surprised": "惊讶", "neutral": "中性"}
DIALECTS = {"Sichuanese": "川渝话", "Cantonese": "广粤话"}
SPEEDS = {"slow": "慢速", "medium": "中速", "fast": "快速"}
LEVELS = {"low": "低", "medium": "中", "high": "高"}


def control_instruction(task, values, tiny):
    dialect = values.get("dialect")
    if dialect is not None:
        dialect = DIALECTS[dialect] if tiny else {"Sichuanese": "chuanyu dialect", "Cantonese": "cantonese"}[dialect]
    fields = {
        "序号": 1, "说话人": "speaker_1", "方言": dialect,
        "风格": values.get("style"), "语速": SPEEDS.get(values.get("speed")),
        "基频": LEVELS.get(values.get("pitch")), "音量": LEVELS.get(values.get("volume")),
        "情感": EMOTIONS.get(values.get("emotion")),
        "BGM": {key: None for key in ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")},
        "IP": values.get("voice"),
    }
    if tiny and task != "emotion":
        order = ("序号", "说话人", "方言", "情感", "风格", "语速", "基频", "音量", "BGM", "IP")
        fields = {key: fields[key] for key in order}
    if not tiny and task == "dialect":
        names = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion", "BGM", "ip")
        fields = dict(zip(names, fields.values()))
    return json.dumps({"audio_sequence": [fields]}, ensure_ascii=False)


def build_prompt(tokenizer, config, task, text, values, speaker_count, latent_tokens, reference_text):
    tiny = config["model_type"] == "lingv3_tts"
    instruction = None
    if task == "freeform":
        instruction = values["instruction"]
        if tiny and not instruction.startswith("Speaking instruction:"):
            instruction = "Speaking instruction: Speak in this style: " + instruction
    elif task in ("voice_design", "ip", "basic", "emotion", "dialect"):
        instruction = control_instruction(task, values, tiny)
    language = values.get("language") if task == "clone" else None
    if tiny:
        if task == "podcast":
            return (*tiny_prompt.build_multispeaker_tts_prompt_ids(
                tokenizer, text, config["tts_config"], speaker_count=speaker_count,
                prompt_text=reference_text, prompt_latent_tokens=latent_tokens,
            ),)
        ids, position = tiny_prompt.build_tts_prompt_ids(
            tokenizer, text, config["tts_config"], instruction=instruction,
            prompt_text=reference_text, prompt_latent_tokens=latent_tokens,
            speaker_slot=True, assistant_instruction=True,
            target_language=language, include_target_language=task == "clone",
        )
        return ids, [position]
    prompt = "Please generate speech based on the following description.\n"
    if language:
        prompt += f"Target language: {language}.\n"
    ids, positions, _ = dense_prompt.build_input_ids(
        tokenizer, "dense", prompt, text, num_speakers=1, instruction=instruction,
        prompt_text=reference_text, prompt_latent_steps=latent_tokens,
    )
    return ids, positions


def podcast_text(text, references):
    mapping = {reference["name"]: i for i, reference in enumerate(references, 1)}
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.match(r"\s*([^:]+):\s*(.*)", line)
        if match is None:
            raise ValueError("Podcast script needs 'speaker_name: text' on every line.")
        name, content = match.groups()
        index = mapping.get(name)
        if index is None and re.fullmatch(r"speaker_\d+", name):
            index = int(name.split("_")[1])
        if index is None or not 1 <= index <= len(references):
            raise ValueError(f"Podcast speaker has no reference: {name}")
        lines.append(f" speaker_{index}:{content}\n")
    transcript = "".join(f" speaker_{i}:{ref['text'].strip()}\n" for i, ref in enumerate(references, 1))
    return "".join(lines), transcript
