"""Split speech text without rewriting punctuation, whitespace or vocal tags."""

import re


_CJK = (
    "\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff"
    "\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7a3"
    "\ud7b0-\ud7ff\uf900-\ufaff\uff66-\uff9d\U00020000-\U000323af"
)
_WORD = rf"[^\W_{_CJK}]"
_LETTERS = rf"(?:{_WORD}|[\u0300-\u036f])"
_TOKEN = re.compile(
    r"\[[A-Za-z]+(?:-[A-Za-z]+)*\]"
    r"|\d+(?:[.,:/]\d+)*"
    r"|(?:[A-Za-z]\.){2,}"
    rf"|[{_CJK}][\u0300-\u036f\u3099\u309a\uff9e\uff9f]*"
    rf"|{_LETTERS}+(?:['’\-]{_LETTERS}+)*"
    r"|\s+|.",
    re.DOTALL,
)
_UNSPLIT_TASKS = frozenset(("voice_design", "freeform", "podcast"))


def split_text(text, task, max_length=50):
    """Bound fragments by CJK characters or words, preferring sentence breaks.

    Tags, numbers and dotted initials are atomic units. Fragments concatenate to the
    exact input, so short prompts and language-specific punctuation stay intact.
    Utterance-wide style instructions and multi-speaker dialogue stay together.
    """
    if max_length < 1:
        raise ValueError("Text fragment length must be positive.")
    if task in _UNSPLIT_TASKS:
        return [text]
    tokens = list(_TOKEN.finditer(text))
    fragments, start = [], 0
    while start < len(tokens):
        length, end, pending_break = 0, start, 0
        sentence_end = clause_end = None
        while end < len(tokens):
            token = tokens[end].group()
            has_content = token[0].isalnum() or token[0] == "["
            if has_content:
                if length >= max(1, max_length // 2):
                    if pending_break == 2:
                        sentence_end = end
                    elif pending_break == 1:
                        clause_end = end
                if length == max_length:
                    break
                length += 1
                pending_break = 0
            elif any(char in token for char in ".!?。！？\n\r"):
                pending_break = 2
            elif any(char in token for char in ",;:，、；："):
                pending_break = max(pending_break, 1)
            end += 1
        if end == len(tokens):
            fragments.append(text[tokens[start].start():])
            break
        end = sentence_end or clause_end or end
        fragments.append(text[tokens[start].start():tokens[end].start()])
        start = end
    return fragments or [text]
