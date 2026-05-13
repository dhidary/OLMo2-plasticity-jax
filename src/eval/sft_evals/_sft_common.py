"""Shared helpers for SFT-suite eval scripts.

Mirrors the chat + multiturn-fewshot prompt construction in OLMES
`oe_eval/tasks/base_task.py:459-576` and `oe_eval/run_eval.py:220-230`,
plus per-task answer-extraction logic ported verbatim from the
`oe_eval/tasks/oe_eval_tasks/*.py` files.

Always use `tokenizer.apply_chat_template` so each model's own template
definition wins (e.g. allenai/OLMo-2-0425-1B-SFT's no-leading-BOS variant).
"""

from __future__ import annotations

import re
import string
from typing import Optional

import numpy as np


# --- Prompt construction (mirrors oe_eval/tasks/base_task.py:517-576) ---

def _concat_with_space(s1: str, s2: str) -> str:
    """Mirror oe_eval/utils.py:371 concat_with_space."""
    s1 = s1.strip()
    s2 = s2.strip()
    if s1 and s2:
        return s1 + " " + s2
    return s1 or s2

def build_messages(
    system: Optional[str],
    fewshot: list[tuple[str, str]],
    user: str,
    multiturn: bool,
    description: Optional[str] = None,
    final_description: Optional[str] = None,
    assistant_prefix: Optional[str] = None,
) -> list[dict]:
    """Mirror OLMES `fewshot_context()` for chat-format multi-turn / single-turn.

    Args:
        system: optional system message
        fewshot: list of (user_text, assistant_text) pairs
        user: the test question (already formatted via doc_to_text)
        multiturn: if True, each shot becomes a separate user/assistant turn
                   (OLMES `fewshot_as_multiturn=True`)
        description: prepended to first user message
        final_description: appended to every user message
        assistant_prefix: appended to assistant turns (in multiturn mode);
                          stripped from end of any user msg if present
    """
    description = description or ""
    final_description = final_description or ""

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    if not fewshot:
        messages.append({
            "role": "user",
            "content": description + user + final_description,
        })
        return messages

    if multiturn:
        # Each shot becomes its own user/assistant turn.
        for idx, (u, a) in enumerate(fewshot):
            msg = u
            if idx == 0:
                msg = description + msg
            if assistant_prefix:
                msg = re.sub(r"\s*" + re.escape(assistant_prefix) + r"\s*$", "", msg)
            messages.append({"role": "user", "content": msg + final_description})
            messages.append({
                "role": "assistant",
                "content": _concat_with_space(assistant_prefix or "", a),
            })
        # Final test turn.
        msg = user
        if assistant_prefix:
            msg = re.sub(r"\s*" + re.escape(assistant_prefix) + r"\s*$", "", msg)
        messages.append({"role": "user", "content": msg + final_description})
    else:
        # Concatenate all shots into one user message (OLMES base_task.py:548-558).
        labeled = "\n\n".join(u + a for u, a in fewshot) + "\n\n"
        msg = description + labeled + user
        if assistant_prefix:
            msg = re.sub(r"\s*" + re.escape(assistant_prefix) + r"\s*$", "", msg)
        messages.append({"role": "user", "content": msg + final_description})

    return messages


# OLMo-2-0425-1B-SFT chat template (from allenai/OLMo-2-0425-1B-SFT
# tokenizer_config.json). Used as a fallback when our SFT'd checkpoints
# saved tokenizers without a chat_template field.
OLMO2_SFT_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    "{% endfor %}"
)


def format_prompt(
    tokenizer,
    messages: list[dict],
    assistant_prefix: Optional[str] = None,
) -> str:
    """Mirror OLMES `oe_eval/run_eval.py:220-230`. Falls back to the
    OLMo-2-0425-1B-SFT chat template if the tokenizer doesn't have one
    set (some of our SFT'd ckpts save tokenizers without chat_template)."""
    template = getattr(tokenizer, "chat_template", None) or OLMO2_SFT_CHAT_TEMPLATE
    out = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=template,
    )
    if assistant_prefix:
        out = out + assistant_prefix
    return out


# --- exact_match (port of oe_eval/dependencies/hf_evaluate/exact_match.py) ---

def exact_match_hf(
    predictions: list[str],
    references: list[str],
    regexes_to_ignore: Optional[list[str]] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
) -> float:
    if regexes_to_ignore:
        for s in regexes_to_ignore:
            predictions = [re.sub(s, "", x) for x in predictions]
            references = [re.sub(s, "", x) for x in references]
    pa = np.asarray(predictions)
    ra = np.asarray(references)
    if ignore_case:
        pa = np.char.lower(pa)
        ra = np.char.lower(ra)
    if ignore_punctuation:
        tbl = string.punctuation.maketrans("", "", string.punctuation)
        pa = np.char.translate(pa, table=tbl)
        ra = np.char.translate(ra, table=tbl)
    return float(np.mean(pa == ra))


# --- GSM8K (port of oe_eval/tasks/oe_eval_tasks/gsm8k.py:232-266) ---

GSM8K_REGEXES_TO_IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]


def extract_gsm8k_number(continuation: str) -> str:
    """Last numeric substring; matches OLMES default _extract_answer."""
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    return output


# --- MMLU CoT — verbatim port of OLMES GenericMMLU_OneTurnCoT._extract_answer ---

# Strict format from the :0shot_cot::tulu3 config; OLMES picks the LAST match.
_MMLU_COT_STRICT = re.compile(r"Therefore, the answer is \(([A-D])\)")

# Backup patterns tried in order via .search() (FIRST match), copied verbatim.
_MMLU_COT_BACKUPS = [
    re.compile(r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?([A-D])\b"),
    re.compile(r"(?i)answer\s*is:?\s*\(?([A-D])\b"),
    re.compile(r"(?i)([A-D])\)?\s+is\s+correct"),
    re.compile(r"\(([A-D])\)"),
    re.compile(r".*\b([A-D])\b"),
]


def extract_mmlu_cot_letter(continuation: str) -> str:
    """Return single letter A-D extracted from MMLU CoT response.

    Mirrors OLMES: strict format regex first taking the LAST match, then
    walks loosening backup patterns taking the FIRST match of each.
    """
    matches = _MMLU_COT_STRICT.findall(continuation)
    if matches:
        return matches[-1]
    for rgx in _MMLU_COT_BACKUPS:
        m = rgx.search(continuation)
        if m:
            return m.group(1).upper()
    return ""


# --- PopQA (port of oe_eval/tasks/oe_eval_tasks/popqa.py:160-180) ---

def popqa_alias_match(prediction: str, aliases: list[str]) -> bool:
    """OLMES alias substring match: label in pred OR label.lower() in pred
    OR label.capitalize() in pred, for any alias."""
    for label in aliases:
        if (label in prediction
                or label.lower() in prediction
                or label.capitalize() in prediction):
            return True
    return False


# --- BBH cot-v1 per-subtask regex extraction (port of OLMES bbh.py) ---

BBH_ANSWER_REGEX = {
    "boolean_expressions": "[tT]rue|[fF]alse",
    "causal_judgement": "[yY]es|[nN]o",
    "date_understanding": "MC",
    "disambiguation_qa": "MC",
    "dyck_languages": r"[\]\)\}\> ]+",
    "formal_fallacies": "[iI]nvalid|[vV]alid",
    "geometric_shapes": "MC",
    "hyperbaton": "MC",
    "logical_deduction_five_objects": "MC",
    "logical_deduction_seven_objects": "MC",
    "logical_deduction_three_objects": "MC",
    "movie_recommendation": "MC",
    "multistep_arithmetic_two": r"-?\d+",
    "navigate": "[nN]o|[yY]es",
    "object_counting": r"\d+",
    "penguins_in_a_table": "MC",
    "reasoning_about_colored_objects": "MC",
    "ruin_names": "MC",
    "salient_translation_error_detection": "MC",
    "snarks": "MC",
    "sports_understanding": "[yY]es|[nN]o",
    "temporal_sequences": "MC",
    "tracking_shuffled_objects_five_objects": "MC",
    "tracking_shuffled_objects_seven_objects": "MC",
    "tracking_shuffled_objects_three_objects": "MC",
    "web_of_lies": "[yY]es|[nN]o",
    "word_sorting": "[a-z ]+",
}

_BBH_PREFIX_REGEXES = [
    r"(?i)So the answer is ($ANS$)\.?",
    r"(?i)answer is ($ANS$)",
    r"(?i)answer:.*?($ANS$)",
    r"(?i)answer\b.*?($ANS$)",
    r"($ANS$)",
]

_BBH_DELIMITERS_TO_STRIP = [
    ("$", "$"), (r"\(", r"\)"), ("(", ")"), ("**", "**"), ("***", "***"),
    (r"\[", r"\]"), ("'", "'"), ("`", "`"), ('"', '"'),
]


def bbh_extract_v1(continuation: str, subtask: str) -> str:
    """Port of GenericBBH._extract_answer_regex (oe_eval/tasks/oe_eval_tasks/bbh.py:211-268)."""
    answer_regex = BBH_ANSWER_REGEX.get(subtask, "MC")
    is_mc = answer_regex == "MC"
    if is_mc:
        answer_regex = r"\([A-Z]\)"

    regexes = list(_BBH_PREFIX_REGEXES)
    if is_mc:
        regexes.append(r"\b([A-Z])\b")
    regexes.append(r"(?i)($ANS$)")

    extracted = ""
    for tmpl in regexes:
        rgx = tmpl.replace("$ANS$", answer_regex)
        found = re.findall(rgx, continuation)
        if found:
            extracted = found[-1]
            break

    for left, right in _BBH_DELIMITERS_TO_STRIP:
        if re.match(answer_regex, left):
            continue
        lr, rr = re.escape(left), re.escape(right)
        extracted = re.sub(rf"^{lr}(.*){rr}$", r"\1", extracted).strip()

    if is_mc and len(extracted) == 1:
        extracted = f"({extracted})"
    return extracted


def bbh_default_extract(continuation: str) -> str:
    """Port of GenericBBH._extract_answer (used by `bbh:cot::tulu`).

    Matches `(?<=the answer is )(.*)(?=.)`. We replicate as a non-greedy
    capture from `the answer is ` until the next period or end.
    """
    m = re.search(r"(?<=the answer is )(.*)(?=.)", continuation)
    if m:
        return m[0].strip(".")
    return ""


# --- TruthfulQA MC2 ---

def mc2_score(loglikelihoods: list[float], n_true: int) -> float:
    """Standard MC2: softmax over (true + false) logprobs, sum the true mass.
    The first `n_true` entries are the true completions."""
    arr = np.asarray(loglikelihoods, dtype=np.float64)
    arr = arr - arr.max()  # numerical stability
    probs = np.exp(arr)
    probs = probs / probs.sum()
    return float(probs[:n_true].sum())
