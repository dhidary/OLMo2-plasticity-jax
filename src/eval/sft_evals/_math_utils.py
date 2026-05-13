"""Vendored Minerva-style MATH answer extraction.

Ported verbatim from `lm_eval.tasks.minerva_math.utils` (Apache 2.0)
to avoid taking a dep on lm-eval-harness on TPU. Sympy is optional;
without it, `is_equiv` falls back to normalized-string compare (drift
~3-5pt on MATH but eval still runs).
"""

from __future__ import annotations

import re
import signal
from typing import Optional

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


class _Timeout:
    def __init__(self, seconds=5, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def _handle(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    if len(parts) > 1:
        for sub in parts[1:]:
            out += "\\frac"
            if sub and sub[0] == "{":
                out += sub
            else:
                if len(sub) < 2:
                    return s
                a, b = sub[0], sub[1]
                if b != "{":
                    rest = sub[2:] if len(sub) > 2 else ""
                    out += "{" + a + "}{" + b + "}" + rest
                else:
                    rest = sub[2:] if len(sub) > 2 else ""
                    out += "{" + a + "}" + b + rest
    return out


def _fix_a_slash_b(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        ai, bi = int(a), int(b)
        if s == f"{ai}/{bi}":
            return f"\\frac{{{ai}}}{{{bi}}}"
    except (ValueError, AssertionError):
        pass
    return s


def _remove_right_units(s: str) -> str:
    if "\\text{ " in s:
        splits = s.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return s


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    splits = s.split("\\sqrt")
    out = splits[0]
    for sp in splits[1:]:
        if sp and sp[0] != "{":
            out += "\\sqrt{" + sp[0] + "}" + sp[1:]
        else:
            out += "\\sqrt" + sp
    return out


def _strip_string(s: str) -> str:
    """Port of lm_eval.tasks.hendrycks_math.utils.strip_string (no sympy)."""
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s


def _hendrycks_is_equiv(s1: str, s2: str) -> bool:
    """Port of lm_eval.tasks.hendrycks_math.utils.is_equiv (string-only)."""
    if s1 is None or s2 is None:
        return False
    try:
        return _strip_string(s1) == _strip_string(s2)
    except Exception:
        return s1 == s2


def _sympy_is_equiv(x1: str, x2: str) -> bool:
    if not _HAS_SYMPY:
        return False
    try:
        with _Timeout(seconds=5):
            try:
                p1 = parse_latex(x1)
                p2 = parse_latex(x2)
            except Exception:
                return False
            try:
                diff = p1 - p2
            except TypeError:
                return False
            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception:
        return False


def is_equiv(x1: str, x2: str) -> bool:
    """OLMES ORs hendrycks (string-norm) and sympy. The hendrycks path
    needs no external deps and catches most cases; sympy is a backup."""
    if x1 == x2:
        return True
    if _hendrycks_is_equiv(x1, x2):
        return True
    return _sympy_is_equiv(x1, x2)


def get_unnormalized_answer(text: str) -> str:
    INVALID = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text = text + end_seq
    m = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.", text
    )
    return m.group(1).strip() if m else INVALID


_SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in _SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def extract_math_answers(raw_answer: str) -> list[str]:
    """Port of GenericMATH.extract_answers (CoT, Minerva style)."""
    boxed = last_boxed_only_string(raw_answer)
    if boxed is not None:
        try:
            boxed = remove_boxed(boxed)
        except AssertionError:
            boxed = None
    answers: list[str] = []
    minerva = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva is not None and minerva != "[invalidanswer]":
        answers.append(minerva)
    if boxed is not None:
        answers.append(normalize_final_answer(boxed))
    if len(answers) == 0:
        dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
        if len(dollars) > 1:
            ans = normalize_final_answer(
                raw_answer[dollars[-2] + 1 : dollars[-1]]
            )
            answers.append(ans)
    if len(answers) == 0:
        answers.append(normalize_final_answer(raw_answer))
    return answers
