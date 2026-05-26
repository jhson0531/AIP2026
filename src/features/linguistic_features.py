import re
import unicodedata
import numpy as np
import pandas as pd
from typing import List

# Hangul unicode ranges
HANGUL_SYLLABLE_RE = re.compile(r"[가-힣]")
HANGUL_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
CHOSUNG_RE = re.compile(r"[ㄱ-ㅎ]")  # consonants only (초성)
LAUGHTER_KK = re.compile(r"ㅋ")
LAUGHTER_HH = re.compile(r"ㅎ")
ENGLISH_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[0-9]")
SPECIAL_RE = re.compile(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]", re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1+")

CHOSUNG_LIST = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
VOWEL_LIST = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def _char_features(text: str) -> dict:
    f = {}
    f["char_len"] = len(text)
    tokens = text.split()
    f["token_count_space"] = len(tokens)
    f["avg_token_len"] = np.mean([len(t) for t in tokens]) if tokens else 0.0

    hangul = len(HANGUL_SYLLABLE_RE.findall(text))
    jamo = len(HANGUL_JAMO_RE.findall(text))
    english = len(ENGLISH_RE.findall(text))
    digit = len(DIGIT_RE.findall(text))
    total = max(len(text), 1)

    f["hangul_char_count"] = hangul
    f["english_char_count"] = english
    f["digit_char_count"] = digit
    f["special_char_count"] = len(SPECIAL_RE.findall(text))

    f["hangul_ratio"] = hangul / total
    f["english_ratio"] = english / total
    f["digit_ratio"] = digit / total
    f["special_ratio"] = f["special_char_count"] / total

    f["has_hangul"] = int(hangul > 0)
    f["has_english"] = int(english > 0)
    f["has_digit"] = int(digit > 0)
    f["is_kor_eng_mixed"] = int(hangul > 0 and english > 0)

    f["has_question_mark"] = int("?" in text)
    f["has_exclamation_mark"] = int("!" in text)
    f["has_tilde"] = int("~" in text)
    f["has_ellipsis"] = int("..." in text or "…" in text)

    chosung_count = len(CHOSUNG_RE.findall(text))
    f["has_chosung"] = int(chosung_count > 0)
    f["chosung_count"] = chosung_count
    f["has_laughter_kk"] = int(bool(LAUGHTER_KK.search(text)))
    f["has_laughter_hh"] = int(bool(LAUGHTER_HH.search(text)))
    f["count_kk"] = len(LAUGHTER_KK.findall(text))
    f["count_hh"] = len(LAUGHTER_HH.findall(text))

    repeat_matches = REPEAT_RE.findall(text)
    all_repeat_spans = [(m.start(), m.end()) for m in REPEAT_RE.finditer(text)]
    f["has_repeated_char"] = int(len(repeat_matches) > 0)
    f["max_repeated_char_len"] = max((m.end() - m.start() for m in REPEAT_RE.finditer(text)), default=0)
    f["repeated_char_count"] = len(repeat_matches)

    return f


def _kiwi_features(text: str, kiwi) -> dict:
    f = {}
    try:
        result = kiwi.analyze(text)[0][0]
        morphs = result
    except Exception:
        morphs = []

    pos_map = {
        "noun": {"NNG", "NNP", "NNB", "NP", "NR"},
        "verb": {"VV", "VX"},
        "adjective": {"VA"},
        "adverb": {"MAG", "MAJ"},
        "interjection": {"IC"},
        "josa": set(),  # J prefix
        "eomi": set(),  # E prefix
    }

    counts = {k: 0 for k in pos_map}
    total_morphs = len(morphs)
    pos_types = set()

    for token in morphs:
        tag = token.tag
        pos_types.add(str(tag))
        tag_str = str(tag)
        if tag_str in pos_map["noun"]:
            counts["noun"] += 1
        elif tag_str in pos_map["verb"]:
            counts["verb"] += 1
        elif tag_str in pos_map["adjective"]:
            counts["adjective"] += 1
        elif tag_str in pos_map["adverb"]:
            counts["adverb"] += 1
        elif tag_str in pos_map["interjection"]:
            counts["interjection"] += 1
        elif tag_str.startswith("J"):
            counts["josa"] += 1
        elif tag_str.startswith("E"):
            counts["eomi"] += 1

    f["morph_count"] = total_morphs
    f["unique_morph_count"] = len(set(str(t.form) for t in morphs))
    f["pos_type_count"] = len(pos_types)

    denom = max(total_morphs, 1)
    for k, v in counts.items():
        f[f"{k}_count"] = v
        f[f"{k}_ratio"] = v / denom

    return f


def _kiwi_fallback_features() -> dict:
    keys = [
        "morph_count", "unique_morph_count", "pos_type_count",
        "noun_count", "verb_count", "adjective_count", "adverb_count",
        "interjection_count", "josa_count", "eomi_count",
        "noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio",
        "interjection_ratio", "josa_ratio", "eomi_ratio",
    ]
    return {k: 0.0 for k in keys}


def _expression_features(text: str, meme_patterns: list) -> dict:
    f = {}
    tokens = text.split()
    n_tokens = len(tokens)
    n_chars = len(text)

    f["is_word_like"] = int(n_tokens == 1 and n_chars <= 10)
    f["is_phrase_like"] = int(1 < n_tokens <= 5)
    f["is_sentence_like"] = int(n_tokens > 5)
    f["is_question_like"] = int(text.endswith("?") or text.endswith("나") or text.endswith("까"))
    f["is_exclamation_like"] = int(text.endswith("!") or "!" in text)
    f["is_command_like"] = int(text.endswith("해") or text.endswith("봐") or text.endswith("자"))

    f["has_abbreviation_pattern"] = int(bool(re.search(r"[ㄱ-ㅎ]{2,}", text)))
    f["has_initialism_pattern"] = int(bool(re.search(r"^[ㄱ-ㅎ]{2,4}$", text.strip())))
    f["has_meme_template_pattern"] = int(any(p in text for p in meme_patterns))
    f["has_quote_like_pattern"] = int(
        bool(re.search(r"[\"'“”‘’]", text)) or
        (text.startswith(""") or text.endswith("""))
    )
    return f


def _rhythm_features(text: str) -> dict:
    f = {}
    syllables = HANGUL_SYLLABLE_RE.findall(text)
    n_syl = len(syllables)
    tokens = text.split()
    f["syllable_count"] = n_syl
    f["avg_syllable_per_token"] = n_syl / max(len(tokens), 1)

    # same syllable repeat
    same_rep = REPEAT_RE.findall("".join(syllables))
    f["same_syllable_repeat_count"] = len(same_rep)
    f["max_same_syllable_repeat_len"] = max(
        (m.end() - m.start() for m in REPEAT_RE.finditer("".join(syllables))), default=0
    )

    # initial consonant repeat score
    initials = []
    for syl in syllables:
        code = ord(syl) - 0xAC00
        if code >= 0:
            initials.append(code // (21 * 28))
    initial_rep = sum(
        1 for i in range(len(initials) - 1) if initials[i] == initials[i + 1]
    )
    f["initial_consonant_repeat_score"] = initial_rep / max(len(initials) - 1, 1) if len(initials) > 1 else 0.0

    # vowel repeat score
    vowels = []
    for syl in syllables:
        code = ord(syl) - 0xAC00
        if code >= 0:
            vowels.append((code % (21 * 28)) // 28)
    vowel_rep = sum(
        1 for i in range(len(vowels) - 1) if vowels[i] == vowels[i + 1]
    )
    f["vowel_repeat_score"] = vowel_rep / max(len(vowels) - 1, 1) if len(vowels) > 1 else 0.0

    # final consonant ratio
    has_final = sum(1 for syl in syllables if (ord(syl) - 0xAC00) % 28 != 0)
    f["final_consonant_ratio"] = has_final / max(n_syl, 1)
    f["no_final_consonant_ratio"] = 1.0 - f["final_consonant_ratio"]

    # rhythm_score (heuristic composite)
    short_bonus = max(0.0, (6 - n_syl) / 6) if n_syl > 0 else 0.0
    balanced_bonus = 1.0 / max(np.std([len(t) for t in tokens]) + 1, 1) if tokens else 0.0
    f["rhythm_score"] = (
        f["same_syllable_repeat_count"] * 0.3
        + f["initial_consonant_repeat_score"] * 0.3
        + f["vowel_repeat_score"] * 0.2
        + short_bonus * 0.1
        + balanced_bonus * 0.1
    )
    f["pronunciation_simplicity_score"] = f["no_final_consonant_ratio"] * 0.5 + (1 - min(n_syl / 10, 1)) * 0.5

    return f


def extract_linguistic_features(texts: pd.Series, cfg: dict) -> pd.DataFrame:
    lc = cfg["linguistic"]
    use_kiwi = lc.get("use_kiwi", True)
    meme_patterns = lc.get("meme_template_patterns", [])

    kiwi = None
    if use_kiwi:
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
        except ImportError:
            print("kiwipiepy not available, skipping kiwi features")
            use_kiwi = False

    records = []
    for text in texts:
        text = str(text)
        row = {}
        row.update(_char_features(text))
        if use_kiwi and kiwi is not None:
            row.update(_kiwi_features(text, kiwi))
        else:
            row.update(_kiwi_fallback_features())
        row.update(_expression_features(text, meme_patterns))
        row.update(_rhythm_features(text))
        records.append(row)

    return pd.DataFrame(records)
