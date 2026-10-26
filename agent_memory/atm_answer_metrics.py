"""Pinned ATM-Bench deterministic number/list answer metrics, offline only.

This module preserves the upstream deterministic normalization and list Jaccard
algorithms. It does not implement or claim the official aggregate ATM/QS score.
Only standard-library dependencies are imported; no model or I/O is invoked.

Porting changes: inline the exact ABSTENTION_PHRASES constant, keep the complete
normalizer, and select only the deterministic evaluator function closure.
Original source identities and license are retained below.

MIT License

Copyright (c) 2026 Jingbiao Mei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from collections import Counter

UPSTREAM_PROVENANCE = {'repository': 'https://github.com/JingbiaoMei/ATM-Bench', 'revision': 'ef4e5dff1a47ec71213a06e359f02753defa8fb1', 'license': 'MIT', 'copyright': 'Copyright (c) 2026 Jingbiao Mei', 'files': [{'path': 'memqa/utils/evaluator/normalizer.py', 'url': 'https://raw.githubusercontent.com/JingbiaoMei/ATM-Bench/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/utils/evaluator/normalizer.py', 'sha256': '1980cce4fbd572d11d2317725cb955d99cf55cf20e8645cd4ba6908fbdcf0e3d'}, {'path': 'memqa/utils/evaluator/evaluate_qa.py', 'url': 'https://raw.githubusercontent.com/JingbiaoMei/ATM-Bench/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/utils/evaluator/evaluate_qa.py', 'sha256': '0581963c1c6b9729398d8007f3d40664b2c026499c9cbf0436a92f77dcb1b66e'}, {'path': 'memqa/utils/evaluator/config.py', 'url': 'https://raw.githubusercontent.com/JingbiaoMei/ATM-Bench/ef4e5dff1a47ec71213a06e359f02753defa8fb1/memqa/utils/evaluator/config.py', 'sha256': '2a72ab0c4b18a31570bcfe014d877dac3be648de85393d67f0b09d3e6b9fc179'}, {'path': 'LICENSE', 'url': 'https://raw.githubusercontent.com/JingbiaoMei/ATM-Bench/ef4e5dff1a47ec71213a06e359f02753defa8fb1/LICENSE', 'sha256': 'e0de4445f9fdc0a5189592d7c97c6de1add6b63887fe7110509fc7957b7bcf1f'}], 'ported_evaluator_functions': ['_extract_list_ids_from_text', 'date_component_match', 'date_token_match', 'dates_match', 'tokens_match', '_deterministic_accuracy_core', 'deterministic_accuracy', '_prediction_list_items', '_list_jaccard_core', 'list_jaccard_score'], 'normalizer_policy': 'Full pinned normalizer; inline exact ABSTENTION_PHRASES constant instead of importing evaluator config.', 'behavior_policy': 'Deterministic algorithm unchanged; no model judging, qtype detection, CLI, file IO, provider, or evaluator configuration is imported.', 'official_aggregate_score': False}

# Exact constant from the pinned evaluator/config.py (no configuration import).
ABSTENTION_PHRASES = [
    "unknown",
    "abstention",
    "no information",
    "not available",
    "no evidence",
    "no evidnece",
    "insufficient information",
]

# BEGIN pinned normalizer.py (future/config imports removed only).
#!/usr/bin/env python3
"""Normalization utilities for QA evaluation."""


import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Optional, Tuple


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_PATTERN = "|".join(MONTHS.keys())

RE_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
RE_COMPACT_DATE = re.compile(
    r"\b((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b"
)
RE_MONTH_DAY = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_MONTH_RANGE = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:-|to|–)\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_TIME_RANGE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|to|–|and)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.IGNORECASE,
)
RE_TIME_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
RE_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

RE_CURRENCY_SYMBOL = re.compile(r"[£$€]")
RE_CURRENCY_CODE = re.compile(r"\b(USD|GBP|EUR|AUD|JOD)\b", re.IGNORECASE)
RE_CURRENCY_WORDS = re.compile(
    r"\b(dollars?|usd|pounds?|gbp|euros?|eur)\b", re.IGNORECASE
)
RE_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
RE_CODE = re.compile(r"\b[A-Z0-9]{4,}\b", re.IGNORECASE)
RE_TIME_AMPM_DOTS = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.m\.|p\.m\.)\b", re.IGNORECASE
)
RE_CURRENCY_AMOUNT_PREFIX = re.compile(
    r"(?P<symbol>[£$€])\s*(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
)
RE_CURRENCY_AMOUNT_SUFFIX = re.compile(
    r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?P<code>USD|GBP|EUR|AUD|JOD|dollars?|pounds?|euros?)\b",
    re.IGNORECASE,
)

LIST_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\band\b|/)\s*", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "for",
    "to",
    "and",
    "or",
    "by",
    "from",
    "with",
    "into",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
}

ADDRESS_KEYWORDS = {
    "street",
    "st",
    "road",
    "rd",
    "lane",
    "ln",
    "avenue",
    "ave",
    "drive",
    "dr",
    "place",
    "pl",
    "square",
    "sq",
    "close",
    "court",
    "ct",
    "way",
    "high",
    "highstreet",
    "boulevard",
    "blvd",
    "city",
    "town",
    "village",
    "county",
    "country",
    "oxford",
    "cambridge",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

DAYS_OF_WEEK = {
    "monday",
    "mon",
    "tuesday",
    "tue",
    "tues",
    "wednesday",
    "wed",
    "thursday",
    "thu",
    "thurs",
    "friday",
    "fri",
    "saturday",
    "sat",
    "sunday",
    "sun",
}

DAY_PATTERN = re.compile(r"\b(" + "|".join(DAYS_OF_WEEK) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class DateToken:
    value: str


@dataclass(frozen=True)
class TimeToken:
    value: str


def aggressive_preprocess(text: str) -> str:
    cleaned = text

    # Strip trailing periods from entire text
    cleaned = cleaned.rstrip(".")

    # Normalize comma separators in numbers
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

    # Strip trailing unit words that add no info ("spots", "hours", etc.)
    cleaned = re.sub(
        r"\b(\d+)\s+(spots?|hours?|sessions?|days?)\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", cleaned)
    cleaned = re.sub(r"\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip commas from numbers
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

    cleaned = cleaned.strip("\t\n\r \"'`.,;:!?()[]{}")
    return cleaned


def strip_parenthetical_details(text: str) -> str:
    cleaned = re.sub(r"\([^)]+\)", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_currency_breakdowns(text: str) -> str:
    cleaned = text
    # Remove "totaling X for Y sessions" patterns
    cleaned = re.sub(
        r",?\s*totaling\s+[£$€]\d+(?:\.\d+)?\s+for\s+\w+\s+sessions?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove "comprising X..." patterns
    cleaned = re.sub(
        r",?\s*comprising\s+.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove "per X-minute session" -> "per session"
    cleaned = re.sub(
        r"per\s+\d+[- ]minute\s+session",
        "per session",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_between_to_range(text: str) -> str:
    # Handle "between X and Y" -> "X to Y"
    cleaned = re.sub(
        r"\bbetween\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+and\s+",
        " to ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def strip_leading_articles(text: str) -> str:
    cleaned = re.sub(r"^\s*(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return cleaned


def strip_context_phrases(text: str) -> str:
    cleaned = text.lower()

    context_patterns = [
        r"\bit happened on\b",
        r"\bstarting at\b",
        r"\bfrom\s+\d",
        r"\bonwards\b",
        r"\btotaling\b",
        r"\bcomprising\b",
        r"\bwith pre-drinks at\b",
        r"\bwith pre-\w+\b",
        r"\band dinner at\b",
        r"\bdress code is\b",
        r"\bthe total cost was\b",
        r"\bcode:\s*",
        r"\bexpires?\s*",
    ]

    for pattern in context_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = DAY_PATTERN.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned




def is_abstention(text: str) -> bool:
    normalized = normalize_text(text)
    for phrase in ABSTENTION_PHRASES:
        if phrase in normalized:
            return True
    return False


def _month_to_number(name: str) -> Optional[int]:
    return MONTHS.get(name.lower())


def _format_date(year: Optional[int], month: int, day: int) -> str:
    if year:
        return f"{year:04d}{month:02d}{day:02d}"
    return f"{month:02d}{day:02d}"


def _format_range(year: Optional[int], month: int, start_day: int, end_day: int) -> str:
    if year:
        start = f"{year:04d}{month:02d}{start_day:02d}"
        end = f"{year:04d}{month:02d}{end_day:02d}"
    else:
        start = f"{month:02d}{start_day:02d}"
        end = f"{month:02d}{end_day:02d}"
    return f"{start}-{end}"


def extract_dates(text: str) -> List[DateToken]:
    tokens: List[DateToken] = []

    for match in RE_ISO_DATE.finditer(text):
        year, month, day = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_COMPACT_DATE.finditer(text):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_MONTH_RANGE.finditer(text):
        month = _month_to_number(match.group(1))
        if not month:
            continue
        start_day = int(match.group(2))
        end_day = int(match.group(3))
        year = int(match.group(4)) if match.group(4) else None
        tokens.append(DateToken(_format_range(year, month, start_day, end_day)))

    text_without_ranges = RE_MONTH_RANGE.sub(" ", text)

    for match in RE_MONTH_DAY.finditer(text_without_ranges):
        month = _month_to_number(match.group(1))
        if not month:
            continue
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_DAY_MONTH.finditer(text_without_ranges):
        day = int(match.group(1))
        month = _month_to_number(match.group(2))
        if not month:
            continue
        year = int(match.group(3)) if match.group(3) else None
        tokens.append(DateToken(_format_date(year, month, day)))

    return dedupe_tokens(tokens)


def _normalize_time(hour: int, minute: int, ampm: Optional[str]) -> str:
    if ampm:
        ampm = ampm.lower()
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour += 12
    return f"{hour:02d}:{minute:02d}"


def extract_times(text: str) -> List[TimeToken]:
    tokens: List[TimeToken] = []

    normalized = text.lower()
    normalized = normalized.replace("noon", "12:00")
    normalized = normalized.replace("midday", "12:00")
    normalized = normalized.replace("midnight", "00:00")

    normalized = normalized.replace("a.m.", "am").replace("p.m.", "pm")
    normalized = normalized.replace("a. m.", "am").replace("p. m.", "pm")

    normalized = RE_TIME_AMPM_DOTS.sub(
        lambda m: f"{m.group(1)}:{m.group(2) or '00'}{m.group(3)[0]}m", normalized
    )

    for match in RE_TIME_RANGE.finditer(normalized):
        start_hour = int(match.group(1))
        start_min = int(match.group(2) or 0)
        start_ampm = match.group(3)
        end_hour = int(match.group(4))
        end_min = int(match.group(5) or 0)
        end_ampm = match.group(6)
        if start_ampm is None and end_ampm is not None:
            start_ampm = end_ampm
        start = _normalize_time(start_hour, start_min, start_ampm)
        end = _normalize_time(end_hour, end_min, end_ampm)
        tokens.append(TimeToken(f"{start}-{end}"))

    text_without_ranges = RE_TIME_RANGE.sub(" ", normalized)

    for match in RE_TIME_24H.finditer(text_without_ranges):
        hour = int(match.group(1))
        minute = int(match.group(2))
        tokens.append(TimeToken(_normalize_time(hour, minute, None)))

    for match in RE_TIME_AMPM.finditer(text_without_ranges):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3)
        tokens.append(TimeToken(_normalize_time(hour, minute, ampm)))

    return dedupe_tokens(tokens)


def remove_date_time_text(text: str) -> str:
    cleaned = RE_ISO_DATE.sub(" ", text)
    cleaned = RE_COMPACT_DATE.sub(" ", cleaned)
    cleaned = RE_MONTH_RANGE.sub(" ", cleaned)
    cleaned = RE_MONTH_DAY.sub(" ", cleaned)
    cleaned = RE_DAY_MONTH.sub(" ", cleaned)
    cleaned = RE_TIME_RANGE.sub(" ", cleaned)
    cleaned = RE_TIME_24H.sub(" ", cleaned)
    cleaned = RE_TIME_AMPM.sub(" ", cleaned)
    cleaned = RE_TIME_AMPM_DOTS.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("noon", " ").replace("midday", " ").replace("midnight", " ")
    )
    cleaned = cleaned.replace("a.m.", " ").replace("p.m.", " ")
    cleaned = cleaned.replace("a. m.", " ").replace("p. m.", " ")
    return cleaned


def _normalize_decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _replace_number_words(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(text)
    for token in tokens:
        lower = token.lower()
        if lower in NUMBER_WORDS:
            text = re.sub(rf"\b{re.escape(token)}\b", str(NUMBER_WORDS[lower]), text)
    return text


def extract_currency_amounts(text: str) -> List[Decimal]:
    amounts: List[Decimal] = []

    normalized = _replace_number_words(text)

    # Remove commas from numbers before extraction
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)

    for match in RE_CURRENCY_AMOUNT_PREFIX.finditer(normalized):
        number = _normalize_decimal(match.group("number"))
        if number is not None:
            amounts.append(number.normalize())

    for match in RE_CURRENCY_AMOUNT_SUFFIX.finditer(normalized):
        number = _normalize_decimal(match.group("number"))
        if number is not None:
            amounts.append(number.normalize())

    return dedupe_tokens(amounts)


def extract_numbers(text: str) -> Tuple[List[Decimal], List[str]]:
    values: List[Decimal] = []
    currencies: List[str] = []

    for match in RE_CURRENCY_CODE.finditer(text):
        currencies.append(match.group(1).upper())

    for match in RE_CURRENCY_WORDS.finditer(text):
        word = match.group(1).lower()
        if "usd" in word or "dollar" in word:
            currencies.append("USD")
        elif "gbp" in word or "pound" in word:
            currencies.append("GBP")
        elif "eur" in word or "euro" in word:
            currencies.append("EUR")

    symbol_map = {"£": "GBP", "$": "USD", "€": "EUR"}
    for symbol_match in RE_CURRENCY_SYMBOL.finditer(text):
        symbol = symbol_match.group(0)
        mapped = symbol_map.get(symbol)
        if mapped:
            currencies.append(mapped)

    normalized = _replace_number_words(text)
    for match in RE_NUMBER.finditer(normalized):
        number = _normalize_decimal(match.group(0))
        if number is not None:
            values.append(number.normalize())

    return values, currencies


def extract_codes(text: str) -> List[str]:
    tokens: List[str] = []
    for match in RE_CODE.finditer(text):
        value = match.group(0)
        # Skip patterns that look like dates with ordinal suffixes (e.g., "25TH", "1ST")
        if re.match(r"^\d{1,2}(ST|ND|RD|TH)$", value.upper()):
            continue
        if any(char.isalpha() for char in value) and any(
            char.isdigit() for char in value
        ):
            tokens.append(value.upper())
    return dedupe_tokens(tokens)


def normalize_currency_codes(currencies: List[str]) -> List[str]:
    normalized = set()
    for curr in currencies:
        curr_upper = curr.upper()
        if curr_upper in ["GBP", "POUNDS", "POUND", "£"]:
            normalized.add("GBP")
        elif curr_upper in ["USD", "DOLLARS", "DOLLAR", "$"]:
            normalized.add("USD")
        elif curr_upper in ["EUR", "EUROS", "EURO", "€"]:
            normalized.add("EUR")
        elif curr_upper in ["AUD"]:
            normalized.add("AUD")
        elif curr_upper in ["JOD"]:
            normalized.add("JOD")
        else:
            normalized.add(curr_upper)
    return sorted(list(normalized))


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def split_list_items(text: str) -> List[str]:
    items = [item.strip() for item in LIST_SPLIT_PATTERN.split(text) if item.strip()]
    return [normalize_text(item) for item in items if item]


def token_subset_match(ground_truth: str, prediction: str) -> bool:
    gt_tokens = [token for token in tokenize(ground_truth) if token not in STOPWORDS]
    pred_tokens = set(tokenize(prediction))

    # Empty GT (all stopwords removed) → vacuously true
    if not gt_tokens:
        return True

    if len(gt_tokens) == 1:
        return gt_tokens[0] in pred_tokens
    return all(token in pred_tokens for token in gt_tokens)


def location_token_match(ground_truth: str, prediction: str) -> bool:
    gt_tokens = tokenize(ground_truth)
    if not any(token in ADDRESS_KEYWORDS for token in gt_tokens):
        return False
    filtered = [
        token for token in gt_tokens if token not in STOPWORDS and not token.isdigit()
    ]
    pred_tokens = set(tokenize(prediction))
    if not filtered:
        return False
    return all(token in pred_tokens for token in filtered)


def extract_reference_date(text: str) -> Optional[date]:
    tokens = extract_dates(text)
    for token in tokens:
        value = token.value
        if len(value) == 8:
            year = int(value[0:4])
            month = int(value[4:6])
            day = int(value[6:8])
            return date(year, month, day)
    return None


def resolve_relative_dates(text: str, reference: Optional[date]) -> str:
    if not reference:
        return text
    mapping = {
        "today": reference,
        "tonight": reference,
        "tomorrow": reference + timedelta(days=1),
        "yesterday": reference - timedelta(days=1),
    }
    resolved = text
    for key, value in mapping.items():
        iso_value = value.strftime("%Y-%m-%d")
        month_day = value.strftime("%B %d")

        # Replace "Tomorrow" with both ISO date and readable format
        if key in resolved.lower():
            # Try to preserve capitalization
            for pattern in [key, key.capitalize(), key.upper()]:
                if pattern in resolved:
                    # Replace with month-day format for better matching
                    resolved = resolved.replace(pattern, month_day)
                    break

    # Remove parenthetical date explanations after relative dates
    resolved = re.sub(
        r"(today|tonight|tomorrow|yesterday)\s*\([^)]+\)",
        r"\1",
        resolved,
        flags=re.IGNORECASE,
    )

    return resolved


def semantic_units_match(ground_truth: str, prediction: str) -> bool:
    gt_dates = extract_dates(ground_truth)
    pr_dates = extract_dates(prediction)
    gt_times = extract_times(ground_truth)
    pr_times = extract_times(prediction)
    gt_amounts = extract_currency_amounts(ground_truth)
    pr_amounts = extract_currency_amounts(prediction)

    # If GT has dates, PR must have matching dates
    if gt_dates:
        if not pr_dates:
            return False
        gt_date_vals = {d.value for d in gt_dates}
        pr_date_vals = {d.value for d in pr_dates}
        # Allow flexible date matching (e.g., "0725" matches "20220725")
        if not any(
            gt_d[-4:] == pr_d[-4:] for gt_d in gt_date_vals for pr_d in pr_date_vals
        ):
            return False

    # If GT has times, PR should have matching times (but not required)
    if gt_times and pr_times:
        gt_time_vals = {t.value for t in gt_times}
        pr_time_vals = {t.value for t in pr_times}
        if not gt_time_vals.intersection(pr_time_vals):
            return False

    # If GT has currency amounts, PR must have them
    if gt_amounts:
        if not pr_amounts:
            return False
        if set(gt_amounts) != set(pr_amounts):
            return False

    # Token overlap check
    gt_tokens = set(tokenize(ground_truth))
    pr_tokens = set(tokenize(prediction))

    # Remove stopwords
    gt_content = {t for t in gt_tokens if t not in STOPWORDS and len(t) > 2}
    pr_content = {t for t in pr_tokens if t not in STOPWORDS and len(t) > 2}

    if not gt_content:
        return True

    # If 80%+ of GT content tokens are in PR, it's a match
    overlap = len(gt_content.intersection(pr_content))
    if overlap / len(gt_content) >= 0.8:
        return True

    return False


def dedupe_tokens(tokens: Iterable) -> List:
    seen = set()
    result = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


# END pinned normalizer.py.

# BEGIN selected pinned evaluate_qa.py deterministic functions.

_RE_MEDIA_ID = re.compile(r"\b(\d{8}_\d{6})\b")

_RE_EMAIL_ID = re.compile(r"\b(email\d+)\b", re.IGNORECASE)

def _extract_list_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for m in _RE_MEDIA_ID.finditer(text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            ids.append(v)
    for m in _RE_EMAIL_ID.finditer(text):
        v = m.group(1).lower()
        if v not in seen:
            seen.add(v)
            ids.append(v)
    return ids

def date_component_match(gt: str, pred: str) -> bool:
    if len(gt) == 4 or len(pred) == 4:
        return gt[-4:] == pred[-4:]
    return gt == pred

def date_token_match(gt_token: str, pred_token: str) -> bool:
    if "-" in gt_token:
        if "-" not in pred_token:
            return False
        gt_start, gt_end = gt_token.split("-", 1)
        pred_start, pred_end = pred_token.split("-", 1)
        return date_component_match(gt_start, pred_start) and date_component_match(
            gt_end, pred_end
        )
    return date_component_match(gt_token, pred_token)

def dates_match(gt_tokens: List[str], pred_tokens: List[str]) -> bool:
    remaining = list(pred_tokens)
    for gt_token in gt_tokens:
        matched = False
        for idx, pred_token in enumerate(remaining):
            if date_token_match(gt_token, pred_token):
                matched = True
                remaining.pop(idx)
                break
        if not matched:
            return False
    return True

def tokens_match(expected: List, actual: List) -> bool:
    return Counter(expected) == Counter(actual)

def _deterministic_accuracy_core(
    ground_truth: str, prediction: str, question: Optional[str] = None
) -> tuple[bool, str]:
    is_abst_gt = is_abstention(ground_truth)
    is_abst_pred = is_abstention(prediction)

    reference_date = extract_reference_date(question or "")
    ground_truth_resolved = resolve_relative_dates(ground_truth, reference_date)
    prediction_resolved = resolve_relative_dates(prediction, reference_date)

    ground_truth_prep = aggressive_preprocess(ground_truth_resolved)
    prediction_prep = aggressive_preprocess(prediction_resolved)

    ground_truth_clean = strip_parenthetical_details(ground_truth_prep)
    prediction_clean = strip_parenthetical_details(prediction_prep)

    ground_truth_clean = strip_currency_breakdowns(ground_truth_clean)
    prediction_clean = strip_currency_breakdowns(prediction_clean)

    ground_truth_clean = normalize_between_to_range(ground_truth_clean)
    prediction_clean = normalize_between_to_range(prediction_clean)

    ground_truth_clean = strip_leading_articles(ground_truth_clean)
    prediction_clean = strip_leading_articles(prediction_clean)

    gt_normalized = normalize_text(ground_truth_clean)
    pred_normalized = normalize_text(prediction_clean)

    if is_abst_gt or is_abst_pred:
        return (is_abst_gt and is_abst_pred, pred_normalized)

    gt_codes = extract_codes(ground_truth_clean)
    if gt_codes:
        pred_upper = prediction_clean.upper()
        pred_stripped = re.sub(
            r"\b(?:CODE|ID|REFERENCE|REF|NUMBER|NUM|LABEL):\s*", "", pred_upper
        )
        if not all(code in pred_stripped for code in gt_codes):
            return False, pred_normalized

    gt_dates = [token.value for token in extract_dates(ground_truth_clean)]
    pred_dates = [token.value for token in extract_dates(prediction_clean)]

    if gt_dates:
        if not pred_dates:
            return False, pred_normalized
        if not dates_match(gt_dates, pred_dates):
            return False, pred_normalized

    gt_times = [token.value for token in extract_times(ground_truth_clean)]
    pred_times = [token.value for token in extract_times(prediction_clean)]
    if gt_times and pred_times:
        if not tokens_match(gt_times, pred_times):
            return False, pred_normalized

    if gt_dates or gt_times:
        gt_remainder = normalize_text(remove_date_time_text(ground_truth_clean))
        pred_remainder = normalize_text(remove_date_time_text(prediction_clean))

        gt_remainder_stripped = strip_context_phrases(gt_remainder)
        pred_remainder_stripped = strip_context_phrases(pred_remainder)

        if not gt_remainder_stripped:
            return True, pred_normalized
        if token_subset_match(gt_remainder_stripped, pred_remainder_stripped):
            return True, pred_normalized

    gt_clean = remove_date_time_text(ground_truth_clean)
    pred_clean = remove_date_time_text(prediction_clean)
    gt_numbers, gt_currencies = extract_numbers(gt_clean)
    pred_numbers, pred_currencies = extract_numbers(pred_clean)
    gt_currency_amounts = extract_currency_amounts(gt_clean)
    pred_currency_amounts = extract_currency_amounts(pred_clean)

    if gt_currencies:
        gt_norm = normalize_currency_codes(gt_currencies)
        pred_norm = normalize_currency_codes(pred_currencies)
        if pred_norm and not tokens_match(gt_norm, pred_norm):
            return False, pred_normalized

    if gt_currency_amounts:
        if not pred_currency_amounts:
            return False, pred_normalized
        if not tokens_match(gt_currency_amounts, pred_currency_amounts):
            return False, pred_normalized
        gt_numbers = [num for num in gt_numbers if num not in gt_currency_amounts]
        pred_numbers = [num for num in pred_numbers if num not in pred_currency_amounts]

    if gt_numbers:
        if not pred_numbers:
            return False, pred_normalized
        if not tokens_match(gt_numbers, pred_numbers):
            return False, pred_normalized

    gt_items = split_list_items(ground_truth_clean)
    if len(gt_items) >= 2:
        if not all(item in pred_normalized for item in gt_items):
            return False, pred_normalized
        return True, pred_normalized

    if gt_normalized.startswith("yes") or gt_normalized.startswith("no"):
        if pred_normalized in {"yes", "no"}:
            return gt_normalized.startswith(pred_normalized), pred_normalized

    if location_token_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    if len(gt_normalized) >= 3 and gt_normalized in pred_normalized:
        return True, pred_normalized

    if len(pred_normalized) >= 3 and pred_normalized in gt_normalized:
        return True, pred_normalized

    if token_subset_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    # Final fallback: semantic units match
    if semantic_units_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    return gt_normalized == pred_normalized, pred_normalized

def deterministic_accuracy(
    ground_truth: str, prediction: str, question: Optional[str] = None
) -> bool:
    return _deterministic_accuracy_core(ground_truth, prediction, question)[0]

def _prediction_list_items(prediction: str) -> list[str]:
    pred_ids = _extract_list_ids_from_text(prediction)
    if pred_ids:
        return pred_ids
    return [item for item in split_list_items(prediction) if item]

def _list_jaccard_core(ground_truth: str, prediction: str) -> tuple[float, list[str]]:
    gt_items = {item for item in split_list_items(ground_truth) if item}
    pred_items_list = _prediction_list_items(prediction)
    pred_items = set(pred_items_list)
    if not gt_items and not pred_items:
        return 1.0, pred_items_list
    if not gt_items:
        return 0.0, pred_items_list
    union = gt_items | pred_items
    if not union:
        return 0.0, pred_items_list
    return len(gt_items & pred_items) / len(union), pred_items_list

def list_jaccard_score(ground_truth: str, prediction: str) -> float:
    return _list_jaccard_core(ground_truth, prediction)[0]
