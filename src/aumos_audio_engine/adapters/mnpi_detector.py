"""MNPI (Material Non-Public Information) detection adapter.

Implements MNPIDetectorProtocol. Combines regex/keyword matching with
context-window analysis to detect sensitive financial information in
transcripts, aligned with SEC Rule 10b-5 and Regulation FD requirements.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aumos_common.observability import get_logger

from aumos_audio_engine.settings import Settings

logger = get_logger(__name__)


# ── Risk classification thresholds ──────────────────────────────────────────

_RISK_THRESHOLDS: dict[str, tuple[float, float]] = {
    "critical": (0.90, 1.01),
    "high":     (0.75, 0.90),
    "medium":   (0.50, 0.75),
    "low":      (0.25, 0.50),
    "none":     (0.00, 0.25),
}

# ── Default MNPI keyword categories ─────────────────────────────────────────

_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "earnings per share", "eps", "revenue miss", "beat expectations",
        "guidance", "forecast update", "quarterly results", "annual results",
        "preliminary results", "unaudited results", "non-gaap", "adjusted ebitda",
    ],
    "mergers_acquisitions": [
        "merger", "acquisition", "takeover", "buyout", "tender offer",
        "letter of intent", "loi", "term sheet", "due diligence",
        "hostile takeover", "friendly acquisition", "spin-off", "divestiture",
        "asset sale",
    ],
    "regulatory": [
        "sec investigation", "doj inquiry", "regulatory action",
        "consent order", "cease and desist", "subpoena", "grand jury",
        "settlement", "fine", "penalty", "enforcement action",
        "securities fraud", "insider trading",
    ],
    "corporate_events": [
        "bankruptcy", "chapter 11", "chapter 7", "restructuring",
        "debt restructuring", "covenant breach", "default",
        "board resignation", "ceo departure", "cfo departure",
        "restatement", "accounting irregularity", "material weakness",
        "going concern", "liquidity crisis",
    ],
    "product_pipeline": [
        "fda approval", "clinical trial results", "phase 3 results",
        "patent grant", "patent expiry", "product recall", "safety issue",
        "material contract", "major customer", "contract termination",
    ],
    "financing": [
        "secondary offering", "share buyback", "stock repurchase",
        "dividend cut", "dividend increase", "debt offering", "bond issuance",
        "credit facility", "equity raise", "convertible note",
    ],
}


@dataclass
class _DetectionMatch:
    """Internal representation of a single keyword match with context."""

    keyword: str
    category: str
    start_char: int
    end_char: int
    context_text: str
    raw_confidence: float
    regulatory_ref: str = ""
    redaction_recommended: bool = False
    flagged_phrases: list[str] = field(default_factory=list)


class MNPIDetector:
    """Material Non-Public Information detector for financial transcript analysis.

    Performs multi-layer MNPI detection:
    1. Keyword/phrase matching across configurable category lists.
    2. Context-window analysis (50-token window around each match).
    3. Co-occurrence scoring (multiple categories = higher risk).
    4. Regulatory classification referencing SEC Rule 10b-5 / Reg FD.
    5. Redaction recommendations for flagged segments.

    Thread-safe: the keyword lists are loaded once and are read-only after that.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize MNPI detector with service settings.

        Args:
            settings: Audio engine settings with mnpi_keywords_path, context window,
                and confidence threshold parameters.
        """
        self._settings = settings
        self._keyword_patterns: dict[str, list[re.Pattern[str]]] = {}
        self._raw_keywords: dict[str, list[str]] = {}
        self._keywords_loaded = False

        # Load defaults at construction time (async load can supplement later)
        self._load_default_keywords()

    def _load_default_keywords(self) -> None:
        """Compile default MNPI keyword patterns into case-insensitive regex patterns."""
        for category, keywords in _DEFAULT_KEYWORDS.items():
            self._raw_keywords[category] = list(keywords)
            self._keyword_patterns[category] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
        self._keywords_loaded = True
        logger.info(
            "Default MNPI keywords loaded",
            categories=list(self._keyword_patterns.keys()),
            total_keywords=sum(len(kws) for kws in self._raw_keywords.values()),
        )

    async def load_keywords(self, keywords_path: str) -> int:
        """Load supplemental MNPI keywords from a JSON file.

        The JSON file must be structured as:
        ``{"category_name": ["keyword1", "keyword2", ...], ...}``

        Merges with default keywords — existing categories are extended,
        new categories are added.

        Args:
            keywords_path: Absolute path to the JSON keyword file.

        Returns:
            Total number of keywords now loaded (across all categories).

        Raises:
            FileNotFoundError: If keywords_path does not exist.
            ValueError: If the JSON structure is invalid.
        """
        path = Path(keywords_path)
        if not path.exists():
            raise FileNotFoundError(f"MNPI keywords file not found: {keywords_path}")

        loop = asyncio.get_running_loop()
        total = await loop.run_in_executor(None, self._load_keywords_from_file, keywords_path)

        logger.info(
            "MNPI keywords loaded from file",
            keywords_path=keywords_path,
            total_keywords=total,
        )
        return total

    def _load_keywords_from_file(self, keywords_path: str) -> int:
        """Synchronous keyword file loading (runs in thread pool)."""
        with open(keywords_path, encoding="utf-8") as fp:
            raw_data: Any = json.load(fp)

        if not isinstance(raw_data, dict):
            raise ValueError("MNPI keywords JSON must be a dict of category -> list[str]")

        for category, keywords in raw_data.items():
            if not isinstance(keywords, list):
                raise ValueError(f"Keywords for category '{category}' must be a list")

            existing = self._raw_keywords.get(category, [])
            new_keywords = [kw for kw in keywords if kw not in existing]
            self._raw_keywords[category] = existing + new_keywords

            existing_patterns = self._keyword_patterns.get(category, [])
            new_patterns = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in new_keywords
            ]
            self._keyword_patterns[category] = existing_patterns + new_patterns

        return sum(len(kws) for kws in self._raw_keywords.values())

    async def scan(
        self,
        text: str,
        tenant_id: str,
        context_metadata: dict | None,
    ) -> dict:
        """Scan text for MNPI content using keyword matching and context analysis.

        Args:
            text: Transcript text to scan (may be multi-sentence / multi-paragraph).
            tenant_id: Tenant identifier for jurisdiction-specific ruleset selection
                (reserved for future per-tenant keyword overrides).
            context_metadata: Optional metadata dict that can contain:
                - 'meeting_type': e.g. 'earnings_call', 'board_meeting'
                - 'speaker_role': e.g. 'cfo', 'analyst'
                - 'jurisdiction': e.g. 'US', 'EU'

        Returns:
            Dict with keys:
                - 'mnpi_detected': bool — True if any MNPI patterns found above threshold.
                - 'confidence': float — Overall confidence score [0.0, 1.0].
                - 'matched_keywords': list[str] — Distinct matched keyword strings.
                - 'flagged_segments': list[dict] — Each dict contains:
                    - 'text': str — Context snippet.
                    - 'reason': str — Human-readable reason for flagging.
                    - 'confidence': float — Per-segment confidence.
                    - 'category': str — MNPI category name.
                    - 'regulatory_ref': str — Applicable regulatory reference.
                    - 'redaction_recommended': bool.
                - 'risk_level': str — 'none' | 'low' | 'medium' | 'high' | 'critical'.
                - 'categories_matched': list[str] — MNPI categories that triggered.
        """
        if not text.strip():
            return self._empty_result()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._scan_sync,
            text,
            tenant_id,
            context_metadata or {},
        )

        logger.info(
            "MNPI scan complete",
            tenant_id=tenant_id,
            mnpi_detected=result["mnpi_detected"],
            risk_level=result["risk_level"],
            matched_keyword_count=len(result["matched_keywords"]),
            flagged_segment_count=len(result["flagged_segments"]),
        )

        return result

    def _scan_sync(
        self,
        text: str,
        tenant_id: str,
        context_metadata: dict,
    ) -> dict:
        """Synchronous MNPI scanning pipeline."""
        matches: list[_DetectionMatch] = []
        matched_keywords: list[str] = []

        for category, patterns in self._keyword_patterns.items():
            for pattern in patterns:
                for match_obj in pattern.finditer(text):
                    keyword_text = match_obj.group(0)
                    context_snippet = self._extract_context(
                        text, match_obj.start(), match_obj.end()
                    )
                    raw_confidence = self._score_match(
                        keyword_text, category, context_snippet, context_metadata
                    )
                    regulatory_ref = self._regulatory_reference(category)
                    redaction_recommended = raw_confidence >= self._settings.mnpi_confidence_threshold

                    detection = _DetectionMatch(
                        keyword=keyword_text.lower(),
                        category=category,
                        start_char=match_obj.start(),
                        end_char=match_obj.end(),
                        context_text=context_snippet,
                        raw_confidence=raw_confidence,
                        regulatory_ref=regulatory_ref,
                        redaction_recommended=redaction_recommended,
                    )
                    matches.append(detection)

                    if keyword_text.lower() not in matched_keywords:
                        matched_keywords.append(keyword_text.lower())

        # Deduplicate overlapping matches — keep highest confidence per span
        matches = self._deduplicate_matches(matches)

        # Aggregate confidence: co-occurrence bonus for multiple categories
        categories_matched = list({m.category for m in matches})
        overall_confidence = self._aggregate_confidence(matches, categories_matched)

        # Apply context_metadata modifiers
        overall_confidence = self._apply_context_modifiers(
            overall_confidence, context_metadata
        )

        risk_level = self._classify_risk(overall_confidence)
        mnpi_detected = (
            len(matches) > 0
            and overall_confidence >= self._settings.mnpi_confidence_threshold
        )

        flagged_segments = [
            {
                "text": m.context_text,
                "reason": self._reason_text(m.keyword, m.category),
                "confidence": round(m.raw_confidence, 4),
                "category": m.category,
                "regulatory_ref": m.regulatory_ref,
                "redaction_recommended": m.redaction_recommended,
                "keyword_matched": m.keyword,
            }
            for m in matches
            if m.raw_confidence >= self._settings.mnpi_confidence_threshold
        ]

        return {
            "mnpi_detected": mnpi_detected,
            "confidence": round(overall_confidence, 4),
            "matched_keywords": matched_keywords,
            "flagged_segments": flagged_segments,
            "risk_level": risk_level,
            "categories_matched": categories_matched,
            "total_matches": len(matches),
        }

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """Extract a context window around a keyword match.

        Uses character-based windowing approximating the configured token window.
        At ~5 chars per token, window_tokens * 5 gives the character radius.

        Args:
            text: Full source text.
            start: Match start character index.
            end: Match end character index.

        Returns:
            Context snippet string.
        """
        chars_per_token = 5
        window_chars = self._settings.mnpi_context_window_tokens * chars_per_token
        context_start = max(0, start - window_chars)
        context_end = min(len(text), end + window_chars)

        snippet = text[context_start:context_end].strip()
        if context_start > 0:
            snippet = "..." + snippet
        if context_end < len(text):
            snippet = snippet + "..."
        return snippet

    def _score_match(
        self,
        keyword: str,
        category: str,
        context: str,
        context_metadata: dict,
    ) -> float:
        """Score a single keyword match for MNPI relevance.

        Base score is 0.60 for any keyword hit. Boosters applied for:
        - High-risk categories (regulatory, corporate_events): +0.15
        - Context contains financial quantifiers (amounts, percentages): +0.10
        - Context contains temporal urgency indicators: +0.08
        - Context contains confidentiality markers: +0.12

        Args:
            keyword: The matched keyword text.
            category: MNPI category the keyword belongs to.
            context: Surrounding text window.
            context_metadata: Meeting metadata for additional context scoring.

        Returns:
            Raw confidence score, capped at 1.0.
        """
        base_score = 0.60

        # Category-level boost
        high_risk_categories = {"regulatory", "corporate_events", "mergers_acquisitions"}
        if category in high_risk_categories:
            base_score += 0.15

        # Financial quantifier detection in context
        financial_pattern = re.compile(
            r"\$[\d,.]+[MBTmbt]?|\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\b",
            re.IGNORECASE,
        )
        if financial_pattern.search(context):
            base_score += 0.10

        # Temporal urgency patterns
        urgency_pattern = re.compile(
            r"\b(?:before\s+(?:the\s+)?(?:market|announcement|close|open)|"
            r"prior\s+to\s+disclosure|ahead\s+of\s+(?:the\s+)?(?:earnings|results|announcement)|"
            r"not\s+yet\s+(?:public|disclosed|announced))\b",
            re.IGNORECASE,
        )
        if urgency_pattern.search(context):
            base_score += 0.08

        # Confidentiality markers
        confidential_pattern = re.compile(
            r"\b(?:confidential|privileged|non[- ]?public|not\s+for\s+distribution|"
            r"embargoed|under\s+nda|do\s+not\s+share)\b",
            re.IGNORECASE,
        )
        if confidential_pattern.search(context):
            base_score += 0.12

        # Meeting type boost — board meetings and earnings calls are highest risk
        meeting_type = context_metadata.get("meeting_type", "").lower()
        if meeting_type in {"board_meeting", "earnings_call", "analyst_briefing"}:
            base_score += 0.05

        return min(1.0, base_score)

    def _aggregate_confidence(
        self, matches: list[_DetectionMatch], categories_matched: list[str]
    ) -> float:
        """Aggregate per-match scores into an overall confidence score.

        Applies a co-occurrence bonus when multiple MNPI categories are matched
        simultaneously, which is a strong signal of material information disclosure.

        Args:
            matches: All deduplicated match objects.
            categories_matched: Unique categories with at least one match.

        Returns:
            Aggregated confidence score [0.0, 1.0].
        """
        if not matches:
            return 0.0

        max_score = max(m.raw_confidence for m in matches)

        # Co-occurrence bonus: +0.05 per additional category beyond the first
        co_occurrence_bonus = max(0, len(categories_matched) - 1) * 0.05

        return min(1.0, max_score + co_occurrence_bonus)

    def _apply_context_modifiers(self, confidence: float, context_metadata: dict) -> float:
        """Apply context-based confidence modifiers.

        Args:
            confidence: Base aggregated confidence.
            context_metadata: Meeting/speaker metadata.

        Returns:
            Modified confidence score [0.0, 1.0].
        """
        speaker_role = context_metadata.get("speaker_role", "").lower()
        # Insider roles amplify the risk
        insider_roles = {"ceo", "cfo", "coo", "general_counsel", "board_member", "insider"}
        if speaker_role in insider_roles:
            confidence = min(1.0, confidence + 0.08)

        return confidence

    @staticmethod
    def _deduplicate_matches(matches: list[_DetectionMatch]) -> list[_DetectionMatch]:
        """Remove overlapping matches, retaining the highest-confidence one per span.

        Args:
            matches: Raw list of all match objects (may overlap).

        Returns:
            Deduplicated list sorted by start character position.
        """
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda m: (m.start_char, -m.raw_confidence))
        deduplicated: list[_DetectionMatch] = []
        last_end = -1

        for match in sorted_matches:
            if match.start_char >= last_end:
                deduplicated.append(match)
                last_end = match.end_char

        return deduplicated

    @staticmethod
    def _classify_risk(confidence: float) -> str:
        """Map a confidence score to a named risk level.

        Args:
            confidence: Aggregated confidence score [0.0, 1.0].

        Returns:
            Risk level string: 'none' | 'low' | 'medium' | 'high' | 'critical'.
        """
        for level, (low, high) in _RISK_THRESHOLDS.items():
            if low <= confidence < high:
                return level
        return "none"

    @staticmethod
    def _regulatory_reference(category: str) -> str:
        """Return the primary regulatory reference for an MNPI category.

        Args:
            category: MNPI keyword category name.

        Returns:
            Regulatory citation string.
        """
        references: dict[str, str] = {
            "earnings":            "SEC Rule 10b-5; Regulation FD (17 CFR 243.100)",
            "mergers_acquisitions": "SEC Rule 10b-5; SEC Rule 14e-3",
            "regulatory":          "SEC Rule 10b-5; 15 U.S.C. § 78j(b)",
            "corporate_events":    "SEC Rule 10b-5; Regulation FD (17 CFR 243.100)",
            "product_pipeline":    "SEC Rule 10b-5; Regulation FD (17 CFR 243.100)",
            "financing":           "SEC Rule 10b-5; Securities Act Section 5",
        }
        return references.get(category, "SEC Rule 10b-5")

    @staticmethod
    def _reason_text(keyword: str, category: str) -> str:
        """Generate a human-readable reason string for a flagged segment.

        Args:
            keyword: Matched keyword.
            category: MNPI category.

        Returns:
            Reason explanation string.
        """
        category_display = category.replace("_", " ").title()
        return (
            f"Keyword '{keyword}' matched under {category_display} MNPI category. "
            f"This content may constitute material non-public information subject to "
            f"SEC disclosure requirements."
        )

    @staticmethod
    def _empty_result() -> dict:
        """Return a clean empty result dict for empty input text."""
        return {
            "mnpi_detected": False,
            "confidence": 0.0,
            "matched_keywords": [],
            "flagged_segments": [],
            "risk_level": "none",
            "categories_matched": [],
            "total_matches": 0,
        }

    async def health_check(self) -> bool:
        """Return True if the MNPI detector is operational."""
        return self._keywords_loaded and len(self._keyword_patterns) > 0
