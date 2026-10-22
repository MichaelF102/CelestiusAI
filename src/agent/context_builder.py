"""Context Builder and Question Intent Classifier for the AI Analyst.

Performs:
1. Deterministic classification of analytical questions.
2. Variable extraction and disambiguation with fuzzy matching.
3. Analytical conversational memory tracking with dataset version scoping.
4. Prompt context compaction to control token usage and prevent stale references.
"""

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import pandas as pd

from src.ingestion.loader import ManagedDataset
from src.interpretation.interpretation_models import AnalystResponse

logger = logging.getLogger(__name__)

IntentCategory = Literal[
    "dataset_overview",
    "descriptive_statistics",
    "distribution_outlier",
    "group_comparison",
    "correlation_association",
    "hypothesis_testing",
    "regression_interpretation",
    "data_quality_investigation",
    "temporal_patterns",
    "hypothesis_generation",
    "recommendation_next_steps",
    "followup_question",
    "general_query",
]


@dataclass
class ConversationTurn:
    """Historical conversation exchange within an active dataset session."""

    turn_id: int
    question: str
    intent_category: str
    dataset_version: int
    response: AnalystResponse
    referenced_columns: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class QuestionClassification:
    """Structured result of question intent analysis."""

    raw_question: str
    intent: IntentCategory
    relevant_columns: list[str]
    ambiguous_matches: dict[str, list[str]] = field(default_factory=dict)
    requires_group_comparison: bool = False
    requires_regression: bool = False
    requires_correlation: bool = False
    is_followup: bool = False
    clarification_needed: str | None = None


class AnalyticalContextBuilder:
    """Manages question classification, conversational memory, and token-efficient context assembly."""

    def __init__(self, max_history_turns: int = 5) -> None:
        self.max_history_turns = max_history_turns
        self.history: list[ConversationTurn] = []

    def clear_history(self) -> None:
        """Reset analytical conversation memory."""
        self.history.clear()

    def record_turn(
        self,
        question: str,
        intent: str,
        dataset_version: int,
        response: AnalystResponse,
        referenced_columns: list[str],
    ) -> None:
        """Add turn to session memory and enforce bounded history window."""
        turn = ConversationTurn(
            turn_id=len(self.history) + 1,
            question=question,
            intent_category=intent,
            dataset_version=dataset_version,
            response=response,
            referenced_columns=referenced_columns,
        )
        self.history.append(turn)
        if len(self.history) > self.max_history_turns:
            self.history.pop(0)

    def classify_question(
        self,
        question: str,
        managed_dataset: ManagedDataset,
    ) -> QuestionClassification:
        """Classify user question and extract relevant dataset columns."""
        q_lower = question.lower().strip()
        df: pd.DataFrame = managed_dataset.working_df
        columns = list(df.columns)

        # 1. Identify mentioned columns
        matched_cols: list[str] = []
        ambiguous: dict[str, list[str]] = {}

        # Exact and token-based match
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower)
        for col in columns:
            col_l = col.lower()
            if col_l in q_lower or col in words:
                if col not in matched_cols:
                    matched_cols.append(col)

        # Fuzzy check for words not matched exactly
        for w in words:
            if len(w) >= 4 and not any(w in c.lower() for c in matched_cols):
                close = difflib.get_close_matches(w, [c.lower() for c in columns], n=2, cutoff=0.8)
                if close:
                    actual_cols = [c for c in columns if c.lower() in close]
                    ambiguous[w] = actual_cols

        # 2. Check if it's a follow-up referring to earlier context
        is_followup = any(
            phrase in q_lower
            for phrase in [
                "why might this",
                "compare that",
                "previous result",
                "what about",
                "explain why",
                "after accounting for",
                "that difference",
                "those findings",
            ]
        )

        # Inherit columns from previous turn if follow-up lacks columns
        if is_followup and not matched_cols and self.history:
            last_turn = self.history[-1]
            matched_cols = list(last_turn.referenced_columns)

        # 3. Classify Intent
        intent: IntentCategory = "general_query"
        req_group = False
        req_reg = False
        req_corr = False

        if any(
            w in q_lower
            for w in ["overview", "shape", "summary", "columns", "dimensions", "how many rows"]
        ):
            intent = "dataset_overview"
        elif any(
            w in q_lower
            for w in ["compare", "difference between", "differ", "higher in", "versus", "vs."]
        ):
            intent = "group_comparison"
            req_group = True
        elif any(w in q_lower for w in ["correlat", "associated with", "relationship between"]):
            intent = "correlation_association"
            req_corr = True
        elif any(
            w in q_lower for w in ["regress", "predict", "impact of", "elasticity", "coefficient"]
        ):
            intent = "regression_interpretation"
            req_reg = True
        elif any(
            w in q_lower for w in ["missing", "null", "quality", "clean", "duplicate", "invalid"]
        ):
            intent = "data_quality_investigation"
        elif any(
            w in q_lower for w in ["outlier", "skew", "distribution", "kurtosis", "histogram"]
        ):
            intent = "distribution_outlier"
        elif any(w in q_lower for w in ["hypothes", "theorize", "conjecture"]):
            intent = "hypothesis_generation"
        elif any(w in q_lower for w in ["recommend", "what next", "should we", "decision"]):
            intent = "recommendation_next_steps"
        elif any(
            w in q_lower for w in ["test", "p-value", "significan", "t-test", "anova", "chi-square"]
        ):
            intent = "hypothesis_testing"
        elif any(
            w in q_lower
            for w in ["average", "mean", "median", "standard deviation", "std", "min", "max"]
        ):
            intent = "descriptive_statistics"
        elif is_followup:
            intent = "followup_question"

        clarification: str | None = None
        if ambiguous:
            opts = [f"'{k}' might refer to {v}" for k, v in ambiguous.items()]
            clarification = f"Clarification suggestion: {'; '.join(opts)}"

        return QuestionClassification(
            raw_question=question,
            intent=intent,
            relevant_columns=matched_cols,
            ambiguous_matches=ambiguous,
            requires_group_comparison=req_group,
            requires_regression=req_reg,
            requires_correlation=req_corr,
            is_followup=is_followup,
            clarification_needed=clarification,
        )

    def build_conversation_context(self, current_dataset_version: int) -> str:
        """Summarize recent conversation history, flagging stale versions."""
        if not self.history:
            return "No previous conversation context in this session."

        lines = ["Recent Session Exchanges:"]
        for turn in self.history:
            stale_tag = (
                " [STALE VERSION]" if turn.dataset_version != current_dataset_version else ""
            )
            lines.append(f"User: {turn.question} (Dataset v{turn.dataset_version}{stale_tag})")
            lines.append(f"Analyst Answer: {turn.response.direct_answer[:150]}...")
            if turn.response.key_findings:
                top_finding = turn.response.key_findings[0].claim
                lines.append(f"Key Finding: {top_finding}")
        return "\n".join(lines)
