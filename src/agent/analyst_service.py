"""Central AI Analyst Service coordinating evidence generation, LLM interpretation, and claim verification.

Flow:
1. Classify question and extract target variables
2. Execute approved deterministic tools to compute missing evidence
3. Assemble scoped Evidence Pack (grounded truth)
4. Query LLM provider with Deep Interpretation Framework prompt
5. Validate quantitative claims and check for contradictions
6. Record session memory and audit traces
7. Return structured AnalystResponse
"""

import logging
import time
from typing import Any

import pandas as pd

from src.agent.context_builder import AnalyticalContextBuilder, QuestionClassification
from src.agent.llm_provider import LLMProvider, MockLLMProvider
from src.agent.prompt_templates import (
    DEEP_INTERPRETATION_USER_PROMPT,
    SYSTEM_ANALYST_INSTRUCTION,
)
from src.agent.provider_factory import ProviderFactory
from src.agent.tool_registry import AnalyticalToolRegistry
from src.config.settings import AppSettings, get_settings
from src.ingestion.loader import ManagedDataset
from src.interpretation.claim_validation import ClaimValidator
from src.interpretation.contradiction_detection import ContradictionDetector
from src.interpretation.evidence_pack import EvidencePackBuilder
from src.interpretation.hypothesis_generator import HypothesisGenerator
from src.interpretation.interpretation_models import (
    AnalystFindingClaim,
    AnalystResponse,
    EvidencePack,
    TraceMetadata,
)

logger = logging.getLogger(__name__)


class AnalystService:
    """Orchestrates evidence gathering, LLM interpretation, and claim validation."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or ProviderFactory.get_provider(settings=self.settings)
        self.tool_registry = AnalyticalToolRegistry()
        self.context_builder = AnalyticalContextBuilder()
        self.claim_validator = ClaimValidator()
        self.contradiction_detector = ContradictionDetector()

    def set_provider(self, provider: LLMProvider) -> None:
        """Dynamically update provider (useful in UI model selection or tests)."""
        self.provider = provider

    def reset_conversation(self) -> None:
        """Clear session conversational memory."""
        self.context_builder.clear_history()

    def analyze_question(
        self,
        question: str,
        managed_dataset: ManagedDataset,
    ) -> tuple[AnalystResponse, EvidencePack]:
        """Execute complete evidence-grounded analysis workflow."""
        start_time = time.perf_counter()
        dataset_version = getattr(managed_dataset, "version", 1)
        df: pd.DataFrame = managed_dataset.working_df

        # 1. Classify question and detect variables
        classification: QuestionClassification = self.context_builder.classify_question(
            question=question,
            managed_dataset=managed_dataset,
        )

        # 2. Execute necessary deterministic tools based on classification
        tool_results: list[dict[str, Any]] = []
        executed_tool_traces: list[dict[str, Any]] = []

        try:
            if (
                classification.requires_group_comparison
                and len(classification.relevant_columns) >= 2
            ):
                # Find which is categorical and which is continuous
                grp_col, meas_col = None, None
                for c in classification.relevant_columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        meas_col = c
                    else:
                        grp_col = c
                if grp_col and meas_col:
                    res = self.tool_registry.execute_tool(
                        "compare_groups",
                        managed_dataset,
                        {"group_col": grp_col, "measure_col": meas_col},
                    )
                    tool_results.append(res)
                    executed_tool_traces.append({"tool": "compare_groups", "status": "success"})

            elif classification.requires_correlation and len(classification.relevant_columns) >= 2:
                v1, v2 = classification.relevant_columns[0], classification.relevant_columns[1]
                if pd.api.types.is_numeric_dtype(df[v1]) and pd.api.types.is_numeric_dtype(df[v2]):
                    res = self.tool_registry.execute_tool(
                        "calculate_correlation",
                        managed_dataset,
                        {"var1": v1, "var2": v2},
                    )
                    tool_results.append(res)
                    executed_tool_traces.append(
                        {"tool": "calculate_correlation", "status": "success"}
                    )

            elif classification.requires_regression and len(classification.relevant_columns) >= 2:
                outcome = classification.relevant_columns[0]
                preds = [
                    c
                    for c in classification.relevant_columns[1:]
                    if pd.api.types.is_numeric_dtype(df[c])
                ]
                if pd.api.types.is_numeric_dtype(df[outcome]) and preds:
                    res = self.tool_registry.execute_tool(
                        "fit_regression",
                        managed_dataset,
                        {"outcome_col": outcome, "predictor_cols": preds},
                    )
                    tool_results.append(res)
                    executed_tool_traces.append({"tool": "fit_regression", "status": "success"})

        except Exception as tool_err:
            logger.warning("Optional tool execution skipped: %s", tool_err)
            executed_tool_traces.append({"error": str(tool_err)})

        # 3. Assemble scoped Evidence Pack
        evidence_pack = EvidencePackBuilder.build_pack(
            managed_dataset=managed_dataset,
            relevant_columns=classification.relevant_columns,
            tool_results=tool_results,
        )

        # 4. Handle Offline / No-Provider Mode gracefully
        if not self.provider.is_configured and not isinstance(self.provider, MockLLMProvider):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response = self._build_deterministic_fallback_response(
                question=question,
                classification=classification,
                evidence_pack=evidence_pack,
                dataset_version=dataset_version,
                duration_ms=duration_ms,
                tool_traces=executed_tool_traces,
            )
            # Record turn in session memory
            self.context_builder.record_turn(
                question=question,
                intent=classification.intent,
                dataset_version=dataset_version,
                response=response,
                referenced_columns=classification.relevant_columns,
            )
            return response, evidence_pack

        # 5. Build prompts
        conv_context = self.context_builder.build_conversation_context(dataset_version)
        evidence_text = evidence_pack.summary_for_prompt()
        tool_defs_str = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tool_registry.get_tool_definitions()
        )

        user_prompt = DEEP_INTERPRETATION_USER_PROMPT.format(
            question=question,
            conversation_context=conv_context,
            evidence_pack_text=evidence_text,
            available_tools=tool_defs_str,
        )

        # 6. Request interpretation from LLM
        parsed_response, llm_result = self.provider.generate_structured(
            prompt=user_prompt,
            response_model=AnalystResponse,
            system_instruction=SYSTEM_ANALYST_INSTRUCTION,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # If LLM generation failed or returned invalid response, use deterministic fallback
        if parsed_response is None or llm_result.status != "success":
            logger.warning(
                "LLM generation unsuccessful (%s: %s). Falling back to deterministic synthesis.",
                llm_result.status,
                llm_result.error_message,
            )
            response = self._build_deterministic_fallback_response(
                question=question,
                classification=classification,
                evidence_pack=evidence_pack,
                dataset_version=dataset_version,
                duration_ms=duration_ms,
                tool_traces=executed_tool_traces,
            )
        else:
            response = parsed_response
            response.dataset_version = dataset_version
            response.trace_metadata.provider = self.provider.provider_name
            response.trace_metadata.model = self.provider.model_name
            response.trace_metadata.execution_time_ms = round(duration_ms, 2)
            response.trace_metadata.tool_executions = executed_tool_traces
            response.trace_metadata.evidence_ids_used = list(
                {eid for f in response.key_findings for eid in f.evidence_ids}
            )

        # 7. Validate claims and flag contradictions
        response.key_findings = self.claim_validator.validate_all(
            claims=response.key_findings,
            evidence_pack=evidence_pack,
            active_dataset_version=dataset_version,
        )

        contradictions = self.contradiction_detector.detect_contradictions(
            response=response,
            evidence_pack=evidence_pack,
            active_dataset_version=dataset_version,
        )
        response.contradictions_detected.extend(contradictions)

        # Ensure exploratory hypotheses exist if none were provided
        if not response.hypotheses:
            response.hypotheses = HypothesisGenerator.generate_hypotheses_from_evidence(
                evidence_pack=evidence_pack
            )

        # 8. Record in conversational memory
        self.context_builder.record_turn(
            question=question,
            intent=classification.intent,
            dataset_version=dataset_version,
            response=response,
            referenced_columns=classification.relevant_columns,
        )

        return response, evidence_pack

    def _build_deterministic_fallback_response(
        self,
        question: str,
        classification: QuestionClassification,
        evidence_pack: EvidencePack,
        dataset_version: int,
        duration_ms: float,
        tool_traces: list[dict[str, Any]],
    ) -> AnalystResponse:
        """Construct grounded, deterministic response when LLM credentials are not configured."""
        findings: list[AnalystFindingClaim] = []

        # Convert top evidence items into verifiable findings
        for item in evidence_pack.items[:5]:
            findings.append(
                AnalystFindingClaim(
                    claim=f"Deterministic calculation observed: {item.metric_name} = {item.formatted_value}.",
                    evidence_ids=[item.id],
                    reported_values={item.metric_name: item.actual_value}
                    if isinstance(item.actual_value, (int, float, str))
                    else {},
                    relevant_variables=item.relevant_variables,
                    sample_size=item.sample_size,
                    method=item.calculation_method,
                    caveats=item.caveats,
                    validation_status="verified",
                    validation_notes=["Computed deterministically by statistical engine."],
                )
            )

        hypotheses = HypothesisGenerator.generate_hypotheses_from_evidence(evidence_pack)

        direct_ans = (
            f"Analysis of '{question}' based on {len(evidence_pack.items)} verified evidence items from dataset v{dataset_version}. "
            f"Deterministic calculations completed successfully."
        )

        return AnalystResponse(
            question=question,
            intent_category=classification.intent,
            direct_answer=direct_ans,
            key_findings=findings,
            interpretation=(
                "Operating in Deterministic Analytics mode. To enable deep AI qualitative synthesis, "
                "configure GEMINI_API_KEY in your environment or .env file."
            ),
            alternative_explanations=[
                "Observed patterns may be influenced by unobserved confounding factors.",
                "Potential selection bias or non-random missingness in sample collection.",
            ],
            uncertainty_and_limitations=[
                f"Findings strictly limited to the current dataset version (v{dataset_version}).",
                "Observational data cannot establish causal mechanisms.",
            ],
            recommended_next_analyses=[
                "Inspect regression diagnostics for multicollinearity and heteroscedasticity.",
                "Perform subgroup breakdown to verify consistency across segments.",
            ],
            hypotheses=hypotheses,
            contradictions_detected=[],
            trace_metadata=TraceMetadata(
                provider="deterministic_engine",
                model="deterministic-fallback-v1",
                execution_time_ms=round(duration_ms, 2),
                evidence_ids_used=[f.evidence_ids[0] for f in findings if f.evidence_ids],
                tool_executions=tool_traces,
            ),
            dataset_version=dataset_version,
        )
