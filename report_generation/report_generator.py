"""End-to-end report generation from vision model output to final report text


Chains everything from Phase 4 : vision output → structured findings →
prompt → LLM call → hallucination check → uncertainty flag → final report


One entry point, one Report back, full traceability
everything is traced in the Report dataclass, which is JSON-serializable and can be stored for auditing

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from report_generation.feature_extractor import extract_findings
from report_generation.hallucination_checker import check_report
from report_generation.llm_client import LLMClient
from report_generation.prompts import build_prompt
from report_generation.uncertainty import assess_uncertainty, build_uncertainty_note
@dataclass
class Report:
    """Everything the API needs to return, everything auditing needs to trace."""

    text: str                             # generated report text (with uncertainty note if flagged)
    findings: dict[str, Any] = field(default_factory=dict)       # StudyFindings.to_dict()
    hallucination: dict[str, Any] = field(default_factory=dict)  # HallucinationReport.to_dict()
    uncertainty: dict[str, Any] = field(default_factory=dict)    # UncertaintyAssessment.to_dict()

    # provenance — always populated
    vision_model: str = ""
    prompt_version: str = ""
    llm_model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "findings": self.findings,
            "hallucination": self.hallucination,
            "uncertainty": self.uncertainty,
            "provenance": {
                "vision_model": self.vision_model,
                "prompt_version": self.prompt_version,
                "llm_model": self.llm_model,
                "latency_ms": round(self.latency_ms, 2),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached": self.cached,
            },
        }


class ReportGenerator:
    """Orchestrator : It takes model output and returns a validated report (with uncertainty note if needed)



    Usage:
    
        gen = ReportGenerator(llm_client=LLMClient()) 
        report = await gen.generate(probabilities, model_name="dinov2")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_version: str | None = None,
        confidence_threshold: float = 0.5,#above this, we consider the model confident in its prediction
        uncertainty_threshold: float = 0.3,
    ):
        self.llm_client = llm_client
        self.prompt_version = prompt_version
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    async def generate(
        self,
        probabilities: np.ndarray,
        segmentation_masks: np.ndarray | None = None,
        model_name: str = "unknown",
    ) -> Report:
        """Full pipeline. Vision probabilities in, Report out."""

        # 1. structured findings
        study = extract_findings(
            probabilities=probabilities,
            segmentation_masks=segmentation_masks,
            model_name=model_name,
            confidence_threshold=self.confidence_threshold,
            uncertainty_threshold=self.uncertainty_threshold,
        )
        findings_dict = study.to_dict()

        # 2. uncertainty assessment
        assessment = assess_uncertainty(
            probabilities=probabilities,
            confidence_threshold=self.confidence_threshold,
            uncertainty_threshold=self.uncertainty_threshold,
        )

        # 3. prompt
        prompt = build_prompt(findings_dict, version=self.prompt_version)

        # 4. LLM call
        llm_response = await self.llm_client.generate(prompt)

        # 5. hallucination check
        hallucination = check_report(llm_response.text, findings_dict)

        # 6. append uncertainty note if flagged
        final_text = llm_response.text
        note = build_uncertainty_note(assessment)
        if note:
            final_text = f"{final_text}\n\n{note}"
        # 7. return everything in a Report dataclass

        #so that the caller can inspect the findings, hallucination report, uncertainty assessment, and provenance of the LLM call
        # we can store this Report in a database for auditing, and we can serialize it to JSON for API responses

        return Report(
            text=final_text,
            findings=findings_dict,
            hallucination=hallucination.to_dict(),
            uncertainty=assessment.to_dict(),
            vision_model=model_name,
            prompt_version=llm_response.prompt_version,
            llm_model=llm_response.model,
            latency_ms=llm_response.latency_ms,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cached=llm_response.cached,
        )
    



    def generate_sync(
        self,
        probabilities: np.ndarray,
        segmentation_masks: np.ndarray | None = None,
        model_name: str = "unknown",
    ) -> Report:
        # sync wrapper for notebooks and quick tests (sync wrapper is useful for testing in Jupyter notebooks where async is not easily handled)

       
        import asyncio
        return asyncio.run(self.generate(probabilities, segmentation_masks, model_name))
