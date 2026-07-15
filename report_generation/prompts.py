"""Versioned prompt templates for clinical report generation.

Each prompt version is a frozen snapshot. When you change the prompt,
you add a new version — never edit an old one. This way every generated
report traces back to the exact template that produced it.

The prompts are designed to constrain the LLM: it can only describe
what the vision model detected, must acknowledge what's absent, and
must flag uncertainty. That's the whole anti-hallucination strategy


"""

from __future__ import annotations


from typing import Any

# Registry of all prompt versions. Key = version string, value = builder function.
_REGISTRY: dict[str, type["PromptTemplate"]] = {}

CURRENT_VERSION = "v3"


def register(cls: type["PromptTemplate"]) -> type["PromptTemplate"]:
    """Decorator that adds a prompt version to the registry """
    _REGISTRY[cls.version] = cls
    return cls


def get_template(version: str | None = None) -> "PromptTemplate":
    """Get a prompt template by version. Defaults to CURRENT_VERSION  """
    v = version or CURRENT_VERSION
    if v not in _REGISTRY:
        raise ValueError(f"unknown prompt version '{v}', available: {list(_REGISTRY)}")
    return _REGISTRY[v]()


class PromptTemplate:
    """Base class. Subclasses set `version` and implement `build`."""

    version: str = ""

    def build(self, findings_dict: dict[str, Any]) -> dict[str, str]:
        """Return {"system": ..., "user": ...} ready for the LLM client."""
        raise NotImplementedError

    def _format_findings_block(self, findings: list[dict]) -> str:
        """Human-readable summary of detected findings for the user prompt."""
        if not findings:
            return "No pathological findings detected."

        lines = []
        for i, f in enumerate(findings, 1):
            parts = [
                f"{i}. {f['pathology']}",
                f"location: {f['location']}",
                f"confidence: {f['confidence']:.0%}",
                f"severity: {f['severity']}",
            ]
            if f.get("area_fraction") is not None:
                parts.append(f"area: {f['area_fraction']:.1%} of image")
            lines.append(" | ".join(parts))
        return "\n".join(lines)


# --- v1: baseline, minimal constraints ---

@register
class PromptV1(PromptTemplate):
    """First version. Structured input, basic instructions.
    Kept as a baseline for prompt ablation experiments.
    """

    version: str = "v1"

    def build(self, findings_dict: dict[str, Any]) -> dict[str, str]:
        system = (
            "You are a radiology report assistant. Write a structured "
            "chest X-ray report based on the provided findings. "
            "Use standard radiology terminology."
        )

        user = (
            f"Findings from automated analysis:\n"
            f"{self._format_findings_block(findings_dict['findings'])}\n\n"
            f"Write a radiology report with sections: "
            f"FINDINGS, IMPRESSION, RECOMMENDATION."
        )

        return {"system": system, "user": user, "version": self.version}


# --- v2: adds explicit negatives ---

@register
class PromptV2(PromptTemplate):
    """Adds negative findings to the prompt. The LLM now knows what
    the vision model did NOT detect, making it harder to hallucinate.
    """

    version: str = "v2"

    def build(self, findings_dict: dict[str, Any]) -> dict[str, str]:
        system = (
            "You are a radiology report assistant. Write a structured "
            "chest X-ray report based ONLY on the provided findings. "
            "Do not mention or imply any pathology not listed below. "
            "Use standard radiology terminology."
        )

        negatives = findings_dict.get("negatives", [])
        neg_line = (
            f"Pathologies explicitly NOT detected: {', '.join(negatives)}."
            if negatives
            else "All screened pathologies showed some signal."
        )

        user = (
            f"Findings from automated analysis "
            f"(model: {findings_dict.get('model_name', 'unknown')}):\n\n"
            f"{self._format_findings_block(findings_dict['findings'])}\n\n"
            f"{neg_line}\n\n"
            f"Write a radiology report with sections: "
            f"FINDINGS, IMPRESSION, RECOMMENDATION."
        )

        return {"system": system, "user": user, "version": self.version}


# --- v3: full constraints — negatives, uncertainty, grounding rules ---

@register
class PromptV3(PromptTemplate):
    """Production prompt. Three anti-hallucination mechanisms:
    1. Explicit negatives — what was NOT found
    2. Uncertainty flagging — findings the model isn't sure about

    3. Grounding rule — every sentence must reference a specific finding
    """

    version: str = "v3"

    def build(self, findings_dict: dict[str, Any]) -> dict[str, str]:
        findings = findings_dict.get("findings", [])
        negatives = findings_dict.get("negatives", [])
        uncertain = findings_dict.get("uncertain", [])
        model_name = findings_dict.get("model_name", "unknown")
        has_spatial = findings_dict.get("has_spatial_info", False)

        system = self._build_system()
        user = self._build_user(findings, negatives, uncertain, model_name, has_spatial)

        return {"system": system, "user": user, "version": self.version}

    def _build_system(self) -> str:
        return (

            "You are a radiology report assistant generating structured "
            "chest X-ray reports. Follow these rules strictly:\n\n"
            "1. ONLY describe findings provided in the input. Never invent, "
            "infer, or suggest pathologies not explicitly listed.\n"
            "2. Every statement in the report must trace to a specific "
            "finding from the input. If you cannot ground a sentence in "
            "a provided finding, do not write it.\n"
            "3. When findings are marked uncertain, say so explicitly — "
            "use phrases like 'cannot be excluded' or 'equivocal'.\n"
            "4. Use standard radiology report structure and terminology.\n"
            "5. Be concise. One paragraph per section maximum.\n"
            "6. End with actionable recommendations tied to the findings."

        )

    def _build_user(
        self,
        findings: list[dict],
        negatives: list[str],
        uncertain: list[str],
        model_name: str,
        has_spatial: bool,
    ) -> str:
        sections = [f"Automated chest X-ray analysis (model: {model_name}):\n"]

        # detected findings
        sections.append("DETECTED FINDINGS:")
        if findings:
            sections.append(self._format_findings_block(findings))
        else:
            sections.append("None — no pathological findings detected.")

        # explicit negatives
        if negatives:
            sections.append(
                f"\nNOT DETECTED (explicitly excluded by the model): "
                f"{', '.join(negatives)}."
            )

        # uncertain
        if uncertain:
            sections.append(
                f"\nUNCERTAIN (model confidence between thresholds — "
                f"mention these as equivocal): {', '.join(uncertain)}."
            )

        # spatial info note
        if has_spatial:
            sections.append(
                "\nNote: locations are derived from segmentation masks "
                "and reflect the primary region of each finding."
            )
        else:
            sections.append(
                "\nNote: no segmentation available — locations are "
                "reported as 'diffuse'. Do not fabricate specific locations."
            )

        sections.append(
            "\nGenerate a radiology report with sections: "
            "FINDINGS, IMPRESSION, RECOMMENDATION."
        )

        return "\n".join(sections)



def list_versions() -> list[str]:
    """All registered prompt versions, sorted."""
    return sorted(_REGISTRY.keys())


def build_prompt(
        
    findings_dict: dict[str, Any],
    version: str | None = None,
) -> dict[str, str]:
    """Convenience function. Get template, build prompt, done."""
    return get_template(version).build(findings_dict)
