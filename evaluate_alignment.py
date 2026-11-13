"""
Sociological & STS Evaluation Pipeline for Generative AI Alignment
Author: Evangelos Sergis (PhD Candidate, University of the Aegean | Associate Researcher, PHEPOC Lab NKUA)
Description: Computational evaluation script analyzing RLHF datasets for algorithmic bias,
digital labor extraction, and subcultural performativity (drill/rap personas, platform capitalism).
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_rlhf_dataset(filepath: str = "rlhf-dataset-sample.json") -> List[Dict[str, Any]]:
    """Loads the sociological RLHF evaluation dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{filepath}' not found in repository.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_sociological_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes quantitative and qualitative alignment metrics across socio-technical dimensions:
    - Epistemic & Factual Accuracy (STS / Algorithmic Governance)
    - Ethical Alignment & Stereotype Mitigation (Cultural Criminology / Subcultural Performativity)
    - Overall Sociological Alignment Score
    """
    total_tasks = len(records)
    if total_tasks == 0:
        return {}

    scores = []
    factual_passes = 0
    ethical_passes = 0
    thematic_flags = {
        "algorithmic_curation": 0,
        "digital_labor_prosumer": 0,
        "subcultural_performativity_drill_rap": 0,
    }

    for item in records:
        eval_data = item.get("data_trainer_evaluation", {})
        score = eval_data.get("overall_score", 0)
        max_score = eval_data.get("max_score", 5)
        scores.append(score / max_score if max_score else 0.0)

        if eval_data.get("factual_accuracy", "").lower() == "pass":
            factual_passes += 1
        if eval_data.get("ethical_alignment", "").lower() == "pass":
            ethical_passes += 1

        # Thematic classification based on prompt & sociological critique
        critique_text = (
            item.get("prompt", "") + " " + eval_data.get("sociological_critique", "")
        ).lower()
        if "algorithmic" in critique_text or "curation" in critique_text:
            thematic_flags["algorithmic_curation"] += 1
        if "digital labor" in critique_text or "prosumer" in critique_text:
            thematic_flags["digital_labor_prosumer"] += 1
        if "drill" in critique_text or "rap" in critique_text or "gangster" in critique_text:
            thematic_flags["subcultural_performativity_drill_rap"] += 1

    mean_normalized_score = sum(scores) / total_tasks
    return {
        "total_evaluated_tasks": total_tasks,
        "mean_alignment_index": round(mean_normalized_score, 3),
        "factual_accuracy_pass_rate": round(factual_passes / total_tasks, 3),
        "ethical_alignment_pass_rate": round(ethical_passes / total_tasks, 3),
        "thematic_coverage": thematic_flags,
    }


def main() -> None:
    dataset = load_rlhf_dataset()
    metrics = compute_sociological_metrics(dataset)
    print("=== STS & Sociological AI Alignment Evaluation Report ===")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
