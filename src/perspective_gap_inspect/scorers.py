from __future__ import annotations

import json
from typing import Any

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from perspective_gap.scoring import score_prompt_writing, score_role_assignment


def _value_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value == CORRECT:
        return 1.0
    return 0.0


@scorer(metrics=[accuracy(), stderr()])
def role_assignment_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        reference_need_sets = json.loads(meta["reference_need_sets"])
        distractor_id = meta.get("distractor_id")

        result = score_role_assignment(
            state.output.completion,
            reference_need_sets,
            distractor_id,
        )

        return Score(
            value=CORRECT if result["pass"] else INCORRECT,
            answer=state.output.completion,
            explanation=json.dumps(result["per_role"]),
            metadata=result["metrics"],
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def prompt_writing_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        fragments = json.loads(meta["fragments"])
        reference_need_sets = json.loads(meta["reference_need_sets"])
        distractor_id = meta.get("distractor_id")

        result = score_prompt_writing(
            state.output.completion,
            fragments,
            reference_need_sets,
            distractor_id,
        )

        return Score(
            value=CORRECT if result["pass"] else INCORRECT,
            answer=state.output.completion,
            explanation=json.dumps({
                role: {
                    "pass": info["pass"],
                    "detected": info["detected"],
                    "missing": info["missing"],
                    "extra": info["extra"],
                }
                for role, info in result["per_role"].items()
            }),
            metadata=result["metrics"],
        )

    return score
