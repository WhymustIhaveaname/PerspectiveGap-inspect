from __future__ import annotations

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate

from perspective_gap_inspect.scorers import role_assignment_scorer

HF_DATASET = "sun1245/PerspectiveGap"
HF_REVISION = "main"


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["role_assignment_prompt"],
        target="unused",
        id=record["evaluation_id"],
        metadata={
            "reference_need_sets": json.dumps(record["reference_need_sets"]),
            "distractor_id": record["distractor_id"],
        },
    )


@task
def role_assignment() -> Task:
    """PerspectiveGap role-fragment assignment task.

    The model receives a multi-agent scenario with labeled information fragments
    and must output a JSON object mapping each role to the fragment IDs it needs.
    """
    return Task(
        dataset=hf_dataset(
            path=HF_DATASET,
            split="test",
            sample_fields=_record_to_sample,
            revision=HF_REVISION,
        ),
        solver=generate(),
        scorer=role_assignment_scorer(),
    )
