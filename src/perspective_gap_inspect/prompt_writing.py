from __future__ import annotations

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate

from perspective_gap_inspect.scorers import prompt_writing_scorer

HF_DATASET = "sun1245/PerspectiveGap"
HF_REVISION = "89ee514670438b7d1b25fe958ea6335ca69934b0"


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["prompt_writing_prompt"],
        target="unused",
        id=record["evaluation_id"],
        metadata={
            "fragments": json.dumps(record["fragments"]),
            "reference_need_sets": json.dumps(record["reference_need_sets"]),
            "distractor_id": record["distractor_id"],
        },
    )


@task
def prompt_writing() -> Task:
    """PerspectiveGap free-form prompt writing task.

    The model receives a multi-agent scenario with labeled information fragments
    and must write one prompt per role, including only the relevant fragments verbatim.
    """
    return Task(
        dataset=hf_dataset(
            path=HF_DATASET,
            split="test",
            sample_fields=_record_to_sample,
            revision=HF_REVISION,
        ),
        solver=generate(),
        scorer=prompt_writing_scorer(),
    )
