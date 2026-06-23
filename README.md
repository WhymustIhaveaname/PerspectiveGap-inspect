# PerspectiveGap for Inspect AI

[Inspect AI](https://inspect.aisi.org.uk/) implementation of the [PerspectiveGap](https://github.com/WhymustIhaveaname/PerspectiveGap) benchmark ([arXiv:2606.08878](https://arxiv.org/abs/2606.08878)).

PerspectiveGap evaluates LLMs' ability to compose orchestration prompts for multi-agent systems. It tests whether a model can decide what each sub-agent needs to know without leaking irrelevant context.

## Tasks

- **role_assignment** — The model receives a multi-agent scenario with labeled information fragments and must output a JSON object mapping each role to the fragment IDs it needs.
- **prompt_writing** — The model receives the same scenario and must write one prompt per role (as markdown with h1 headers), including only the relevant fragments verbatim.

## Installation

```bash
pip install git+https://github.com/WhymustIhaveaname/PerspectiveGap-inspect.git
```

Inspect AI treats model providers as optional dependencies.
Install the client library for the provider you want to use:

```bash
pip install openai      # for OpenAI models
pip install anthropic   # for Anthropic models
```

## Usage

```bash
# Run role-fragment assignment task
inspect eval perspective_gap_inspect/role_assignment --model openai/gpt-5.5

# Run free-form prompt writing task
inspect eval perspective_gap_inspect/prompt_writing --model openai/gpt-5.5

# Limit to a subset of samples
inspect eval perspective_gap_inspect/role_assignment --model openai/gpt-5.5 --limit 20
```

## Validation

Results from this implementation match the paper (all within 95% CI):

| | role_assignment | prompt_writing |
|---|---|---|
| **gpt-5.5** | 60.9% (paper 55.5%) | 65.5% (paper 68.6%) |
| **gpt-5.4** | 23.2% (paper 25.5%) | 4.5% (paper 2.3%) |

## Dataset

Uses the [sun1245/PerspectiveGap](https://huggingface.co/datasets/sun1245/PerspectiveGap) dataset from Hugging Face (220 evaluations: 110 scenarios x 2 shuffle seeds).

## Scoring

Scoring uses the original PerspectiveGap scorer from the benchmark repository. Metrics reported per sample:

| Metric | Description |
|--------|-------------|
| strict_pass | 1 if all roles are assigned/written correctly, 0 otherwise |
| net_match_score | (TP - FP - FN) / expected, clipped to [0, 1] |
| required_coverage | TP / (TP + FN) |
| boundary_precision | TP / (TP + FP) |
| distractor_leakage | Number of times the distractor fragment was incorrectly included |

The primary metric reported by `inspect eval` is **accuracy** (percentage of samples with strict_pass = 1).

## Citation

```bibtex
@misc{sun2026perspectivegapbenchmarkmultiagentorchestration,
      title={PerspectiveGap: A Benchmark for Multi-Agent Orchestration Prompting},
      author={Youran Sun and Xingyu Ren and Kejia Zhang and Xinpeng Liu and Jiaxuan Guo},
      year={2026},
      eprint={2606.08878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.08878},
}
```
