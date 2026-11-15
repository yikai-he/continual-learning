# ContinualManipulation

ContinualManipulation is an independently installable benchmark for continual
robotic manipulation. It is inspired by the architecture and metrics of
Continual World, but it is not an official Continual World release.

Package version `0.1.0` identifies the Python distribution. It is not a
scientific benchmark-protocol version.

## Installation

From this repository:

```bash
python -m pip install packages/continual-manipulation
```

MetaWorld-v3 is the primary backend and is installed as a required dependency.
The tested environment uses `metaworld==3.1.1` and `gymnasium==1.3.0`.

KUKA-v3 is optional. Install the separately maintained compatible
`metaworld_kuka` distribution before selecting `backend="kuka-v3"`. This
package does not bundle KUKA code or assets. Because the current KUKA project
also provides a distribution identified as `metaworld==3.1.1`, it is not
declared as a package extra; the lazy backend import keeps MetaWorld-only use
independent of it.

## Public API

```python
from continual_manipulation import (
    EvaluationMatrix,
    TaskSequence,
    TaskSwitcher,
    evaluate_stage,
    resolve_named_sequence,
)

sequence = TaskSequence.from_names(resolve_named_sequence("cm3"))

with TaskSwitcher(sequence, backend="metaworld-v3") as switcher:
    first_environment = switcher.switch(0, seed=0)
    # Train an external policy on the first task.
    second_environment = switcher.switch(1, seed=1)

matrix = EvaluationMatrix(sequence)
evaluate_stage(policy, matrix, stage=1, evaluation_mode="fixed-tasks")
```

External policies implement one method:

```python
def act(self, observation, deterministic=True):
    ...
```

The package exposes the current `cm1`, `cm3`, `cm5`, `cm10`, and
`kuka_cw10_inspired_5` sequence identities. The KUKA sequence is project-defined
and is not an official CW10 benchmark. Repeated-task/CW20 sequences are not
implemented.

## Evaluation and metrics

`evaluate_task` and `evaluate_stage` preserve deterministic actions,
consecutive episode reset seeds, sampled or fixed-task evaluation, and ordered
task-bank provenance. `EvaluationMatrix` stores `R[k,j]`, complete evaluation
cells, stage provenance, and the existing JSON schema.

The provided metrics are average performance, task-end forgetting, and
normalized forward transfer from explicitly aligned learning/reference curves.
Missing measurements remain missing rather than being converted to zero.
