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

## Named task sequences

Continual Manipulation (CM) provides fixed Sawyer / MetaWorld-v3 benchmark
sequences and project-defined KUKA / MetaWorld-KUKA-v3 sequences:

| Alias | Backend | Stages | Ordered definition |
|---|---|---:|---|
| `cm1` | Sawyer | 1 | `hammer-v3` |
| `cm3` | Sawyer | 3 | `hammer-v3`, `push-wall-v3`, `faucet-close-v3` |
| `cm5` | Sawyer | 5 | CM10 tasks 1--5 |
| `cm10` | Sawyer | 10 | `hammer-v3`, `push-wall-v3`, `faucet-close-v3`, `push-back-v3`, `stick-pull-v3`, `handle-press-side-v3`, `push-v3`, `shelf-place-v3`, `window-close-v3`, `peg-unplug-side-v3` |
| `kuka_cm4` | KUKA | 4 | `kuka-drawer-close-v3`, `kuka-window-close-v3`, `kuka-door-open-v3`, `kuka-peg-unplug-side-v3` |
| `kuka_cm5` | KUKA | 5 | KUKA CM10 tasks 1--5 |
| `kuka_cm10` | KUKA | 10 | `kuka-reach-v3`, `kuka-push-cube-v3`, `kuka-hammer-v3`, `kuka-handle-press-side-v3`, `kuka-button-press-topdown-v3`, `kuka-faucet-open-v3`, `kuka-drawer-close-v3`, `kuka-window-close-v3`, `kuka-door-open-v3`, `kuka-peg-unplug-side-v3` |
| `kuka_cm5_legacy` | KUKA | 5 | Historical KUKA CM4 followed by `kuka-reach-v3` |
| `kuka_cw10_inspired_5` | KUKA | 5 | Backward-compatible project sequence at selected CW10 positions |

CM5 follows the five-task subset used by DISTR. CM10 uses the original
Continual World CW10 task selection and ordering, adapted only to MetaWorld-v3
identifiers. CM1 and CM3 are project-defined prefixes of CM10, not official
Continual World protocols. The KUKA curricula are custom: canonical KUKA CM10
uses the simulated 40 mm cube push task, and KUKA CM5 is its five-task prefix.
`kuka_cm5_legacy` preserves the former KUKA CM5 meaning. Explicit ordered task
lists remain supported for both backends.

Matching the original CW task order does not reproduce the complete Continual
World experimental protocol. MetaWorld version, observations, rewards, SAC and
continual-learning algorithms, policy architecture, and evaluation can differ,
so direct numerical comparability requires separate validation.

`cm20` is not implemented. Repeating CM10 requires separate stage, semantic
task, generator-condition, and evaluation identities; the current unique-task
sequence and square evaluation-matrix contracts do not encode those semantics.
No partial repeated-task alias is exposed.

## Evaluation and metrics

`evaluate_task` and `evaluate_stage` preserve deterministic actions,
consecutive episode reset seeds, sampled or fixed-task evaluation, and ordered
task-bank provenance. `EvaluationMatrix` stores `R[k,j]`, complete evaluation
cells, stage provenance, and the existing JSON schema.

The provided metrics are average performance, task-end forgetting, and
normalized forward transfer from explicitly aligned learning/reference curves.
Missing measurements remain missing rather than being converted to zero.
