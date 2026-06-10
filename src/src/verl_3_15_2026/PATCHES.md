# Local patches to vendored verl

This directory is a snapshot of [verl](https://github.com/volcengine/verl)
(version `0.8.0.dev`, vendored 2026-03-15), licensed under Apache-2.0. The
upstream `LICENSE` and `Notice.txt` are retained unchanged.

We carry a few small local patches required to (a) evaluate on a **mixed test
set** spanning five bias benchmarks, and (b) train **small LLMs** where some
metric values can be empty. They are listed here for transparency and so the
changes can be re-applied to a clean upstream checkout if you prefer not to use
this vendored copy.

## 1. `verl/protocol.py` — allow heterogeneous `data_source` / `extra_info`

Our combined test set mixes items from five datasets, so `data_source` and
`extra_info` are not identical across the batch. Skip them in the equality/union
check that otherwise assumes homogeneous fields:

```python
if key in ["data_source", "extra_info"]:
    continue
```

## 2. `verl/trainer/ppo/ray_trainer.py` — JSON-serialize numpy scalars

Validation/rollout dumps contain numpy integers, which raise
`TypeError: Object of type int64 is not JSON serializable`. Add a `default`
serializer:

```python
lines.append(json.dumps(
    entry,
    ensure_ascii=False,
    default=lambda x: x.item() if hasattr(x, "item") else str(x),
))
```

## 3. `verl/trainer/ppo/metric_utils.py` — guard empty metric values

For smaller LLMs a metric's values can be `None` or a dict, which broke the
variance aggregation. Broaden the skip condition:

```python
if not var_vals or isinstance(var_vals[0], str) or not var_vals[0] or isinstance(var_vals[0], Dict):
    continue
```

---

To reproduce from upstream instead of using this vendored copy: check out verl
`0.8.0.dev`, then apply the three edits above.
