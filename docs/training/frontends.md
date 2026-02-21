# Frontends

## Notion

<iframe src="https://mipcandy.notion.site/ebd/26582340f41480699034d510d42cf874?v=26582340f41481adb702000cbab06660" width="100%" height="600" frameborder="0" allowfullscreen></iframe>

Duplicate [this template](https://mipcandy.notion.site) to your account.

In your duplication, click "..." -> "Copy Link" in the top right corner. You will get a link like
"https://www.notion.so/{DATABASE_ID}?v={VIEW_ID}".

Click "..." -> "Connections" -> "Develop Integration" in the top right corner. Create a new integration. After created,
copy the "Integration Secret". This is your `API_KEY`. In the access tab, select your database.

```shell
mipcandy -c secret -kv notion_api_key {API_KEY} -kv notion_database_id {DATABASE_ID}
```

```python
from mipcandy import NotionFrontend

trainer = ...
trainer.set_frontend(NotionFrontend)
```

## WandB

[Weights & Biases](https://wandb.ai) provides experiment tracking with real-time metric logging and visualization dashboards.

### Setup

Create a WandB account and obtain your entity and project names. Store them as secrets:

```shell
mipcandy -c secret -kv wandb_entity {ENTITY} -kv wandb_project {PROJECT}
```

`wandb_entity` is your WandB username or team name. `wandb_project` is the project under which experiments will be logged.

### Usage

```python
from mipcandy import WandBFrontend

trainer = ...
trainer.set_frontend(WandBFrontend)
```

`WandBFrontend` logs experiment configuration on creation (experiment ID, trainer, model, note, MACs, parameters, epochs) and streams metrics at each epoch update. The run is finalized when the experiment completes.

:::{note}
`WandBFrontend` requires the `wandb` package. If `wandb` is not installed, the class falls back to the base `Frontend` (a no-op).
:::

## Hybrid Frontend

Use `create_hybrid_frontend()` to combine multiple frontends into a single frontend class. All lifecycle events are forwarded to each provided frontend instance.

```python
from mipcandy import create_hybrid_frontend

def create_hybrid_frontend(*frontends: Frontend) -> type[Frontend]:
```

### Parameters

- `*frontends`: One or more `Frontend` instances to combine.

### Returns

A new `Frontend` subclass that delegates all calls (`on_experiment_created`, `on_experiment_updated`, `on_experiment_completed`, `on_experiment_interrupted`) to each provided frontend.

### Usage

```python
from mipcandy import NotionFrontend, WandBFrontend, create_hybrid_frontend

secrets = {...}
notion = NotionFrontend(secrets)
wandb = WandBFrontend(secrets)

HybridFrontend = create_hybrid_frontend(notion, wandb)

trainer = ...
trainer.set_frontend(HybridFrontend)
```

## Frontend Protocol

All frontends implement the `Frontend` base class from `mipcandy.frontend.prototype`:

```python
class Frontend:
    def on_experiment_created(self, experiment_id, trainer, model, note,
                              num_params, num_macs, num_epochs, early_stop_tolerance) -> None: ...
    def on_experiment_updated(self, experiment_id, epoch, metrics, early_stop_tolerance) -> None: ...
    def on_experiment_completed(self, experiment_id) -> None: ...
    def on_experiment_interrupted(self, experiment_id, error) -> None: ...
```

Secrets are passed to the constructor as a `Settings` dict. Use `require_nonempty_secret(entry, *, required_type)` within your frontend to validate required configuration.