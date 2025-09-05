# Frontends

## Notion

<iframe src="https://mipcandy.notion.site/ebd/26582340f41480699034d510d42cf874?v=26582340f41481adb702000cbab06660" width="100%" height="600" frameborder="0" allowfullscreen></iframe>

Create an integration [here](https://www.notion.so/profile/integrations). After created, copy the "Integration Secret".
This is your `API_KEY`.

Copy [this template](https://mipcandy.notion.site) to your account.

In your duplication, click "..." -> "Copy Link" in the top right corner. You will get a link like
"https://www.notion.so/{DATABASE_ID}?v={VIEW_ID}".

```shell
mipcandy -c secret -kv notion_api_key {API_KEY} -kv notion_database_id {DATABASE_ID}
```

```python
from mipcandy import NotionFrontend

trainer = ...
trainer.set_frontend(NotionFrontend)
```