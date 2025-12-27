# Dora echo example

Make sure to have, dora, uv and cargo installed.

```bash
uv venv -p 3.11 --seed
# Make sure you have checked out dora to the same folder as dora-hub.
uv pip install -e ../../../dora/apis/python/node --reinstall
dora build dataflow.yml --uv
dora run dataflow.yml --uv
```
