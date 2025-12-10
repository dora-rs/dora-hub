# Dora record & replay example

Make sure to have, dora, uv and cargo installed.

```bash
uv venv -p 3.11 --seed
uv pip install -e ../../apis/python/node --reinstall
dora build dataflow-record.yml --uv

# Record data
dora run dataflow-record.yml --uv

# Replay data
dora run dataflow-replay.yml --uv
```
