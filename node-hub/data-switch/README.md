# data-switch

A Dora node to forward inputs as is, optionally record all inputs to an MCAP file, or replay inputs from an MCAP file.

This node is still experimental.

## Getting Started

```bash
cargo install data-switch --locked
```

## Adding to existing graph:

```yaml
- id: switch
  path: data-switch
  env:
    - MODE: record # Options: disable, record, replay; default is disable
    - MCAP_FILE: path/to/your/record.mcap # Optional, default is record.mcap
  inputs:
    image: webcam/image
    text: webcam/text
  outputs:
    - image
    - text
```
