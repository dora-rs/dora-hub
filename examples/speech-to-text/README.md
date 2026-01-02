# Speech to Text Example

## Overview

This dataflow creates a complete speech-to-text pipeline:

```
Microphone -> VAD -> Whisper (STT) -> Rerun (Display)
```

The pipeline captures audio from your microphone, detects when you're speaking, transcribes your speech to text using the Whisper model, and displays the results in the Rerun viewer.

## Nodes

- **dora-microphone**: Captures audio from microphone
- **dora-vad**: Voice Activity Detection - detects when you're speaking
- **dora-distil-whisper**: Speech-to-text using Distil-Whisper model
- **dora-rerun**: Visualizes transcription in Rerun viewer

## Prerequisites

- Python 3.11+
- dora-rs
- Microphone
- uv (Python package manager)

## Getting Started

### 1. Install dora

```bash
# Install dora CLI
cargo install dora-cli

# Or install Python package (must match CLI version)
pip install dora-rs
```

### 2. Build and Run

```bash
cd examples/speech-to-text

# Create virtual environment
uv venv --seed -p 3.11

# Build dataflow
dora build whisper.yml --uv

# Run dataflow
dora run whisper.yml --uv
```

### 3. View Results

```bash
# Connect to Rerun viewer
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

## Configuration

### Whisper Node Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TARGET_LANGUAGE` | Target language for transcription | `english` |

## Dataflow Variants

- `whisper.yml`: Production version using pre-packaged nodes
- `whisper-dev.yml`: Development version for local development

## Architecture

```
+------------+     +---------+     +------------------+
| Microphone | --> |   VAD   | --> | distil-whisper   |
+------------+     +---------+     | (Speech-to-Text) |
                                   +------------------+
                                            |
                                            v
                                       +--------+
                                       | rerun  |
                                       |(Display)|
                                       +--------+
```

## Troubleshooting

### Microphone Issues
- Check system microphone permissions
- Verify correct audio input device is selected
- Test microphone in other applications first

### Model Download Slow
- First run downloads the Whisper model which may take time
- Ensure stable internet connection
- Model is cached after first download

### Rerun Version Mismatch
- If you see version warnings, install matching Rerun SDK:
  ```bash
  pip install rerun-sdk==<version>
  ```

## Source Code

- [dora-microphone](https://github.com/dora-rs/dora-hub/tree/main/node-hub/dora-microphone)
- [dora-vad](https://github.com/dora-rs/dora-hub/tree/main/node-hub/dora-vad)
- [dora-distil-whisper](https://github.com/dora-rs/dora-hub/tree/main/node-hub/dora-distil-whisper)
- [dora-rerun](https://github.com/dora-rs/dora-hub/tree/main/node-hub/dora-rerun)
