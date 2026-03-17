# dora-tflite

A Dora node for running TensorFlow Lite inference on edge and constrained devices (Raspberry Pi, microcontrollers).

## Installation

```bash
pip install dora-tflite
```

## Usage

```yaml
nodes:
  - id: tflite-inference
    path: dora-tflite
    inputs:
      tensor: source/tensor
    outputs:
      - inference
    env:
      MODEL: path/to/model.tflite
```

## Inputs

- `tensor`: Input tensor as a flat array (PyArrow)

## Outputs

- `inference`: Output tensor as a flat array (PyArrow)

## Environment Variables

- `MODEL`: Path to the .tflite model file (default: model.tflite)
