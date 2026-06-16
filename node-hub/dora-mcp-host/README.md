# dora-mcp-host

A Model Context Protocol (MCP) host node. It exposes an OpenAI-compatible HTTP
API, routes chat completions to multiple AI providers (OpenAI, Gemini, Deepseek)
or back into the dataflow, and connects to MCP servers to expose their tools.

## Behavior

On start, the node reads a config file (path from the `CONFIG` env var,
defaulting to `config.toml`) and exits if the file is missing. It then starts an
OpenAI-compatible HTTP server on the configured `listen_addr` (default
`0.0.0.0:8008`) exposing `/v1/models` and `/v1/chat/completions`.

Incoming chat requests are routed by model id to a configured provider:

- **openai / gemini / deepseek** providers call the upstream API directly over
  HTTP.
- A **dora** provider forwards the prompt into the dataflow. The node sends the
  prompt text as a UTF-8 string array on the provider's configured output id,
  tagged with a `__dora_call_id` metadata parameter. It then waits for an input
  carrying the same `__dora_call_id`, wraps that text in a chat-completion
  response, and returns it to the HTTP caller.

MCP servers declared in the config (stdio or streamable HTTP transports) are
started and their tools registered with the chat session.

The node stops on a Dora `Stop` event or when the HTTP server task ends.

## Inputs

- `tick`: reply stream for the Dora provider. A node wired here receives the
  prompt on the configured output and must reply with a UTF-8 string array
  carrying the same `__dora_call_id` metadata parameter. The input is matched by
  metadata, not by id, so any input id may be wired; inputs without a matching
  `__dora_call_id` are warned and ignored.

## Outputs

- `tick`: the chat prompt forwarded to the dataflow when a request routes to the
  Dora provider, emitted as a UTF-8 string array with a `__dora_call_id`
  metadata parameter. The actual output id is set by the `output` field of the
  Dora provider in the config file.

## Environment variables

- `CONFIG` (default `config.toml`): path to the config file (toml, json, or
  yaml).
- `GEMINI_API_KEY`, `GEMINI_API_URL`: fallback Gemini credentials/URL when not
  set in the config.
- `DEEPSEEK_API_KEY`, `DEEPSEEK_API_URL`: fallback Deepseek credentials/URL when
  not set in the config.
- `OPENAI_API_KEY`, `OPENAI_API_URL`: fallback OpenAI credentials/URL when not
  set in the config.

In the config file, an `api_key` value prefixed with `env:` is read from the
named environment variable.

## Usage

```yaml
nodes:
  - id: dora-echo
    hub: dora-echo
    inputs:
      echo: dora-mcp-host/text
    outputs:
      - echo

  - id: dora-mcp-host
    hub: dora-mcp-host
    outputs:
      - text
    inputs:
      text: dora-echo/echo
    env:
      CONFIG: mcp_host.toml
```

Example `mcp_host.toml`:

```toml
listen_addr = "0.0.0.0:8118"
endpoint = "v1"

[[providers]]
id = "dora"
kind = "dora"
output = "text"

[[providers]]
id = "moonshot"
kind = "openai"
api_key = "env:MOONSHOT_API_KEY"
api_url = "https://api.moonshot.cn/v1"

[[models]]
id = "kimi-latest"
route = { provider = "moonshot", model = "kimi-latest" }

[[mcp.servers]]
name = "amap-maps"
protocol = "stdio"
command = "npx"
args = ["-y", "@amap/amap-maps-mcp-server"]
envs = { AMAP_MAPS_API_KEY = "your_amap_maps_api_key_here" }

[[mcp.servers]]
name = "local"
protocol = "streamable"
url = "http://127.0.0.1:8228/mcp"
```

The Dora provider's `output` field must match the output id wired in the
dataflow (`text` above).

## Build

```bash
cargo build --release --target-dir target
```
