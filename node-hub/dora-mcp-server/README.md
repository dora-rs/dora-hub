# dora-mcp-server

A Model Context Protocol (MCP) server node. It exposes the surrounding dataflow
as MCP tools over HTTP: incoming tool calls are forwarded to other dora nodes as
outputs, and their replies are returned to the MCP client as tool results.

## Behavior

On startup the node loads a config file (path from the `CONFIG` env var,
default `config.toml`) describing the server identity and the list of MCP tools
to expose. If the file is missing or invalid the node exits with an error.

It then binds an HTTP server (default `0.0.0.0:8008`, path `/mcp`) and connects
as a dora node, serving two request paths:

- **HTTP MCP requests**: an MCP client POSTs JSON-RPC requests (`initialize`,
  `ping`, `tools/list`, `tools/call`). For `tools/call`, the node looks up the
  tool's configured `output`, sends the call parameters on that dora output with
  a generated `__dora_call_id` in the metadata, and waits for the matching reply
  before returning the result to the client.

- **`request` dora input**: a JSON-RPC MCP request supplied as a dora input
  string. The result is emitted on the `response` output.

When a tool is invoked, the downstream node is expected to send its result back
as a dora input carrying the same `__dora_call_id` metadata key. The node uses
that id to match the reply to the pending call and resolve it.

## Inputs

- `request`: a JSON-RPC MCP request as a UTF-8 string. Its result is emitted on
  `response`.
- `tick`: tool-call replies from downstream nodes. Each reply must carry the
  originating `__dora_call_id` metadata key so it is matched to the pending MCP
  call; the reply body is returned to the MCP client as the tool result. Reply
  input ids are config-driven (one per tool output target), so they are declared
  under the single catch-all `tick` input.

## Outputs

- `response`: the JSON-RPC MCP result (UTF-8 string) for a `request` input.

Tool calls additionally emit on per-tool output ids defined in the config file
(`mcp_tools[].output`). These are config-driven and therefore not fixed in this
contract; wire each tool's output node accordingly.

## Environment variables

- `CONFIG` (string, default `config.toml`): path to the MCP server config file.
  The format is inferred from the extension (`.toml`, `.json`, `.yaml`/`.yml`).
  It defines the server name/version, listen address, HTTP endpoint, and the MCP
  tools to expose.

## Usage

```yaml
nodes:
  - id: dora-mcp-server
    hub: dora-mcp-server@^0.3
    inputs:
      request: client/request
      tick: my-tool-node/result
    outputs:
      - response
    env:
      CONFIG: config.toml
```

## Build

```bash
cargo build --release --target-dir target
```
