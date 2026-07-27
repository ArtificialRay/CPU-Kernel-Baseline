# nanobot-kernel-session — nanobot-specific setup

For starting an `mcp_app` session and syncing results back, see
[`skills/README.md`](../../README.md). This covers only what's
nanobot-specific.

## 0. Install nanobot

nanobot is a Python package — this repo assumes it's already installed but
doesn't pin it:

```bash
pip install nanobot-ai          # or: uv tool install nanobot-ai
nanobot onboard                  # creates ~/.nanobot/config.json + workspace
```

Set the model/provider in `~/.nanobot/config.json` (`agents.defaults.model` +
the matching `providers.<name>.apiKey`) — this is separate from any keys the
`eval/` harness uses.

## 1. Where `SKILL.md` goes

nanobot loads skills from `<workspace>/skills/<name>/SKILL.md`
(`agents.defaults.workspace` in `~/.nanobot/config.json`, default
`~/.nanobot/workspace`). Symlink this directory in:

```bash
ln -s <repo>/skills/nanobot/nanobot-kernel-session \
      <workspace>/skills/nanobot-kernel-session
```

The directory name is the lookup key, not the frontmatter `name:` field.
`metadata.nanobot.always: true` means its full content is injected into the
system prompt every turn — no trigger phrase needed.

## 2. Configure the MCP server entry

General mechanism: nanobot's [MCP tools
guide](https://github.com/HKUDS/nanobot/blob/main/docs/guides/mcp-tools-for-ai-agents.md).
`launch`/`prepare-session` prints an **SSE endpoint** (`tunnel up:
http://127.0.0.1:<port>/sse` — it tunnels to the remote server itself, it
does *not* hand you an ssh spawn command), so the `tools.mcpServers` entry in
`~/.nanobot/config.json` is the SSE form, using that URL:

```json
{
  "tools": {
    "ssrfWhitelist": ["127.0.0.0/8"],
    "mcpServers": {
      "cpu-kernel-baseline": {
        "type": "sse",
        "url": "http://127.0.0.1:<port>/sse",
        "toolTimeout": 600
      }
    }
  }
}
```

- **`ssrfWhitelist` is required** for the tunneled endpoint. nanobot's SSRF
  guard blocks private/loopback addresses by default and will silently refuse
  to connect (`blocked unsafe URL ... resolves to private/internal address`,
  then `No MCP servers connected`) — whitelist the loopback CIDR or the run
  starts with zero MCP tools and the agent flails on builtins.
- `toolTimeout`: 300–600s, not the 30s default — first `compile()` per
  definition may trigger a slow baseline collection.
- **Don't set `enabledTools`.** It defaults to `["*"]` (tools + resources +
  prompts); any explicit subset disables resources entirely. This skill
  needs `list_resources()`/`read_resource()` (reading reference kernels and
  your own earlier versions) — restricting to just
  `compile`/`evaluate`/`disassemble`/`submit` would silently break that.
- Config is process-wide — restart nanobot after editing it. Pin
  `--local-port` on `launch` so the `url` above stays stable across relaunches.

## 3. Start nanobot

- One-shot: `nanobot agent -m "<task>"`
- Long-lived: `nanobot gateway` (`--background` to daemonize; `status`/
  `stop`/`restart`/`logs` to manage it)

Give the agent a task naming the dataset/definitions to work on —
`SKILL.md` takes it from there.
