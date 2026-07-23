# nanobot-kernel-session — nanobot-specific setup

For starting an `mcp_app` session and syncing results back, see
[`skills/README.md`](../../README.md). This covers only what's
nanobot-specific.

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
Add the command `launch`/`prepare-session` printed to
`tools.mcpServers` in `~/.nanobot/config.json`:

```json
{
  "tools": {
    "mcpServers": {
      "cpu-kernel-baseline": {
        "command": "ssh",
        "args": ["...as printed by launch/prepare-session..."],
        "toolTimeout": 600
      }
    }
  }
}
```

- `toolTimeout`: 300–600s, not the 30s default — first `compile()` per
  definition may trigger a slow baseline collection.
- **Don't set `enabledTools`.** It defaults to `["*"]` (tools + resources +
  prompts); any explicit subset disables resources entirely. This skill
  needs `list_resources()`/`read_resource()` (reading reference kernels and
  your own earlier versions) — restricting to just
  `compile`/`evaluate`/`disassemble`/`submit` would silently break that.
- Config is process-wide — restart nanobot after editing it.

## 3. Start nanobot

- One-shot: `nanobot agent -m "<task>"`
- Long-lived: `nanobot gateway` (`--background` to daemonize; `status`/
  `stop`/`restart`/`logs` to manage it)

Give the agent a task naming the dataset/definitions to work on —
`SKILL.md` takes it from there.
