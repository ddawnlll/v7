# CLAUDE.md — v7

Read [RULES.md](RULES.md) first. The rules override everything, including this file.

## How to work here

- **Small diffs.** One concern per commit. If a change touches more than ~3 files, split it.
- **Every claim ships with a verify command** the operator runs on the remote box.
  Never claim completion from your own run alone (RULES §11).
- **Execution happens on the remote SSH box; local is for audit and small edits.**
- **Write nothing speculative.** No helpers "for later", no config options nobody asked for,
  no abstractions with one caller (RULES §3).
- **Salvage flow:** audit line-by-line → write tests → move to `lab/` → note the audit
  in the commit message. Never import `salvage/` (RULES §2).
- Repo language: English. No new top-level directories without updating README layout.

## What this repo is not

- Not a continuation of `v7-engine` — that repo is frozen, read-only, and never imported.
- Not a trading system. It is a measurement instrument. Execution code does not exist here
  until an edge survives out-of-sample (see ROADMAP Phase 4+).
