# Developer documents

Planning, decisions and research for `@luma-contributors/*`. User documentation lives in `docs/`.

## Current

| Document | What it is |
| --- | --- |
| [roadmap.md](./roadmap.md) | The single current roadmap: prioritized backlog by domain, with IDs, current limits and sources |
| [contributor-contract.md](./contributor-contract.md) | The rules every contributor follows: shape, parallelism, iteration budget, parameters, context, no-data, output layout, status, limits |
| [coverage-matrix.md](./coverage-matrix.md) | Function-by-function coverage and the entry point each symbol lives in |
| [upstream-workarounds.md](./upstream-workarounds.md) | Capabilities dropped, worked around or regressed by depending on published luma.gl, and the upgrade procedure for each `@luma.gl` alpha |
| [upstream-first-plan.md](./upstream-first-plan.md) | Rules and phases for depending only on published luma.gl |
| [upstream-first-pending.md](./upstream-first-pending.md) | Replacements for the quarantined luma.gl extensions |
| [upstream-first-foundation.md](./upstream-first-foundation.md) | Kernel-node helpers and test utilities on published luma.gl |
| [upstream-first-recording.md](./upstream-first-recording.md) | Computation records built from public `GPUCommandGraph` data |
| [upstream-proposal-gpu-resident-iteration.md](./upstream-proposal-gpu-resident-iteration.md) | Draft proposal, not sent upstream: batched indirect dispatch (PLAT-04) |
| [upstream-proposal-adapter-limits-subgroups.md](./upstream-proposal-adapter-limits-subgroups.md) | Draft proposal, not sent upstream: adapter limits, subgroups, buffer_view (PLAT-05) |

## Research

[research/](./research/README.md) holds evidence that still informs decisions: kepler sandbox review,
implementation sources and licensing, kepler SQL ledger, condensed performance findings, the
showcase overhaul notes, and the small evidence JSON cited by tests and docs. Large artifacts are
listed with checksums in manifests and are not stored in git.

## Archive

[archive/](./archive/README.md) keeps the planning documents from the luma.gl `map-contributors`
branch verbatim, each with a banner. They are the evidence trail behind the roadmap. Do not update
them; change [roadmap.md](./roadmap.md) instead.

## Maintenance rules

- Plans go in `roadmap.md`, not in new roadmap files. Finished items are deleted from it.
- A new regression or workaround caused by luma.gl goes in `upstream-workarounds.md` in the same change.
- A superseded document moves to `archive/` with a banner naming its replacement.
- Research artifacts over 1 MB go in a release asset with a sha256 manifest, never in git.
