# Developer documents

Plans, rules and research for `@luma-contributors/*`. User documentation lives in `docs/`.

| Document | What it is |
| --- | --- |
| [roadmap.md](./roadmap.md) | The one roadmap: open work by domain with IDs and current limits, decisions waiting for the maintainer, and a short log of finished passes |
| [contributor-contract.md](./contributor-contract.md) | The rules every contributor follows: shape, parallelism, iteration budget, parameters, no-data, output layout, status, limits, WGSL numerics |
| [coverage-matrix.md](./coverage-matrix.md) | Function-by-function coverage and the entry point each symbol lives in |
| [upstream-first-plan.md](./upstream-first-plan.md) | The rules for depending only on published luma.gl, what the split replaced or dropped, and the shared helpers that stand on the published API |
| [upstream-first-recording.md](./upstream-first-recording.md) | How computation records are built from public `GPUCommandGraph` data |
| [upstream-workarounds.md](./upstream-workarounds.md) | Every capability dropped, worked around or regressed by depending on published luma.gl, and the procedure for each `@luma.gl` bump. The website reads its tables |
| [upstream-proposals.md](./upstream-proposals.md) | Feedback and proposed pull requests for luma.gl, with reproductions. Nothing has been filed |
| [research/](./research/README.md) | Dated reviews, measurements and the reference data that scripts, specs and docs read |

## Maintenance rules

- Plans go in `roadmap.md`. A finished item is deleted from it; a finished pass gets one line in its log.
- A new regression or workaround caused by luma.gl goes in `upstream-workarounds.md` in the same change. Keep one table row per capability and keep the status word first in the last cell: `website/src/components/meta/pages/upstream-first/generate-data.ts` counts rows by it.
- Something luma.gl should change goes in `upstream-proposals.md` with a reproduction.
- A superseded document is deleted, not archived; git history keeps it. Fix the references to it in the same change.
- State what is true now. Dated narration of how a document got there belongs in commit messages.
