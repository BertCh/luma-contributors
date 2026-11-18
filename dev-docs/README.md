# Developer documents

Plans, rules and research for `@luma-contributors/*`. User documentation lives in `docs/`.

| Document | What it is |
| --- | --- |
| [roadmap.md](./roadmap.md) | The one roadmap: open work by domain with IDs and current limits, decisions waiting for the maintainer, and a short log of finished passes |
| [contributor-contract.md](./contributor-contract.md) | The rules every contributor follows: shape, parallelism, iteration budget, parameters, no-data, output layout, status, limits, WGSL numerics |
| [authoring-walkthrough.md](./authoring-walkthrough.md) | A worked coregionalization contributor: props, validation, kernels, status, CPU twin, exports, catalog, audits, specs, docs and lenses |
| [testing-gates.md](./testing-gates.md) | Numeric comparisons, required devices, pipeline identities, shader memory estimates and structural coverage |
| [script-gates.md](./script-gates.md) | Fresh package artifacts, live typecheck ownership, oracle execution manifests, architecture rules and per-site baseline allowances |
| [regression-loop-inventory.md](./regression-loop-inventory.md) | Reviewed regression loop axes, enforced bounds, execution routes, and the kernel-wrapper checker scope |
| [coverage-matrix.md](./coverage-matrix.md) | Function-by-function coverage and the entry point each symbol lives in |
| [release-package.md](./release-package.md) | Alpha installation contract, tested dependency target versus peer range, shipped licences and offline packed-consumer verification |
| [upstream-first-plan.md](./upstream-first-plan.md) | The rules for depending only on published luma.gl, what the split replaced or dropped, and the shared helpers that stand on the published API |
| [upstream-first-recording.md](./upstream-first-recording.md) | How computation records are built from public `GPUCommandGraph` data |
| [upstream-workarounds.md](./upstream-workarounds.md) | Every capability dropped, worked around or regressed by depending on published luma.gl, and the procedure for each `@luma.gl` bump. The website reads its tables |
| [upstream-proposals.md](./upstream-proposals.md) | Feedback and proposed pull requests for luma.gl, with reproductions. Nothing has been filed |
| [research/](./research/README.md) | Dated reviews, measurements and the reference data that scripts, specs and docs read |

## Maintenance rules

- Plans go in `roadmap.md`. A finished item is deleted from it; a finished pass gets one line in its log.
- A new regression or workaround caused by luma.gl goes in `upstream-workarounds.md` in the same change. Keep one table row per capability and keep the status word first in the last cell: `website/src/components/meta/pages/upstream-first/generate-data.ts` counts rows by it.
- Something luma.gl should change goes in `upstream-proposals.md` with a reproduction.
- A superseded coordination document can be deleted once its decisions and open work have a maintained home. Before moving or deleting research, identify its consumers and classify the relationship using the retention table below; absence of a literal-path reference is not evidence that an artifact is unused. Move executable inputs and their consumers together, preserve measured baselines and provenance, and update rationale citations to a durable source before deleting their target.
- State what is true now. Dated narration of how a document got there belongs in commit messages.

## Research retention

Record the consumer, relationship and replacement before retiring a research artifact. A dated
snapshot remains evidence of its recorded revision; current requirements belong in maintained guides.
The [scalability-limits audit](./research/scalability-limits-2026-11-15/README.md) is retained decision
evidence for the roadmap's scale work and classification of limits.

| Relationship | Retained artifact and current consumer | Retirement condition |
| --- | --- | --- |
| Executable input | `research/showcase-overhaul-2026-10-25/fire-topology-reference.json` and `fire-topology-extended-reference.json`, parsed by `examples/deck/spatial-analysis-showcase/scripts/check-data-topology-reference.mjs` | Move fixtures with the script paths and verify the consumer; preserve reference provenance and licensing. |
| Maintained tool | `scripts/upstream/{exports,usage,diff}.mjs`, invoked by the [upgrade procedure](./upstream-workarounds.md#5-upgrade-procedure-for-each-lumagl-alpha-bump) | Update maintained tools and their consumer together; keep measured inventories, baselines and original script snapshots under `research/luma-usage-review-2026-11-11/` with their dated provenance. |
| Implementation rationale | `research/extended-precision-2026-11-11/`, cited by `modules/analysis/src/analysis-core/kernels/wgsl-kernel-nodes.ts`; shader-compiler findings cited by the contributor contract | Keep the explanation and reproductions, or replace each citation with equivalent durable evidence before removal. |
| Decision evidence | `research/scalability-limits-2026-11-15/` and library-review findings consumed by `roadmap.md` and the contract | Resolve the decisions and preserve the supporting measurements and constraints in their maintained home before pruning. |
| Historical snapshot | Closed review coordination and pass narratives with no remaining execution or rationale dependency | Keep a completion reference in Git history; delete only after checking nonliteral references and resolving the other relationship classes. |
