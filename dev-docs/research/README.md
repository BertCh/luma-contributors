# Research

Reviews, measurements and reference data that still inform decisions or that code, tests and docs
read. Each report is a dated snapshot: it describes the tree on its date and is not updated. Open
work that came out of a report is tracked in [../roadmap.md](../roadmap.md); luma.gl gaps are in
[../upstream-workarounds.md](../upstream-workarounds.md) and [../upstream-proposals.md](../upstream-proposals.md).

Reports may link to documents removed on 2026-11-13 (`dev-docs/archive/`, the per-stream pass
reports, the card interaction audit, the performance research). Those resolve in git history at
`d9d56f0f`.

## Reviews of this repository

| Directory | What it holds | Feeds |
| --- | --- | --- |
| [split-review-2026-11-08](./split-review-2026-11-08/README.md) | What the split from luma.gl dropped or changed without saying so: eighteen findings | roadmap, open findings |
| [fork-comparison-2026-11-13](./fork-comparison-2026-11-13/README.md) | What the luma.gl fork changed in luma.gl-owned code (graph, core, engine, deck layers), whether published luma.gl has it, what stands in for it here, and the measurements the fork recorded | proposals, workarounds register, roadmap |
| [kepler-workflows-2026-11-14](./kepler-workflows-2026-11-14/README.md) | How to support user-defined, composable analysis workflows in the Kepler host: prior art, internal assets, execution model, Kepler integration, a draft workflow spec and the KE-WF phased plan | roadmap KE-WF |
| [analysis-workflow-ux-2026-11-14](./analysis-workflow-ux-2026-11-14/README.md) | The analysis experience beyond a workflow builder: five reviews of the existing analysis UI, catalog metadata, recipes, lenses, website and examples; eight web research notes (GIS analysis UIs, web map platforms, type-driven recommendation, workflow synthesis, node builders, step lists, curated templates, assistants); the design and the AX plan; section 9 records the library layer that landed the same day, how it was reviewed and its measured limits; `binding-defects.md` lists defects found in existing catalog bindings | roadmap AX |
| [library-review-2026-11-17](./library-review-2026-11-17/README.md) | The analysis and lenses packages from the outside: ergonomics and naming census, 36 wrong-result defects confirmed by independent refutation, gates that pass when nothing ran, release readiness and a phased plan; thirteen area reports, the verification record and the thinking-partner dialogue; `implementation.md` records what was implemented on 2026-11-18, how it was checked and what is left | roadmap LR |
| [scalability-limits-2026-11-15](./scalability-limits-2026-11-15/README.md) | Classification of data-size limits, their execution routes and remaining scale work | roadmap scale work; [retention relationships](../README.md#research-retention) |
| [library-review-2026-11-09](./library-review-2026-11-09/README.md) | The analysis package against its own contract: one verdict per contributor, duplicated machinery, missing capabilities per domain | [../contributor-contract.md](../contributor-contract.md), roadmap |
| [spatiotemporal-review-2026-11-09](./spatiotemporal-review-2026-11-09/README.md) | Five code reports and five web research reports on spatial and spatiotemporal work; two float32 experiment scripts that a spec and two API pages cite | roadmap |
| [math-correctness-review-2026-11-10](./math-correctness-review-2026-11-10/README.md) | The math of every contributor against its reference method: seventeen domain reports with their resolutions, the comparison with published luma.gl classes (`luma-official.md`), the shader-compiler findings, the behaviour changes for callers. Specs cite the closing fix reviews | workarounds register, proposals, `CHANGELOG.md` |
| [cpu-suitability-2026-11-11](./cpu-suitability-2026-11-11/README.md) | Which contributors suit the CPU, how CPU stages mix with GPU contributors in a `GPUGraph`, boundary-cost measurements, one verdict per contributor in `survey.json` | roadmap |
| [luma-usage-review-2026-11-11](./luma-usage-review-2026-11-11/README.md) | Whether the code uses published luma.gl as well as it can: eight reports, the bind-group and local-frame probes, and the export and usage inventory scripts the upgrade procedure runs | workarounds register |
| [extended-precision-2026-11-11](./extended-precision-2026-11-11/README.md) | What published luma.gl offers for 64-bit emulation and a probe of the variants on the test GPU | workarounds register |
| [new-contributors-2026-11-11](./new-contributors-2026-11-11/README.md) | Directions for new contributors: a census, five web research reports, three GPU probes, ranked candidates | roadmap |
| [picture-rescore-2026-11-10](./picture-rescore-2026-11-10/README.md) | Independent score of the website run pictures and gestures, with `scores.json` | roadmap WEB-PIC-12 |

## Sources and reference data

| Directory | What it holds | Read by |
| --- | --- | --- |
| [gpu-geospatial-implementations-2026-10-23](./gpu-geospatial-implementations-2026-10-23/README.md) | Survey of reference implementations by domain, with licensing findings | the upstream-first policy |
| [gpu-geospatial-performance-2026-10-23](./gpu-geospatial-performance-2026-10-23/integration-notes.md) | Condensed performance findings for analysis kernels | `docs/api/gpu-spatial-analysis/interpolation-expansion.md` |
| [kepler-sandbox-2026-10-28](./kepler-sandbox-2026-10-28/README.md) | Review of the Kepler sandbox: capability matrix (regenerated by a script), verification log | `examples/deck/kepler-compute` |
| [showcase-overhaul-2026-10-25](./showcase-overhaul-2026-10-25/README.md) | Reference and oracle JSON for the showcase | `examples/deck/spatial-analysis-showcase/scripts/check-*.mjs` |
| [evidence](./evidence/README.md) | Measurement captures | module test READMEs, API pages, the kepler example |

## Rules

- A new review gets its own dated folder with a README that states what was checked, on which commit, and what was run as opposed to read.
- Before retiring a closed review, identify its consumers and distinguish executable inputs, maintained tools, implementation rationale, decision evidence and historical snapshots using [the retention rules](../README.md#research-retention). Move executable consumers and inputs together; preserve measurements and provenance; replace rationale citations with equivalent durable evidence before removing their targets. A missing literal-path reference alone does not establish that an artifact is unused. Closed coordination snapshots with no remaining dependency may be recovered from Git history.
- Nothing over 1 MB, and no archives or bytecode, in git.
