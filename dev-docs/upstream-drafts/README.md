# First four luma.gl filing drafts

These are ready-to-file issue bodies, in the requested filing order. **Nothing has been filed,
posted or sent.** The proposed APIs are requests, not exports available today.

Alpha.16 disposition (2026-11-18): all four filings remain **not satisfied**. Packed transfer
still copies rows; equal-width histogram arithmetic remains unchanged; graph algorithms still
require physical topology/output vectors; raw-buffer import and typed-array CPUVector convenience
constructors are still absent. `getGPUVectorWGSLLayout` from `@luma.gl/graph/gpu-data` partly helps
draft 4's layout metadata, but supplies neither requested constructor. Newly public segmented
reduction, PCG and grid WGSL satisfy other requests in [the proposal inventory](../upstream-proposals.md#alpha16-request-disposition-2026-11-18).

| Order | Draft | Kind | Evidence |
| --- | --- | --- | --- |
| 1 | [Packed CPU/GPU transfers](./01-packed-cpu-gpu-transfer.md) | Performance | Alpha.16 copy loops verified in installed code; alpha.12 timings retained with provenance; standalone benchmark included |
| 2 | [Histogram edges and GPU domains](./02-histogram-edges-and-domains.md) | Correctness | Alpha.16 shader verified; committed alpha.12 GPU observations and desired assertions separated |
| 3 | [Algorithms on caller CSR with masks](./03-caller-csr-and-masks.md) | API extension | Alpha.16 declarations; typechecked completeness contract, two-hop and sentinel fixtures; headless baseline cases supplied for the lead |
| 4 | [Typed buffer imports and CPU array construction](./04-typed-buffer-and-cpu-vector.md) | Ergonomics | Alpha.16 public imports and constructors; executable CPU example and raw-buffer example |

All four were checked offline against installed `@luma.gl/graph`, `@luma.gl/core` and
`@luma.gl/shaders` **10.0.0-alpha.16**. Package `exports` and `dist` implementations/declarations are
the authority for the current surface. The runnable examples import only published entry points.
Examples that need a GPU accept an application-created WebGPU `Device`; they require no downstream
package or test harness. Proposed type declarations and the numeric layout precheck are explicitly
marked as proposals; blocks marked `text` contain method-call sketches or expected output.

No GPU benchmark or headless spec was run while preparing these drafts. The historical performance
and histogram observations are for the stated older release and adapter; retained source expressions
do not establish identical performance or GPU behavior on alpha.16. No numerical algorithm fix is
claimed here. The histogram spec intentionally pins some *incorrect observed* results; the second
draft gives the assertions an upstream fix should satisfy instead.

Review regressions are in `modules/analysis/test/analysis-core/upstream-drafts.node.spec.ts` and
`upstream-drafts.spec.ts`. The Node suite executes the layout guard and benchmark validation,
including a deliberately stale first-snapshot implementation, and checks the CSR example against
installed and proposed types. The headless file checks the exact BFS, sentinel and overflow fixtures
through existing implementations; it cannot establish support for the proposed view API.

Each issue body is self-contained through its alternatives and acceptance criteria. The final
“Downstream deletion” section records this repository's retirement plan and can accompany the filing.
The [proposal inventory](../upstream-proposals.md) remains the broader list; these drafts implement
[library-review decision 8](../research/library-review-2026-11-17/README.md#decisions-for-the-maintainer).
