# luma.gl usage review (2026-11-11)

Question: is the repository using published luma.gl (`@luma.gl/core`, `@luma.gl/shaders`,
`@luma.gl/graph`, and `@luma.gl/engine` in the examples, the website and the tests) and its
built-in features as well as it can. Baseline: `10.0.0-alpha.13`, to which the pins moved on the
same day with no source change.

It extends the comparison of 2026-11-10
([luma-official.md](../math-correctness-review-2026-11-10/luma-official.md), against alpha.10) to
what that one left out: `modules/kepler`, `modules/lenses`, the website, the examples, the test
helpers, `@luma.gl/engine`, gather, run-length and point-to-segment code, and the execution
machinery. The plan and the outcome are in [roadmap.md](../../roadmap.md) section 16; the
corrections went into [upstream-workarounds.md](../../upstream-workarounds.md).

## Method

1. An inventory made by script: every public export of every entry point of the four packages
   (TypeScript checker over the declaration files), and every `@luma.gl/*` import in the
   repository by area. `scripts/exports.mjs`, `scripts/usage.mjs`, `scripts/diff.mjs`; results in
   `usage-summary.txt` and `usage-report.json`. Rerun them at each luma.gl bump (the upgrade
   procedure in the register says how).
2. Eight read-only reviewers, one per capability area, each comparing published features with the
   local code that does the same job, by reading both sides ([brief.md](./brief.md)). They ran
   nothing on a GPU.
3. Implementors changed the code for the findings chosen in roadmap 16.2, one owner per file set.
4. Every change was then read by an independent reviewer that did not write it, told to find what
   was wrong, and the defects were fixed and reviewed again where the fix was not trivial. The
   lead ran every GPU spec.

## Reports

| Report | Area | Findings |
| --- | --- | --- |
| [primitives-data.md](./primitives-data.md) | Gather, scatter, scan, compaction, sort, runs, search, hashing, aggregation of `gpu-core` against all of `modules/analysis/src` | PD-01 to PD-11 |
| [primitives-numeric.md](./primitives-numeric.md) | Dense and sparse linear algebra, FFT, convolution, stencils, spatial indexes, pairwise geometry, vector search, time series | PN-01 to PN-09 |
| [graph-machinery.md](./graph-machinery.md) | Building, compiling, running, iterating, inspecting and recording GPU work | GM-01 to GM-08 |
| [shaders-core.md](./shaders-core.md) | WGSL composition and device use against `@luma.gl/shaders` and `@luma.gl/core`; the design note for shader modules in `createWGSLKernelNode` | SC-01 to SC-07 |
| [raster-terrain.md](./raster-terrain.md) | `raster-analysis` and `gpu-terrain` against `gpu-raster`; class table in [raster-terrain-class-table.md](./raster-terrain-class-table.md) | RT-01 to RT-09 |
| [network-cloud-projection.md](./network-cloud-projection.md) | Network, point cloud, projection and discrete global grids | NC-01 to NC-07 |
| [kepler-lenses.md](./kepler-lenses.md) | `modules/kepler` and `modules/lenses` | KL-01 to KL-12 |
| [website-examples-tests.md](./website-examples-tests.md) | Website, examples, test helpers, documentation snippets | WE-01 to WE-14 |

[measurements.md](./measurements.md) holds the GPU measurements behind two decisions not to adopt a published class as it is (the local-frame projection route, with its probe `local-frame-route-probe.spec.ts.txt`, and fill holes on connected components).

`bind-group-probe.spec.ts.txt` is the measurement behind roadmap LUM-03 (bind groups per encode and
encode time for the published `Kernel` with and without pipeline-held bindings). It is a probe, not
a test: it reports by throwing, so do not add it to the test tree as written.

## What the reports are not

- A reviewer's finding is a reading, not a measurement. Several did not survive: the published
  `GPUTranspose` rejects large degenerate shapes the local kernel accepts; the published projection
  program is 6.5 to 29 times slower than the local route; `GPURasterConnectedComponents` cannot
  report the flood depth a status word promises. Roadmap 16.6 lists what was adopted and what was
  not, with the reason.
- An independent review of a change was also wrong once: it reported that a column output sharing
  a buffer with the row output "worked before" in `GPUPairGather`. WebGPU refuses one buffer bound
  read-only and read-write in the same pass, so it never did. The GPU run settled it.
- Line numbers in the reports are those of 2026-11-11, before the changes of roadmap 16.2.
