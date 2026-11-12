# Measurements made while deciding roadmap section 16 (2026-11-11 and 2026-11-12)

All on the development machine: headless Chromium WebGPU on an Apple GPU (Metal). Each section is the implementor's own record, copied from its report. The bind-group table is in the header of [bind-group-probe.spec.ts.txt](./bind-group-probe.spec.ts.txt) and in roadmap section 16.6.

## 1. `GPULocalFrameProjection`: local plan route against published `planProjection` with `destinationOrigin` (NC-01, LUM-11)

Not adopted. The probe is [local-frame-route-probe.spec.ts.txt](./local-frame-route-probe.spec.ts.txt) (it reports by throwing; do not add it to the test tree as written).

- `gpu-local-frame-projection.ts` is unchanged (never edited). Read: published
  `projection-planning`, `projection-program`, `gpu-projection`, `gpu-projection-program`, the
  local class, its operation form (`gpu-local-frame-projection-operation.ts`), CPU twin
  (`cpu-local-frame-projection-operation.ts`), constructors (showcase `hot-spots.compute.ts`,
  lenses example, website run, kernel-audit cases, 3 GPU specs).
- Probe: a temporary headless spec that builds both routes directly with the published classes
  (old: `compileProjectionPlan` + `GPUProjection` with `destinationOrigin` overwritten to `[0,0]`;
  new: `planProjection({precision: 'local-f32', destinationOrigin: [0,0], inputFormat: 'uint32x4', onUnsupported: 'throw'})`
  + `GPUProjectionProgram`), raw binary64 input, tolerance 0.01, 65,536 seeded uniform rows inside
  square bounds centred on the frame origin; error = max |GPU - JS float64 `frame.project`| over
  both axes of valid rows. Run twice; numbers agree. Probe kept beside
  this file as `local-frame-route-probe.spec.ts.txt`.

Accuracy (max error in metres, patch count; mid = origin 8.5417, 47.3769; high = 18.9553, 69.6496):

| frame | lat | half-width | old route | new route |
| --- | --- | --- | --- | --- |
| AEQD | mid | 0.25 | 1 patch, 0.0039 | 1 patch, 0.0011 |
| AEQD | mid | 1 | 10 patches, 0.0123 | 4 patches, 0.0053 |
| AEQD | mid | 2 | throws (max subdivision depth) | 16 patches, 0.0093 |
| AEQD | mid | 4 | throws | 64 patches, 0.0172 |
| AEQD | high | 0.25 | 1, 0.0032 | 1, 0.0010 |
| AEQD | high | 1 | 16, 0.0119 | 4, 0.0048 |
| AEQD | high | 2 | throws | 16, 0.0088 |
| AEQD | high | 4 | throws | 64, 0.0166 |
| LAEA | mid | 0.25 | 1, 0.0036 | 1, 0.0010 |
| LAEA | mid | 1 | 10, 0.0120 | 4, 0.0053 |
| LAEA | mid | 2 | throws | 16, 0.0092 |
| LAEA | mid | 4 | throws | 64, 0.0171 |
| LAEA | high | 0.25 | 1, 0.0035 | 1, 0.0010 |
| LAEA | high | 1 | 7, 0.0111 | 4, 0.0048 |
| LAEA | high | 2 | throws | 16, 0.0088 |
| LAEA | high | 4 | throws | 64, 0.0166 |

No invalid rows on either route.

Speed (AEQD, mid latitude, headless Chromium WebGPU on this Mac, shared GPU; median of 9 timed
submit-and-`onSubmittedWorkDone` runs after one warm-up; two runs of the probe, each with two
repeats per route, ranges given; timer resolution about 0.1 ms):

| bounds | rows | old median | new median | ratio | plan (CPU) old / new | `graph.compile()` old / new | first run (incl. pipeline creation) old / new |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ±0.25 | 65,536 | 0.4-0.6 ms | 3.0-4.8 ms | ~7-10x | 0.5-1.2 / 0.7-0.8 ms | 5.8-10.4 / 6.4-6.5 ms | 1.2-2.2 / 3.3-5.1 ms |
| ±0.25 | 1,000,000 | 1.0-3.0 ms | 28.8-29.0 ms | ~10-29x | 0.5-0.6 / 0.7-0.8 ms | 5.6-6.1 / 6.4-6.6 ms | 5.4-89.5 / 32.4-322 ms |
| ±1 | 65,536 | 0.4-0.5 ms | 2.6-3.0 ms | ~6x | 4.4-4.8 / 2.0-2.1 ms | 5.4-5.9 / 6.3-6.6 ms | 1.3-1.6 / 3.2-3.4 ms |
| ±1 | 1,000,000 | 3.1-4.5 ms | 29.1-29.3 ms | ~6.5-9.5x | 4.4-6.4 / 2.0-2.3 ms | 5.5-6.5 / 6.4-6.9 ms | 7.3-129 / 32.6-300 ms |

(`graph.compile()` does not create pipelines; the first-run column carries the shader compile, the
large values being the first build of each shader in the session.)

- Decision: the new route is better on accuracy everywhere measured (about 2.3x smaller max error
  at 1 degree, and it handles 2 and 4 degree bounds the old route refuses), but at 1,000,000 rows it
  costs about 29 ms against 1-4.5 ms, 6.5x to 29x, well past the "about three times" limit. The
  cost is the per-row double-single (integer-emulated fp64, `fp64Arithmetic: 'platform'`) program;
  it is flat in patch count. `GPUProjectionProgram` has no option to evaluate the adaptive stage in
  f32. Per the task, the old code stays and nothing was changed in the class.
- Note for the lead: the old route's measured max error at ±1 degree (0.011-0.012 m) is above its
  0.01 m tolerance, as the class documentation already says (float32 rounding of absolute frame
  coordinates); the existing specs bound it at 0.02 / 0.03 m. A possible luma.gl request: a
  `local-f32` program option that evaluates adaptive patches in f32 relative to the destination
  origin (fit validated for that origin), which would give the accuracy without the double-single cost.


## 2. `GPURasterFillHoles`: border flood against published `GPURasterConnectedComponents` (RT-01, LUM-09)

Not changed; a decision on the meaning of `iterationCount` is open (roadmap 16.7).


**RT-01 `GPURasterFillHoles` on published `GPURasterConnectedComponents`: NOT DONE.** No source file
changed (`gpu-raster-reconstruction.ts` and `raster-expansion-utils.ts` are untouched).

Reason: the task allows the change only if every output is bit-identical wherever the current class
converges, status words included. `output.status.iterationCount` cannot stay the same, and a spec
checks it exactly:

- Today `iterationCount` is the number of flood rounds that ran. That is the largest geodesic distance
  from the border through the background, plus one round that confirms nothing changed. The CPU twin
  `computeRasterFillHolesOnCPU` (`raster-morphology/cpu-raster-fill-holes-operation.ts:30-31, 69`)
  documents and returns that number. The headless spec
  `modules/analysis/test/geospatial-expansion/raster-fill-holes-operation.spec.ts:43-44` requires the
  GPU `rounds` port to equal the CPU one exactly, on every fixture.
- The published class's `iterationCount` counts union-find rounds (hook, compress, convergence check).
  That number has nothing to do with the flood depth: 3 on a 255 x 255 spiral, where the flood needs
  thousands of rounds. Getting the old number back needs the breadth-first distance, which is the
  flood we would be removing.
- A second spec would also change, though the task expected this one: at
  `raster-terrain-expansion.spec.ts:830-862`, `maximumIterations: 1` gives a partial fill, and the spec
  checks `filled[12] = 0` and `iterationCount = 2`.

Under brief rule 3 (a delegation that changes a status word a spec checks), I stopped here.

Everything else the task asked me to check was in favour of the change (section 4). The change is
worth making if the lead accepts a new meaning for `iterationCount`. That decision touches files
outside my list:

- the CPU twin's `iterationCount` and its doc;
- `cpu-raster-fill-holes-operation.node.spec.ts:45-54, 93`, which checks layer counts 4, 1 and 4;
- the comparison at `raster-fill-holes-operation.spec.ts:43-44`.

Options:

- (a) `iterationCount` becomes the labelling rounds, and the CPU twin reports the same thing or stops
  being compared on that port.
- (b) `iterationCount` is documented as "not comparable across implementations". The comparison spec
  drops the `rounds` port, and the CPU twin keeps its layer count.

Either way, the planned design holds:

1. A local kernel inverts the mask into a transient `uint32` band (it needs a band `id`).
2. `new GPURasterConnectedComponents({..., connectivity}).addToGraph` runs inside
   `captureGraphCommandNodes`.
3. A kernel sets `flag[label - 1] = 1` for border pixels.
4. A publish kernel writes `input != 0 || (label != 0 && flag[label - 1] == 0)`, and
   `iterationLimitReached = 1 - converged`.


## 4. Checking the review's claims (for the register)

- **Connectivity.** These match. The published hook pass joins the left and up neighbours, plus up-left
  and up-right at 8 (`gpu-raster-connected-components.js`, hook kernel). The union is symmetric, so this
  is the full 4 or 8 neighbourhood. The local flood uses the cross at 4 and the 3 x 3 square at 8
  (`gpu-raster-reconstruction.ts:141-183`), with `props.connectivity ?? 4` (`:300`). Both default to 4.
- **What it outputs.** Each label is the sparse "minimum row-major pixel index + 1" of its component,
  not a dense number, and 0 means background or invalid. At convergence the label is exact and
  deterministic: the minimum pixel is always its own root, and a round with no change implies a flat
  forest that agrees across every edge. Validity comes from `input.validity` and `noDataValue`.
  `GPURasterFillHoles` takes no validity or no-data input, so neither applies.
- **Limits.** The published raster limit is `MAXIMUM_RASTER_PIXEL_COUNT = 0xffffffff`, the same as the
  local `validateRasterGrid`. The round budget runs from 1 to 64, default `ceil(log2 N) + 2`. The class
  needs a WebGPU device, as the local WGSL kernels do. It needs a band `id`, which the review did not
  mention. It only has `addToGraph`, with no `getCommandNodes`.
- **Iteration count.** Rounds that ran, including the round that confirmed nothing changed. It is not
  the flood depth, and it can vary with scheduling, because hooks inside a pass see one another's
  concurrent writes. The final labels do not vary.
- **Non-convergence.** If the budget runs out, every label and every validity value is cleared and
  `converged = 0` (publish kernel). Root chasing is capped at 32 pointer steps per pass, but that cannot
  produce a wrong converged result: if compression changes nothing, the forest is already flat. No input
  failed to converge, the spiral and serpentine included. The review's "about log2(N) rounds" is
  conservative: we measured 3 to 5 rounds.
- **Cost.**
  - Today: `2 + getRasterJacobiLoopNodeCount(65) + 2 = 89` command nodes at the default of 64, and
    169 at the showcase's 128. Scratch is about 6N words (marker, mask, reconstructed and validity,
    plus two ping-pong buffers).
  - Planned: `1 + 3B + 1 + 3` nodes. At 256² (B = 18) that is 59; at 3 x 3 (B = 6) it is 23, against
    14 today with `maximumIterations: 4`. Scratch is about 5N words (inverted, labels, validity, parents,
    flags).
  - Not a blocker.
- **`maximumIterations`.** The local class accepts up to 26,207; the published budget is capped at 64.
  The plan was to keep accepting the prop and pass `min(maximumIterations, 64)`, or ignore it and use
  the published default. Either changes what the prop means, which the review already said.

Review claims that were wrong or missing:

- "Status contract stays": `converged` stays, `iterationCount` does not.
- The class needs a band `id`.
- In practice it converges in about 3 to 5 rounds, not about log2(N).

What callers could simplify if this lands later:

- The showcase's `maximumIterations: 128` (`missing-observations.compute.ts:736`) and the kernel-audit
  case's 128 rounds (`cases-expansion.ts:102`) would no longer be needed.
- The `converged = 0` caveat about partial fills in the class comment would go.

