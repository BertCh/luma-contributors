# Extended precision: what published luma.gl offers (2026-11-11)

Written in this repository. Findings only; nothing here is a plan yet, and no source file was changed.

The question: for the computations where float32 against float64 is an issue, do luma.gl or deck.gl have a better way of emulating 64-bit than the local cures?

The answer: yes for part of the category. Published `@luma.gl/shaders` `10.0.0-alpha.12` has three tools this repository does not use. deck.gl adds nothing beyond them.

## 1. What luma.gl has

Read from `node_modules/@luma.gl/shaders/src/modules/math/fp64` and `node_modules/@luma.gl/graph/dist`. `10.0.0-alpha.13` is identical for these files.

- **Integer-controlled double-single.** `fp64arithmetic` has three WGSL variants chosen by defines: classic (with `LUMA_FP64_CODE_ELIMINATION_WORKAROUND`), `LUMA_FP64_HYBRID_ARITHMETIC` and `LUMA_FP64_INTEGER_ARITHMETIC`. The integer variant does two-sum and two-product on integer significands, so a compiler that reassociates floating point cannot remove it. The classic variant with its workaround uses an integer OR with a runtime zero, which is the device of the local double-float in `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts`.
- **Exact difference from raw float64 bits.** `sub_fp64u32_to_f32` and `sub_fp64u32_to_fp64` take binary64 values as `vec2u(high, low)`, subtract exactly and round once. A `Float64Array` viewed as a `Uint32Array` is low word first. `@luma.gl/graph` accepts such positions as `uint32x4` rows in `GPUProjection`, `GPUHaversineDistance` and the pairwise geospatial classes.
- **Double-single functions in WGSL.** The `fp64` module has `exp_fp64`, `log_fp64`, `sin_fp64`, `cos_fp64` and `tan_fp64`.

luma.gl's guide, `docs/api-guide/shaders/gpu-floating-point-precision.md`, chooses by failure mode: origin rebasing or the exact difference for large coordinates, fixed point for accumulations, integer double-single for general arithmetic that needs about 14 digits. It says a source-level barrier (a runtime zero, a `let`, a `bitcast`) is not guaranteed by WGSL. The local checkout of the guide predates alpha.12 and still says the `fp64` functions are GLSL only.

deck.gl (9.4 beta and the v10 build) uses origin rebasing with a `positions64Low` residual; its `Fp64Extension` is built on luma.gl's `fp64` module.

## 2. Measurement

Apple Metal, headless Chromium, the user's Mac. Bits are `-log2` of the largest relative error against float64 over 1,048,576 random double-single pairs. The loop is 64 chained rounds of product, sum and quotient per row (192 million operations); its time includes buffer upload and readback.

| Variant | Sum | Product | Quotient | Loop |
| --- | --- | --- | --- | --- |
| Plain float32 | 22.1 | 22.5 | 22.3 | 120 ms |
| Published classic, no define | 22.1 | 23.0 | 22.5 | 125 ms |
| Published classic + workaround | 46.6 | 46.1 | 45.4 | 145 ms |
| Published hybrid | 46.5 | 46.1 | 45.4 | 235 ms |
| Published integer | 47.4 | 46.7 | 46.8 | 380 ms |
| Local double-float (H3 extended) | 46.5 | 46.0 | 46.1 | 132 ms |

- Classic with no define collapses to float32, which confirms the compiler behaviour recorded in `upstream-workarounds.md`.
- The integer variant is the most accurate and costs about three times the wall time of the local code in this loop. The float32 row bounds the transfer cost, so the ratio of the arithmetic alone is higher.
- The hybrid variant costs more than classic and gains nothing here.
- `sub_fp64u32_to_f32` matched `Math.fround(a - b)` on 65,536 of 65,536 rows near 5.1e6. Narrowing both operands to float32 first is off by up to 0.46.

The probe is [fp64-variants-probe.spec.ts.txt](./fp64-variants-probe.spec.ts.txt). To rerun it, copy it to `modules/analysis/test/utils/` without the `.txt` and run `yarn vitest run --project headless` on it. It fails on purpose so that the report is printed; do not leave it in the test tree. It replaces the module's uniform block with a private variable filled from a buffer, so the arithmetic text is the published one.

Not measured: any GPU other than this one, and shader compile time. luma.gl's source says the integer variant compiles impractically slowly on D3D12, where it selects classic.

## 3. Where it is a better way

Site counts come from a sampled sweep of `modules/*/src` and `docs/api` and are approximate.

| Group | Sites | Cure today | Verdict |
| --- | --- | --- | --- |
| Coordinates and times where only differences matter | about 26 | Origin subtraction on the CPU before upload; a hi/lo split for timestamps and the datum grid | Better where the input starts as float64: upload the raw words and take the exact difference in the kernel. `GPUPointToCell` with `precision: 'extended'` still returns the cell of the float32-rounded position, and float32 degrees carry a 0.4 to 1.7 m quantum. Candidates: `GPUPointToCell`, geodesic pairs (published `GPUHaversineDistance`), neighbor search, `dwithin`, the datum grid transform. |
| General double-single arithmetic | 1 (H3 extended) | Local double-float with an OR barrier | The published integer variant is robust by construction and slightly more accurate, and slower. Adopting it retires the "Double-float WGSL" row of `upstream-workarounds.md`. |
| Small dense solves and likelihoods (regression, kriging, Vecchia) | about 10 | A condition-number warning, or a float64 CPU twin | Not tried. A double-single solve on published `fp64arithmetic` is the one place general emulation could add something new. |
| Long sums and moments | about 13 | Fixed-point accumulator, sorted-segment sums, `getAdditionRoundingError` | No. luma.gl has no compensated reduction, and its guide recommends fixed point. |
| Exact predicates | about 12 | Exact integer orientation and lattice kernels | No. The published classifier reports `uncertain`; the local kernels decide. |
| Counts over 2^24 and 64-bit words | about 5 rows, many files | Named limits; local `(high, low)` words | No. The published `fp64_u64_*` helpers have no multiply or divide. |

## 4. What stands in the way

- `createWGSLKernelNode` builds raw `Kernel` source and does not run luma.gl's shader assembler. Using a published module needs it to accept modules and defines; the module also declares a uniform at `@binding(auto)`. Done on 2026-11-12 (roadmap LUM-02): `createWGSLKernelNode` takes `modules` and `defines`, assembles with the published `WGSLShaderAssembler` and `getPlatformInfo`, and forces the integer variant whenever an fp64 module is present, so the uniform is never read and nothing is bound. Measured through the factory on the test GPU (Apple, `modules/analysis/test/analysis-core/wgsl-kernel-node-fp64.spec.ts`): add 7.4e-16, multiply 3.5e-15, divide 2.7e-15 relative; sine 7.1e-15 and cosine 2.4e-15 absolute; assembly about 14 ms per kernel, first compile and run 165 to 194 ms with a cold shader cache. No kernel uses it yet (LUM-20).
- The public route exists: `fp64arithmetic` and `preprocess` from `@luma.gl/shaders`, `WGSLShaderAssembler` from `@luma.gl/shaders/wgsl`. The assembler selects the integer variant only for Apple WebGPU and sets no barrier define for classic on other vendors. The helpers that choose per device, `getPlatformFP64Arithmetic` and `getClassicFP64Defines`, are `@internal` in `@luma.gl/graph`. The upstream request in `upstream-workarounds.md` narrows to exporting them.
- The exact difference changes the input contract of a contributor from `float32x2` to `uint32x4` rows.

## 5. Suggested order

1. Raw float64 input for `GPUPointToCell`, and published `GPUHaversineDistance` for geodesic pairs. This is the gain in precision.
2. Benchmark the H3 extended kernel on the published integer variant before replacing the local arithmetic.
3. Trial a double-single dense solve against the float64 twins.
4. Leave sums, predicates and the 2^24 limits as they are.
