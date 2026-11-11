# Math correctness review: shader-compiler behaviour (`compiler`)

Date: 2026-11-11 (closing pass). Scope: the four behaviours of the test GPU's shader pipeline that
the first pass recorded in `dev-docs/contributor-contract.md` section 11,
`dev-docs/upstream-workarounds.md` section 6 and `dev-docs/upstream-proposal-math-review.md`
section 13, for which "the cause inside the compiler was not found". No library source was changed.

Scratch scripts (a Playwright launcher, a Swift program that compiles Metal source, the variant
matrices) are in the session scratchpad under `compiler/` and are not kept; section 7 says how to
rebuild each from this report. Nothing was sent or filed anywhere.

## 1. Result

| | Behaviour | Cause | Evidence |
| --- | --- | --- | --- |
| a | Kahan, Neumaier and two-sum compensation terms are removed | Metal fast math | The Metal source Tint generates, compiled outside the browser: compensation 0 with `fastMathEnabled = true`, the IEEE value bit for bit with `false` |
| b | `(value - shift) - small` is regrouped | Metal fast math | Same run: 16 of 16 residuals regrouped with fast math, 16 of 16 as written without |
| c | A local integer table reads back as zeros | A miscompile in the Apple GPU back end (not Tint, not the Metal front end, not fast math) | The generated Metal source is right; the front end's intermediate code is right; the result is wrong with fast math on and off; `#pragma clang loop unroll(disable)` on the fill loop cures it |
| d | Division is not correctly rounded; a loop sum is not the IEEE serial sum | Metal fast math | Same run: 551 of 2048 quotients one unit in the last place off with fast math, 0 without; the first-pass kernel's plain sum is the IEEE serial sum without |

Three things in the first-pass documents were wrong or too narrow for (c) and are corrected in
section 3.8: it is not about identity tables, `var<private>` at module scope is not a cure, and the
size window is wider than stated.

The browser offers no way out of fast math: the Dawn of Chromium 151 has no toggle for the Metal
math mode (section 4.2).

## 2. Environment

| | |
| --- | --- |
| Machine | MacBook Pro `Mac15,7`, Apple M3 Pro, 18 GPU cores, Metal 3 |
| Operating system | macOS 14.1 (23B2073) |
| GPU driver | `AGXMetalG15X_B0`, bundle version 276.56 |
| Metal front end | `Apple metal version 32023.26 (metalfe-32023.26)`, AIR 2.6.0, language Metal 3.1 (from the `llvm.ident` and `air.*` metadata of the compiled library) |
| Browser | Chrome for Testing 151.0.7922.34 (Playwright 1.62.1, revision `chromium-1234`), `channel: 'chromium'`, headless, flags `--enable-unsafe-webgpu --ignore-gpu-blocklist` (the launch options of `vitest.config.ts`) |
| User agent | `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/151.0.0.0 Safari/537.36` |
| WebGPU adapter | `GPUAdapterInfo`: vendor `apple`, architecture `metal-3`, device and description empty |
| Native runner | Swift 5.10 (Command Line Tools), `MTLDevice.makeLibrary(source:options:)` |

One machine, one operating-system version, one browser version. macOS 14.1 is two major versions
behind the current one; whether a later macOS still has (c) was not tested.

## 3. (c) The lost table fill

### 3.1 Reduced shader

Plain WebGPU, one storage binding and one uniform. `table[37]` is 38 by the language; the test GPU
returns 0.

```wgsl
struct Parameters { index: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@compute @workgroup_size(1)
fn main() {
  var table: array<u32, 64>;
  for (var entry = 0u; entry < 64u; entry++) { table[entry] = entry + 1u; }
  output[0] = table[parameters.index];   // uniform holds 37: expected 38, actual 0
}
```

The ingredients, each necessary on this machine:

1. a local array of `u32` or `i32` (function scope, or `var<private>`, which Tint also emits as a
   local of the entry point) of 33 to about 250 entries;
2. the zero initialisation WGSL requires (Tint emits `= {}`), or any earlier complete write;
3. a loop with a constant trip count of 3 to 96 that stores to the entry at the loop index (any
   value: the index, a constant, a sentinel, a value read from a buffer);
4. no store to the table at an index computed at run time;
5. a read the compiler cannot resolve at compile time (an index or a loop bound from a uniform or a
   buffer, or a loop nest it does not fold).

The stores of the loop are then lost and the read returns what the table held before the loop.

### 3.2 What Tint generates: correct Metal source

Obtained with `--enable-dawn-features=dump_shaders`, which prints the WGSL and the generated MSL to
the page console. The entry point for the shader above (the `tint_array` wrapper is a struct around
a C array with `operator[]`):

```cpp
kernel void dawn_entry_point_…(device tint_array<uint, 1>* v_1 [[buffer(0)]],
    const constant tint_struct* v_2 [[buffer(1)]], const constant tint_struct_1* v_3 [[buffer(30)]]) {
  tint_struct_2 const v_4 = tint_struct_2{.tint_member_2=v_1, .tint_member_3=v_2, .tint_member_4=v_3};
  tint_array<uint, 64> v_5 = {};
  {
    uint v_6 = 0u;
    while(true) {
      if ((v_6 < 64u)) {
      } else {
        break;
      }
      v_5[v_6] = (v_6 + 1u);
      {
        v_6 = (v_6 + 1u);
      }
    }
  }
  uint const v_7 = (tint_struct_3{.tint_member_5=(((*v_4.tint_member_4).tint_member_1[0u].x +  tint_volatile_zero) / 4u)}.tint_member_5 - 1u);
  device uint* const v_8 = (&(*v_4.tint_member_2)[min(uint(0), v_7)]);
  (*v_8) = v_5[min((*v_4.tint_member_3).tint_member, 63u)];
}
```

Zero initialisation, the loop, the robustness clamp `min(index, 63u)` and the read are all as they
should be. Tint did not transform the loop. A `var<private>` table comes out as
`thread tint_array<uint, 64> v_4 = {};` at the top of the entry point: the same shape.

### 3.3 What the Metal front end generates: correct intermediate code

The dumped MSL, compiled by a small Swift program with `MTLDevice.makeLibrary(source:options:)` and
run once, gives the same wrong answer outside the browser, so Chromium, Dawn and the WebGPU command
stream are not involved. The front end's output (AIR, LLVM bitcode) was taken from an
`MTLBinaryArchive` of the pipeline and disassembled with `clang -x ir -S -emit-llvm`. For the
hand-written Metal shader of section 5.2:

```llvm
entry:
  %table = alloca [64 x i32], align 4
  %0 = bitcast [64 x i32]* %table to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(256) %0, i8 0, i64 256, i1 false)
  br label %for.body
for.cond.cleanup:
  %1 = load i32, i32 addrspace(2)* %index, align 4
  %2 = tail call i32 @air.min.u.i32(i32 %1, i32 63)
  %idxprom2 = zext i32 %2 to i64
  %arrayidx3 = getelementptr inbounds [64 x i32], [64 x i32]* %table, i64 0, i64 %idxprom2
  %3 = load i32, i32* %arrayidx3, align 4
  store i32 %3, i32 addrspace(1)* %output, align 4
  ret void
for.body:
  %entry1.09 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %add = add nuw nsw i32 %entry1.09, 1
  %arrayidx = getelementptr inbounds [64 x i32], [64 x i32]* %table, i64 0, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %exitcond.not = icmp eq i32 %add, 64
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !31
```

A `memset` to zero, a scalar loop of 64 stores, one load. Correct, and not unrolled or vectorised
at this stage.

### 3.4 Where it goes wrong: the Apple GPU back end, when it unrolls the fill loop

The wrong result therefore arises between AIR and the machine code of the GPU (the `applegpu` slice
of the archive, produced by the `AGXMetalG15X_B0` driver's compiler at pipeline creation). That
code was not disassembled (no tool for it on the machine). What narrows it:

| Change to the hand-written Metal shader (64 entries, read at index 37) | Result |
| --- | --- |
| as written | 0 (wrong) |
| `fastMathEnabled = false` | 0 (wrong): not fast math |
| language version 2.3, 2.4, 3.0, 3.1 (tried on the dumped MSL of the first-pass shape) | zeros (wrong) |
| `#pragma clang loop unroll(disable)` on the fill loop | 38 (right). The AIR differs from the wrong one only by the loop metadata `llvm.loop.unroll.disable`, which only the back end can be acting on |
| `#pragma clang loop vectorize(disable)`, `interleave(disable)`, `vectorize_width(2)` | 0 (wrong) |
| no zero initialisation (`uint table[64];`) | 38 (right) |
| zero initialisation by a loop instead of `= {}` | 0 (wrong) |
| initialised to 5 by a loop, then the fill loop | 5: the earlier write wins, the later loop is lost |
| the 64 stores written out as 64 statements | 38 (right) |
| initializer list `{1u, 2u, ...}` | 38 (right) |
| no clamp on the read index | 0 (wrong): not the robustness clamp |
| stored value `entry + 1u + parameters->index` (not a constant) | lost as well |
| `float`, `half`, `bool`, `uchar`, `ushort`, `ulong`, `uint2`, `uint3`, `uint4`, `int2`, `float2` entries | right at 36, 64 and 96 entries |
| `int` entries | lost, as `uint` |
| a store at a run-time index after the loop, also inside a branch that is never taken | right |
| a store at a constant index after the loop | 0 (wrong) |

Narrowest statement: the Apple GPU back end loses the stores of a fully unrolled constant-count
loop into a stack array of 32-bit integers when the array was completely written just before (the
zero initialisation) and is between 132 and about 1008 bytes. A conjecture that fits everything
above and was not verified: the zero `memset` of that size class and the merged stores of the
unrolled loop are both expanded to 32-bit integer vector stores, and a later store-merging or
dead-store step keeps the wrong one of the two; other element types are not merged with the zero
fill and so survive.

A side observation from the same runs: a hand-written Metal shader with a local array of
`struct { uint; uint; }` filled by such a loop makes pipeline creation fail with "Compiler
encountered an internal error" (`AGXMetalG15X_B0`, code 3) at 24, 32, 48, 50, 64 and 96 entries and
not at 8, 16, 17, 33, 49, 51, 97 or 128. The WGSL equivalent through Tint compiles and reads right
at all of those sizes, so WebGPU callers do not meet it.

### 3.5 One thing at a time (WGSL, through Chromium)

Base shape: `var table: array<u32, N>` at function scope; identity loop with a constant bound; the
table read by the inner index of a two-deep loop nest (`for row { for column <= row { last =
table[column]; } output[row] = last + 1000u; }`); no other write. Sizes tried: 8, 16, 24, 32, 33,
34, 35, 36, 40, 48, 56, 64, 72, 80, 96, 97, 100, 112, 128, 200. "Lost" below means the same sizes
as the base (33, 34, 36, 40, 48, 56, 64, 72, 80, 96) unless sizes are given; "right" means right at
all twenty sizes.

| Varied | Variant | Result |
| --- | --- | --- |
| (base) | | lost at 33, 34 (the first 32 entries), 36 to 96 (all entries); right at 8 to 32, 35, 97 and above |
| Element type | `i32` | lost |
| | `f32` (`f32(i)`) | right |
| | `vec2<u32>` | right |
| Fill | constant loop (`table[i] = 7u`) | lost |
| | `i + 1u`; `3u * i + 5u`; `N - 1 - i`; `i * i` | lost |
| | identity loop counting down | lost at 34 to 96, in different partial patterns |
| | `i + zero` with the zero from a uniform | lost at 36 to 96 |
| | values copied from a storage buffer (`table[i] = u32(source[i])`, also at a run-time base), read by one loop with a uniform bound | lost at 36, 48, 64, 96 of 16, 32, 33, 36, 48, 64, 96, 128, 200 |
| | read-modify-write in the loop (`table[i] \|= value`), same read and sizes | lost at 36, 48, 64, 96 |
| | a counter-indexed store in a constant-count loop (`if (keep) { table[kept] = value; kept++; }`), same read and sizes | right |
| | the loop in a helper function taking `ptr<function, array<u32, N>>` | lost |
| | literal initializer (`var table = array<u32, N>(0u, 1u, ...)`) | right |
| | module-scope `const table = array<u32, N>(...)` | right |
| | `var<private>` at module scope, identity loop | **lost**, same sizes as function scope |
| Fill loop bound | a uniform (`i < parameters.count`) | right |
| | `min(parameters.count, N)` | right |
| Read | one loop with a constant bound (copy loop), also reversed | right |
| | constant indices (1, N / 2, N - 1) | right |
| | indices from a uniform | lost |
| | one loop with a uniform bound | lost |
| | nest, the outer index reads, inner loop does not touch the table | lost at 40 to 96 |
| | nest depth 2 (base) and depth 3, inner index reads | lost |
| | nest with uniform bounds | lost |
| Write after the fill | `table[3] = 3u` (constant index) | lost |
| | `table[parameters.index] = value` (index from a uniform) | right |
| | swap of two entries at uniform indices | right |
| | a second, unrelated array filled after the table | lost (no change) |

The first-pass shape (`order` and `groups`, two rounds, `continue`) loses the fill at 40 to 96 and
the first 32 entries at 33, as the first pass measured for 40 to 96.

### 3.6 Sizes

Full fill (trip count equal to the array length), every length from 1 to 140 and 160, 192, 200,
256, 512, in WGSL through Chromium and in hand-written Metal, with the same result:

- right up to 32 entries and from 97;
- 33 to 96: all entries lost at every multiple of 4; none lost at 35, 39, ... 95; at the other
  lengths a leading block is lost and then one or two of every four entries (33: the first 32;
  37: the first 28 and then every fourth).

Partial fill (hand-written Metal; rows are the array length, columns the trip count of the loop;
`L` lost, `.` right; the WGSL grid through Chromium agrees except that 252 entries is right there):

| Array \ trip | 1 | 2 | 3 | 4 | 5 | 8 | 16 | 17 | 32 | 64 | 96 | 97 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 to 32 | . | . | . | . | . | . | . | . | . | | | |
| 33 to 252 | . | . | L | L | . | L | L | . | L | L | L | . |
| 253 and above | . | . | . | . | . | . | . | . | . | . | . | . |

Trip counts of 6, 7, 10, 11, 18, 19, 20 and 95 are lost as well and 9 is right (tried at 128 to
252 entries). The middle row is not uniform at its edges: 35 entries with a 32-iteration loop is
right, and 129 or 193 entries with a 96-iteration loop lose 72 of the 96.

So a four-iteration loop that sets the first four entries of a 64-entry table is lost as well. The
window is the array's size (33 to about 250 entries of 4 bytes) together with a trip count the back
end unrolls completely (3 to 96 here, for a loop body of one store; trip counts of 5, 9 and 17 into
a longer array were right). The first pass measured 12 to 24 entries inside the Louvain kernel; the
reduced shader does not reproduce a window that low (two tables, a struct of two arrays and a large
neighbouring array were tried). The library audit (section 9.2) ran that kernel again: its table
is lost from 12 entries up, at every size tried to 48, and from 33 only when the helper that
declares two float arrays is not called. So the lower limit of 33 belongs to the reduced shader,
not to the fault.

### 3.7 Dawn toggles

The toggle names were read from the strings of the Chromium binary. With the base shape:

| Toggle (`--enable-dawn-features=` unless noted) | Result |
| --- | --- |
| `disable_robustness` | the base and every loop-index read become right; a read at a uniform index, a loop with a uniform bound and the outer-index nest (48 to 96) are still lost. Without the clamp the back end resolves those reads at compile time; the fault is hidden, not removed |
| `disable_workgroup_init`, `disable_symbol_renaming`, `use_user_defined_labels_in_backend`, `metal_use_argument_buffers`, `enable_integer_range_analysis_in_robustness`, `disable_polyfills_on_integer_div_and_mod`, `decompose_uniform_buffers`, `skip_validation` | no change |
| disabled: `metal_fix_u32_div_mod`, `metal_use_argument_buffers`, `enable_integer_range_analysis_in_robustness`, `lazy_clear_resource_on_first_use` | no change |

No toggle is available to a web page in any case.

### 3.8 What this corrects in the first-pass documents

- "written only by an identity loop": any stored value is lost, including a sentinel fill
  (`0xffffffff`) and values copied from a buffer.
- "Declare lookup and order tables at module scope as `var<private>`": not a cure. Tint emits a
  private variable as a zero-initialised local of the entry point and the reduced shader loses it at
  the same sizes. It was right in the first-pass shape, which is evidence for that shader only.
- "40 to 96 entries": the array is 33 to about 250 entries and the loop 3 to 96 iterations.
- "read by a loop index inside a loop nest": any read the compiler cannot resolve, a single read at
  an index from a uniform included.
- `f32` tables being right is confirmed (also `vec2<u32>`); a store at a run-time index after the
  fill ("written again by data") being right is confirmed.

### 3.9 The rule for authors, and kernels to look at again

A local `array<u32, N>` or `array<i32, N>` must not get its contents only from constant-count
loops at the loop index, at any size (33 entries is the lower limit of the reduced shader; the old
Louvain kernel lost a 12-entry table, section 9.2). Reliable by mechanism: a literal initializer or a
module-scope `const` (there is no second write), or a fill loop whose bound the compiler cannot
know (it is not unrolled). Observed right, mechanism not known: a store at a run-time index
somewhere after the fill, and float or vector entries.

The first pass searched `modules/analysis/src` for identity fills only. A text screen for the wider
shape (a local or private `u32`/`i32` array whose every store is at the index of a loop with a
literal, capital-case or interpolated bound, and stores something other than zero) finds these
with a length that can fall in the window. None was run in this section; their loop bodies are
larger than the reduced one. Section 9 (the library audit) ran all eight and the rest of the
library: none loses its table.

- `gpu-network/network-optimization/gpu-network-rational-access.ts:198` (`minima`, `maxima`)
- `gpu-network/network-optimization/gpu-network-louvain.ts:215` (`groups`)
- `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:1425` (`caseSums`)
- `gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts:250` (`assignments`)
- `gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics.ts:294` (`pooledRow`)
- `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts:195` (`nearestSample`)
- `raster-analysis/remote-sensing/gpu-raster-temporal.ts:181` (`dates`)
- `gpu-network/map-matching/gpu-map-matching.ts:1825` (`newBack`)

The screen is by regular expression over one-line WGSL and will have missed tables whose index is
an expression of the loop variable.

## 4. (a), (b) and (d): one cause, Metal fast math

### 4.1 Measurement

One WGSL shader (in `modules/analysis/test/analysis-core/shader-compiler-fast-math.spec.ts`): a
million terms summed plainly, by Kahan and by Neumaier; the two-sum of 2^24 + 1; sixteen residuals
`(value - shift) - small` with `shift = 10000` and `small` three quarters of the float32 step there,
as one expression and as two statements; 2048 quotients. It was run in Chromium, and its dumped MSL
was compiled natively with `MTLCompileOptions.fastMathEnabled` true and false. Expected values are
IEEE float32 arithmetic with `Math.fround`.

| | Chromium 151 | Native, fast math on | Native, fast math off | IEEE float32 |
| --- | --- | --- | --- | --- |
| Kahan compensation at the end | 0 | 0 | 14.662109375 | 14.662109375 |
| Kahan sum | 1,230,312,832 (the plain sum) | 1,230,312,832 | 1,230,341,248 | 1,230,341,248 |
| Neumaier compensation | 0 | 0 | 28,401.392578125 | 28,401.392578125 |
| Two-sum error of 2^24 + 1 | 0 | 0 | 1 | 1 |
| Residuals as written / regrouped, one expression | 0 / 16 | 0 / 16 | 16 / 0 | 16 / 0 |
| Residuals as written / regrouped, two statements | 0 / 16 | 0 / 16 | 16 / 0 | 16 / 0 |
| Quotients not correctly rounded, of 2048 | 551, each by 1 ULP | 551 | 0 | 0 |

The true sum is 1,230,341,233.34. Chromium and the native fast-math run agree in all 2086 words of
the output (the run without fast math differs in 587), so Chromium compiles with fast math on this
operating system.

The loop sum of (d): in this shader the plain sum is the IEEE serial sum in all three runs. The
first-pass kernel (`test/utils/addition-rounding-error-wgsl.spec.ts`, rebuilt on plain WebGPU from
the same WGSL modules) reproduces its numbers, and without fast math its plain sum is the IEEE
serial sum and its float Neumaier correction equals the integer one bit for bit:

| | Chromium 151 | Native, fast math on | Native, fast math off |
| --- | --- | --- | --- |
| Plain sum | 1,234,025,472 | 1,234,025,472 | 1,234,033,152 (the IEEE serial sum) |
| Float Neumaier compensation | 0 | 0 | 8,435.263671875 (equal to the integer form) |

So (a), (b) and both halves of (d) are what fast math permits (reassociation, removal of
algebraically zero terms, approximate division), and (c) is not: it is the same with fast math off.

### 4.2 No way out through the browser

- The Dawn of Chromium 151 has no toggle for the Metal math mode: the binary's toggle list has
  nothing for it (the only strictness toggle is `d3d_disable_ieee_strictness`, for Direct3D).
- The binary contains the string `#pragma METAL fp math_mode(` with `safe` and `relaxed`, so this
  Dawn can emit a math-mode pragma; the MSL dumped on macOS 14.1 carries none. Prepending
  `#pragma METAL fp math_mode(safe)` by hand changes nothing here (the macOS 14.1 compiler ignores
  it). Whether Chromium on macOS 15 or later compiles in a stricter mode was not tested.
- The cures of the contract therefore stand: integer or bit operations the compiler cannot
  simplify.

## 5. Bug report drafts (not filed)

The defect is in Apple's compiler, so the report that can lead to a fix is the Apple one. The
Chromium report asks for a workaround in Tint and for the Dawn team to forward it.

### 5.1 Chromium / Dawn (issues.chromium.org, component Blink > WebGPU, or dawn.googlesource.com "tint")

**Title.** Metal: stores of a constant-count loop into a zero-initialised local `array<u32, N>` are
lost (Apple GPU back-end miscompile of correct MSL), Apple M3 Pro, macOS 14.1

**Reproduction.** Paste into the DevTools console of any `https:` page:

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const code = `
struct Parameters { index: u32 }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@compute @workgroup_size(1)
fn main() {
  var table: array<u32, 64>;
  for (var entry = 0u; entry < 64u; entry++) { table[entry] = entry + 1u; }
  output[0] = table[parameters.index];
}`;
const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {module: device.createShaderModule({code}), entryPoint: 'main'}
});
const output = device.createBuffer({size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
const staging = device.createBuffer({size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
const uniform = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
device.queue.writeBuffer(uniform, 0, new Uint32Array([37, 0, 0, 0]));
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    {binding: 0, resource: {buffer: output}},
    {binding: 1, resource: {buffer: uniform}}
  ]
});
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(1);
pass.end();
encoder.copyBufferToBuffer(output, 0, staging, 0, 4);
device.queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ);
console.log('table[37] =', new Uint32Array(staging.getMappedRange())[0], '(expected 38)');
```

**Expected.** `table[37] = 38`.

**Actual.** `table[37] = 0`. Every entry reads 0 for array lengths that are multiples of 4 from 36
to 96; lengths 33 to 96 other than 35, 39, ... 95 lose part of the table; up to 32 and from 97 it is
right. `array<i32, N>` behaves the same; `array<f32, N>` and `array<vec2<u32>, N>` are right. A
`var<private>` table behaves the same as a function-scope one.

**Analysis.** The MSL from `--enable-dawn-features=dump_shaders` is correct
(`tint_array<uint, 64> v_5 = {};`, a `while(true)` loop storing `v_6 + 1u`, a clamped read). The
same MSL compiled with `MTLDevice newLibraryWithSource:` in a standalone program returns the same
wrong value with `fastMathEnabled` on and off. The AIR extracted from an `MTLBinaryArchive` is
correct (a 256-byte `llvm.memset` and a scalar loop of 64 stores). Adding
`#pragma clang loop unroll(disable)` to the loop in the MSL gives the right value, and the AIR then
differs only by `llvm.loop.unroll.disable`. Removing the zero initialisation gives the right value.
So the Apple GPU back end loses the stores when it unrolls the loop after the zero fill.
`disable_robustness` hides it for reads by a loop index and not for a read at a uniform index.

**Possible workaround in Tint.** For a function-scope or private array of 32-bit integers above 32
entries, keep the back end from unrolling a loop that stores to it (for instance
`#pragma clang loop unroll(disable)` on loops that store to such an array), or emit the zero
initialisation in a form the back end does not merge with later stores.

**Versions.** Chrome for Testing 151.0.7922.34, headless, `--enable-unsafe-webgpu
--ignore-gpu-blocklist`; macOS 14.1 (23B2073); MacBook Pro Mac15,7, Apple M3 Pro; adapter vendor
`apple`, architecture `metal-3`; driver `AGXMetalG15X_B0` 276.56; Metal front end `metalfe-32023.26`.
Not tested on a later macOS or on another Apple GPU.

### 5.2 Apple (Feedback Assistant, Metal)

**Title.** Metal shader compiler (AGX back end, M3 Pro, macOS 14.1): stores of a constant-count
loop into a zero-initialised `uint` stack array are lost

**Reproduction.** `swiftc -O reproduction.swift && ./reproduction`:

```swift
import Foundation
import Metal

func run(_ label: String, _ loopPragma: String, fastMath: Bool) throws {
  let source = """
  #include <metal_stdlib>
  using namespace metal;
  struct Parameters { uint index; };
  kernel void reproduction(device uint* output [[buffer(0)]],
                           const constant Parameters* parameters [[buffer(1)]]) {
    uint table[64] = {};
    \(loopPragma)
    for (uint entry = 0u; entry < 64u; entry++) { table[entry] = entry + 1u; }
    output[0] = table[min(parameters->index, 63u)];
  }
  """
  let device = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices()[0]
  let options = MTLCompileOptions()
  options.fastMathEnabled = fastMath
  let library = try device.makeLibrary(source: source, options: options)
  let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "reproduction")!)
  let output = device.makeBuffer(length: 4, options: .storageModeShared)!
  var index: UInt32 = 37
  let parameters = device.makeBuffer(bytes: &index, length: 4, options: .storageModeShared)!
  let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(output, offset: 0, index: 0)
  encoder.setBuffer(parameters, offset: 0, index: 1)
  encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  let value = output.contents().load(as: UInt32.self)
  print("\(device.name), \(label), fastMathEnabled=\(fastMath): table[37] = \(value) (expected 38)")
}

try run("as written", "", fastMath: true)
try run("as written", "", fastMath: false)
try run("unroll disabled", "#pragma clang loop unroll(disable)", fastMath: false)
```

**Expected.** 38 three times.

**Actual.**

```
Apple M3 Pro, as written, fastMathEnabled=true: table[37] = 0 (expected 38)
Apple M3 Pro, as written, fastMathEnabled=false: table[37] = 0 (expected 38)
Apple M3 Pro, unroll disabled, fastMathEnabled=false: table[37] = 38 (expected 38)
```

**Notes for the report.** The AIR is correct (section 3.3) and the first and third variants differ
in AIR only by `llvm.loop.unroll.disable`. Affected: `uint` and `int` arrays of 33 to 252 entries
with a loop of 3 to 96 iterations (section 3.6); `float`, `half`, `bool`, 8-, 16- and 64-bit
integers and vectors are not. Without `= {}` the result is right; with an earlier loop that fills
the array with 5 the read returns 5. A local array of `struct { uint; uint; }` filled the same way
fails pipeline creation with "Compiler encountered an internal error" at 24, 32, 48, 50, 64 and 96
entries. Versions as in 5.1.

## 6. Specs

| Spec | What it pins | Tests |
| --- | --- | --- |
| `modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts` | The reduced shader on plain WebGPU: lost at 33, 36, 64 and 96 of the sizes 16, 32, 33, 35, 36, 64, 96, 97, 128, with the first 32 entries lost at 33 (skips on an adapter that keeps the fill); `var<private>`, `i32`, a constant, a sentinel, a helper function and a later constant-index store are lost at the same sizes as the reduced shader; a literal initializer, a module-scope `const`, a fill bounded by a uniform, a later store at a uniform index, `f32` and `vec2<u32>` entries are right at every size (asserted on every adapter) | 3 |
| `modules/analysis/test/analysis-core/shader-compiler-fast-math.spec.ts` | (a) compensation exactly 0 and both compensated sums equal to the plain sum; (b) 16 of 16 residuals equal to the regrouped value in both forms; (d) more than 100 of 2048 quotients off, each by exactly one unit in the last place. Each test skips on an adapter that computes the expression as written | 3 |
| `modules/analysis/test/analysis-core/shader-compiler-raw-webgpu.ts` | Helper: one compute shader on the raw `GPUDevice` of the shared test device | |

Run: `npx vitest run --project headless modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts modules/analysis/test/analysis-core/shader-compiler-fast-math.spec.ts`
(2 files, 6 tests passed, none skipped, on the machine of section 2).

## 7. How to repeat the investigation

- **Generated MSL.** Launch the Playwright Chromium (`chromium.launch({channel: 'chromium',
  headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
  '--enable-dawn-features=dump_shaders']})`), serve a blank page on an `https:` URL through
  `page.route`, run the shader with `navigator.gpu` and collect `page.on('console')`: the message
  that starts `/* Dumped generated MSL */` is the Metal source. `vitest.config.ts` was not edited.
- **Native run.** A Swift program as in section 5.2 that reads the MSL from a file, binds the
  output at buffer 0, a uniform or input at buffer 1 and Tint's buffer-size uniform at buffer 30
  (a `uint4` whose components are the byte lengths of the runtime-sized arrays).
- **AIR.** `device.makeBinaryArchive`, `addComputePipelineFunctions`, `serialize(to:)`; the file is
  a fat file with an `applegpu` Mach-O and an `air64` metallib. The bitcode starts at the wrapper
  magic `0x0B17C0DE` (offset and size are the third and fourth words of the wrapper); `clang -x ir
  -S -emit-llvm -Xclang -disable-llvm-passes` prints it. `xcrun metal` is not installed.
- **Toggle names.** `strings` on the Chromium framework binary, around `disable_robustness`.

## 8. Not verified

- The machine code of the GPU was not read, so the faulty pass is inferred from what cures the
  fault, not seen. The store-merging conjecture of section 3.4 is a conjecture.
- One machine, one macOS version (14.1), one Chromium version. No later macOS, no other Apple GPU,
  no Safari or Firefox, no non-Apple adapter.
- Why the old Louvain kernel loses a table from 12 entries and the reduced shader from 33 is not
  explained. It was measured again and narrowed to the calls of one helper (section 9.2).
- What the library audit did not verify is listed in section 9.6.
- Whether Chromium compiles with a stricter math mode on macOS 15 or later (section 4.2).
- The lag-ML, GM and 2SLS residual chains that the register lists as untested for (b) were not
  tested here either; the finding that (b) is fast math says they are exposed in principle.

## Resolution

### Closing pass (2026-11-11)

| Item | Outcome | Files | Spec | Evidence |
| --- | --- | --- | --- | --- |
| (c) reduced to a standalone shader | verified, no source change | new spec and helper only | `shader-compiler-table-fill.spec.ts` (3 tests) | Plain WebGPU in headless Chromium 151, and the same Metal source compiled natively; expectations are the WGSL meaning of the shader |
| (c) cause | found as far as the Apple GPU back end: correct MSL, correct AIR, wrong with fast math on and off, cured by `unroll(disable)` on the fill loop or by removing the zero initialisation | this report | same | Dumped MSL, disassembled AIR, native runs (sections 3.2 to 3.4). The GPU machine code was not read |
| (c) conditions in the documents | corrected: any stored value, 33 to about 250 entries, `var<private>` is not a cure | `dev-docs/contributor-contract.md` section 11, `dev-docs/upstream-workarounds.md` section 6, `dev-docs/upstream-proposal-math-review.md` section 13.3 (one or two sentences each) | same | Sections 3.5, 3.6, 3.8 |
| (a), (b), (d) under fast math | verified: one cause, Metal fast math; no browser toggle | this report; the same three documents | `shader-compiler-fast-math.spec.ts` (3 tests) | Native compile of the dumped MSL with `fastMathEnabled` true and false against IEEE float32 by `Math.fround` (section 4) |
| Bug report | drafted for Chromium/Dawn and for Apple, not filed | this report, section 5 | | The console snippet and the Swift program were both run as printed |
| Library kernels against the wider condition of (c) | verified, no change, for every kernel that holds such a table: none loses it (table audit stream; section 9) | new specs only | `test/analysis-core/table-audit/table-audit-strict.spec.ts` (8 tests), `table-audit-network.spec.ts` (5), `table-audit-raster.spec.ts` (4) | Each contributor run on the GPU at the table sizes its limits allow, against a closed form, a property of the construction, exact integer arithmetic or a brute-force reference written apart from the kernel |
| Seeded visit order of `GPUNetworkLouvain` (the one table the first pass moved to `var<private>`) | fixed by mechanism: a literal initializer replaces the identity loop; the order is unchanged. The replaced form was not measured wrong | `modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain.ts` | `gpu-network-louvain-visit-order.spec.ts` (the order at 13 sizes by 5 seeds against a TypeScript Fisher-Yates, before and after), `shader-compiler-table-shapes.spec.ts` (the kernel as it was before the first pass fails; a literal initializer cures it) | Section 9.4 |
| Size window and the exposed shapes | corrected: no size is safe (12 entries lost inside the old Louvain kernel); only a loop that stores nothing but local tables is exposed | this report, sections 3.6, 3.9 and 9; `dev-docs/contributor-contract.md` section 11, `dev-docs/upstream-workarounds.md` section 6, `dev-docs/upstream-proposal-math-review.md` section 13.3 (the passages rewritten, not appended to); comments in `gpu-network-louvain.ts`, `gpu-natural-neighbor.ts`, `gpu-raster-phenology-metrics.ts`; `test/utils/function-scope-table-readback.spec.ts` | `shader-compiler-table-shapes.spec.ts` (4 tests) | Plain WebGPU and the old Louvain kernel on the GPU; expectations are the WGSL meaning of each shader |

No caller-visible behaviour changed, so `behaviour-changes.md` has no row from this stream or from the table audit.

## 9. Library audit (2026-11-11)

Scope: every WGSL array of `u32` or `i32` in `modules/analysis/src` that is not a storage or
uniform binding, against the condition of section 3: contents that come from loops with a constant
trip count, read at an index the compiler cannot resolve. Done by the table audit stream after the
sections above were written.

### 9.1 How the candidates were found

A script listed every declaration `name: array<u32|i32, size>` in `modules/analysis/src` (330 in
123 files, struct members, function parameters and pointer parameters included) and, for each
name, every statement that stores to it with the loop header above it. The stores were then read
by eye. The WGSL is in TypeScript template strings, so a size is a literal, a WGSL constant or an
interpolated expression; each range below is the limit the constructor enforces.

The declarations fall into four groups:

| Group | Count | What it is | Treatment |
| --- | --- | --- | --- |
| `var<workgroup>` arrays | 149 | Shared scratch of cooperative kernels | Not local memory. The reduced shader with a workgroup table is right at every size (section 9.2) |
| A. Every store is at the index of a constant-count loop, or at a constant index | table A | The condition of section 3 in full | Each run through its contributor (9.3, table A) |
| B. A constant-count fill, then stores at data-dependent indices | table B | Sentinel or zero fill followed by an insertion, a relaxation, a hash insert or a sort | Each run through its contributor, or covered by its existing GPU specs (9.3, table B) |
| C. No constant-count fill | the rest | Stacks, counter-indexed lists (`table[count] = value; count++`), insertion lists without a fill, loops bounded by a run-time count, literal initializers and `const` tables | Not candidates. A counter-indexed store in a constant-count loop is right in the reduced shader (section 3.5) |

Limits of the screen: it is textual. A table whose declaration is assembled from pieces that do
not contain `array<u32,` or `array<i32,` on one line would be missed (a search for an interpolated
element type, a type alias, a nested integer array and a declaration split over lines found
none). The directories that other streams were editing (`spatial-regression`,
`polygon-coverage-topology`, `ring-assembly`, `polygon-triangulation`, `order-statistic-wgsl.ts`,
`gpu-column-quantiles.ts`, `gpu-space-time-diggle-test.ts`) were screened and hold no candidate:
their local integer arrays are literals, `const` tables, a counter-indexed list
(`gpu-restricted-regime-regression.ts:594`, `selected`) and a nine-entry list filled at a counter
(`gpu-segment-ring-assembly.ts`, `seen`).

### 9.2 What the reduced shaders add to section 3

`modules/analysis/test/analysis-core/shader-compiler-table-shapes.spec.ts`, plain WebGPU, sizes 8,
12, 16, 20, 24, 32, 33, 36, 40, 48, 64, 96, 128, 200, 250. The table is read through a loop whose
bound comes from a uniform unless noted.

| Shape | Result |
| --- | --- |
| Counted fill, nothing else | lost at 33, 36, 40, 48, 64, 96 |
| A `u32` sentinel and an `f32` fill in one loop, no later store | lost at 33, 36, 40, 48 |
| Values copied from a storage buffer by a counted loop | lost at 36, 40, 48, 64, 96 |
| `var<workgroup>` table, counted fill | right |
| `var<private>` table with a literal initializer | right |
| Member of a local struct, counted fill | right |
| Table returned from a function that fills it by a counted loop | right |
| Fill bounded by `N * min(arrayLength(&binding), 1u)`, or by `min(N, arrayLength(&binding))` | right |
| Fill loop that also writes storage, table read in a loop nest | right |
| Store at the loop index under a data condition | right |
| Sums accumulated by a counted loop inside a run-time loop | right |
| Identity fill, then a compare-exchange network at loop indices | right |
| Sentinel fill, then insertion at a data position | right |
| `i32` sentinel fill, then stores at data indices | right |
| Limbs copied between regions of one workspace through a pointer, constant limb counts | right |
| Bit mask set by a counted loop | right |

So the exposed shape is narrower than the condition of section 3 reads: a loop that stores
nothing but local tables (a constant, the index, a value read from a buffer), with no store at a
data-dependent index anywhere after it. The "right" rows are observations, not guarantees, except
the literal initializer and the run-time bound, which leave no unrolled loop.

The size window is not narrower, it is wider. The Louvain kernel as it was before the first pass
(`git show HEAD:modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain.ts`,
ascending visits, `var order: array<u32, N>` filled by `order[vertex] = vertex`), rebuilt in the
same spec with the table read out through an extra binding where the sweep uses it:

| Variant of the old kernel | Sizes 4, 8, 10, 12, 16, 24, 32, 40 |
| --- | --- |
| As it was | lost at 12, 16, 24, 32, 40 (all zeros); right at 4, 8, 10. Also lost at 14, 18, 20, 22, 26, 28, 30, 36, 48 in an earlier run |
| Calls to `objective` removed (the helper declares two `array<f32, N>`) | lost at 40 only: the window of the reduced shader |
| Candidate loop removed (which also removes the call inside the sweep) | right |
| `order` with a literal initializer | right |
| Fill loop bounded by `NODE_COUNT * min(arrayLength(&weights), 1u)` | right |
| `var<private> order` with the same counted fill | right in this kernel (lost in the reduced shader, section 3.5) |

A 12-entry table is 48 bytes, far below the 132 bytes of section 3.4. The float arrays of the
inlined helper, zero-initialised inside the sweep, move the limit; how is not known. The first
pass reported 12 to 24 entries; this run finds no upper end up to 48. It also shows where the
first pass's belief in `var<private>` came from: a private table does read right in that kernel.

### 9.3 Candidates and measured results

Every row was run on the machine of section 2. "Sizes" are table sizes. A wrong result at any
size would be listed; there is none.

Table A: every store at the index of a constant-count loop.

| Site | Table | Size and range | Stores | Read | Run | Result | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gpu-network/network-optimization/gpu-network-louvain.ts:222` | `groups: u32` | `nodeCount`, 1 to 32 | identity fill (the loop also writes `communities`); copy from `communities` at the end of a level | loop indices in the sweep nest | `GPUNetworkLouvain`, 2 to 32 vertices by no seed, seeds 0 and 7: disjoint cliques (the answer is the cliques; modularity by Newman's formula), and for even sizes from 12 a ring of pairs (a correct run must join pairs at the second level, by the closed-form gain); 12, 16, 24, 32 vertices by 2, 8, 32 levels by 4, 16, 64 sweeps by directed or not by seed or not | right | none (comment added) |
| same file, `:62` | `louvainVisitOrder: u32`, `var<private>` (the first-pass conversion) | `nodeCount`, 1 to 32 | was: identity loop, then Fisher-Yates swaps at drawn indices | loop index in the sweep nest | the runs above; `gpu-network-louvain-visit-order.spec.ts` (2 to 512 entries by 5 seeds, the whole order against a TypeScript Fisher-Yates) | right before and after | literal initializer in place of the identity loop (9.4) |
| `gpu-network/network-optimization/gpu-network-rational-access.ts:198` | `minima`, `maxima: u32` | demand rows, 1 to 64 | conditional store at the origin index in the cycle loop | origin index in a later loop | `GPUNetworkRationalAccess`, 4, 8, 12, 16, 20, 24, 28, 32, 33, 36, 40, 48, 64 origins, two facilities, nearer facility 0 or 1: one unit per origin must move and the run must converge (hand computation) | right | none |
| `gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics.ts:294` | `pooledRow: u32` | `classCount`, 1 to 64 | store at the class index | class index in a nest | `GPUTransitionDiagnostics`, the same 13 sizes: disjoint 2 by 2 tables, Q is the table total and LR `2 (a ln((a + b) / a) + b ln((a + b) / b))` per from-state | right | none |
| `gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts:250` | `assignments: u32` | design rows, 2 to 256 | sentinel at the row index, then the chosen cluster at the row index | row index | `GPUSpatialCrossValidation`, environmental design, 4 to 256 rows (17 sizes): two separated groups must be the two folds | right | none |
| `gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.ts:263` | `work: u32` (and `negatives`, 6) | 140, fixed | limb copies, sums, differences and products over constant limb counts at constant offsets | limb indices of constant loops | `incircleExact` on 512 quadruples on or within a few float32 steps of a common circle at four scales, against the determinant in BigInt; and the class's GPU specs (`gpu-delaunay-tessellation.spec.ts`, `delaunay-property.spec.ts`, `delaunay-mixed-scale.spec.ts`) | right | none |
| `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:1425` | `caseSums: u32` | `timeBuckets`, 1 to 32 | accumulated at the bucket index inside a run-time loop | bucket index in a nest | `GPUSpatialScanStatistic`, 8, 12, 16, 20, 24, 28, 32 buckets: a planted block must be the most likely cluster, with Kulldorff's likelihood ratio in closed form | right | none |
| `gpu-spatial-analysis/label-point/gpu-label-point.ts:443` (the cooperative path) | `topCell: u32` | `refinementCandidates`, 1 to 16 | sentinel fill, then conditional stores at the slot index | slot 0 and slot indices | `GPULabelPoint`, 1, 2, 4, 8, 10, 12, 13, 14, 15, 16 candidates, a comb whose pole of inaccessibility is (13, 10) at distance 10 and whose scanline seeds are elsewhere; a feature of 8 vertices (serial path, which keeps its candidates in a `vec3<f32>` table) and one of 48 (cooperative path) | right | none |
| `analysis-core/kernels/dense-solve-wgsl.ts:294` | `pivots: u32`, member of a struct | capacity | identity at the row index; the pivot row at the column index (bound is the size argument, a constant in some callers) | row index | `function-scope-table-readback.spec.ts`: capacities 8 to 64, with and without exchanges, in a kernel of the spec (not through a contributor) | right | none |
| `analysis-core/kernels/exact-lattice-wgsl.ts:152` | `terms: u32` | 12, fixed | two copies of six limbs | limb indices | `snap-rounded-noding.spec.ts` (the class that uses it, against its reference) | right | none |
| Limb, digit and flag tables of 4 to 8 entries: `exact-orientation-wgsl.ts`, `polygon-overlap-metric.ts`, `overlay-face-labelling-wgsl.ts`, `gpu-geomorphons.ts` (`digits`), `gpu-terrain-spike-repair.ts` (`runBits`, `var<private>`), `isobands-wgsl.ts` (`bandStates`, `var<private>`), `gpu-raster-skeleton.ts` (`neighborhood`), `gpu-map-matching.ts` (`pendingTargets`, `newBack`, at most 8) | `u32`, `i32` | 4 to 8 | computed values at the index of a constant loop | constant loops | their existing GPU specs: `gpu-geomorphons.spec.ts`, `gpu-terrain-spike-repair.spec.ts`, `spike-repair-component-labels.spec.ts`, `gpu-isobands.spec.ts`, `polygon-overlap-metric.spec.ts`, `polygon-overlay-snap-rounding.spec.ts`, `gpu-map-matching.spec.ts` | right. Below the smallest size ever measured lost (12). `gpu-raster-skeleton.ts` has no GPU spec of its own that was run | none |

Table B: a constant-count fill, then stores at data-dependent indices.

| Site | Table | Size and range | Run | Result |
| --- | --- | --- | --- | --- |
| `gpu-network/network-optimization/gpu-network-flow.ts:219` | `distances: i32`, `predecessor: u32` | `nodeCount`, 2 to 128 | `GPUNetworkFlow`, 16, 32, 33, 36, 40, 48, 64, 96, 128 nodes: a chain in reverse edge order plus one direct edge; flow, cost, augmentations and cut by construction | right |
| `gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts:132` | `connected: u32` | `vertexCount`, 2 to 256 | resistance connectivity recipe, 16 to 256 vertices (11 sizes): a chain of unit resistors; potentials, currents and effective resistance in closed form | right |
| `gpu-spatial-analysis/urban-morphology/visibility-graph.ts:302` | `distances: i32` | points, 1 to 256 | visibility graph recipe, 16 to 256 points (11 sizes): a path; depth sums, maximum depth and betweenness in closed form | right |
| `gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan.ts:736` | `order: u32` (and `ordered`) | neighbours, at most 64 | `GPUExpectationBasedSpaceTimeScan`, 16, 32, 33, 36, 40, 48, 64 neighbours: five planted zones must be the cluster, in count order | right |
| `gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts:278` | `neighbors`, `remaining: u32` | NNDM rows, at most 64 | `GPUSpatialCrossValidation`, NNDM, the same 7 sizes, two settings whose radii are brute-force distances | right |
| `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts:195` | `nearestSample`, `votes: u32` | `neighborCount` at most 16, `classCount` at most 32 | k of 1, 8, 16, both metrics, against a brute-force sort; 17 is refused | right. The window of the reduced shader cannot be reached |
| `raster-analysis/remote-sensing/gpu-raster-temporal.ts:181` | `dates: u32` | `sliceCount`, 1 to 256 | minimum, maximum and median composites, 16 to 256 slices (11 sizes), against `Array.prototype.sort` | right |
| `pointcloud-analysis/gpu-pointcloud-neighborhoods.ts:206` | `nearestIds: u32` | `capacity`, 1 to 64 | serial and cooperative, 16, 32, 33, 36, 40, 48, 64, against a brute-force sort | right |
| `raster-analysis/raster-time-series/gpu-raster-bfast-monitor.ts:240` | `slicesUsed: u32` (counter-indexed) | slices, at most 256 | 16 to 256 slices (11 sizes): the break must be reported at its original slice | right |
| `gpu-network/map-matching/gpu-map-matching.ts:1438` | `tableHash: u32`, `var<private>`, cleared by a constant loop before every search | hash size from `routeNodeBudget`, 8 to 128 | existing specs: `gpu-map-matching.spec.ts`, `gpu-map-matching-route-budget.spec.ts`, `gpu-map-matching-robustness.spec.ts`, `parity-wp3-map-matching.parity.spec.ts` | right |
| `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:1291` | `remaining: u32` | `timeBuckets`, 1 to 32 | `gpu-spatial-scan-statistic.spec.ts` (replicate maxima against its oracle) | right |
| `scan-statistic-nearest-zones.ts:63` (`histogram`, 256), `gpu-azp-regions.ts:821` (`localSlots`, 256; `localVisited`, 4) | `u32` | fixed | `gpu-spatial-scan-statistic.spec.ts`, `gpu-azp-regions.spec.ts` | right. 256 entries is above the window of the reduced shader |
| `gpu-raster-cold-monitor.ts:326` (`includedMask`, `var<private>`), `gpu-raster-hants.ts:245` (`keptMask`) | `u32` bit masks | at most 8 words | not run in this pass | not run: 8 words, and the bit-mask shape is right in the reduced shader |

The eight candidates of section 3.9 are rows 3, 1, 7, 5 and 4 of table A, rows 6 and 7 of table
B, and `newBack` in the last row of table A.

### 9.4 The one source change

`getGPUNetworkLouvainVisitOrderSource` declared `var<private> louvainVisitOrder: array<u32, N>;`
and filled it with `louvainVisitOrder[vertex] = vertex` before the shuffle. It now declares
`var<private> louvainVisitOrder: array<u32, N> = array<u32, N>(0u, 1u, ...);` and the loop is
gone. The replaced form read right (it is a row of table A), for two reasons that are not
understood: `var<private>` reads right in that kernel, and the shuffle stores at drawn indices.
The literal initializer is right by mechanism and is right in the reduced shader and in the old
kernel. The order is the same: `gpu-network-louvain-visit-order.spec.ts` compares all of it with a
TypeScript Fisher-Yates over the same Philox draws at 13 sizes by 5 seeds, and passed before and
after the change, as did `gpu-network-optimization.spec.ts` and the Louvain runs of table A.

Only the seeded kernel's WGSL changed. The lens record
`docs/lens-records/gpu-network.network-louvain.json` holds the unseeded kernel (it has no
`louvainVisitOrder`), so it is not affected.

Nothing else was changed in source: no other site loses its table, and the rule of this pass is
to leave what does not reproduce.

### 9.5 Specs and commands

| Spec | Tests |
| --- | --- |
| `modules/analysis/test/analysis-core/shader-compiler-table-shapes.spec.ts` | 4 |
| `modules/analysis/test/analysis-core/table-audit/table-audit-strict.spec.ts` | 8 |
| `modules/analysis/test/analysis-core/table-audit/table-audit-network.spec.ts` | 5 |
| `modules/analysis/test/analysis-core/table-audit/table-audit-raster.spec.ts` | 4 |

`npx vitest run --project headless modules/analysis/test/analysis-core/table-audit
modules/analysis/test/analysis-core/shader-compiler-table-shapes.spec.ts
modules/analysis/test/analysis-core/shader-compiler-table-fill.spec.ts
modules/analysis/test/utils/function-scope-table-readback.spec.ts
modules/analysis/test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts
modules/analysis/test/gpu-network/network-optimization/gpu-network-optimization.spec.ts`:
8 files, 45 tests passed, none skipped.

Existing specs run as evidence for the rows that name them: 15 files, 246 tests passed (the
Delaunay, snap-rounded noding, geomorphons, spike repair, map matching, isobands, polygon overlap,
SMRF, H3 boundary, polygon overlay, visit order and table readback specs), and 6 files, 122 tests
passed (map-matching route budget, robustness and parity, the scan statistic, AZP regions and the
table readback spec after its edit).

### 9.6 Not verified

- No test was made to fail on purpose by zeroing a table in a kernel. That a zeroed table gives a
  different answer is argued in each test's comment, and shown by a run only for Louvain (the old
  kernel fails the same clique check).
- A table that reads right is evidence for the kernels that were compiled: the sizes listed, the
  default props otherwise. A different combination of props is a different shader.
- The dense LU pivot table was run in a kernel of the spec, not through each contributor that
  solves with it.
- The swap behaviour of NNDM has no independent reference here; the two settings that were run
  have radii that follow from distances alone. The `training` and `testing` masks of that design
  were not checked.
- The k-nearest-neighbour classifier skips pixels whose k-th and next distances differ by less
  than 1e-4.
- The two bit masks of table B and `gpu-raster-skeleton.ts` were not run.
- Why the old Louvain kernel loses a 12-entry table is not known (9.2).
- One machine, one macOS version, one Chromium version, as in section 8.
