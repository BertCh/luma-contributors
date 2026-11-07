# County population classification verifier

This lane adds a CPU oracle and a serialized-slot browser check for `county-population-classification`. It reads the bundled `ncovr-new-jersey` manifest and `population1990.bin`; the 21 float32 values are exact integral counts and are converted to uint32 by the scene. It does not alter scene or runtime source.

## Independent reference

`reference-county-classification.py` enumerates every contiguous five-class split of the 21 sorted observed values (`20 choose 4 = 4,845`) and computes within-class SSE with `math.fsum`. The optimum is **27,607,880,702**, with edges `[65,294, 224,327, 395,066, 493,819, 671,780, 825,380]` and class frequencies `[6, 4, 4, 4, 3]`. There is one minimum partition for this data, but the browser check verifies the resulting SSE and assignment-to-edge consistency rather than relying on a fragile floating-point tie identity.

The independent approximate reference mirrors the current 128-bin path: each float32 value is assigned to an equal-width range bin and one of 256 fixed-point subpositions within that bin. Equal quantized positions are weighted observations in the coarse dynamic program. The position resolution is `range / (128 × 256)`; equivalently, each occupied subposition is represented by its cell center (`position + 0.5`). The GPU cost omits that same half-cell translation from every value, which does not change any partition SSE. The coarse DP is then followed by single-cut coordinate refinement on the original integer counts. All 21 values occupy distinct positions here. Its final cuts equal the exhaustive optimum for this dataset. That is a measured fixture result, not a general guarantee of approximate Jenks.

The palette is the explicit light-ground, five-class ColorBrewer YlOrBr table from the application source. Packing follows its documented RGBA8 layout: R is the low byte and A the high byte. Numeric colors independently apply `integerValues: 'numeric'` quantize semantics, including float32 conversion and `floor(t × 5)` clamped to index 4. Ordinal colors index the same palette with the actual class ID.

## Browser check

`check-county-classification.mjs` reuses `statistics-browser-check.mjs`, which captures and verifies the pinned showcase bundle before and after its browser slot. It checks exact/count, exact/class, approximate/count, approximate/class, and a rapid approximate-to-exact return. Each case checks five occupied classes, all 21 FIPS IDs and both packed colors, assignments against returned edges, class frequencies, and returned edge reasonableness. Exact SSE must match the exhaustive optimum within 0.5 count-squared units. Approximate SSE may be at most 1% above that optimum, and its edges must remain within one 128-bin width of the independent exact edges. The final rapid-return readout is checked again after a short settling interval so a stale result cannot replace the latest generation.

## Commands and handoff

Run the CPU oracle and syntax-only check from the repository root:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/reference-county-classification.py
node --check dev-docs/research/showcase-overhaul-2026-10-25/check-county-classification.mjs
```

The browser script is prepared for the parent-authorized serialized browser/GPU slot; it was deliberately not run in this numerical-only lane. During that slot, with the showcase already served from the pinned build, run:

```sh
node dev-docs/research/showcase-overhaul-2026-10-25/check-county-classification.mjs http://127.0.0.1:5313/ /tmp/luma-county-classification
```

The expected browser report has five passing cases, zero page/console/GPU errors, and exact and approximate interior edges `224327 | 395066 | 493819 | 671780` for this dataset. The explicit class palette packed uint32 values are `0xffd4ffff`, `0xff8ed9fe`, `0xff2999fe`, `0xff0e5fd9`, `0xff043499`. On the 21-row fixture approximate matches exact; the checker allows a 1% SSE gap and one histogram-bin edge movement to avoid turning the approximation check into a bitwise tie test.

The current scene already names the readouts `GPU Jenks edges`, `GPU class assignments`, and `Per-unit IDs and packed GPU colors`. The evidence row labels are `FIPS name: population people · class N · numeric 0x........ · ordinal 0x........`; those actual labels and the generation gate are used directly by the check.

Only the CPU oracle was executed here, along with `node --check`. Per the authorized lane constraints, no GPU/browser, root build/lint/install, or runtime source check was run.
