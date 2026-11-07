# Grid-alignment projection and support diagnostic

## Wave13 measured result

The production browser run used bundle SHA256 `658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d`, verified unchanged before and after the run. Registration recovered both known shifts exactly, with zero aligned reflectance residual. The reprojection comparison is **not numerically equivalent to the independent Rasterio/GDAL transform**. This is a measured precision limitation of the explicitly selected Float32 analytic projection stages, not a sampling-mask defect or an accepted arbitrary count tolerance.

All 32,768 GPU validity decisions (4,096 cells × eight date/method/policy cases) agree exactly with independent scalar sampling at the actual GPU source coordinates. The maximum axis coordinate difference from Rasterio is 3.239092 m on this 60 m grid. The declared longitude/latitude center versus an exact UTM-center conversion changes coordinates by at most 0.000097 m. Reconstruction of the affine source-grid coordinates from the GPU projected offsets contributes at most 0.000519 m in easting and 0.000306 m in northing. The metre-scale difference is already present in the projection before source-grid affine reconstruction.

| Source date and sampling | Rasterio valid count | GPU valid count | Spatially different validity cells |
| --- | ---: | ---: | ---: |
| July nearest, strict | 4,091 | 4,090 | 1 |
| July nearest, renormalize | 4,091 | 4,090 | 1 |
| July bilinear, strict | 4,068 | 4,067 | 1 |
| July bilinear, renormalize | 4,096 | 4,096 | 0 |
| September nearest, strict | 3,486 | 3,486 | 28 |
| September nearest, renormalize | 3,486 | 3,486 | 28 |
| September bilinear, strict | 2,979 | 2,978 | 7 |
| September bilinear, renormalize | 3,891 | 3,890 | 3 |

Matching aggregate counts therefore do not establish matching spatial support. In particular, September nearest loses and gains 14 cells each.

At zero-based destination row 0, column 58, the Rasterio source coordinate is `(89.328819763, -0.495865458)`; the GPU coordinate is `(89.325523376, -0.524999976)`. The source row crosses the nearest-support boundary at −0.5. Rasterio northing is 4,456,039.751927 m, while the GPU projection before affine reconstruction is 4,456,041.5 m. July support is consequently valid in the reference and invalid on the GPU, exactly as each coordinate implies.

At row 0, column 35, source row changes from `0.015359367` to `-0.008333308`, crossing the bilinear full-support boundary at zero. Other September differences cross internal cloud/quality-support boundaries. The complete report identifies every changed cell; these are not inferred from a count difference.

## Numerical contract and acceptance

`makeReprojectionProjection` explicitly chooses `arithmetic: 'float32'` for the inverse Web Mercator and forward transverse Mercator stages. `precision: 'local-f32'` subtracts the destination origin only at the final output boundary; it does not upgrade those analytic stages. This is stated in `modules/graph/src/gpu-project/projection-program.ts` and the GPU project README. Changing the output format alone would not remove the analytic precision loss.

The exact sampling-policy comparison passes at actual GPU coordinates. Rasterio coordinate/support equivalence remains false and must stay separately reported. The observed 3.239092 m error is evidence for this particular window, device and compiled pipeline, not a general mathematical error bound. No core shader, support-count assertion, tolerance, or source geometry was changed to obtain this result. A more precise adaptive projection would require its own source change and verification.

## Reproduction and evidence

`check-grid-alignment-browser.mjs` enables the test-only `gridSupportDiagnostics` query, collects actual GPU source coordinates, projected offsets, and all eight masks, and saves the served build manifest before and after execution. Normal scene use does not allocate these diagnostic buffers.

`compare-grid-alignment-support.py` independently reconstructs the destination grid using the actual declared center, transforms it through Rasterio, samples source validity at both reference and GPU coordinates, and fails on any GPU mask decision unexplained by its actual coordinate. `diagnose-grid-alignment-support.py` is the earlier candidate-ranking utility; its predictions alone were not evidence of actual mismatches.

```sh
node examples/deck/spatial-analysis-showcase/scripts/check-grid-alignment-browser.mjs \
  http://127.0.0.1:5313/ /tmp/luma-showcase-overhaul-20261025/grid-wave13
/tmp/luma-raster-fill-oracle312/bin/python \
  examples/deck/spatial-analysis-showcase/scripts/compare-grid-alignment-support.py \
  /tmp/luma-showcase-overhaul-20261025/grid-wave13/gpu-support.json \
  /tmp/luma-showcase-overhaul-20261025/grid-wave13/support-comparison.json
```

Evidence directory: `/tmp/luma-showcase-overhaul-20261025/grid-wave13/`. `support-comparison.json` contains every spatial mismatch; `support-summary.json` provides the compact count/precision summary; `build-verification.json` proves the bundle remained unchanged. The browser was closed and the GPU slot released before CPU comparison.

## Approved correction in progress

The parent approved replacing the default analytic GPU chain with the existing bounded adaptive
projection infrastructure. The provider evaluates the same analytic transform in JavaScript
binary64, then returns UTM offsets from the source tile’s northwest origin. An adaptive plan is
fitted at a requested sampled error of 0.001 m, evaluated with double-single intermediates, and
published as local Float32 coordinates. The source raster affine is rebased equivalently so the
sampler never adds a global northing before converting to source cells. The source data, QA,
destination grid and display geography remain the same.

The sampled fit certificate is not the acceptance test: every destination center will also be
compared independently with Rasterio, with a 0.01 m Euclidean error requirement and exact support
masks. The comparison script’s `--require-accurate` mode rejects the previous Float32 baseline
(maximum Euclidean difference 3.624963 m), confirming the test catches the diagnosed failure.
GPU verification of the corrected source is pending a new coordinated build and slot. A separate
reference audit finds the closest support-changing coordinate boundary at 0.027892 m
(September bilinear strict, destination row 31, column 42). The 0.01 m acceptance requirement
is tighter than that measured boundary margin; exact mask equality remains independently required.

The CPU transverse-Mercator provider uses the repository’s sixth-order Krüger implementation.
The coefficient formulation and geographic interpretation are documented by
[PROJ’s transverse Mercator reference](https://proj.org/en/stable/operations/projections/tmerc.html).
That published formulation motivates the provider; its theoretical accuracy is not substituted
for the independent measurements of this particular compiled scene.

### Independent CPU review of the corrected source

The corrected provider and plan were exported at all 4,096 destination centers using the exact
runtime source origin `[662800, 4456040]`. One cubic patch has a sampled fit certificate of
`1.6052345582760234e-8 m`, against the requested `0.001 m` tolerance. Separately, Rasterio comparison
of every center finds a maximum Euclidean provider error of `5.155240610587534e-9 m` and CPU plan
error of `1.5375639294366404e-8 m`. Simulating final local Float32 storage and the contributor’s
exact affine expression order yields a maximum Euclidean coordinate error of
`0.0006553139597480028 m`; all eight validity masks match the independent reference exactly.

These are CPU-only results, not a claim of successful GPU execution. The tracked
`check-grid-alignment-cpu-reference.py` regenerates this comparison from the exported provider and
plan coordinates. Its report is `/tmp/luma-showcase-overhaul-20261025/grid-adaptive-cpu-reference.json`.
The previous Float32 GPU failure remains preserved in the wave13 evidence directory.

The source correction is ready for the coordinated wave14 build. Showcase TypeScript, scoped
Biome, story checks and the raster WGSL scan (798 templates) pass. Reproduce CPU review with:

```sh
node examples/deck/spatial-analysis-showcase/scripts/export-grid-alignment-cpu-reference.mjs \
  /tmp/grid-alignment-adaptive-projection.json
/tmp/luma-raster-fill-oracle312/bin/python \
  examples/deck/spatial-analysis-showcase/scripts/check-grid-alignment-cpu-reference.py \
  /tmp/grid-alignment-adaptive-projection.json \
  /tmp/luma-showcase-overhaul-20261025/grid-adaptive-cpu-reference.json
```

The exporter reads the pinned source manifest, invokes the same scene helper and destination
center function, and bundles only a temporary Node helper using installed esbuild. It does not
build the application or replace the independent Rasterio oracle. On the next production GPU
run, append `--require-accurate` to the support-comparison command to require the 0.01 m Euclidean
limit and exact spatial support. The fit and CPU simulation alone are not GPU acceptance.

## Wave14 production acceptance

The corrected production GPU pipeline passes the strict independent gate on unchanged bundle
`872af62d507f3cae0bed2749a0b9f8c7736739cf2d99cab71ba8658bf5e817ae`. Across all 4,096 centers the maximum
Euclidean difference from Rasterio is **0.000763259622 m**, below the required 0.01 m. All 32,768
validity decisions match the independent reference exactly, including the September cases whose
equal aggregate counts had concealed spatial differences in the rejected Float32 baseline.
Both controlled registration experiments also pass with zero displayed aligned residual.

Durable evidence is `earth-grid-adaptive-gpu-review.json`, containing exact mask comparisons,
registration and sampling records, source hashes, raw report hashes, and verified build identity.
Raw coordinate arrays and screenshots remain in `/tmp/luma-showcase-overhaul-20261025/grid-wave14/`.
The earlier failed pipeline is retained as `earth-grid-float32-baseline.json`; its comparison fails
the same strict gate. This establishes the correction without changing the accuracy requirement.
