# Oracle fixtures of the math pass, rerun on the pinned set (2026-11-11)

Closing pass, stream "fixtures" (roadmap 12.6). The math pass of 2026-11-10 had no pinned Python
environment: it regenerated some oracle fixtures with the system Python 3.9.6 (NumPy 2.0.2, SciPy
1.13.1, pandas 2.2.2, Shapely 2.0.6) and recomputed a few without the reference library. This
report reruns every one of them on the pinned set of `modules/analysis/test/oracles/requirements.txt`
and, where the first pass had no library, runs the library.

Result in one paragraph: **no fixture value was wrong.** Twelve fixtures were rerun. Ten reproduce
the committed values to the last digit, one (the default-degree RBF fixture) differs by 3.9e-14
relative, and one (the noding fixture) is byte for byte the committed file. Shapely 2.2.0 gives
the two hand-patched region fixtures exactly, PyGeodesy 26.9.9 gives the hand-recomputed rhumb
midpoints to 4.6e-10 degrees, and mapclassify 2.11.0 with giddy 2.3.9 gives the right-closed
classes and transition counts exactly. Eleven fixtures were replaced by their pinned run so that
the recorded versions are the pinned ones; no contributor needs a change.

## Environment

- CPython 3.14.8, NumPy 2.5.4, SciPy 1.16.3, pandas 3.0.6, Shapely 2.2.0 (GEOS 3.14.1), pyproj
  3.8.0 (PROJ 9.8.1), PyGeodesy 26.9.9, mapclassify 2.11.0, giddy 2.3.9, scikit-learn 1.9.1,
  libpysal 4.15.0, scikit-image 0.26.0, networkx 3.7. The sibling fixtures of the same generators
  record CPython 3.14.5; a rerun changes that string in them and no value (second table).
- Not in the shared environment, installed into a scratch `--target` directory with `--no-deps`
  and put on `PYTHONPATH` for the one generator that needs them: pointpats 2.6.0 and Pillow 12.3.0
  (both pinned; the shared environment has Pillow 13.0.0) for `parity-wp1-round4.py`; rasterio
  1.5.1, randomgen 2.3.0 and affine 3.0.1 (pinned) for the `parity-wp4-*` generators;
  `opencv-python-headless` 4.12.0.88 for `raster-stretch-references.py`. OpenCV is not in
  `requirements.txt`, and its metadata asks for NumPy below 2.3; it imports and runs with 2.5.4.
- Each generator wrote into a scratch directory (`--output-dir`, or a copy of the script in a
  scratch tree with copies of its inputs). The committed fixture was compared leaf by leaf with a
  scratch script, and the final state was confirmed with the repository runner,
  `node scripts/run-oracle-fixture.mjs <generator> --python <pinned python>`, which regenerates in
  a sandbox and diffs: it reports `IDENTICAL` for every replaced fixture it can run (it passes
  `-I`, so the three generators that need the scratch `--target` packages were not run through it).

## Fixtures of the math pass

"Leaves" are numeric values in the decoded JSON. The consuming specs were run after the
replacement; counts are tests passed, none failed or skipped.

| Fixture | Generator | What the first pass did | Pinned result | Maximum difference | Action | Consuming spec and result |
| --- | --- | --- | --- | --- | --- | --- |
| `parity-wp6-group-statistics.json` | `parity-wp6-dataframe-reductions.py` | Regenerated on Python 3.9.6, NumPy 2.0.2, pandas 2.2.2, SciPy 1.13.1 after the float sum and mean moved from the fixed-point words to the pandas values (8 values changed) | 29,460 leaves equal | 0 | Replaced; records pandas 3.0.6 and the pinned set | `gpu-dataframe/group-statistics/parity-wp6-group-statistics.parity.spec.ts` 34 (GPU); `parity-wp6-reduction-fixtures.parity.node.spec.ts` 4 |
| `parity-wp1-focal-small.json` | `parity-wp1-round4.py` (`generateFocal`) | Hand-patched, then regenerated on Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1 with an empty stand-in for pointpats (non-finite samples became nodata) | Payload SHA-256 equal (1,749 leaves). Against the fixture at `HEAD` 42 cells differ (1, 4, 8, 9, 4, 0, 16 in the seven cases), as the first pass reported | 0 | Replaced; records pointpats 2.6.0, Pillow 12.3.0 and the pinned set | `gpu-spatial-analysis/spatial-interpolation/parity-wp1-focal-statistics.parity.spec.ts` 23 (GPU); `parity-wp1-round4.parity.node.spec.ts` 30 |
| `raster-stretch-references.json` | `raster-stretch-references.py` | New; OpenCV 4.12.0 and NumPy 2.0.2 on Python 3.9.6 | 1,995 leaves equal (OpenCV `equalizeHist`, `numpy.percentile(method='lower')`, the curve) | 0 | Replaced; records NumPy 2.5.4, OpenCV 4.12.0 | `gpu-raster/raster-stretch/gpu-raster-stretch.spec.ts` 12 (GPU) |
| `rts02-whittaker-sparse-exact.json` | `rts02-whittaker-sparse-exact.py` | New; exact rationals (`fractions`) on Python 3.9.6 | 31,881 leaves equal (120 cases); rational arithmetic, so equality is the expected result | 0 | Replaced; records Python 3.14.8 | `gpu-raster/raster-time-series/rts02-whittaker-precision.spec.ts` 15 (GPU) |
| `distribution-class-assignment-numpy.json` | `distribution-class-assignment-numpy.py` | New; `numpy.digitize` on NumPy 2.0.2, Python 3.9.6. mapclassify and giddy not run | 1,604 leaves equal. mapclassify 2.11.0: `bin1d` on the 400 fixed-edge values and `UserDefined` on the 398 finite ones equal the `'right'` classes (and differ from `'left'` in 38 rows); `Quantiles(k=4)` has bins `[3, 5, 9, 11]`, the fixture's percentile edges, and the `'right'` classes (26 rows differ from `'left'`). giddy 2.3.9 `Markov` transitions equal the `'right'` counts | 0 | Replaced. The generator now asserts the mapclassify and giddy results and records their versions | `gpu-spatial-analysis/distribution-dynamics/gpu-distribution-dynamics.spec.ts` 13 (GPU) |
| `scoring-rules-fair.json` | `scoring-rules-fair.py` | New; NumPy 2.0.2, SciPy 1.13.1 | 5,004 leaves equal (members, fair and standard scores, closed forms, bands) | 0 | Replaced; records NumPy 2.5.4, SciPy 1.16.3 | `gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-fair.spec.ts` 2 (GPU) |
| `trajectories-lines-numpy-checks.json` | `trajectories-lines-numpy-checks.py` | New; NumPy 2.0.2, Python 3.9.6 | 39 leaves equal | 0 | Replaced; records NumPy 2.5.4, Python 3.14.8 | `trajectory-interpolation/gpu-trajectory-resample.spec.ts` 4 and `trajectory-home-range/home-range-recursion-mob07.spec.ts` 15 (GPU) |
| `interpolation-rbf-default-degree.json` | `interpolation-rbf-default-degree.py` | New; SciPy 1.13.1 `RBFInterpolator`, NumPy 2.0.2 | 30 of 32 expected values differ in the last digits (the 32-site dense solve) | 9.8e-15 absolute, 3.9e-14 relative; the spec compares to 5e-4 | Replaced; records NumPy 2.5.4, SciPy 1.16.3 | `gpu-spatial-analysis/spatial-interpolation/interpolation-advanced.spec.ts` 33 (GPU) |
| `parity-wp1-round5-region-small.json` | `parity-wp1-round5.py` | Lasso rows patched by hand with exact rationals, Shapely not run (inclusive lasso; 4 rows of `concave-boundaries` changed) | Shapely 2.2.0 `covers`: 259 leaves equal, masks, ids, counts, statistics and histograms of all six queries | 0 | Replaced by the generator's output (the `provenance.lassoBoundaryRule` note of the hand patch is gone; nothing reads it) | `region-statistics/parity-wp1-region-statistics.parity.spec.ts` 24 and `parity-wp1-region-selection-modes.parity.spec.ts` 40 (GPU); `neighborhood-summary/parity-wp1-round5.parity.node.spec.ts` 13 |
| `parity-wp1-round5-region-real.json` | `parity-wp1-round5.py` | As above (2 boundary rows of 43,557 changed in `concave-lasso-selection-histogram`) | Shapely 2.2.0 `covers`: 411,651 leaves equal | 0 | Replaced, as above | The three specs of the row above |
| `parity-wp3-noding-showcase-large-snap.json` | `parity-wp3-topology-modes.py`, through `generate` of `parity-wp3-noding.py` | CSR arrays patched by hand to list each of six self-loops once; Shapely not run | Shapely 2.2.0 `unary_union`: the output is byte for byte the committed file | 0 | None needed | `gpu-network/network-noding/parity-wp3-noding.parity.spec.ts` 34 (GPU); `parity-wp3-noding.parity.node.spec.ts` 10 |
| `parity-wp2-round8-geodesics.json` | `parity-wp2-round8-geodesics.py` | Rhumb midpoints of the two rhumb cases recomputed with the generator's closed form in Python 3.15 (`math` only); PyGeodesy not run, generator not rerun | Full generator run: 11,990 leaves equal. PyGeodesy 26.9.9 `LatLon.rhumbMidpointTo(exact=False)`, the spherical loxodrome, on all 133 pairs of both cases: 0 pairs outside the fixture tolerance | Against PyGeodesy: 4.6e-10 degrees of longitude (26 micrometres, on a near-parallel pair), 7.1e-15 degrees of latitude; the fixture tolerance is 1.2e-4 degrees | Replaced by the full run (the `notes` field of the hand recomputation and the unused `versions.shapely` are gone). The generator now asserts every rhumb midpoint against PyGeodesy to 1e-8 degrees | `geometry-measures/parity-wp2-round8-geodesics.parity.spec.ts` 6 (GPU); `parity-wp2-round8-fixtures.parity.node.spec.ts` 3 |

Totals of the consuming specs after the replacement: 13 GPU spec files, 255 tests passed; 5 node
spec files, 60 tests passed; `oracles/oracle-fixture-runner-plat06.node.spec.ts` 4 passed.

## Generators the math pass edited without replacing a fixture

These were run to check that the edited script still reproduces what is committed. Nothing was
replaced.

| Generator | First-pass edit | Pinned result against the committed fixtures |
| --- | --- | --- |
| `parity-wp1-round4.py` | Non-finite samples are nodata in `computeFocal` | The other 21 fixtures (zonal, geographic, sixteen Grand Canyon focal stripes): every leaf equal; only `versions.python` differs |
| `parity-wp1-round5.py` | Inclusive lasso | `round5-neighborhood` and `round5-partition`: payload SHA-256 equal |
| `parity-wp6-dataframe-reductions.py` | pandas float sum and mean | `earthquake-input`, `temporal-reduction`, `histogram`: every leaf equal |
| `parity-wp6-round5.py` | pandas float sum and mean in the key join | Six fixtures, every leaf equal. Their `conventions` field still holds the generator docstring from before the edit |
| `parity-wp3-noding.py` | A self-loop has one CSR entry | `chicago-loop`, `chicago-west`, `cross-overlap`, `snap-cells`: every leaf equal (none holds a self-loop) |
| `parity-wp4-contours.py` | Bilinear saddle decider | Every leaf equal; one provenance path string differs (`modules/graph/...` in the committed file) |
| `parity-wp4-isobands.py` | Bilinear saddle decider | Identical apart from `versions.python`. The 1e-16 area differences the first pass saw came from Shapely 2.0.6 and scikit-image 0.24.0, as it said |
| `parity-wp4-lic.py` | World-space tangent | 47,266 of 47,267 leaves equal; the diagnostic `minimumPixelMargin` of the wind case differs by 7.1e-15 (2e-9 relative), the value the first pass reported. Run with real rasterio 1.5.1 and randomgen 2.3.0 instead of the first pass's stand-ins |
| `parity-wp4-lic-modes.py` | World-space tangent | The three fixtures are byte for byte the committed files |

## Not rerun, and why

- `nested-validation-oracle.py` asserts NumPy 2.2.6 and SciPy 1.15.3 (the older set named in the
  header of `requirements.txt`), so it stays on its recorded set. Its first-pass edit replaced a
  hash by the constant `[1, 1, 0, 0, 1, 0]`; evaluating the removed hash for blocks 0 to 5 gives
  the same six values, so the fixture is unaffected.
- `statistics-blockcv.R`: the first pass changed a comment only. R is not installed.
- `line-integral-convolution-cpu.ts` is a TypeScript twin, not a reference-library generator.

## Changes made

- Fixtures replaced by their pinned run (eleven): every fixture of the first table except the
  noding fixture, which was already the pinned output. The replaced files are the generators'
  output, copied from the scratch run.
- `distribution-class-assignment-numpy.py`: imports mapclassify and giddy, asserts `bin1d`,
  `UserDefined`, `Quantiles` and `giddy.Markov` against the NumPy values, records both versions.
  The stored values are still NumPy's.
- `parity-wp2-round8-geodesics.py`: asserts each rhumb midpoint against PyGeodesy
  `rhumbMidpointTo`; the stored value stays the closed form, which is continuous with the origin
  across the date line.
- Docstrings and comments that named the system Python as the recorded environment:
  `parity-wp1-round4.py`, `parity-wp1-round5.py`, `interpolation-rbf-default-degree.py`,
  `raster-stretch-references.py`, `scoring-rules-fair.py`, `trajectories-lines-numpy-checks.py`.
- `GENERATORS.md`: the entries of the six fixtures the first pass added, and the noding
  showcase fixture moved from `parity-wp3-noding.py` to `parity-wp3-topology-modes.py`, the script
  that writes it (the repository runner reads this file to know what a generator should write).

## Not verified, and open

- rio-color was not run: the stretch curve is a NumPy evaluation of its formula, as before.
- OpenCV has no entry in `requirements.txt`. `opencv-python-headless==4.12.0.88` should be added
  there with the `--no-deps` note (the file is outside this stream).
- PyGeodesy `rhumbMidpointTo(exact=True)` on the same spherical points returns latitude 32.52 for
  (10, 20) to (40, 50), where the mean latitude is 35. It is the ellipsoidal rhumb path of the
  library and is not what the generator or the kernel claims; it was not investigated further.
- Two spec comments still name the first-pass versions: `gpu-distribution-dynamics.spec.ts:261`
  ("NumPy 2.0.2") and `gpu-ensemble-scores-fair.spec.ts:11` ("NumPy 2.0.2, SciPy 1.13.1"). The
  specs belong to other streams.

## Commands

```
PY=<scratch>/venv314/bin/python
$PY -I modules/analysis/test/oracles/parity-wp6-dataframe-reductions.py --output-dir <scratch>
$PY -I modules/analysis/test/oracles/parity-wp6-round5.py --output-dir <scratch>
$PY modules/analysis/test/oracles/parity-wp1-round5.py --output-dir <scratch>
$PY modules/analysis/test/oracles/parity-wp3-noding.py --output-dir <scratch>
PYTHONPATH=<scratch target> $PY modules/analysis/test/oracles/parity-wp1-round4.py --output-dir <scratch>
node scripts/run-oracle-fixture.mjs <generator> --python $PY
npx vitest run --project headless <the thirteen GPU specs of the table>
npx vitest run --project node <the five node specs of the table>
```

The generators without `--output-dir` were run from a copy in a scratch tree.
