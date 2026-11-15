# AX-70 journey baseline (measured in a browser)

Measured on 2026-11-14 with `examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs`, against the Kepler host example served by Vite. This replaces the "counted from code" numbers in `../README.md` sections 2.2 and 6 with counts from a driven browser. Raw data: `journeys.json` (every step, selector, count, time, observation) and `manifest.json` (environment).

Re-run after each release:

```sh
cd examples/deck/kepler-compute && yarn start --host 127.0.0.1 --port 5391   # in one terminal
node scripts/ax-journey-baseline.mjs --url http://127.0.0.1:5391/ --out <folder>   # in another
```

The script exits non-zero and names the journey, the step and the selector when anything is missing; a failed journey is recorded as `outcome: "failed"`, never skipped. `--only <id prefix>` runs a subset (it overwrites the output files with that subset, so use a scratch `--out`).

## What was run

- Chromium 151.0.7922.34, headless, launched with the repository's WebGPU options (`--enable-unsafe-webgpu --ignore-gpu-blocklist`). Adapter reported by `navigator.gpu`: vendor apple, architecture metal-3, not a fallback adapter (a real GPU, not SwiftShader).
- Apple M3 Pro (Mac15,7), 36 GiB, macOS Darwin 23.1.0, Node 22.23.4.
- Commit `fbb78ec5` (branch `upstream-first`, read from `.git/HEAD`). The measured application is the working tree, which had uncommitted changes in the host (see the git status of the session), so the numbers describe that tree and not the commit alone.
- URL `http://127.0.0.1:5391/?rows=256`: the built-in "Resident observations" dataset is 256 rows instead of the default 100,000, so that startup is short. The journeys do not depend on its size. Viewport 1440 by 1000.
- 14 journeys plus one derived record (15 entries), one fresh page each, all in one browser, 63 seconds in total. All passed; no selector was missing. Console noise on every page: a React 19 `element.ref` deprecation warning, and on pages that open the Add Data dialog a "Cannot update a component (MemoryManagerPanel) while rendering" warning. Neither stopped a journey.

## Counting rules

An action is one click, one select-option, one fill, one key press or one file selection performed by the script. Navigation clicks are counted. Reads of the page, waits and the legend probe after a journey ends are not counted. Each step in `journeys.json` has a `phase`: `setup` is loading the sample through the gallery (Add Data, Prepare data, open "Try a sample dataset", choose the sample unless it is the gallery default, Prepare sample, Check schema and precision, Publish selected fields), `analysis` is everything after. Time runs from the first action to the stated end condition and includes real waits (GPU upload, analysis dispatch) but no fixed sleeps. Times are from a script with no reading or hesitation, so they are a floor, not a user time.

## 1. First analysis result per gallery sample

Shortest route found: after the sample is published, choose it in the analysis panel's Dataset select, then choose Operation "Standardize · z-score". End condition: the GPU analysis panel shows a live block "N analyzed · M undefined" with a "Transformed value range". Predicted (review and plan): setup 4 actions for the default sample and 5 for others (open, prepare, check, publish; plus a selection); analysis 1 (choose operation). Predicted total 5 or 6.

| Sample | Setup actions | Analysis actions | Total | Predicted total | Time (ms) | Default numeric field | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| h3-cells | 7 | 2 | 9 | 6 | 1076 | measurement | 6 analyzed, range -0.968 to 1.614 |
| points-time | 6 | 2 | 8 | 5 | 947 | longitude | 256 analyzed, range -1.624 to 1.624 |
| districts | 7 | 2 | 9 | 6 | 1157 | events | 16 analyzed |
| connections | 7 | 2 | 9 | 6 | 1020 | longitude | 24 analyzed |
| trips | 7 | 2 | 9 | 6 | 1038 | duration_seconds | 8 analyzed, range 0 to 0 (degenerate: every duration is equal) |
| trip-observations | 7 | 2 | 9 | 6 | 1055 | longitude | 48 analyzed |
| numeric-grid | 7 | 2 | 9 | 6 | 967 | value | 253 analyzed (3 missing cells, excluded) |
| directed-network | 7 | 2 | 9 | 6 | 974 | cost_seconds | 48 analyzed |

`noAnalysisAvailable` did not occur: every sample offers at least the standardization, classification and risk operations on its numeric fields. The interface does not offer the sample-specific analysis the sample description hints at (for example lattice hot spots for numeric-grid, network analysis for directed-network); those need the hand-written panel or the catalog.

Measured facts that differ from the prediction:

- The gallery path is 6 actions for the default sample (points-time) and 7 for the other seven, not 4 and 5. The review counted "open, prepare, check, publish"; Add Data and Prepare data (the route into the gallery) are two further clicks, and opening the "Try a sample dataset" disclosure is a third.
- Publishing a sample does not select it in the analysis panel. The Dataset select stays on "Resident observations" (the built-in dataset) after the sample is published (`analysisDatasetSelectedAfterLoad` in `journeys.json`), so choosing the dataset is one more action in every route. The plan's "1 action: choose operation" is 2 from a loaded sample.
- The default numeric field is the first column, which is a coordinate for points-time, connections and trip-observations (`longitude`). The shortest route therefore produces a result on longitude. Getting a result on a measurement (reading, volume, speed) needs one more select action for those three samples.
- Table-only analysis result: the result block is text in the sidebar. After the result the map legend panel (opened by one extra click, not counted) lists only the layers' "Fill color" swatches for the sample and the built-in layer; no analysis legend entry appeared in it (probe text stored per journey). The map screenshot `first-result-points-time.jpg` shows part of the points recolored along longitude.

## 2. Point statistics through a hand-written panel (points sample)

Route: choose the dataset, open "Configure hot spots on geographic points", choose Hot-spot longitude field, choose Hot-spot latitude field, press "Apply neighbors and run point hot spots". End condition: the analysis panel shows "N analyzed · M undefined" and a "Hot-spot z-score range" line.

| Journey | Setup | Analysis | Total | Predicted | Time (ms) | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| point-hotspots:default-field | 6 | 5 | 11 | 4 after choosing the dataset | 1081 | 256 analyzed, z-score range -4.298 to 4.298 |
| point-hotspots:reading-field | 6 | 6 | 12 | 4 (5 with a field change) | 1248 | 195 analyzed, 52 undefined, range -5.002 to 5.017 |

The prediction of 4 holds: open, longitude, latitude, apply are 4 of the 5 analysis actions. The fifth is choosing the dataset (see above), and the default run analyses the coordinate `longitude`, which is meaningless; the reading field takes a sixth. Fields that start blank (value empty): the longitude field and the latitude field of the panel, and nothing else. Defaults present: geographic coordinate system, topology knn, 8 nearest neighbors, minimum neighbors 8, self weight 1, islands excluded, Benjamini-Hochberg correction (full list in `observations.formControlsBeforeEditing`). The reading field has 52 missing values and those rows are undefined in the result.

## 3. The generated catalog

Observation 3a: after the districts sample (polygons with a GeoJSON `geometry` column) was prepared through the gallery, the catalog's first input was switched to "Native dataset" and its dataset select held one option, "Choose dataset". The prepared dataset "District counts and populations" is absent from the catalog, while the analysis panel lists it. Prediction confirmed for gallery-prepared data.

Observation 3b: the same after a two-polygon GeoJSON file was added through Add Data (file select plus the dialog's Add Data button, 3 setup actions) and auto-prepared (`autoPrepared: true`; the analysis panel lists `ax-two-squares.geojson`, Kepler shows "Datasets(2)"). The catalog dataset select again held only "Choose dataset". The data most users load natively is not reachable from the catalog. Prediction confirmed. No route to the catalog's native-dataset branch was found in this run, so the "about 8 for an eligible non-resident single-input dataset" figure in the plan could not be measured.

Shortest route by which a catalog operation can run today (built-in sample input, default operation `geometry-validity`): press Run, press "Stage in Sources", Add Data, Prepare data, Check schema and precision, Publish selected fields: 6 actions, 780 ms to a new valid `geojson` layer in the compute state. Predicted 5 (operation, run, stage, check, publish in the review's reading). Measured 6 because opening the staged table needs Add Data and Prepare data (two navigation clicks); no operation select is needed for the default operation, so choosing another operation makes it 7. The result block shows "featureCount: 3 · invalidCount: 2" and a preview table before staging.

## 4. A catalog result as published

`catalog-published-layer-legend.jpg` (legend opened with one extra click; journey `catalog-result:published-style`). The published layer "Geometry validity" is a single colour: the valid polygon and the two invalid polygons (status `invalid`, error counts 1) share one cyan fill and outline. The legend lists "Fill color" and "Outline" swatches only, with no field or scale. A pixel census of the map area shows one dominant non-background colour (about 82,000 pixels at RGB near 64,160,192). The result columns (`status`, `error_count`, `issues`) are not used for styling. This confirms F7 for the layer colour; the legend exists but carries no classes. A summary sentence is not part of the published layer (the catalog panel shows raw keys).

## 5. Interpolation defaults

On the points sample, the IDW panel ("Configure inverse-distance interpolation grid") and the kriging panel both open with fixed bounds west -74.1, south 40.65, east -73.9, north 40.85, a 32 by 32 grid, search radius 10,000 (m). The points-time sample's actual extent is west -74.04000, south 40.70000, east -73.95000, north 40.79000. Prediction confirmed in that the bounds are constants and not the data's extent. Nuance: all four gallery samples that carry coordinates (points-time, connections, trip-observations, numeric-grid) lie inside the fixed New York box (`observations.sampleExtents`), so on the samples the fixed bounds contain the data, with a margin of at least 0.05 degrees on every side. The defect shows on user data outside New York, which this baseline did not run. Also blank at start: the longitude and latitude fields of both panels; kriging adds two intentionally blank optional fields (maximum lag, confidence level).

## Predictions in the plan that turned out wrong or incomplete

1. "4 actions from a loaded sample to a first result for point statistics" (section 6 measure): 4 holds only after the dataset is chosen; from a loaded sample it is 5, and 6 for a meaningful field, because the default field is a coordinate.
2. "1 action: choose operation" for standardization and classification (review table): 2, because publishing a sample does not select it in the analysis panel.
3. Gallery preparation "four actions" (review): 6 for the default sample and 7 for the others.
4. Catalog with built-in input "5 actions": 6 to a published layer.
5. Not wrong but understated: the default numeric field is the first column, a coordinate for three of the eight samples, so the first result on those samples is on longitude.
6. IDW bounds: confirmed as fixed, but they happen to contain every gallery sample, so the samples do not reveal the problem.

No prediction was contradicted in kind: the catalog cannot see prepared datasets (confirmed twice), point forms start with blank coordinates (confirmed), IDW and kriging use fixed New York bounds (confirmed), and the catalog result is one colour (confirmed).

Not measured: the cartogram (predicted 9 actions), lattice hot spots on numeric-grid (predicted 4), OLS, DBSCAN, polygon statistics, and the catalog route for a non-resident dataset with a GeoJSON field.

## Limits of the measurement

- Headless Chromium on one machine (Apple M3 Pro, real Metal GPU). No other browser, GPU or operating system.
- A script takes the shortest path and knows the selectors; a person must discover the Analysis section, the Prepare data dialog and the dataset select. The counts are lower bounds on user actions and say nothing about errors, hesitation or reading. Times (about 1 s per journey, 0.8 s of it loading the sample) are lower bounds that mostly measure the app and Playwright, not users.
- The built-in dataset is 256 rows and the samples are 6 to 256 rows. Latency at scale is not covered.
- The working tree had uncommitted changes from other sessions; the manifest records the commit only. A rerun on a different tree can differ.
- The end conditions are DOM text and compute-runtime state, not pixels, except the colour census in journey 4. "Result visible" means the panel reports a result; whether the map layer was recolored for each sample was checked by eye for points-time only.
- Selector fragility: the journeys use element ids and aria labels of the current host (for example `#analysis-dataset`, `.sample-gallery select`, `.analysis-point-hotspot-settings`). A rename fails the journey loudly.
- The Vite CLI in this repository does not accept `--cacheDir`; the server was started with a configuration file kept outside the repository that wraps the example's `vite.config.ts` and only sets `cacheDir`.
