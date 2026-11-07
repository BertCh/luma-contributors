# Native Kepler fixtures and saved-map loading

Test-support notes for `scripts/kepler-fixtures/` and `scripts/native-saved-map.mjs`.

## Contents

| File | Purpose |
| --- | --- |
| `generate.mjs` | Deterministic (seeded) generator for everything in `data/`. `node scripts/kepler-fixtures/generate.mjs` |
| `validate-saved-maps.mjs` | Node-only validation through Kepler's own `KeplerGlSchema` / `processKeplerglJSON` |
| `load-kepler-schemas.mjs` | Loads the installed CJS `@kepler.gl/schemas` and `@kepler.gl/layers` in node (see below) |
| `data/*.csv`, `data/*.geojson` | Raw uploads mirroring Kepler demo schemas (earthquakes, NYC taxi, SF zip polygons, H3, arcs/flows, trips, S2/geohash/A5 tokens, icons, wind U/V) |
| `data/*.kepler.json` | Native saved maps (`{datasets:[{version,data}], config:{version:'v1',config}, info}`) |

Saved maps are produced by `KeplerGlSchema.save` from real Kepler layer instances (`LayerClasses[type]`)
and datasets processed by `processCsvData` / `processGeojson`, i.e. the same code Load Files runs. Nothing
is hand-written in the export format. `all-layer-types.kepler.json` contains 17 layers: point, heatmap,
hexagon, grid, cluster, 3D, arc, line, geojson, hexagonId, trip, s2, geohash, a5, icon, flow, flowField.
The remote-source types (vectorTile, rasterTile, wms, tile3d, bitmap) need hosted tiles/imagery and are
covered by the TILES scripts instead.

Run order: `generate.mjs` (only when changing fixtures) -> `validate-saved-maps.mjs` (node, no browser) ->
`node scripts/native-saved-map.mjs [url]` against a dev server (browser, WebGPU).

## Running Kepler's schema in node

`require('@kepler.gl/schemas')` transitively imports deck.gl / luma.gl GPU modules, which fail in plain node
(two deck.gl core copies 9.4.0 vs 9.4.1 trigger the version check, and `@luma.gl/webgl/constants` is a Vite
alias). `load-kepler-schemas.mjs` stubs `@deck.gl/*`, `@luma.gl/*`, `@loaders.gl/*`, `@math.gl/*`,
`@flowmap.gl/*` and `@kepler.gl/deckgl-*` with an inert proxy while the CJS graph loads. Schema,
layer class metadata, and processors then run unmodified. Nothing here renders.

## How Kepler Load Files handles a saved map (installed 3.3.0-alpha.18)

1. `processFileData` (`@kepler.gl/processors` file-handler): a JSON object with `datasets`, `config` and
   `info.app === 'kepler.gl'` is detected by `isKeplerGlMap` -> format `keplergl`, processor
   `processKeplerglJSON`. `KeplerTable` defines no static `getFileProcessor`, and the host table class
   (`ResidentKeplerTable`) does not either, so the default branch is taken.
2. `processKeplerglJSON` returns `{datasets:[{data:{fields,rows}, info:{id,label,color,type}}], config, info}`
   with `config` already parsed by `KeplerGlSchema.parseSavedConfig` (channels folded into `layer.config`,
   no `version`). Dataset rows are plain row arrays (a GeoJSON dataset has the Feature object in column 0).
3. `filesToDataPayload` sets `options.centerMap = !config.mapState`; saved maps with a `mapState` keep their
   own viewport.
4. `loadFilesSuccessUpdater` calls `addDataToMapUpdater` per payload (maps first, then plain datasets).
   Because `config` is not in `{version, config}` form, `isValidConfig` is false and it is used as parsed.

So a saved map reaches the host as one `ADD_DATA_TO_MAP` whose `datasets` array and `config` reference the
same dataset ids. Both ids and layer ids are fixed by the file (raw files get a filename hash id).

## Host review (examples/deck/kepler-compute/src)

* `ResidentKeplerTable.getInputDataValidator` -> `createResidentInputValidator(validateInputData)` only
  changes behavior for inputs marked by `markResidentArrowInput`; saved-map datasets and raw uploads take
  the pinned native `validateInputData` path unchanged. Saved fields already carry `format` and
  `analyzerType`, so the native validator does not re-infer them. No blocker.
* `applyNativeConfig` / `attachPublishedDataset` / `publishSource` do not run for Load Files; they serve
  sandbox JSON import and the Prepare data workflow. No blocker (not on this path).
* `getResidentActionRejection` runs `assertSingleViewConfiguration` on `ADD_DATA_TO_MAP` /
  `RECEIVE_MAP_CONFIG` payloads. Fixtures use no split maps (`splitMaps: []`, `isSplit: false`), so they
  pass; a real Kepler export with split maps is rejected with `SINGLE_VIEW_REASON` (intended host limit).
* No host code intercepts `LOAD_FILES*`, `stagedToAdd`, or `processFileData`; the stock modal is used and
  `NativeLoadDataModalFactory` only appends the Prepare data and SQL methods.

### Requirements this places on auto-prepare (not blockers in current code)

* Dataset id and layer ids must be preserved when an auto-prepare replaces the native dataset with a
  resident one: saved layers, filters, tooltip `fieldsToShow` and charts reference dataset ids, and the
  acceptance script asserts `saveNativeConfig()` returns the same layer ids, `dataId`s, columns, filter ids
  and tooltip fields as the file.
* The config arrives in the same action as the datasets, so layers for not-yet-resident datasets exist
  before auto-prepare completes. Layer admission must tolerate that interval or auto-prepare must run
  from the `ADD_DATA_TO_MAP` action itself.
* Datasets are row arrays: GeoJSON rows (`_geojson` column) hold Feature objects, timestamps are strings
  in the field `format` (e.g. `YYYY-M-D H:m:s`), H3 / S2 / geohash / A5 tokens are strings. Auto-prepare
  conversion must handle each (`h3` field type, string tokens, `geojson` object cells).
* One map adds several datasets and many layers at once (`all-layer-types`: 9 datasets, 17 layers);
  auto-prepare must not publish them one at a time against a stale snapshot.
* Several layers can share one dataset with different columns (S2/geohash/A5 over `cell-tokens`).

## Known limits

* Resident rendering of 3D (scenegraph) and flowField layers is not verified; the acceptance script lists
  misses per layer (`--report-only` downgrades only the all-layer-types misses to a report).
* The icon layer fetches Kepler's hosted svg catalog; the script serves a local three-icon catalog on
  `**/icons/svg-icons.json`.
* `renderedLayers()` ids of host-created resident layers may carry suffixes; the script matches a saved
  layer when a rendered id contains the saved layer id.
* Biome ignores `examples/**/scripts`, so these files are not auto-formatted by `yarn lint`.
