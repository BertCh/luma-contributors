# Assistant tool mapping

The assistant router (`src/assistant-operation-router.ts`) accepts calls shaped like the tools
published by `@kepler.gl/ai-assistant`. Source: `npm view` and `npm pack` of
`@kepler.gl/ai-assistant@3.3.0-alpha.8`, the latest published version (dist-tag `latest`). The
host app runs Kepler 3.3.0-alpha.18, and no alpha.18 of this package exists. The package is not
installed and is not a dependency of this repo. Its tool definitions were read from the tarball only
(`dist/esm/tools/**`). `@openassistant/*` (zod schemas of the SQL, chart and geo tools) were not
fetched, so those tool parameters are not mirrored.

Accepted call shapes: `{toolName | tool | name, args | arguments | input | parameters}` or a flat
`{type: <toolName>, ...args}`. Normalization lives in `src/assistant-published-tools.ts` and runs
before the router's guards, so approval, memory, population, precision and export contracts still
apply. Spec: `modules/kepler/test/assistant-published-tools.node.spec.ts`.

## Mapped

| Published tool | Published arguments | Routed as | Notes |
| --- | --- | --- | --- |
| `addLayer` | `datasetName` (display name), `layerType`, `latitudeColumn`, `longitudeColumn`, `layerName` | native `addLayer` | Dataset name resolves through dataset `label` (id accepted as fallback). Lat/lng go to `columns.lat/lng`. `geojson` uses `_geojson` when that field exists. `layerName` becomes the layer label. Layer types outside the resident renderer (`cluster`, `h3`, `s2`, `a5`) are rejected. Published `guessDefaultLayer` column inference is not replicated: missing columns are not guessed. |
| `addLayer` with `colorBy` / `colorMap` | `colorBy` (field), `colorMap` (value -> `#rrggbb`, object or `[{value,color}]`) | native `addLayer` with `colorField`, `colorScale` and optional custom `colorRange` | Numeric fields use a quantile scale, string/boolean fields ordinal. `colorMap` requires a categorical field. `colorType` is ignored. |
| `colorBy` / `colorMap` (standalone, `layerId`) | `layerId`, field, optional `colorMap` | native `layerVisualChannelConfigChange` (channel `color`) | Same validation as above, against the layer's dataset. |
| `updateLayerColor` | `layerId`, `numberOfColors`, `customColors` | native `layerVisConfigChange` with `colorRange` | Count mismatch is rejected like the published tool. Published tool keeps and recolors the existing `colorMap`; here the range is replaced with a custom palette and no `colorMap`. |
| `basemap` | `styleType` (9-value enum) | native `mapStyleChange` | The enum is validated; whether the style exists in this app's map style list is left to Kepler. |
| `loadData` | `url` | host capability `source-workspace` | Published args carry no size, so the router assumes the per-load byte maximum (conservative memory guard). The capability only stages a notice; it does not stream data. |

| `mapBoundary` | optional `datasetName` | host capability `map-boundary` | Stages a notice; no assistant read route returns bounds to the model. |
| `genericQuery`, `filterDataset`, `tableTool`, `mergeTables` | `sql`/`query`, optional `datasetName` | host capability `sql-query` | One read-only `SELECT`/`WITH` only (no mutation keywords or file readers). Source dataset must be complete. The last three create a dataset, so the per-load byte maximum is checked against the memory budget. Row cap is the export row limit. Host `executeSql` is an alias without dataset creation. The capability stages a notice; it does not run DuckDB for the model. |
| `histogramTool`, `boxplotTool`, `scatterplotTool`, `bubbleChartTool`, `pcpTool` | `datasetName`, `variableName(s)`/`xVariable`/`yVariable`/`sizeVariable` | host capability `chart-widget` | Needs a complete dataset and numeric fields; parameter names are inferred since `@openassistant/*` schemas were not fetched. Stages a notice. |
| `saveDataToMap` with `url` | `url` | `loadData` | Without a `url` it is rejected. |

## Not mapped

| Published tool | Reason |
| --- | --- |
| `classifyTool`, `weightsTool`, `regressionTool`, `lisaTool`, `gridTool`, `bufferTool`, `centroidTool`, `dissolveTool` and other geo tools | Run CPU geo-analysis (GeoDa) on retained rows; resident datasets have no equivalent route. Use `runAnalysis` recipes. |
| `saveDataToMap` without `url` | Persists geo-tool outputs; there are no geo-tool outputs here. |

## Host-defined operations with no published tool

`setFilter`, `runAggregation`, `runAnalysis`, `exportData`, `setMapSplit` and `setLayerStyle` remain
the host vocabulary; the published package has no filter, group-by, split-map or export tool.
`setFilter` and `runAggregation` map to `createOrUpdateFilter` and the Group By middleware.

## Limits

Nothing here runs a model or the published package. The tool schemas were read statically and are
alpha. Browser behavior of `mapStyleChange` and label passthrough is untested.
