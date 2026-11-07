# County rate-mapping showcase lane

## Existing evidence and source

`rate-smoothing` uses the archived `us-county-births` table aligned by FIPS to 3,109 contiguous-US county-equivalent polygons. The source manifest documents 2021–2023 live births by maternal residence and the sum of Census PEP July 1 estimates of women aged 15–44 as estimated woman-years at risk. It also records that the latest birth year is partly estimated and that the exposure is model-based. See `examples/deck/spatial-analysis-showcase/public/data/us-county-births/manifest.json` and the dataset descriptor `src/data/datasets/us-county-births.dataset.ts`.

The `us-counties` source manifest documents 9,102 Queen-contiguity edges, 5.9 mean neighbors, and two true Queen islands: Nantucket, Massachusetts and San Juan, Washington (`examples/deck/spatial-analysis-showcase/public/data/us-counties/manifest.json`). The scene computes the current graph's no-neighbor rows from its GPU CSR offsets and reports the names only when the weights graph has no capacity overflow. It does not construct substitute island rows.

## Added GPU outputs

- `addExcessRiskRecipe` receives the source `births` and `womenYears` buffers and returns observed / expected births at the pooled births-per-woman-year rate. The output is dimensionless and displayed as a ratio centered on 1. It is descriptive; it is not a p-value or significance test.
- `addSpatialRateRecipe` receives those same event and exposure buffers plus the exact Queen/Rook weights from `addRateClusterMapRecipe`. The output is `1000 * sum(events in focal + binary neighbors) / sum(exposure in focal + binary neighbors)`, not an average of county-specific rates. It is shown in births per 1,000 woman-years.
- Two recipe outputs run on the same GPU graph: `islandPolicy: 'undefined'` and `islandPolicy: 'own-rate'`. The first returns NaN for rows with no included nonself neighbors; the second uses the focal row's own count and exposure. The scene compares finite/NaN status from both GPU outputs on the actual rows flagged as isolates. Existing raw, global empirical-Bayes, spatial empirical-Bayes, standardized, and inference outputs remain as before.

Implementation is in `src/scenes/statistics/rate-smoothing.{scene,compute}.ts`. No new dataset or scene metadata is needed.
