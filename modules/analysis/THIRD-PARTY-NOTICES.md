# Analysis source licences and notices

The package as a whole is distributed under `MIT AND Apache-2.0 AND BSD-3-Clause`.
Individual files keep their SPDX expressions; this package expression does not replace them.
Files with `MIT` are covered by [LICENSE](LICENSE). The remaining shipped source files are
listed below. Their compiled ESM, CommonJS and declaration counterparts retain the same provenance.
Dependencies and peers retain their own licences and notices in their respective packages.

| Source path | SPDX expression | Attribution and licence text |
| --- | --- | --- |
| `src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-boundary-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-index-tables.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-index-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-measures-fp64-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-indexing/h3-measures-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-topology/h3-local-ij-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-topology/h3-neighbor-tables.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-spatial-analysis/cell-topology/h3-neighbor-wgsl.ts` | `MIT AND Apache-2.0` | H3; see below |
| `src/gpu-terrain/relief-visualization/gpu-local-dominance.ts` | `MIT AND Apache-2.0` | RVT; see below |
| `src/gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts` | `MIT AND Apache-2.0` | RVT; see below |
| `src/gpu-terrain/relief-visualization/gpu-simple-local-relief.ts` | `MIT AND Apache-2.0` | RVT; see below |
| `src/gpu-spatial-analysis/point-density/point-density-d3-hexagon.ts` | `BSD-3-Clause` | D3 hexbin; see below |
| `src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation.ts` | `BSD-3-Clause` | scikit-learn; see below |
| `src/gpu-spatial-analysis/spatial-clustering/gpu-optics-xi.ts` | `BSD-3-Clause` | scikit-learn; see below |
| `src/raster-analysis/raster-contrast-equalization/contrast-equalization-wgsl.ts` | `MIT AND BSD-3-Clause` | scikit-image; see below |

## H3

H3 tables and algorithms are derived or adapted from Uber's H3 and h3-js, including generated
tables and GPU/CPU adaptations described in the source files. Original source:
[H3](https://github.com/uber/h3) and [h3-js](https://github.com/uber/h3-js).

- Copyright 2016-2024 Uber Technologies, Inc.
- Copyright 2020-2021 Uber Technologies, Inc.
- Copyright (c) vis.gl contributors

The [Apache-2.0 licence text](LICENSES/Apache-2.0.txt) and
[h3-js upstream NOTICE](LICENSES/H3-NOTICE.txt) accompany these adaptations.

## Relief Visualization Toolbox (RVT)

Terrain relief formulas are ported or adapted from the Apache-2.0
[Relief Visualization Toolbox](https://github.com/EarthObservation/RVT_py), with the changes and
numerical differences documented in each source file.

- Copyright (c) Relief Visualization Toolbox (RVT) contributors
- Copyright (c) vis.gl contributors

See the [Apache-2.0 licence text](LICENSES/Apache-2.0.txt).

## BSD adaptations

The [BSD licence texts](LICENSES/BSD-3-Clause.txt) retain the component-specific notices,
redistribution conditions and disclaimers:

- D3 hexbin: Copyright (c) 2012-2016 Mike Bostock. Adapted hexagonal bin selection from
  [d3-hexbin](https://github.com/d3/d3-hexbin).
- OPTICS Xi extraction: Copyright (c) 2007-2024 The scikit-learn developers. Adapted from
  scikit-learn 1.7.2 `_xi_cluster` and predecessor correction. The original notice is also
  retained in `src/gpu-spatial-analysis/spatial-clustering/optics-xi-license.txt`.
- Histogram clipping: Copyright (c) 2009-2022 the scikit-image team. Adapted from
  scikit-image 0.24 `clip_histogram` redistribution.
- Adaptations: Copyright (c) vis.gl contributors.
