// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {TiledObservationsOptions} from './tiled-observations.compute';
import {TILED_TOPOLOGY_PALETTE} from './tiled-observation-topology-style';

export default defineScene<TiledObservationsOptions>({
  id: 'tiled-observations',
  title: 'What happens at tile seams?',
  chapter: 'raster',
  order: 18,
  summary:
    'A bounded strict-QA Sentinel reflectance experiment compares tile-local and reconciled regions before returning to filtering, global populations, overviews and resampling.',
  contributors: [
    'GPURasterTileReader',
    'GPURasterTileCache',
    'GPURasterTileHaloAssembler',
    'GPURasterTileHaloFill',
    'GPURasterGaussianBlur',
    'GPURasterTileCoreExtract',
    'GPURasterConnectedComponents',
    'GPURasterDenseComponents',
    'GPURasterRegionMeasurements',
    'GPURasterCrossTileComponents',
    'addRasterRegionGraphTiles',
    'GPURasterGlobalInitialize',
    'GPURasterGlobalStatisticsMerge',
    'GPURasterGlobalHistogramMerge',
    'GPURasterGlobalPercentile',
    'GPURasterOverview',
    'GPURasterCategoricalOverview',
    'GPURasterTileReprojection'
  ],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Calibrated Sentinel-2 NIR reflectance and strict native-pixel QA masks'
    }
  ],
  initialView: {longitude: -120.999, latitude: 40.176, zoom: 12.5, pitch: 0},
  basemap: ground('relief'),
  furniture: {
    title: {
      title: 'Tile seams and support',
      subtitle: '60 m Sentinel-2 NIR · Dixie Fire study area'
    },
    scaleBar: {units: 'metric'},
    credit:
      'Contains modified Copernicus Sentinel data 2021 · Sentinel-2 L2A via AWS Open Data Registry',
    caveat:
      'July 13 is ignition day, not a pre-fire baseline. Strict QA reduces contamination but does not guarantee cloud-free observations.'
  },
  options: [
    {
      kind: 'select',
      id: 'date',
      label: 'Acquisition',
      group: 'Source',
      apply: 'compile',
      default: 'july',
      options: [
        {value: 'july', label: '2021-07-13 · ignition day'},
        {value: 'september', label: '2021-09-21 · later scene'}
      ],
      help: 'Selects the calibrated reflectance array and its strict native-pixel QA mask.'
    },
    {
      kind: 'select',
      id: 'step',
      label: 'Investigation',
      group: 'Story',
      apply: 'param',
      default: 'ownership',
      options: [
        {value: 'ownership', label: 'Tile ownership'},
        {value: 'seams', label: 'Halo and seams'},
        {value: 'population', label: 'Global population'},
        {value: 'histogram', label: 'Histogram replay'},
        {value: 'overview', label: 'Weighted overviews'},
        {value: 'resampling', label: 'Tiled resampling'}
      ],
      help: 'Jumps between the six linked views in this processing investigation.'
    },
    {
      kind: 'select',
      id: 'mapLayer',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'blur',
      options: [
        {value: 'source', label: 'Source reflectance'},
        {value: 'topology', label: 'Threshold regions · 64² focus'},
        {value: 'blur', label: 'Halo-processed cores'},
        {value: 'isolated', label: 'Isolated-core blur'},
        {value: 'difference', label: 'Halo minus isolated'},
        {value: 'wholecropDifference', label: 'Tiled minus whole-crop GPU blur'},
        {value: 'overview', label: 'Weighted overview'},
        {value: 'support', label: 'Overview support'},
        {value: 'categorical', label: 'NIR-bin categorical overview'},
        {value: 'resampled', label: 'Half-cell-shift tiled resampling'},
        {value: 'resampleDifference', label: 'Tiled minus untiled resampling'}
      ],
      help: 'Chooses a computed raster output; the difference map isolates cross-core context effects.'
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Gaussian radius',
      group: 'Neighborhood',
      apply: 'compile',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      unit: 'pixels',
      format: value => `${value} px · ${value * 60} m`,
      help: 'Sets the Gaussian receptive field and halo width in 60 m cells.'
    },
    {
      kind: 'select',
      id: 'edgePolicy',
      label: 'Crop-edge border',
      group: 'Neighborhood',
      apply: 'compile',
      default: 'nodata',
      options: [
        {value: 'nodata', label: 'Outside is nodata'},
        {value: 'clamp', label: 'Clamp'},
        {value: 'reflect', label: 'Reflect'}
      ],
      help: 'Applies the same true crop-edge rule to the halo reference and tile calculations.'
    },
    {
      kind: 'select',
      id: 'noDataPolicy',
      label: 'Missing neighbors',
      group: 'Neighborhood',
      apply: 'compile',
      default: 'ignore-renormalize',
      options: [
        {value: 'ignore-renormalize', label: 'Ignore and renormalize'},
        {value: 'propagate', label: 'Propagate missing'}
      ],
      help: 'Invalid center cells always remain missing. For other neighbors, propagate removes the result; ignore omits their weights and renormalizes.'
    },
    {
      kind: 'select',
      id: 'binCount',
      label: 'Histogram bins',
      group: 'Global distribution',
      apply: 'compile',
      default: '32',
      options: [
        {value: '32', label: '32 bins'},
        {value: '128', label: '128 bins'}
      ],
      help: 'Changes the GPU histogram resolution; quantiles remain approximate within the selected bin.'
    },
    {
      kind: 'select',
      id: 'percentile',
      label: 'Global percentile',
      group: 'Global distribution',
      apply: 'compile',
      default: '0.5',
      options: [
        {value: '0', label: 'Minimum'},
        {value: '0.1', label: '10th'},
        {value: '0.5', label: '50th'},
        {value: '0.9', label: '90th'},
        {value: '1', label: 'Maximum'}
      ],
      help: 'Rebuilds the GPU percentile query for this probability; endpoints use the exact observed extrema.'
    },
    {
      kind: 'select',
      id: 'overviewScale',
      label: 'Overview reduction',
      group: 'Overviews',
      apply: 'compile',
      default: '2',
      options: [
        {value: '2', label: '2 × 2'},
        {value: '4', label: '4 × 4'}
      ],
      help: 'Computes mean reflectance and exact valid-cell support for each overview pixel.'
    },
    {
      kind: 'select',
      id: 'categoricalPolicy',
      label: 'NIR-bin overview policy',
      group: 'Overviews',
      apply: 'compile',
      default: 'mode',
      options: [
        {value: 'nearest', label: 'Nearest center'},
        {value: 'mode', label: 'Mode'}
      ],
      help: 'Selects the upper-left nearest-center bin or most frequent valid integer bin; mode ties choose the lower bin.'
    },
    {
      kind: 'select',
      id: 'resampling',
      label: 'Tile resampling',
      group: 'Resampling',
      apply: 'compile',
      default: 'bilinear-renormalize',
      options: [
        {value: 'nearest', label: 'Nearest'},
        {value: 'bilinear-strict', label: 'Bilinear · strict'},
        {value: 'bilinear-renormalize', label: 'Bilinear · available taps'}
      ],
      help: 'Controls interpolation support in the half-cell-shift alignment experiment.'
    },
    {
      kind: 'toggle',
      id: 'omitEast',
      label: 'Remove east tiles',
      group: 'Resampling',
      apply: 'compile',
      default: false,
      help: 'Makes the eastern half absent so strict and renormalized support can be inspected.'
    }
  ],
  story: [
    {
      id: 'ownership',
      title: 'Can tile-owned regions be joined?',
      headline: 'Four local cores reconcile into one 8-connected population',
      body: 'The central 64 × 64 focus is split into four owned 32 × 32 cores. GPU components are labeled inside each core, then reconciled across seams for calibrated NIR below or above 0.2. **Map shows** colors those stable component IDs; **Acquisition** changes the measured source date. The white square marks the focus and its inner lines mark ownership seams.',
      evidence: '**{{topologyComponents}}** · **{{topologyPopulation}}** · **{{regionAdjacency}}**',
      textAlternative:
        'A categorical map shows globally reconciled component identities in a central 64 by 64 focus divided into four 32 by 32 cores; readouts compare local components with reconciled components and report region contacts.',
      caveat:
        'The 0.2 split is a reflectance threshold, not a validated land-cover or burn class. Components use eight-neighbor connectivity; adjacency contacts count four-neighbor shared edges. The focus is a bounded interior subwindow, not a complete scene.',
      options: {step: 'ownership', mapLayer: 'topology'},
      controls: ['date', 'mapLayer'],
      readouts: ['topologyComponents', 'topologyPopulation', 'regionAdjacency', 'cropSupport']
    },
    {
      id: 'seams',
      title: 'What changes when a filter crosses a seam?',
      headline: 'A filter radius sets the context a tile must borrow',
      body: 'A Gaussian needs neighboring cells outside its output core. **Gaussian radius** below grows the required halo; compare the halo-processed map with the deliberately isolated-core result. A real crop edge still follows **Crop-edge border** below.',
      evidence:
        'Both comparisons are GPU-derived; **{{seamError}}** contrasts isolated cores and **{{wholecropSeamError}}** contrasts the untiled whole-crop reference.',
      textAlternative:
        'A diverging raster shows the difference between halo-aware filtering and filtering each core alone.',
      caveat:
        'The halo is context and never enters the owned-core population. A seam difference measures this filter on the selected crop, not a general quality guarantee.',
      options: {step: 'seams', mapLayer: 'wholecropDifference'},
      controls: ['radius', 'edgePolicy', 'noDataPolicy', 'mapLayer'],
      readouts: ['seamError', 'wholecropSeamError']
    },
    {
      id: 'population',
      title: 'Can tiles describe one population?',
      headline: 'Four valid cores contribute to one global source summary',
      body: 'Each owned source core contributes once to a persistent GPU accumulator. The displayed population and mean refer to valid NIR reflectance cells; halo cells do not count. **Acquisition** below swaps the strict-QA date.',
      evidence: '**{{validCells}}** valid owned cells · mean **{{globalMean}}**',
      textAlternative:
        'A calibrated NIR reflectance raster with missing strict-QA cells shown transparently.',
      caveat:
        'A sum and count are order-independent within floating-point rounding. They summarize this bounded crop, not the full acquisition.',
      options: {step: 'population'},
      controls: ['date'],
      readouts: ['validCells', 'globalMean']
    },
    {
      id: 'histogram',
      title: 'Why replay the histogram?',
      headline: 'A stable range is known before bins are merged',
      body: 'The first core pass fixes one global extent. A second pass bins each core against that final domain, and a percentile query reads the finished histogram. **Histogram bins** changes the approximation width; **Global percentile** chooses the requested probability.',
      evidence: '**{{percentile}}** · histogram counts include only valid core samples.',
      textAlternative: 'A histogram chart shows counts binned over the stable global range.',
      caveat:
        'Percentiles are bin-based estimates, except endpoints which use the observed extrema. The histogram is not an exact quantile index.',
      options: {step: 'histogram'},
      controls: ['binCount', 'percentile'],
      readouts: ['percentile', 'histogram']
    },
    {
      id: 'overview',
      title: 'What should a coarse pixel preserve?',
      headline: 'Weighted means retain missing-cell support',
      body: 'A mean alone loses how much source support contributed. The overview carries sums and valid counts to the next level; **Overview reduction** lets you compare a direct factor-four block with two factor-two aggregations.',
      evidence: '**{{overviewAgreement}}** · **{{categoricalOverview}}**',
      textAlternative:
        'A coarser raster summarizes reflectance and counts valid source cells per overview pixel.',
      caveat:
        'This crop has even dimensions. The experiment does not claim ragged overview edges were exercised here.',
      options: {step: 'overview', mapLayer: 'overview'},
      controls: ['overviewScale', 'categoricalPolicy', 'mapLayer'],
      readouts: ['overviewAgreement', 'categoricalOverview']
    },
    {
      id: 'resampling',
      title: 'How does a shifted grid cross tile edges?',
      headline: 'Interpolation needs adjacent-tile samples for shifted cells',
      body: 'A half-cell shift makes interpolation tap across core boundaries. **Tile resampling** controls nearest or bilinear support; **Remove east tiles** exposes missing coverage on the same identity-projection lattice.',
      evidence: '**{{resampleSupport}}** destination cells retain support.',
      textAlternative:
        'A shifted-grid map compares how neighboring tiles contribute to destination samples.',
      caveat:
        'This is a local grid-alignment experiment. It does not transform CRS, fetch missing tiles, or measure remote source performance.',
      options: {step: 'resampling', mapLayer: 'resampled'},
      controls: ['resampling', 'omitEast', 'mapLayer'],
      readouts: ['resampleSupport']
    }
  ],
  legends: state =>
    state.mapLayer === 'topology'
      ? [
          {
            kind: 'categories',
            title: 'Reconciled component identity',
            entries: TILED_TOPOLOGY_PALETTE.map((color, index) => ({
              color,
              label: `Palette slot ${index}`
            })),
            note: 'GPU-derived global IDs 0–127 for NIR <0.2 and 128–255 for NIR ≥0.2; colors cycle modulo 15 and are identity markers, not magnitude.'
          }
        ]
      : state.mapLayer === 'categorical'
        ? [
            {
              kind: 'categories',
              title: 'NIR reflectance bin',
              entries: [
                {color: [102, 194, 165, 255], label: '0 · <0.1'},
                {color: [252, 141, 98, 255], label: '1 · [0.1, 0.2)'},
                {color: [141, 160, 203, 255], label: '2 · [0.2, 0.3)'},
                {color: [231, 138, 195, 255], label: '3 · ≥0.3'}
              ],
              note: `${state.categoricalPolicy} aggregation; integer bin labels, not land cover.`
            }
          ]
        : [
            {
              kind: 'ramp',
              title:
                state.mapLayer === 'difference' ||
                state.mapLayer === 'wholecropDifference' ||
                state.mapLayer === 'resampleDifference'
                  ? 'GPU difference · NIR reflectance'
                  : state.mapLayer === 'support'
                    ? 'Valid source cells'
                    : 'NIR reflectance',
              ramp:
                state.mapLayer === 'difference' ||
                state.mapLayer === 'wholecropDifference' ||
                state.mapLayer === 'resampleDifference'
                  ? 'diverging'
                  : 'batlow',
              extent:
                state.mapLayer === 'difference' ||
                state.mapLayer === 'wholecropDifference' ||
                state.mapLayer === 'resampleDifference'
                  ? [-0.2, 0.2]
                  : [0, 0.5],
              ...(state.mapLayer === 'difference' ||
              state.mapLayer === 'wholecropDifference' ||
              state.mapLayer === 'resampleDifference'
                ? {midpoint: 0, midpointLabel: 'No difference'}
                : {}),
              unit: state.mapLayer === 'support' ? 'cells' : 'reflectance'
            }
          ],
  readouts: [
    {id: 'probe', label: 'Selected cell', help: 'Source reflectance at the clicked crop cell.'},
    {
      id: 'cropSupport',
      label: 'Crop and ownership',
      help: 'Bounded source crop and disjoint output core layout.'
    },
    {
      id: 'topologyComponents',
      label: 'Local and reconciled components',
      help: 'Eight-neighbor components before and after GPU seam reconciliation, gated on every local/global status and overflow flag.'
    },
    {
      id: 'topologyPopulation',
      label: 'Threshold population',
      help: 'GPU region-measurement populations for finite QA-valid NIR below and above 0.2.'
    },
    {
      id: 'regionAdjacency',
      label: 'Cross-class adjacency',
      help: 'Unique low/high region pairs and four-edge contact totals after component reconciliation and fixed-threshold region-graph convergence.'
    },
    {id: 'tileCache', label: 'Tile cache', help: 'Actual reader/cache residency and pin counters.'},
    {
      id: 'seamError',
      label: 'Isolated-core seam error',
      help: 'GPU difference between halo-aware tile output and deliberately isolated core processing.'
    },
    {
      id: 'wholecropSeamError',
      label: 'Tiled vs whole-crop Gaussian',
      help: 'GPU comparison against an untiled whole-crop Gaussian output.'
    },
    {id: 'validCells', label: 'Global valid population', format: 'integer'},
    {id: 'globalMean', label: 'Owned-core mean', format: 'decimal'},
    {id: 'sourceSumCheck', label: 'GPU sum vs CPU source reference'},
    {id: 'percentile', label: 'Histogram estimate'},
    {id: 'histogram', label: 'Global histogram', kind: 'chart'},
    {id: 'resampleSupport', label: 'Resampling support', format: 'integer'},
    {id: 'histogramDetails', label: 'Histogram domain, width, population and overflow'},
    {id: 'overviewAgreement', label: 'Direct/cascaded overview agreement'},
    {id: 'categoricalOverview', label: 'Categorical NIR-bin labels and support'}
  ],
  snippet: state =>
    `GPURasterTileHaloAssembler(radius=${state.radius}) → GPURasterGaussianBlur → GPURasterTileCoreExtract → GPU global merge and histogram replay`,
  about: {
    what: 'A small, reproducible processing study of calibrated Sentinel-2 near-infrared reflectance around the Dixie Fire.',
    why: 'Tile ownership and support rules affect seams, global summaries and coarse representations. The controls expose where those rules enter the computation.',
    howToRead:
      'The topology view colors GPU-reconciled component IDs inside a bounded central focus. Colors cycle and do not encode magnitude; white lines mark four owners. Other views show NIR reflectance, differences or support. Missing strict-QA cells remain missing.'
  },
  create: async ctx => (await import('./tiled-observations.compute')).createTiledObservations(ctx)
});
