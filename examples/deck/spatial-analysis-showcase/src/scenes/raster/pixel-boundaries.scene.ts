// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {PixelBoundariesOptions} from './pixel-boundaries.compute';

export default defineScene<PixelBoundariesOptions>({
  id: 'pixel-boundaries',
  title: 'What happens when a pixel crosses a boundary?',
  chapter: 'raster',
  order: 10,
  summary:
    'Four analyst-drawn queries cross a 60 m grid of Sentinel-2 reflectance averaged from native 10/20 m bands. Exact fractional coverage, reflectance distributions and conservative remapping show how boundary placement and missing support change the answer.',
  contributors: [
    'GPURasterFractionalCoverage',
    'GPURasterFractionalZonalDistribution',
    'GPURasterFractionalZonalStatistics',
    'GPURasterFractionalCategoricalStatistics',
    'GPURasterConservativeRemapping'
  ],
  datasets: [
    {
      id: 'dixie-spectral',
      role: 'Sentinel-2 calibrated NIR reflectance and strict per-date validity'
    }
  ],
  initialView: {longitude: -121.0005, latitude: 40.1759, zoom: 12.2},
  basemap: ground('relief'),
  annotations: [
    {kind: 'point', coordinate: [-120.95, 40.14], text: 'Greenville', rank: 'context'},
    {kind: 'area', coordinate: [-121.23, 40.31], text: 'Chester', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'A pixel crosses the boundary',
      subtitle: 'Dixie Fire landscape · Sentinel-2 NIR at 60 m'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data 2021',
    caveat:
      'Outlined polygons are analyst-defined queries, not observed burn perimeters or administrative boundaries. Reflectance bins are numeric NIR intervals, not land-cover or burn-severity classes. Missing support stays missing.'
  },
  options: [
    {
      kind: 'select',
      id: 'sourceDate',
      label: 'Source acquisition',
      group: 'Observation',
      apply: 'param',
      default: 'after',
      help: 'Switch the same query and remapping grids between the pinned July 13 ignition-day and September 21 post-ignition Sentinel-2 scenes.',
      options: [
        {value: 'before', label: '13 Jul 2021 · ignition day'},
        {value: 'after', label: '21 Sep 2021 · post-ignition'}
      ]
    },
    {
      kind: 'select',
      id: 'mapLayer',
      label: 'Map shows',
      group: 'Evidence',
      apply: 'param',
      default: 'nir',
      help: 'Every layer is computed from this crop or from contributor outputs.',
      options: [
        {value: 'nir', label: 'NIR reflectance'},
        {value: 'zone-coverage', label: 'Selected query coverage'},
        {value: 'categorical', label: 'NIR reflectance bins'},
        {value: 'remapped-values', label: '240 m remapped reflectance'},
        {value: 'support-fraction', label: 'Valid support fraction'}
      ]
    },
    {
      kind: 'select',
      id: 'selectedZone',
      label: 'Query geometry',
      group: 'Fractional zones',
      apply: 'param',
      default: '2',
      help: 'Selects one of four analyst-drawn query shapes; the third contains an interior hole and the fourth overlaps other zones.',
      options: [
        {value: '0', label: 'Rectangle'},
        {value: '1', label: 'Oblique transect'},
        {value: '2', label: 'Polygon with hole'},
        {value: '3', label: 'Overlapping query'}
      ]
    },
    {
      kind: 'select',
      id: 'observationSupport',
      label: 'Statistical observation support',
      group: 'Distribution',
      apply: 'param',
      default: 'selected-date',
      help: 'Paired support gives weight zero to cells missing on either acquisition. It changes weighted statistics, not source-valid area, categorical summaries or remapping. These are support weights, not estimated confidence.',
      options: [
        {value: 'selected-date', label: 'All valid cells on selected date'},
        {value: 'paired-dates', label: 'Only cells valid on both dates'}
      ]
    },
    {
      kind: 'select',
      id: 'quantileMethod',
      label: 'Weighted quantiles',
      group: 'Distribution',
      apply: 'param',
      default: 'exactextract',
      help: 'The GPU evaluates both exactextract-style interpolation and inverse weighted CDF; the selected convention determines reported quantiles.',
      options: [
        {value: 'exactextract', label: 'Exactextract interpolation'},
        {value: 'inverse-cdf', label: 'Inverse weighted CDF'}
      ]
    },
    {
      kind: 'slider',
      id: 'probability',
      label: 'Selected probability',
      group: 'Distribution',
      apply: 'param',
      min: 0.05,
      max: 0.95,
      step: 0.05,
      default: 0.5,
      help: 'Requests this weighted quantile alongside the 25th, 50th and 75th percentiles.'
    },
    {
      kind: 'select',
      id: 'gridPhase',
      label: 'Destination grid phase',
      group: 'Conservative remapping',
      apply: 'param',
      default: '0',
      help: 'Moves the 240 m destination grid by 0 or 120 m in both axes. Separate overlap-weight tables are prepared for each phase.',
      options: [
        {value: '0', label: 'Aligned · 0 m'},
        {value: '120', label: 'Offset · 120 m'}
      ]
    },
    {
      kind: 'select',
      id: 'normalization',
      label: 'Remapping normalization',
      group: 'Conservative remapping',
      apply: 'param',
      default: 'destination-area',
      help: 'Destination-area divides by full destination area; covered-area divides by valid support; extensive distributes source reflectance·m² cell totals.',
      options: [
        {value: 'destination-area', label: 'Destination area'},
        {value: 'covered-area', label: 'Valid covered area'},
        {value: 'extensive', label: 'Extensive reflectance·m²'}
      ]
    }
  ],
  readouts: [
    {id: 'coverage', label: 'Coverage tuples and validity'},
    {id: 'zoneSummary', label: 'Selected fractional summary'},
    {
      id: 'quantityContract',
      label: 'Intensive values and extensive cell integrals',
      help: 'The quantity reducer sums fractions sequentially in float32. Displayed decimals are computational diagnostics, not physical measurement precision; small area rounding differs from the distribution reducer.'
    },
    {id: 'distributionChart', label: 'NIR quantile curve', kind: 'chart'},
    {id: 'categories', label: 'Reflectance-bin areas and missing support'},
    {id: 'remapping', label: 'Remapping support and integral'},
    {id: 'pixel', label: 'Clicked source pixel'}
  ],
  legends: state =>
    state.mapLayer === 'zone-coverage'
      ? [
          {
            kind: 'ramp',
            title: 'Fraction of the source cell inside query',
            ramp: 'cividis',
            extent: [0, 1],
            unit: 'fraction'
          }
        ]
      : state.mapLayer === 'categorical'
        ? [
            {
              kind: 'categories',
              title: 'NIR reflectance intervals',
              entries: [
                {label: '< 0.10', color: [50, 78, 125, 255]},
                {label: '0.10–0.20', color: [66, 143, 150, 255]},
                {label: '0.20–0.30', color: [224, 190, 91, 255]},
                {label: '≥ 0.30', color: [197, 103, 66, 255]}
              ]
            }
          ]
        : [
            {
              kind: 'ramp',
              title:
                state.mapLayer === 'support-fraction'
                  ? 'Valid destination-area support'
                  : state.normalization === 'extensive' && state.mapLayer === 'remapped-values'
                    ? 'Extensive NIR integral'
                    : 'NIR reflectance',
              ramp: 'cividis',
              extent:
                state.mapLayer === 'support-fraction'
                  ? [0, 1]
                  : state.normalization === 'extensive' && state.mapLayer === 'remapped-values'
                    ? [0, 36000]
                    : [0, 0.6],
              unit:
                state.mapLayer === 'support-fraction'
                  ? 'fraction'
                  : state.normalization === 'extensive' && state.mapLayer === 'remapped-values'
                    ? 'reflectance·m²'
                    : 'reflectance'
            }
          ],
  story: [
    {
      id: 'native-pixels',
      headline: 'A raster cell is an area, not a point',
      textAlternative:
        'The map shows a bounded Sentinel-2 near-infrared crop with analyst query outlines.',
      title: 'Start with the 60 m grid',
      body: 'This 128 × 128 crop contains calibrated NIR reflectance averaged from Sentinel-2 native 10/20 m bands onto a 60 m UTM grid. Switch between the July 13 ignition-day and September 21 scenes to see how actual valid support changes.',
      evidence: '**{{coverage}}**',
      caveat:
        'The acquisitions are observations on different dates; their difference is not isolated fire impact. July 13 is the ignition day, so this scene is not described as strictly pre-fire. Source validity follows the supplied scene mask.',
      options: {sourceDate: 'after', mapLayer: 'nir', selectedZone: '2'},
      controls: ['sourceDate', 'mapLayer'],
      readouts: ['coverage', 'pixel']
    },
    {
      id: 'analyst-queries',
      headline: 'Four query shapes ask different spatial questions',
      textAlternative:
        'Exact GPU clipping measures the fraction of each source cell within each query.',
      title: 'Follow a rectangle, transect, hole and overlap',
      body: 'The polygons are analyst-defined queries in local projected metres. They are not mapped fire perimeters or administrative areas. **Query geometry** selects the shape; the third polygon has a hole, while the last overlaps neighboring queries. Exact planar clipping reports tuple capacity and geometry validity.',
      evidence: '**{{coverage}}**',
      caveat:
        'Clipping is planar within this 7.68 km crop. Float32 rounding can add tiny boundary tuples; the 1 m² diagnostic threshold is below this geometry’s smallest exact overlap (5.04 m²) and never removes computed weights.',
      options: {mapLayer: 'zone-coverage', selectedZone: '2'},
      controls: ['selectedZone', 'mapLayer'],
      readouts: ['coverage']
    },
    {
      id: 'weighted-values',
      headline: 'Partial pixels contribute in proportion to their covered area',
      textAlternative:
        'Fractional zonal statistics report NIR means, variance and weighted quantiles.',
      title: 'Compare distributions rather than cell counts',
      body: 'A cell that is 20% inside a query contributes weight 0.2. The GPU reports mean, population and reliability-weighted sample variance, effective support, valid area and quantiles. Switch quantile conventions and add a selected probability to the displayed quartiles. Paired-date support assigns zero observation weight where either date is missing, so a date comparison can use the same spatial support. Explicit cell areas are 3,600 projected m²; area and statistical weights remain separate.',
      evidence: '**{{zoneSummary}}**',
      caveat:
        'Effective support is a weighted diagnostic, not a count of independent physical samples. Source-valid area includes zero-weight cells; statistical support area reports only positive-weight observations. Neither is a confidence or causal-effect estimate.',
      options: {
        mapLayer: 'nir',
        quantileMethod: 'exactextract',
        probability: 0.5,
        selectedZone: '2'
      },
      controls: [
        'quantileMethod',
        'probability',
        'selectedZone',
        'sourceDate',
        'observationSupport'
      ],
      readouts: ['zoneSummary', 'distributionChart', 'quantityContract']
    },
    {
      id: 'reflectance-bins',
      headline: 'Numeric reflectance bins preserve missing area',
      textAlternative:
        'Categorical summaries divide continuous NIR into four explicit reflectance intervals.',
      title: 'Use intervals, not invented land-cover labels',
      body: 'The bins are **< 0.10**, **0.10–0.20**, **0.20–0.30** and **≥ 0.30** reflectance. The GPU reports class areas, proportions, majority interval, least-present interval and missing support for the selected fractional query.',
      evidence: '**{{categories}}**',
      caveat:
        'Absent intervals do not compete for least-present status. Equal-area ties select the highest majority interval and the lowest present minority interval. These NIR intervals are descriptive bins only. They are not vegetation, burn-severity or land-cover classes.',
      options: {mapLayer: 'categorical'},
      controls: ['sourceDate', 'selectedZone', 'mapLayer'],
      readouts: ['categories']
    },
    {
      id: 'new-grid',
      headline: 'A new grid changes the support around every edge',
      textAlternative:
        'Conservative remapping compares a 240 m grid at two phases and keeps valid-area support explicit.',
      title: 'Shift the destination cells and test conservation',
      body: 'The crop is converted explicitly to south-first Cartesian cells before remapping. A 33 × 33 destination grid at 240 m covers the 7,680 m source window with either a 0 m or 120 m phase. All three normalization choices use actual GPU overlap weights; their integrated results recover reflectance·m² over valid support.',
      evidence: '**{{remapping}}**',
      caveat:
        'The extensive result is an area-weighted reflectance integral, not physical mass. Each grid phase prepares its own overlap weights before remapping reuses them.',
      options: {mapLayer: 'remapped-values', gridPhase: '0', normalization: 'destination-area'},
      controls: ['gridPhase', 'normalization', 'sourceDate', 'mapLayer'],
      readouts: ['remapping']
    }
  ],
  create: async ctx => (await import('./pixel-boundaries.compute')).createPixelBoundaries(ctx)
});
