// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {MeuseNaturalNeighborOptions} from './meuse-natural-neighbor.compute';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';

export default defineScene<MeuseNaturalNeighborOptions>({
  id: 'meuse-natural-neighbor',
  title: 'Meuse natural-neighbor interpolation: every site indexed',
  chapter: 'interpolation',
  order: 3,
  summary:
    'An all-site GPU Delaunay tessellation supplies query seeds and triangle adjacency for exact bounded Sibson interpolation across every original Meuse grid location.',
  contributors: ['GPUDelaunayTessellation', 'GPUNaturalNeighbor'],
  datasets: [
    {
      id: 'meuse-validation',
      role: '155 measured topsoil samples, 3,103 grid queries and river outline'
    }
  ],
  initialView: {longitude: 5.74314, latitude: 50.97424, zoom: 12.4},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Meuse natural-neighbor interpolation',
      subtitle: 'Sibson area weights · projected metre coordinates'
    },
    scaleBar: {units: 'metric'},
    credit: 'R package sp 2.2-4 · Meuse data',
    caveat:
      'Every prediction uses all 155 measured sites. Queries outside the convex hull are left blank; the 64-neighbor cavity bound can also report capacity failures.'
  },
  options: [
    {
      kind: 'select',
      id: 'response',
      label: 'Metal',
      group: 'Data',
      apply: 'param',
      default: 'zinc',
      display: 'segmented',
      help: 'Natural-neighbor weights are geometric; only the measured log-metal value at each site changes.',
      options: [
        {value: 'zinc', label: 'Zinc'},
        {value: 'copper', label: 'Copper'},
        {value: 'lead', label: 'Lead'},
        {value: 'cadmium', label: 'Cadmium'}
      ]
    }
  ],
  story: [
    {
      id: 'indexed-sites',
      title: 'Start from the measured sampling design',
      headline: '155 topsoil samples anchor the 3,103 source query locations',
      textAlternative:
        'A river outline, 155 colored metal samples and 3,103 natural-neighbor query predictions cover the original Meuse sampling area.',
      body: 'The field samples are 15 m × 15 m topsoil composites represented by points. This scene uses the original projected Dutch RD coordinates for distance calculations and the full 3,103-point prediction grid from the `sp` package. **Metal** changes only the measured log response; the sample coordinates and query set stay fixed.',
      controls: ['response'],
      readouts: ['sampleCount', 'queryCount', 'triangulation'],
      evidence: 'All 155 observed field sites and all 3,103 original query locations are included.'
    },
    {
      id: 'complete-delaunay',
      title: 'Triangulate all observed sites',
      headline: 'A complete 2D Delaunay mesh provides the spatial index',
      textAlternative:
        'The triangulation status reports the used triangle count, exact required count, capacity overflow and invalid-site totals.',
      body: '**`GPUDelaunayTessellation`** builds a planar triangulation from all 155 unique field sites. Its bounded output holds 305 triangles, the general-position maximum for this sample count; the contributor reports the exact required count and overflow separately. The next stages are gated on complete output, so a truncated mesh cannot silently seed a natural-neighbor calculation.',
      controls: ['response'],
      readouts: ['triangulation'],
      caveat:
        'Delaunay topology is a geometric index. The triangulation itself does not model metal concentration or uncertainty.'
    },
    {
      id: 'query-seeds',
      title: 'Attach each query to its containing triangle',
      headline: 'Two GPU passes provide adjacency and a local cavity seed',
      textAlternative:
        'Every query receives an index seed only when the Delaunay output is complete and the query lies inside a containing triangle.',
      body: 'The contributor emits triangle vertices, while **GPU adapter passes** find reciprocal shared-edge neighbors and the first containing triangle for each query. Queries outside the convex hull get an explicit invalid seed. **`GPUNaturalNeighbor`** traverses from that seed rather than comparing every query with every site; the adapter checks `requiredCount`, used `count` and `overflow` before allowing any seed.',
      controls: ['response'],
      readouts: ['triangulation'],
      evidence:
        'The 64-neighbor limit is on one query cavity; the complete sample set remains indexed.'
    },
    {
      id: 'sibson-weights',
      title: 'Interpolate with Sibson area weights',
      headline: 'Natural neighbors receive weight from stolen Voronoi area',
      textAlternative:
        'A cividis surface shows valid log-metal predictions at original grid locations alongside measured samples.',
      body: 'For each query inside the sample hull, the indexed cavity finds the sites whose Voronoi regions shrink when the query is inserted. **Sibson weights** are those stolen areas divided by the new query-cell area. The weighted sum reproduces observed values at sample hits and gives a smooth local interpolant without fitting a covariance model.',
      controls: ['response'],
      readouts: ['naturalStatus'],
      caveat:
        'Natural-neighbor interpolation is deterministic and local; it does not supply a conditional prediction variance.'
    },
    {
      id: 'hull-and-cavity',
      title: 'Read the interpolation footprint',
      headline: 'Blank queries and bounded cavities stay visible as failures',
      textAlternative:
        'The cividis map shows the log-metal natural-neighbor estimate at valid query locations, with measured sites and the Meuse river outline overlaid.',
      body: 'Sibson weights interpolate between observed values using the area stolen from neighboring Voronoi cells. The map leaves outside-hull and invalid queries transparent. The readout separates outside-hull queries from degenerate topology and cavity-capacity failures, which indicate the 64-neighbor local work limit was reached. Outside-hull counts are interpreted only when the Delaunay status confirms a complete mesh. Hover near a query to inspect its back-transformed geometric estimate; natural-neighbor interpolation has no conditional prediction variance.',
      controls: ['response'],
      readouts: ['naturalStatus', 'naturalDiagnostics'],
      caveat:
        'Exponentiating an interpolated natural-log concentration gives a geometric estimate; it is not a conditional median or expected concentration without a stochastic model. Cadmium zeros in the published source were replaced by 0.2 mg/kg.'
    }
  ],
  legends: state => [
    {
      kind: 'ramp',
      title: `${state.response.charAt(0).toUpperCase()}${state.response.slice(1)} · log concentration`,
      unit: 'log mg/kg',
      ramp: 'cividis',
      extent: MEUSE_LOG_RESPONSE_RANGES[state.response]
    },
    {
      kind: 'categories',
      title: 'Map features',
      entries: [
        {color: [49, 113, 151, 230], label: 'Meuse river outline'},
        {color: [16, 18, 22, 255], label: 'Measured sample outline'}
      ]
    }
  ],
  readouts: [
    {id: 'sampleCount', label: 'Measured field sites', format: 'integer'},
    {id: 'queryCount', label: 'Grid queries', format: 'integer'},
    {id: 'triangulation', label: 'Delaunay capacity status'},
    {id: 'naturalStatus', label: 'Natural-neighbor query status'},
    {id: 'naturalDiagnostics', label: 'Prediction and weight diagnostics'}
  ],
  about: {
    what: 'Indexed Sibson natural-neighbor interpolation uses the complete Meuse sample set and original source query grid.',
    why: 'It makes the Delaunay adjacency, containing-triangle seed, convex-hull mask and bounded local cavity visible as parts of one interpolation method.',
    howToRead:
      'A query is colored only when triangulation completed and the natural-neighbor cavity succeeded. The color shows interpolated log concentration; samples use the same response range. The prediction summary reports the GPU-computed range, maximum neighbor count and maximum weight-sum error.'
  },
  create: async ctx =>
    (await import('./meuse-natural-neighbor.compute')).createMeuseNaturalNeighbor(ctx)
});
