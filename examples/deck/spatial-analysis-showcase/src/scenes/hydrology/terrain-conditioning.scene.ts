// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {TerrainConditioningOptions} from './terrain-conditioning.compute';

const PROFILE_STATUS_LEGEND = [
  {color: [170, 170, 170, 255], label: '0 excluded · elevation background'},
  {color: [245, 195, 66, 255], label: '1 complete'},
  {color: [201, 64, 64, 255], label: '2 unresolved walk'}
] as const;

export default defineScene<TerrainConditioningOptions>({
  id: 'terrain-conditioning',
  title: 'Where does the terrain hold water?',
  chapter: 'hydrology',
  order: 7,
  summary:
    'A block-averaged Grand Canyon DEM traces constrained pit breaching, nested depression stages, multiple-flow routing and terrain profiles on one shared GPU surface.',
  contributors: [
    'GPUTerrainBreach',
    'GPUTerrainDepressionHierarchy',
    'GPUTerrainFlowFractions',
    'GPUTerrainBranchingFlow',
    'GPUTerrainCatchmentSummary'
  ],
  datasets: [
    {id: 'grand-canyon-dem', role: '15 m Terrarium DEM, block-averaged to a 64 × 64 analysis grid'}
  ],
  initialView: {longitude: -112.12, latitude: 36.1, zoom: 10.7},
  basemap: ground('relief'),
  annotations: [
    {kind: 'point', coordinate: [-112.0953, 36.107], text: 'Phantom Ranch', rank: 'context'},
    {kind: 'area', coordinate: [-112.14, 36.055], text: 'South Rim', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'Where does the terrain hold water?',
      subtitle: 'Grand Canyon · block-averaged DEM'
    },
    scaleBar: {units: 'metric'},
    credit: 'Mapzen Terrarium elevation tiles; luma.gl terrain-processing contributors',
    caveat:
      'The 64 × 64 analysis grid is a block average; it smooths narrow channels and small sinks.'
  },
  options: [
    {
      kind: 'select',
      id: 'layer',
      label: 'Terrain product',
      group: 'GPU output',
      apply: 'param',
      default: 'source',
      help: 'Switch among the measured block-average source, canonical filled surface, constrained breach cuts, actual flow and distance products, depression ownership, and profile status.',
      options: [
        {value: 'source', label: 'Source elevation'},
        {value: 'filled', label: 'Filled routing surface'},
        {value: 'breach', label: 'Breach cut depth'},
        {value: 'fractions', label: 'Largest outgoing fraction'},
        {value: 'transport', label: 'Transport from high interior cell'},
        {value: 'distance', label: 'Expected distance to outlet'},
        {value: 'minimum-distance', label: 'Minimum route distance'},
        {value: 'maximum-distance', label: 'Maximum route distance'},
        {value: 'fractional-catchment', label: 'Fractional membership at low boundary outlet'},
        {value: 'hierarchy', label: 'Depression activation owner'},
        {value: 'profile', label: 'Catchment profile status'}
      ]
    },
    {
      kind: 'select',
      id: 'routing',
      label: 'Fraction method',
      group: 'Multiple-flow routing',
      apply: 'param',
      default: 'd-infinity',
      help: 'Select the actual GPU-computed D-infinity, Freeman MFD or Quinn MFD fraction map, all derived from the same filled surface and D8 fallback directions.',
      options: [
        {value: 'd-infinity', label: 'D-infinity'},
        {value: 'mfd-freeman', label: 'MFD · Freeman'},
        {value: 'mfd-quinn', label: 'MFD · Quinn'}
      ]
    },
    {
      kind: 'slider',
      id: 'stage',
      label: 'Absolute elevation query',
      group: 'Depression hierarchy',
      apply: 'param',
      min: 0,
      max: 8,
      step: 1,
      default: 3,
      format: value => `query ${value + 1}/9`,
      help: 'Chooses one of nine GPU queries spaced 200 m apart from the source minimum elevation. The readout reports its absolute elevation, sublevel-set area and geometric volume.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 1,
      step: 0.05,
      default: 0.92,
      help: 'Reveals the shaded terrain under the analysis grid.'
    }
  ],
  readouts: [
    {id: 'grid', label: 'Analysis grid and metric scale'},
    {id: 'pitCandidates', label: 'Constrained breach candidates'},
    {id: 'breachStatus', label: 'Breach outcomes'},
    {id: 'hierarchyStatus', label: 'Hierarchy capacity and stage query'},
    {id: 'hierarchyChart', label: 'Area and geometric volume by elevation', kind: 'chart'},
    {id: 'flowStatus', label: 'Canonical routing status'},
    {id: 'productStatus', label: 'Selected product status'},
    {id: 'catchmentSummary', label: 'Window summary and profile status'},
    {id: 'hypsometryChart', label: 'GPU catchment hypsometry', kind: 'chart'}
  ],
  legends: state =>
    state.layer === 'profile'
      ? [{kind: 'categories', title: 'Profile status', entries: PROFILE_STATUS_LEGEND}]
      : state.layer === 'hierarchy'
        ? [
            {
              kind: 'categories',
              title: 'Hierarchy owner (nominal IDs; colors repeat)',
              entries: [
                {color: [65, 105, 225, 255], label: 'Node ID class 0'},
                {color: [245, 195, 66, 255], label: 'Node ID class 1'},
                {color: [201, 64, 64, 255], label: 'Node ID class 2'},
                {color: [104, 170, 120, 255], label: 'Node ID class 3'}
              ]
            }
          ]
        : [
            {
              kind: 'ramp',
              title:
                state.layer === 'source' || state.layer === 'filled'
                  ? 'Elevation (m)'
                  : state.layer === 'breach'
                    ? 'Excavation depth (m)'
                    : state.layer === 'fractions'
                      ? 'Largest outgoing share'
                      : state.layer === 'fractional-catchment'
                        ? 'Fractional membership'
                        : state.layer === 'transport'
                          ? 'Transport mass'
                          : 'Route distance (m)',
              ramp: state.layer === 'source' || state.layer === 'filled' ? 'isolum' : 'cividis',
              extent:
                state.layer === 'breach'
                  ? [0, 25]
                  : state.layer === 'fractions' || state.layer === 'fractional-catchment'
                    ? [0, 1]
                    : state.layer === 'transport'
                      ? [0, 1]
                      : state.layer === 'distance' ||
                          state.layer === 'minimum-distance' ||
                          state.layer === 'maximum-distance'
                        ? [0, 50000]
                        : [700, 2700],
              unit:
                state.layer === 'breach' ||
                state.layer === 'source' ||
                state.layer === 'filled' ||
                state.layer === 'distance' ||
                state.layer === 'minimum-distance' ||
                state.layer === 'maximum-distance'
                  ? 'm'
                  : 'share'
            }
          ],
  story: [
    {
      id: 'surface',
      headline: 'A compact DEM keeps the landform in view',
      textAlternative:
        'The measured block-average elevation establishes the Grand Canyon analysis window.',
      title: 'Start with the measured terrain',
      body: 'The bundled Terrarium DEM is averaged in **32 × 32 source blocks** to make the 64 × 64 analysis grid. Broad canyon relief remains visible at this scale, while narrow channels and small closed depressions disappear in the averaging.',
      evidence: '**{{grid}}**',
      caveat:
        'The elevation mosaic has mixed source provenance and no single declared vertical datum.',
      options: {layer: 'source'},
      controls: ['layer'],
      readouts: ['grid']
    },
    {
      id: 'breach',
      headline: 'A breach is a bounded cut toward a lower outlet',
      textAlternative:
        'The cut-depth map marks cells changed along feasible routes from selected local minima.',
      title: 'Test a bounded set of candidate pits',
      body: 'Candidates are interior cells no higher than their eight immediate neighbors, ordered by elevation, with a fixed cap. For each one, a shallow local search tries to cut a short route toward a lower cell. The map shows the actual depth changed along successful paths; unresolved candidates stay visible in the summary.',
      evidence: '**{{pitCandidates}}** · **{{breachStatus}}**',
      caveat:
        'This bounded local experiment does not identify every depression or provide a complete hydrologic conditioning pass.',
      options: {layer: 'breach'},
      controls: ['layer'],
      readouts: ['pitCandidates', 'breachStatus']
    },
    {
      id: 'hierarchy',
      headline: 'Nested depressions are queried at absolute elevations',
      textAlternative:
        'The hierarchy map shows nominal node ownership; a chart shows root area and geometric volume by absolute elevation, with the selected query marked.',
      title: 'See how sublevel-set area grows with elevation',
      body: 'The depression hierarchy uses the block-averaged surface and treats the raster edge as explicit outlets. The slider selects one of nine absolute elevation queries spaced 200 m apart from the source minimum. The chart shows the GPU-computed root area and geometric sublevel-set volume as elevation rises; these are not connected flood inundation. The map colors identify nominal owners, not ranks.',
      evidence: '**{{hierarchyStatus}}**',
      caveat:
        'This is a finite raster window; edge outlets are not a regional watershed boundary. Capacity overflow invalidates the hierarchy output.',
      options: {layer: 'hierarchy'},
      controls: ['layer', 'stage'],
      readouts: ['hierarchyStatus', 'hierarchyChart']
    },
    {
      id: 'routing',
      headline: 'One surface supports several plausible flow splits',
      textAlternative: 'The selected multiple-flow routing map displays actual GPU fractions.',
      title: 'Compare how flow is split and carried',
      body: 'Each fraction rule routes over the same filled surface. Change the rule to compare how the largest downslope share changes. Expected distance uses the selected rule; transport, minimum and maximum distance, and outlet membership use D-infinity fractions. Switch **Terrain product** to inspect each actual GPU output, including transport from the highest interior cell and upstream membership for the lowest boundary outlet. Distance colors saturate at 50 km.',
      evidence: '**{{productStatus}}** · **{{flowStatus}}**',
      caveat:
        'Different split rules redistribute flow; none resolves uncertainty in the source DEM.',
      options: {layer: 'fractions', routing: 'd-infinity'},
      controls: ['routing', 'layer'],
      readouts: ['productStatus', 'flowStatus']
    },
    {
      id: 'profiles',
      headline: 'A basin summary depends on explicit dense labels',
      textAlternative:
        'The profile status map shows completed and unresolved downstream walks, and the chart plots cumulative GPU hypsometry.',
      title: 'Inspect a whole-window summary and flow profiles',
      body: 'The summary uses dense zero-based labels for the finite analysis window, a resolved stream mask and the computed D8 directions. Its GPU outputs include valid area, elevation range, stream length, drainage density, cumulative hypsometry and per-cell downstream profiles. The chart shows cumulative area across the returned elevation bins.',
      evidence: '**{{catchmentSummary}}**',
      caveat:
        'This is a window summary rather than a named outlet catchment; the metric conversion uses the centre-row ground-cell scale.',
      options: {layer: 'profile'},
      controls: ['layer'],
      readouts: ['catchmentSummary', 'hypsometryChart']
    },
    {
      id: 'compare',
      headline: 'Conditioning choices stay visible in the outputs',
      textAlternative:
        'The map can return to elevation, breach depth, flow splits, route distance or hierarchy ownership.',
      title: 'Keep the intermediate terrain products together',
      body: 'Switch products while keeping the same source and routing setup. The breached elevation is separate from the canonical filled surface used for routing, so a breach experiment cannot silently change the fraction or catchment calculations.',
      evidence: '**{{breachStatus}}** · **{{flowStatus}}** · **{{hierarchyStatus}}**',
      caveat: 'The mapped products are algorithm outputs, not a field-validated drainage model.',
      options: {layer: 'filled'},
      controls: ['layer', 'routing', 'stage'],
      readouts: ['breachStatus', 'flowStatus', 'hierarchyStatus']
    }
  ],
  about: {
    what: '`GPUTerrainBreach` searches bounded least-excavation paths for explicit pit candidates. `GPUTerrainDepressionHierarchy` writes nested depression ownership and stage curves. `GPUTerrainFlowFractions` materializes D-infinity and MFD fractions, `GPUTerrainBranchingFlow` propagates flow products, and `GPUTerrainCatchmentSummary` computes basin statistics and profiles.',
    why: 'The scene puts several terrain-processing products beside each other so their inputs, support limits and intermediate surfaces remain visible.',
    howToRead:
      'The source terrain is block-averaged. Fraction methods and branching outputs use the filled routing surface, while the breach depth is a separate constrained experiment. Hierarchy overflow and incomplete profile walks are reported as statuses. The catchment summary uses one dense window label and a center-row planar metric approximation.'
  },
  create: async ctx =>
    (await import('./terrain-conditioning.compute')).createTerrainConditioning(ctx)
});
