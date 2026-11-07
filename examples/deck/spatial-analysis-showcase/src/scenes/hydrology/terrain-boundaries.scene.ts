// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {TerrainBoundariesOptions} from './terrain-boundaries.compute';

export default defineScene<TerrainBoundariesOptions>({
  id: 'terrain-boundaries',
  title: 'Where does this terrain window end?',
  chapter: 'hydrology',
  order: 9,
  summary:
    'A measured Grand Canyon elevation window reveals finite-domain peak saddles, a checked D8 drainage recipe and four owned flow tiles with explicit seams.',
  contributors: [
    'GPUTerrainPeakProminence',
    'GPUTerrainFlow',
    'GPUTerrainStreamOrder',
    'GPUTerrainWatersheds',
    'GPUTerrainFlowDistance',
    'GPUTerrainFlowFractions',
    'GPUTerrainBranchingFlow',
    'GPUTerrainStreamNetwork',
    'GPUGroupStatistics',
    'GPUTerrainTiledFlow'
  ],
  datasets: [{id: 'grand-canyon-dem', role: 'Terrarium elevation · 32 × 32 source-block means'}],
  initialView: {longitude: -112.12, latitude: 36.1, zoom: 10.7},
  basemap: ground('relief'),
  annotations: [
    {kind: 'point', coordinate: [-112.0953, 36.107], text: 'Phantom Ranch', rank: 'context'},
    {kind: 'area', coordinate: [-112.14, 36.055], text: 'South Rim', size: 'small'}
  ],
  furniture: {
    title: {
      title: 'Where does this terrain window end?',
      subtitle: 'Grand Canyon · 64 × 64 block-mean DEM'
    },
    scaleBar: {units: 'metric'},
    credit: 'Mapzen Terrarium elevation tiles; luma.gl terrain contributors',
    caveat:
      'The analysis window is closed at its raster edge. It is not a regional watershed; terminal edges and infinite component elders are finite-domain conditions.'
  },
  options: [
    {
      kind: 'select',
      id: 'product',
      label: 'Terrain evidence',
      group: 'Map',
      apply: 'param',
      default: 'prominence',
      help: 'Switch among the GPU peak-prominence surface, D8 stream-order and reach lines, and four-tile owned-cell accumulation.',
      options: [
        {value: 'prominence', label: 'Peak prominence'},
        {value: 'streams', label: 'D8 streams and order'},
        {value: 'tiled', label: 'Four-tile accumulation'}
      ]
    },
    {
      kind: 'select',
      id: 'iterations',
      label: 'Tile DAG sweep budget',
      group: 'Tiled flow',
      apply: 'compile',
      default: '128',
      help: 'Sets the maximum owned-flow DAG sweeps to 4, 32 or 128. Cells with status 2 remain unresolved; the value is not a promise that every route converges.',
      options: [
        {value: '4', label: '4 sweeps'},
        {value: '32', label: '32 sweeps'},
        {value: '128', label: '128 sweeps'}
      ]
    },
    {
      kind: 'select',
      id: 'routing',
      label: 'Tile routing',
      group: 'Tiled flow',
      apply: 'compile',
      default: 'd8',
      help: 'Compare single-receiver D8 ownership with branching flow over globally prepared fractions. Each mode uses a matching whole-window reference.',
      options: [
        {value: 'd8', label: 'D8'},
        {value: 'branching', label: 'Branching flow'}
      ]
    },
    {
      kind: 'select',
      id: 'branchMethod',
      label: 'Branching fractions',
      group: 'Tiled flow',
      apply: 'compile',
      default: 'd-infinity',
      help: 'D-infinity facets or Freeman MFD fractions recompute slopes from the original block-mean elevations; where the filled D8 recipe supplies a fallback direction, its direction is retained. Both tile and whole-window references use the same fractions.',
      options: [
        {value: 'd-infinity', label: 'D-infinity'},
        {value: 'mfd-freeman', label: 'Freeman MFD'}
      ]
    }
  ],
  readouts: [
    {id: 'grid', label: 'Window and scale'},
    {id: 'selectedPeak', label: 'Finite-window peak evidence'},
    {id: 'peakChart', label: 'Selected peak and key saddle', kind: 'chart'},
    {id: 'drainage', label: 'D8 workflow status and reach geometry'},
    {id: 'catchment', label: 'Selected-cell catchment summary'},
    {id: 'tiledFlow', label: 'Owned-tile resolution and seam comparison'}
  ],
  legends: state =>
    state.product === 'prominence'
      ? [
          {
            kind: 'ramp',
            title: 'Finite prominence (m)',
            ramp: 'cividis',
            extent: [0, 500],
            unit: 'm',
            note: 'Finite peaks only; values above 500 m saturate. Infinite elders are diamonds.'
          }
        ]
      : state.product === 'streams'
        ? [
            {
              kind: 'categories',
              title: 'Strahler stream order',
              entries: [
                {color: [48, 130, 166, 255], label: 'Order 1'},
                {color: [16, 89, 126, 255], label: 'Order 2'},
                {color: [9, 52, 92, 255], label: 'Order 3'},
                {color: [232, 157, 75, 255], label: 'Order 4+'}
              ],
              note: 'Reach lines follow resolved D8 stream cells.'
            }
          ]
        : [
            {
              kind: 'ramp',
              title: 'Owned-cell accumulation',
              ramp: 'mako',
              extent: [1, 500],
              unit: 'cells',
              note: 'Fixed display range; larger contributing counts saturate at 500.'
            },
            {
              kind: 'categories',
              title: 'Tile ownership status',
              entries: [
                {color: [251, 205, 93, 255], label: 'Resolved owned cell'},
                {color: [214, 83, 72, 255], label: 'Unresolved walk'}
              ],
              note: 'One-cell halos carry global routes; only owned cells inject runoff.'
            }
          ],
  story: [
    {
      id: 'closed-window',
      headline: 'A peak inherits the boundary of its window',
      textAlternative:
        'The elevation backdrop supports a GPU finite-window peak-prominence surface and selected finite peak.',
      title: 'Ask what a saddle means inside this frame',
      body: 'The **Peak prominence** surface compares a summit with the key saddle where its higher terrain first joins an elder peak. Click a cell to inspect the GPU prominence, saddle and parent link. The relief backdrop remains the measured 32 × 32 block-average source.',
      evidence: '**{{selectedPeak}}**',
      caveat:
        'The raster is a closed finite window: a component elder is assigned +∞ because no higher elder joins it here, not because the summit has infinite prominence at regional scale or stands above sea level.',
      options: {product: 'prominence'},
      controls: ['product'],
      readouts: ['grid', 'selectedPeak', 'peakChart']
    },
    {
      id: 'drainage-recipe',
      headline: 'A drainage map is a chain of dependent decisions',
      textAlternative:
        'Stream-order colors and GPU-generated reach lines sit above the terrain; readouts report fill, routing and geometry status.',
      title: 'Follow the D8 recipe through its checks',
      body: 'The drainage workflow fills depressions, resolves flats, accumulates cell contributions, assigns Strahler stream order and extracts reach geometry. Blue cells are resolved channels; the lines trace successive D8 receivers. Click terrain to inspect the recipe’s catchment cell count and mean downstream distance. A status other than converged or a reach-capacity overflow limits what the display can claim.',
      evidence: '**{{drainage}}**',
      caveat:
        'These are modeled routes over a block-mean DEM, not measured discharge or a surveyed channel network. The outer edge can drain out of this cropped window.',
      options: {product: 'streams'},
      controls: ['product'],
      readouts: ['drainage', 'catchment']
    },
    {
      id: 'owned-tiles',
      headline: 'Ownership can be split without splitting the terrain',
      textAlternative:
        'Four disjoint colored quadrants are solved with one-cell routing halos; a readout compares owned outputs with whole-window D8 accumulation.',
      title: 'Move the work boundary across the same surface',
      body: 'Four non-overlapping 32 × 32 owned quadrants cover the grid, with their seam drawn across the map. One-cell halos carry globally prepared directions or fractions; only owned cells inject one cell of runoff. Choose **D8** or **Branching flow**, then compare the owned-tile result with the matching whole-window reference. D-infinity and Freeman MFD fractions are computed over the measured block-mean surface; they are not mixed with the depression-filled D8 recipe. The sweep budget controls incomplete tile walks.',
      evidence: '**{{tiledFlow}}**',
      caveat:
        'This graph exchanges through a resident global routing DAG. It demonstrates ownership and seam semantics, not out-of-core or distributed hydrology; an unresolved status invalidates that cell.',
      options: {product: 'tiled'},
      controls: ['product', 'iterations', 'routing', 'branchMethod'],
      readouts: ['tiledFlow']
    }
  ],
  about: {
    what: 'A finite-window terrain investigation using peak persistence, a checked D8 drainage recipe and owned-cell D8 or branching flow.',
    why: 'Boundary choices change both the reference for summit prominence and which routes appear to leave a study area.',
    howToRead:
      'The base relief is measured elevation averaged over 32 × 32 source blocks. Product layers and status summaries are outputs of GPU contributors. Infinite elders are marked as finite-window survivors, not assigned an external datum.'
  },
  create: async ctx => (await import('./terrain-boundaries.compute')).createTerrainBoundaries(ctx)
});
