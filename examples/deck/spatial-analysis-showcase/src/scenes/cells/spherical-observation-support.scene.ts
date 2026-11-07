// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SphericalObservationOptions} from './spherical-observation-support.compute';

const LEVEL_OPTIONS = [2, 3, 4, 5].map(level => ({
  value: String(level),
  label: `S2 level ${level}`
}));

export default defineScene<SphericalObservationOptions>({
  id: 'spherical-observation-support',
  title: 'S2 cell area and observation windows',
  chapter: 'cells',
  order: 4,
  summary:
    'An archived USGS earthquake sample shows how S2 resolution changes spatial support and how an exterior or interior spherical-cap cover brackets an analyst-defined search radius.',
  contributors: [
    'GPUPointToCell',
    'GPUNeighborSearch',
    'GPUS2CellMeasures',
    'GPUS2CellHierarchy',
    'GPUS2CapCovering',
    'GPUCellGeometry'
  ],
  datasets: [
    {id: 'poopdeck-earthquakes', role: 'archived USGS ComCat M4+ event catalog, 2020–2024'}
  ],
  initialView: {longitude: 12, latitude: 18, zoom: 1.45},
  basemap: ground('abyss'),
  furniture: {
    title: {
      title: 'How much Earth does a window represent?',
      subtitle: 'S2 cell area and spherical-cap cover'
    },
    credit: true,
    caveat:
      'The cap is a synthetic analyst search window, not a hazard zone. The archive has 77,215 earthquake events among 77,231 catalog rows; it ends 30 Dec 2024 and is not a complete global catalog.'
  },
  options: [
    {
      kind: 'select',
      id: 'level',
      label: 'S2 cell level',
      group: 'Spatial support',
      apply: 'compile',
      default: '4',
      options: LEVEL_OPTIONS,
      help: 'Selects an S2 resolution; each level and cover-rule variant compiles on first use. Each level adds four times as many global cells and generally reduces the area represented by one cell.'
    },
    {
      kind: 'select',
      id: 'coverMode',
      label: 'Cap cover rule',
      group: 'Search window',
      apply: 'compile',
      default: 'exterior',
      options: [
        {value: 'exterior', label: 'Exterior: include intersecting cells'},
        {value: 'interior', label: 'Interior: include fully contained cells'}
      ],
      help: 'Exterior cells may extend past the radius; interior cells stay inside it. Both are uniform-level S2 covers, not adaptive region coverings.'
    },
    {
      kind: 'slider',
      id: 'radiusKm',
      label: 'Search radius',
      group: 'Search window',
      apply: 'param',
      min: 100,
      max: 1500,
      step: 50,
      default: 700,
      unit: 'km',
      help: 'Sets the angular radius of a spherical cap. This is an analyst-chosen query window, not a prediction or hazard boundary.'
    },
    {
      kind: 'toggle',
      id: 'showParents',
      label: 'Show center-cell parent',
      group: 'Spatial support',
      apply: 'param',
      default: false,
      help: 'Adds a geodesically densified outline of the one-level-coarser parent of the selected query center cell.'
    },
    {
      kind: 'select',
      id: 'center',
      label: 'Window center',
      group: 'Search window',
      apply: 'param',
      default: 'Japan',
      options: ['Japan', 'Türkiye', 'Andes', 'Sunda'].map(value => ({value, label: value})),
      help: 'Moves the spherical query window among four earthquake regions. The global event index and area summaries remain global.'
    }
  ],
  story: [
    {
      id: 'catalog',
      title: 'The catalog is global; the sample is explicit',
      headline: 'A bounded sample keeps cell outlines legible',
      textAlternative:
        'The global map shows a deterministic sample of 2,048 warm earthquake dots and the radius circle around the Japan query center; the archive is an observed catalog, not a completeness estimate.',
      body: '`GPUPointToCell` indexes the 77,215 earthquake rows in S2. The map draws a deterministic stride sample of 2,048 events so cell boundaries remain readable; all events still contribute to the area and density summaries. The underlying archive also contains 16 non-earthquake event types, which this view excludes.',
      evidence:
        '**{{events}}** are indexed; the map shows only the declared 2,048-event geometry sample.',
      caveat:
        'ComCat archive rows are an observed catalog, shaped by reporting and detection. These counts do not estimate earthquake risk or catalog completeness.',
      controls: [],
      readouts: ['events'],
      camera: {longitude: 12, latitude: 18, zoom: 1.45, transitionMs: 1200}
    },
    {
      id: 'cell-area',
      title: 'Cells define the support of a count',
      headline: 'Every event is assigned one spherical area',
      textAlternative:
        'Warm sample dots remain visible across the map; the selected center-cell parent is teal, while blue cap-cover outlines and the pale circle show the synthetic query radius.',
      body: '`GPUS2CellMeasures` computes spherical cell area in steradians for each indexed key; the display converts those GPU outputs with the same mean-Earth radius used for the geodesic circle. **S2 cell level** below changes the support unit. `GPUS2CellHierarchy` derives the selected center cell’s parent, outlined in teal.',
      evidence:
        '**{{uniqueCells}}** occupy **{{supportArea}}**; dividing events by that area gives **{{density}}**.',
      caveat:
        'This is a spherical area approximation using one named mean-Earth radius. S2 cells have unequal areas, so counts per cell alone are not comparable as density.',
      controls: ['level'],
      options: {level: '4', showParents: true},
      readouts: ['uniqueCells', 'supportArea', 'density'],
      camera: {longitude: 146, latitude: 34, zoom: 3.1, transitionMs: 1500}
    },
    {
      id: 'coarser-support',
      title: 'A parent cell changes the question',
      headline: 'Coarser cells pool distant local differences',
      textAlternative:
        'At the coarser S2 level, fewer larger supports surround the query center; its parent outline is teal, the cap cover is blue, and warm dots are the map-only event sample.',
      body: 'Move **S2 cell level** below from 4 toward 2. Each cell then represents a larger share of the sphere, and neighboring event clusters merge into fewer supports. Turn on **Show center-cell parent** below to compare the query point’s cell with its one-level-coarser parent. Blue outlines show the cap cover; warm dots show the event sample.',
      evidence:
        'The indexed event count stays fixed while **{{uniqueCells}}** and **{{supportArea}}** respond to level.',
      caveat:
        'A coarser cell smooths spatial variation by construction; it does not add information about events between the observed locations.',
      options: {level: '2', showParents: true},
      controls: ['level', 'showParents'],
      readouts: ['uniqueCells', 'supportArea'],
      camera: {longitude: 146, latitude: 34, zoom: 3.1, transitionMs: 1200}
    },
    {
      id: 'cover-window',
      title: 'A cap cover trades edge error for candidates',
      headline: 'Exterior cells extend beyond the chosen radius',
      textAlternative:
        'The pale geodesic circle marks the selected radius; blue GPU S2 cell outlines show the chosen exterior or interior cover, whose summed area is compared with the analytic cap area.',
      body: '`GPUS2CapCovering` tests the same spherical radius against all candidate cells at the selected level. Choose **Cap cover rule** below to compare exterior cells (every intersecting cell) with interior cells (fully contained cells); change **Search radius** to reshape the synthetic query window.',
      evidence:
        '**{{capEvents}}** events fall within the exact radius (**{{capDensity}}**); **{{coverCells}}** candidate cells represent **{{coverArea}}**, or **{{overcover}}** the analytic cap area.',
      caveat:
        'The GPU great-circle search counts catalog events within the exact radius separately from candidate cover cells. Cover status reports completion; the cap is not a hazard or risk zone.',
      options: {level: '4', coverMode: 'exterior', radiusKm: 700},
      controls: ['coverMode', 'radiusKm', 'center'],
      readouts: ['capEvents', 'coverCells', 'coverArea', 'areaComparison'],
      camera: {longitude: 140, latitude: 35, zoom: 3.1, transitionMs: 1500}
    },
    {
      id: 'takeaway',
      title: 'A zero is still a measured result',
      headline: 'An empty search is still a measured result',
      textAlternative:
        'Near Sunda, the pale 250 km circle marks the query; any blue outlines are cells wholly inside the level-5 interior cover, and an empty interior cover is possible. The GPU search returns zero archived earthquakes, which is a measured archive count rather than missing data or a claim that no earthquakes occurred.',
      body: 'At the selected **Sunda** center and **250 km** radius, this archived selection contains zero earthquakes. Zero is the result of the GPU great-circle query; it does not mean data are missing, and it does not imply no earthquakes occurred. Explore **Window center**, **Search radius**, and **S2 cell level** to see how the support and candidate cover change.',
      evidence:
        '**{{capEvents}}** fall within the window; **{{capDensity}}** divides that event count by the analytic spherical-cap area.',
      caveat:
        'The bundled dataset is a five-year archive selection, not a complete catalog. The 2,048-event display sample is cartographic; reported cell counts and areas use all 77,215 earthquakes.',
      options: {
        level: '5',
        coverMode: 'interior',
        radiusKm: 250,
        center: 'Sunda',
        showParents: false
      },
      controls: ['level', 'radiusKm', 'center'],
      readouts: ['capEvents', 'coverCells', 'areaComparison', 'supportArea'],
      camera: {longitude: 107.5, latitude: -1, zoom: 3.8, transitionMs: 1500}
    }
  ],
  readouts: [
    {id: 'events', label: 'Indexed earthquakes', unit: 'events', emphasis: 'tile'},
    {id: 'capEvents', label: 'Earthquakes within cap', unit: 'events'},
    {id: 'capDensity', label: 'Cap event density', unit: 'events / km²'},
    {id: 'uniqueCells', label: 'Occupied S2 cells', unit: 'cells'},
    {id: 'supportArea', label: 'Summed occupied-cell area', unit: 'km²'},
    {id: 'density', label: 'Area-normalized event density', unit: 'events / km²'},
    {id: 'coverCells', label: 'Cap candidate cells', unit: 'cells'},
    {id: 'coverArea', label: 'Cover area / analytic cap area', unit: 'km²'},
    {id: 'overcover', label: 'Cover to cap area ratio', unit: '×'},
    {
      id: 'areaComparison',
      label: 'Cap and cover area',
      kind: 'chart',
      help: 'Compares analytic spherical-cap area with the summed S2 area of the GPU cover cells.'
    },
    {id: 'status', label: 'GPU cover status', unit: 'status', hood: true}
  ],
  legends: state => [
    {
      id: 'map',
      title: 'Map marks',
      kind: 'categories',
      entries: [
        {color: [235, 112, 57, 170], label: 'sampled earthquake events'},
        {
          color: state.coverMode === 'exterior' ? [58, 137, 178, 205] : [44, 105, 161, 205],
          label: `${state.coverMode} GPU cap-cover cells`
        },
        ...(state.showParents
          ? [
              {
                color: [93, 198, 186, 145] as [number, number, number, number],
                label: 'query-center cell parent'
              }
            ]
          : []),
        {color: [145, 207, 224, 230], label: 'true geodesic radius'}
      ],
      note: `Warm marks are a deterministic display sample; blue outlines show the ${state.coverMode} GPU cap cover.`
    }
  ],
  snippet: state =>
    `graph.add(new GPUPointToCell({family: 's2', resolution: ${state.level}, positions, output: {cells}}));\n` +
    `graph.add(new GPUS2CellMeasures({cells, output: {measures, status}}));\n` +
    `graph.add(new GPUNeighborSearch({\n` +
    `  mode: 'radius', positions, queryPositions: centerPosition, parameters,\n` +
    `  spatialContext: greatCircleMeters, gridSize: [180, 90],\n` +
    `  output: {offsets, neighbors, weights, status}\n` +
    `}));\n` +
    `graph.add(new GPUS2CapCovering({level: ${state.level}, mode: '${state.coverMode}', caps, capacity, output}));`,
  about: {
    what: '`GPUPointToCell` assigns global event positions to S2 keys. `GPUS2CellMeasures` returns cell area in steradians, `GPUS2CellHierarchy` maps the query center’s cell to its parent, and `GPUS2CapCovering` enumerates a fixed-level exterior or interior cover of a spherical cap. A great-circle `GPUNeighborSearch` independently counts events inside the radius.',
    why: 'Spatial counts need an explicit support unit. This scene makes the size of that unit visible and compares the area of an indexed cell set with the analytic area of a geodesic query window.',
    howToRead:
      'Warm points show the deterministic display sample. Thin blue outlines are cap-cover cells, teal is the query center cell parent, and the pale blue line is the true geodesic radius. Read global occupied-cell area and density separately from the exact event count within the cap.'
  },
  create: async ctx =>
    (await import('./spherical-observation-support.compute')).createSphericalObservationSupport(ctx)
});
