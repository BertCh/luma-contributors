// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CellNeighborhoodOptions} from './cell-neighborhoods.types';

export default defineScene<CellNeighborhoodOptions>({
  id: 'cell-neighborhoods',
  title: 'Can an H3 grid path stand in for a journey?',
  chapter: 'cells',
  order: 5,
  summary:
    'Named OpenFlights airport coordinates become H3 cells; GPU grid paths, cell measures and exact edge and vertex identities show what a discrete path says and what it cannot say about travel.',
  contributors: ['GPUPointToCell', 'GPUCellGridPath', 'GPUCellMeasures', 'GPUH3CellIdentities'],
  datasets: [{id: 'openflights', role: 'historical airport locations; route network is not used'}],
  initialView: {longitude: -80.5, latitude: 40.8, zoom: 4.8, pitch: 0, bearing: 0},
  basemap: ground('night'),
  furniture: {
    title: {
      title: 'H3 path between airport anchors',
      subtitle: 'Grid topology, not a travel route'
    },
    credit: 'OpenFlights airport coordinates · ODbL 1.0',
    scaleBar: {units: 'metric'},
    caveat: 'Airport locations only; an H3 grid path does not describe flight service or a journey.'
  },
  options: [
    {
      kind: 'select',
      id: 'focus',
      label: 'Map focus',
      group: 'Display',
      apply: 'param',
      default: 'journey',
      options: [
        {value: 'journey', label: 'Airport pair'},
        {value: 'boundary', label: 'First cell boundary'}
      ],
      help: 'The boundary view zooms to the first two path cells so their shared canonical vertices can be inspected.'
    },
    {
      kind: 'select',
      id: 'destination',
      label: 'Destination airport',
      group: 'Airport anchors',
      apply: 'compile',
      default: 'JFK',
      options: [
        {value: 'MDW', label: 'MDW · Chicago Midway'},
        {value: 'JFK', label: 'JFK · New York'},
        {value: 'LAX', label: 'LAX · Los Angeles'},
        {value: 'SYD', label: 'SYD · Sydney'}
      ],
      help: 'The origin is fixed at ORD. These coordinates are airport anchors; they do not assert that an airline served the selected pair.'
    },
    {
      kind: 'select',
      id: 'resolution',
      label: 'H3 resolution',
      group: 'Grid definition',
      apply: 'compile',
      default: '4',
      options: [
        {value: '3', label: '3 · coarser'},
        {value: '4', label: '4 · default'},
        {value: '5', label: '5 · finer'}
      ],
      help: 'Changes the grid cells and recomputes keys, path, measures, and identities. H3 cell measures are kept within resolutions 3–5, where the documented fallback accuracy remains useful.'
    },
    {
      kind: 'select',
      id: 'pathCapacity',
      label: 'Stored path capacity',
      group: 'Output limit',
      apply: 'compile',
      default: '256',
      options: [
        {value: '16', label: '16 cells'},
        {value: '64', label: '64 cells'},
        {value: '256', label: '256 cells'}
      ],
      help: 'Bounds stored path cells. The GPU also reports the true required path length; when the count exceeds capacity, the map draws only a prefix and marks it as truncated.'
    }
  ],
  readouts: [
    {id: 'airportPair', label: 'Airport anchors', format: 'text', emphasis: 'tile'},
    {
      id: 'originKey',
      label: 'ORD H3 cell key',
      format: 'text',
      help: 'The little-endian 64-bit cell key written by GPUPointToCell.'
    },
    {
      id: 'destinationKey',
      label: 'Destination H3 cell key',
      format: 'text',
      help: 'The little-endian 64-bit cell key written by GPUPointToCell.'
    },
    {
      id: 'gridDistance',
      label: 'H3 grid distance',
      format: 'text',
      unit: 'hops',
      help: 'Number of adjacent-cell hops. Undefined is a property of the H3 grid-distance operation, not evidence about travel.'
    },
    {
      id: 'requiredCells',
      label: 'True required path cells',
      format: 'integer',
      unit: 'cells',
      help: 'GPUCellGridPath reports the complete path count independently of output capacity.'
    },
    {
      id: 'storedCells',
      label: 'Cells stored for display',
      format: 'integer',
      unit: 'cells',
      help: 'The bounded path prefix used for map, measure, and identity inspection.'
    },
    {id: 'pathStatus', label: 'Grid-path result', format: 'text'},
    {id: 'firstCellArea', label: 'First stored cell area', format: 'decimal', unit: 'km²'},
    {id: 'lastCellArea', label: 'Last stored cell area', format: 'decimal', unit: 'km²'},
    {id: 'firstCellPerimeter', label: 'First stored cell perimeter', format: 'decimal', unit: 'km'},
    {id: 'pentagonCells', label: 'Pentagon cells in stored path', format: 'integer'},
    {
      id: 'cellAreaChart',
      label: 'Spherical cell area along path',
      kind: 'chart',
      help: 'GPUCellMeasures output for each stored path cell, in path order; a truncated path shows only its prefix.'
    },
    {
      id: 'sharedVertices',
      label: 'Canonical vertices shared by first adjacent pair',
      format: 'text',
      help: 'Intersection of the two cells’ GPUH3CellIdentities canonical vertex-key sets.'
    },
    {
      id: 'sharedVertexKeys',
      label: 'Shared vertex identity keys',
      format: 'text',
      help: 'Canonical H3 vertex keys from GPUH3CellIdentities. These are vertex identities, not cell IDs.'
    },
    {
      id: 'firstCellDirectedEdges',
      label: 'First cell directed-edge identities',
      format: 'text',
      help: 'All six GPUH3CellIdentities slots for the first path cell are inspected; zero slots are omitted, including the absent pentagon direction.'
    },
    {
      id: 'opposedDirectedEdges',
      label: 'Opposed edge identities across first pair',
      format: 'text',
      help: 'The GPU-produced directed-edge keys are decoded for their origin and destination to find the reciprocal edge between the first two adjacent cells.'
    }
  ],
  story: [
    {
      id: 'airport-anchors',
      title: 'Start with two airport locations',
      headline: 'A named coordinate pair gives the grid a question',
      textAlternative:
        'Two OpenFlights airport locations are shown as separate anchors. The origin is Chicago O’Hare; the destination is selected from Midway, New York, Los Angeles, or Sydney.',
      body: 'The origin is Chicago O’Hare (ORD); choose MDW, JFK, LAX, or SYD as the other airport anchor. The bundled OpenFlights airport coordinates are historical and the route table is not used in this analysis.',
      evidence:
        'The GPU indexes the two airport coordinates as H3 cells: **{{originKey}}** and **{{destinationKey}}**.',
      caveat:
        'These points do not establish a current airport service connection or a path taken by a traveler.',
      controls: ['destination', 'focus'],
      options: {focus: 'journey'},
      readouts: ['airportPair', 'originKey', 'destinationKey']
    },
    {
      id: 'grid-distance',
      title: 'Count adjacent-cell hops',
      headline: 'H3 distance measures cell adjacency, not kilometres',
      textAlternative:
        'A sequence of H3 cells connects the selected airport anchors when GPUCellGridPath returns a stored path. The line follows cell centers and is not a flight or travel route.',
      body: '`GPUPointToCell` writes an H3 key for each airport coordinate. `GPUCellGridPath` then computes the H3 grid distance and a shortest local grid path between those keys. The map joins cell centers only to make the sequence visible.',
      evidence:
        'The selected pair has **{{gridDistance}}** H3 hops and requires **{{requiredCells}}** path cells.',
      caveat:
        'Grid distance counts neighboring cells. It is not a great-circle distance, a road route, an airline route, or evidence that a journey is possible.',
      controls: ['destination', 'resolution', 'focus'],
      readouts: ['gridDistance', 'requiredCells', 'pathStatus'],
      options: {destination: 'JFK', resolution: '4', focus: 'journey'}
    },
    {
      id: 'bounded-path',
      title: 'Inspect the stored path limit',
      headline: 'A stored prefix must not be mistaken for the destination',
      textAlternative:
        'The displayed cell chain is limited by the selected capacity. If the GPU required count exceeds it, the line shows a prefix and the status explicitly says that the stored endpoint is not the destination.',
      body: 'Try a smaller stored path capacity or choose Sydney. The result keeps the GPU-reported required count separate from the number of cells that fit in the output buffer. An undefined distance and a valid distance with an empty local IJ path are also reported separately.',
      evidence:
        '**{{storedCells}}** cells are stored from a GPU-reported requirement of **{{requiredCells}}**. {{pathStatus}}',
      caveat:
        'Some H3 path cases affected by pentagon distortion can return a valid grid distance but no path cells. The contributor reports this as an empty path; no substitute line is invented.',
      controls: ['destination', 'resolution', 'pathCapacity', 'focus'],
      readouts: ['gridDistance', 'requiredCells', 'storedCells', 'pathStatus'],
      options: {destination: 'LAX', resolution: '4', pathCapacity: '16', focus: 'journey'}
    },
    {
      id: 'cell-measures',
      title: 'Measure the actual cells',
      headline: 'Cells at one resolution do not all have the same area',
      textAlternative:
        'A line chart plots GPU-computed spherical area for each stored H3 cell in path order; the map outlines those cells and marks the two airport coordinates.',
      body: '`GPUCellMeasures` reads each GPU-produced H3 cell key and returns its spherical area, perimeter, edge count, and pentagon flag. The chart reports the area of each stored cell instead of assigning one resolution-wide average.',
      evidence:
        'The first stored cell measures **{{firstCellArea}}** and the last stored cell measures **{{lastCellArea}}**; **{{pentagonCells}}** stored cells are pentagons.',
      caveat:
        'Area and perimeter are spherical H3 measurements in km² and km. Hops remain a topological count, and fine-resolution fallback measures outside this story’s 3–5 range are not shown.',
      controls: ['destination', 'resolution', 'pathCapacity', 'focus'],
      options: {focus: 'journey'},
      readouts: ['firstCellArea', 'lastCellArea', 'pentagonCells', 'cellAreaChart']
    },
    {
      id: 'shared-identities',
      title: 'Follow the shared boundary identities',
      headline: 'Adjacent cells share canonical vertices, while edges have direction',
      textAlternative:
        'The first two stored path cells are inspected through GPU-produced canonical vertex identities and directed-edge identities. Shared vertex keys describe boundary topology; they are not cell indexes.',
      body: '`GPUH3CellIdentities` emits directed-edge IDs and canonical vertex IDs for each path cell. The readout intersects the vertex IDs of the first consecutive pair and decodes GPU-produced directed edges to locate the opposed identities across that pair. The boundary view highlights the first two cells and their shared vertices.',
      evidence:
        'The first adjacent pair shares **{{sharedVertices}}** canonical vertex identities: {{sharedVertexKeys}}. Opposed directed edges: {{opposedDirectedEdges}}.',
      caveat:
        'This identity check describes the cell boundary topology only. It does not validate a physical route or a travel path.',
      controls: ['destination', 'resolution', 'pathCapacity', 'focus'],
      readouts: ['sharedVertices', 'sharedVertexKeys', 'opposedDirectedEdges', 'pathStatus'],
      options: {focus: 'boundary'}
    }
  ],
  legends: () => [
    {
      id: 'map',
      title: 'Map marks',
      kind: 'categories',
      entries: [
        {color: [245, 222, 120, 255], label: 'ORD anchor'},
        {color: [240, 134, 99, 255], label: 'destination airport'},
        {color: [63, 164, 194, 160], label: 'stored H3 path cells'},
        {color: [255, 194, 103, 255], label: 'path through H3 cell centers'}
      ]
    }
  ],
  about: {
    what: '`GPUPointToCell` indexes two named airport coordinates; `GPUCellGridPath` reports H3 grid distance and a bounded local path; `GPUCellMeasures` measures the resulting cells; `GPUH3CellIdentities` exposes directed-edge and canonical vertex keys.',
    why: 'A grid can answer adjacency questions and provide reusable spatial identifiers. Seeing path limits, cell size variation, and shared boundary identities makes clear what those answers do and do not say about a journey.',
    howToRead:
      'The source is the bundled OpenFlights airport table and coordinate column. Its route network is unused. Grid paths are H3 topology; they are not observed or recommended travel routes. Some pentagon-affected paths can be empty despite a defined distance, and stored output is bounded.'
  },
  create: async ctx => (await import('./cell-neighborhoods.compute')).createCellNeighborhoods(ctx)
});
