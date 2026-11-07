// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CountyLineworkOptions} from './county-linework-options';

const BORDER_VIEW = {longitude: -123.81, latitude: 42.36, zoom: 8.0};
// Union of the archived Curry and Josephine source longitude/latitude bounds.
const COUNTY_PAIR_BOUNDS: readonly [number, number, number, number] = [
  -124.5524368, 41.9951401, -123.2296219, 42.9549484
];

/** Shared runs, atomic provenance and bounded polygonization for an archived county pair. */
export default defineScene<CountyLineworkOptions>({
  id: 'county-linework',
  title: 'Shared county linework',
  chapter: 'geometry',
  order: 11,
  summary:
    'Inspect one real Oregon county boundary with GPUSharedPaths, GPULineNoding and GPUPolygonize, then compare a direction-only re-encoding and a clearly synthetic open-edge scenario.',
  contributors: ['GPUSharedPaths', 'GPULineNoding', 'GPUPolygonize'],
  datasets: [
    {id: 'us-counties', role: 'archived US county polygon boundaries and FIPS identifiers'}
  ],
  initialView: BORDER_VIEW,
  basemap: ground('paperSheet'),
  furniture: {
    title: {
      title: 'What does shared linework preserve?',
      subtitle: 'Curry + Josephine counties · Oregon'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.usCensus),
    caveat:
      'The Census county geometries are projected here into Albers metres for planar topology operations. The synthetic open-edge option changes one source ring edge only to demonstrate linework diagnostics; it is not a defect in the archived county data.'
  },
  options: [
    {
      kind: 'select',
      id: 'inputEncoding',
      label: 'Input encoding',
      group: 'Source representation',
      apply: 'compile',
      default: 'archived',
      options: [
        {value: 'archived', label: 'Archived county rings'},
        {value: 'reverse-right-ring', label: 'Reverse Josephine ring'},
        {value: 'omit-outer-edge', label: 'Omit Curry edge 0 (synthetic)'}
      ],
      help: 'The reversal preserves the same boundary and changes its stored direction. The omitted-edge scenario removes one non-shared Curry County segment from the synthetic input to create an open chain.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'GPU evidence',
      group: 'View',
      apply: 'param',
      default: 'source',
      options: [
        {value: 'source', label: 'Source boundaries'},
        {value: 'shared-paths', label: 'Shared path runs'},
        {value: 'noded-segments', label: 'Atomic noded edges'},
        {value: 'polygonized-faces', label: 'Polygonized faces'}
      ],
      help: 'Switches the overlay among archived input lines and geometry read back from the corresponding GPU operator. Partial or uncertain outputs are withheld from the map.'
    }
  ],
  story: [
    {
      id: 'county-pair',
      title: 'Two neighboring source rings',
      headline: 'A bounded pair keeps every segment inspectable',
      textAlternative:
        'Two muted county outlines meet along the Curry and Josephine County boundary in southern Oregon. The map uses the bundled archived Census county geometries projected to a planar Albers metre frame.',
      body: 'The source selection is Curry County (FIPS 41015) and Josephine County (FIPS 41033) from the bundled archived US counties dataset. The scene keeps both complete polygon rings, source IDs and closure conventions, then projects their longitude/latitude coordinates to Albers metres for the GPU operators.',
      evidence: '**{{countyLineworkSource}}**',
      caveat:
        'This is a two-county linework example, not a validation of every county boundary or of the current Census release.',
      options: {inputEncoding: 'archived', display: 'source'},
      controls: [],
      readouts: ['countyLineworkSource'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 1000}
    },
    {
      id: 'shared-runs',
      title: 'Find the shared boundary runs',
      headline: 'Direction is measured relative to the left ring',
      textAlternative:
        'Teal and amber line segments mark GPUSharedPaths runs on the county border. The accompanying two-bar chart reports GPU runs whose right ring travels forward or backward relative to the left ring.',
      body: '`GPUSharedPaths` compares the two source rings and returns common line runs with left-line ID, right-line ID and a direction flag. Teal is forward and amber is backward. The reverse-right-ring option changes that direction encoding while leaving the geographic boundary in the same place.',
      evidence: '**{{countyLineworkShared}}**',
      caveat:
        'The operation reports shared linework within its declared precision policy; it does not establish the positional accuracy of the Census source boundaries.',
      options: {inputEncoding: 'archived', display: 'shared-paths'},
      controls: ['inputEncoding', 'display'],
      readouts: ['countyLineworkShared', 'countyLineworkDirection'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 800}
    },
    {
      id: 'reverse-representation',
      title: 'Reverse one ring without moving its boundary',
      headline: 'Shared-run orientation responds to representation',
      textAlternative:
        'The same county outlines remain in place, while the amber and teal shared-run direction chart changes after the Josephine County ring order is reversed.',
      body: 'The synthetic encoding variant reverses the stored Josephine ring order and preserves its vertices and closure. GPUSharedPaths reads the changed orientation from the actual submitted coordinates; compare the forward and backward run counts with the archived encoding.',
      evidence: '**{{countyLineworkShared}}**',
      caveat:
        'Ring direction is a representation property here. Reversal does not move the county boundary or change the Census geography.',
      options: {inputEncoding: 'reverse-right-ring', display: 'shared-paths'},
      controls: ['inputEncoding', 'display'],
      readouts: ['countyLineworkShared', 'countyLineworkDirection'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 800}
    },
    {
      id: 'atomic-edges',
      title: 'Inspect noded segments and their origin',
      headline: 'Each atomic edge retains a source interval',
      textAlternative:
        'Distinct county colors show output segments from GPULineNoding. The readout identifies source FIPS, ring index, source edge index and the normalized start-to-end parameter interval for emitted atomic segments.',
      body: '`GPULineNoding` splits the combined source linework at detected crossings, touches and overlap endpoints. Each output row carries the source feature ID, ring and source-edge ID, plus start and end parameters along that source edge. The displayed segments are decoded from GPU output buffers.',
      evidence: '**{{countyLineworkNoding}}**',
      caveat:
        'The precision policy uses projected float32 coordinates with exact-or-uncertain predicates. Capacity loss, incomplete candidate search or uncertain predicates make the overlay partial and are reported in the status.',
      options: {inputEncoding: 'archived', display: 'noded-segments'},
      controls: ['display'],
      readouts: ['countyLineworkNoding'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 800}
    },
    {
      id: 'polygonized-faces',
      title: 'Polygonize the arbitrary linework',
      headline: 'The operator nodes its own input and returns bounded faces',
      textAlternative:
        'Translucent GPU polygonized faces from the combined two-county linework overlay the source boundaries, and colored linework diagnostics distinguish closed face edges from cuts and dangles. The readout separates bounded output rings from the total assembly requirement, which includes unbounded walks.',
      body: '`GPUPolygonize` accepts arbitrary finite planar linework and internally nodes crossings and overlaps; it does not require the separate GPULineNoding output as input. It emits every bounded face, including faces that would fill holes in a polygon ring, because polygonization does not apply source fill parity. The separate noding operator is displayed as an inspectable provenance stage.',
      evidence: '**{{countyLineworkPolygonize}}**',
      caveat:
        'The assembly requirement counts bounded and unbounded traversals, while the polygon port contains bounded face rings only. Polygonized faces are not a repaired polygon with source hole semantics; output is shown only when capacity, candidate search, uncertainty and noded-segment statuses are complete.',
      options: {inputEncoding: 'archived', display: 'polygonized-faces'},
      controls: ['display'],
      readouts: ['countyLineworkPolygonize', 'countyLineworkDiagnostics'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 800}
    },
    {
      id: 'open-edge-diagnostic',
      title: 'Remove one edge in a labelled scenario',
      headline: 'An open input chain changes polygonization diagnostics',
      textAlternative:
        'A red marker identifies the deliberately omitted Curry County source segment. GPU polygonization displays only complete bounded faces while its diagnostic chart reports open chains and dangling or unused segments in the synthetic linework.',
      body: 'Select **Omit Curry edge 0 (synthetic)** to remove one non-shared outer segment from the Curry ring. The map marks the removed source segment in red. GPUPolygonize processes the remaining arbitrary linework, then reports its actual closed rings, open chains, cuts, dangles and unused edges. Restore the archived input or reverse the Josephine ring to compare.',
      evidence: '**{{countyLineworkPolygonize}}**',
      caveat:
        'This deliberately altered linework is an educational scenario, not a claim that either archived county source ring is open or invalid.',
      options: {inputEncoding: 'omit-outer-edge', display: 'polygonized-faces'},
      controls: ['inputEncoding', 'display'],
      readouts: ['countyLineworkPolygonize', 'countyLineworkDiagnostics'],
      camera: {bounds: COUNTY_PAIR_BOUNDS, padding: 24, transitionMs: 800}
    }
  ],
  readouts: [
    {id: 'countyLineworkSource', label: 'Archived county pair', layout: 'block'},
    {id: 'countyLineworkShared', label: 'GPU shared-path runs', layout: 'block'},
    {id: 'countyLineworkNoding', label: 'GPU noded-segment provenance', layout: 'block'},
    {id: 'countyLineworkPolygonize', label: 'GPU polygonization status', layout: 'block'},
    {id: 'countyLineworkDirection', label: 'Shared runs by direction', kind: 'chart'},
    {id: 'countyLineworkDiagnostics', label: 'Polygonization diagnostics', kind: 'chart'}
  ],
  legends: state => [
    {
      id: 'county-linework',
      title: 'Displayed evidence',
      kind: 'categories',
      entries:
        state.inputEncoding === 'omit-outer-edge'
          ? [
              {color: [145, 126, 112, 190], label: 'Curry County source linework'},
              {color: [116, 143, 149, 190], label: 'Josephine County source linework'},
              {color: [218, 75, 60, 255], label: 'synthetic omitted source edge'},
              {color: [70, 122, 137, 150], label: 'GPU polygonizer diagnostics'}
            ]
          : state.display === 'shared-paths'
            ? [
                {color: [39, 163, 144, 255], label: 'shared run: forward along right line'},
                {color: [221, 159, 73, 255], label: 'shared run: backward along right line'}
              ]
            : state.display === 'noded-segments'
              ? [
                  {color: [221, 133, 79, 255], label: 'Curry County source FIPS 41015'},
                  {color: [60, 155, 174, 255], label: 'Josephine County source FIPS 41033'}
                ]
              : state.display === 'polygonized-faces'
                ? [
                    {color: [112, 137, 144, 150], label: 'bounded face from combined linework'},
                    {color: [218, 75, 60, 255], label: 'dangle edge'},
                    {color: [181, 134, 72, 240], label: 'cut edge'}
                  ]
                : [
                    {color: [145, 126, 112, 190], label: 'Curry County source ring'},
                    {color: [116, 143, 149, 190], label: 'Josephine County source ring'}
                  ],
      note: 'GPU-derived overlays are withheld when the corresponding status is partial, overflowing or uncertain.'
    }
  ],
  snippet: state =>
    `graph.add(new GPUSharedPaths({left: curryLines, right: josephineLines, intersectionCapacity: 512, output: sharedPaths}));\n` +
    `graph.add(new GPULineNoding({lines: countyLines, intersectionCapacity: 512, output: nodedSegments}));\n` +
    `// Rebase globally quantized Albers metres around the selected extent; four local float32 ULPs here are 0.03125 m.\n` +
    `graph.add(new GPUPolygonize({lines: countyLines, intersectionCapacity: 512, precision: {vertexTolerance: localMatchingTolerance, predicates: 'exact-or-uncertain', storage: 'float32'}, output: polygonFaces}));\n` +
    `// Input encoding: ${state.inputEncoding}; each operation has its own output/status ports.`,
  about: {
    what: '`GPUSharedPaths` extracts shared runs and their relative directions, `GPULineNoding` emits atomic segments with source edge intervals, and `GPUPolygonize` internally nodes arbitrary planar linework and emits its bounded faces and diagnostics.',
    why: 'A shared border can be compared as line runs, split into atomic source segments, or interpreted as a planar face graph. These outputs answer different questions even when they start from the same two county boundaries.',
    howToRead:
      'All GPU overlays use the bounded Curry/Josephine source pair projected to Albers metres. The ring reversal changes direction but not location. The red omitted edge belongs only to a synthetic open-line scenario. Polygonize treats both county rings as one linework group, enumerates bounded faces and does not preserve polygon hole parity or repair source polygons.'
  },
  create: async ctx => (await import('./county-linework.compute')).createCountyLinework(ctx)
});
