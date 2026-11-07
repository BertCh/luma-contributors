// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RepairOverlayOptions} from './repair-overlay.compute';

export default defineScene<RepairOverlayOptions>({
  id: 'repair-overlay',
  title: 'Repair and overlay linework',
  chapter: 'geometry',
  order: 11,
  summary:
    'Check a real footprint, deliberately cross a copy of its ring, rebuild odd-parity faces on the GPU, then compare those boundaries with a hypothetical work area.',
  contributors: ['GPUGeometryValidity', 'GPUMakeValid', 'GPUPolygonOverlay'],
  datasets: [
    {id: 'chicago-buildings', role: 'Chicago building footprint outlines with source building IDs'}
  ],
  initialView: {longitude: -87.6365, latitude: 41.882, zoom: 17.4},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Repair the linework; keep the source visible',
      subtitle: 'Chicago Loop · one real footprint and one simulated ring'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cityOfChicago),
    caveat:
      'The selected source footprint is not described as invalid. The crossing is a separate simulation made by reordering its stored corners. The rectangular work area and its ID are synthetic scenario inputs, not municipal features or a proposed project.'
  },
  options: [
    {
      kind: 'select',
      id: 'mapView',
      label: 'Analysis view',
      group: 'View',
      apply: 'param',
      default: 'repair',
      options: [
        {value: 'repair', label: 'Linework repair'},
        {value: 'overlay', label: 'Polygon overlay'}
      ],
      help: 'Switch between the repair example and operations against the hypothetical rectangular work area.'
    },
    {
      kind: 'select',
      id: 'inputVariant',
      label: 'Input boundary',
      group: 'Repair input',
      apply: 'compile',
      default: 'simulated-crossing',
      options: [
        {value: 'source', label: 'Real source ring'},
        {value: 'simulated-crossing', label: 'Simulated crossing'}
      ],
      help: 'The simulated ring swaps two inner corners of the real source coordinates to create crossed linework. It does not alter or diagnose the source record.'
    },
    {
      kind: 'select',
      id: 'repairView',
      label: 'Repair display',
      group: 'Repair input',
      apply: 'param',
      default: 'repaired',
      options: [
        {value: 'input', label: 'Input linework'},
        {value: 'repaired', label: 'GPU repaired faces'}
      ],
      help: 'Compare the selected input boundary with the actual GPU polygon faces. The teal source outline remains as a reference.'
    },
    {
      kind: 'select',
      id: 'overlayOperation',
      label: 'Overlay operation',
      group: 'Polygon overlay',
      apply: 'compile',
      default: 'intersection',
      options: [
        {value: 'intersection', label: 'Intersection'},
        {value: 'union', label: 'Union'},
        {value: 'difference', label: 'Difference'},
        {value: 'symmetric-difference', label: 'Symmetric difference'},
        {value: 'dissolve', label: 'Dissolve selected footprints'}
      ],
      help: 'GPU overlay runs on projected metre coordinates. Dissolve unions the selected real footprints; other operations compare building 814558 with the synthetic work-area rectangle.'
    }
  ],
  story: [
    {
      id: 'source-check',
      title: 'Read the source before changing it',
      headline: 'A validity check does not rewrite the record',
      textAlternative:
        'The teal line is the City of Chicago source outline of building 814558. The GPU validity bitmask reports its listed checks, while the source geometry remains visible and unmodified.',
      body: 'The source building ring is projected into a local metre frame. `GPUGeometryValidity` reports its finite-coordinate, closure, ring, and intersection checks. A clear mask means no listed issue was found by these checks; it is not a guarantee about source accuracy or all possible topology rules.',
      evidence: 'Source validity: **{{validitySource}}**.',
      caveat:
        'This is a bounded set of validity checks on one archived footprint. No source defect is presumed.',
      options: {mapView: 'repair', inputVariant: 'source', repairView: 'input'},
      controls: ['inputVariant', 'repairView'],
      readouts: ['validitySource'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 17.4, transitionMs: 900}
    },
    {
      id: 'simulated-crossing',
      title: 'Make a separate crossed-ring example',
      headline: 'The altered copy creates a validity issue on purpose',
      textAlternative:
        'The teal source outline stays in place. An amber simulated ring connects the same stored corners in a different order, creating a self-crossing linework example; its GPU validity mask reports the detected condition.',
      body: 'For this demonstration, two inner corners are swapped in a separate copy of the selected ring. The altered coordinates reuse the real vertices but form crossing linework. The orange and teal marks therefore have different geometry even though the source coordinates were retained.',
      evidence: 'Simulated input mask: **{{validityCrossing}}**.',
      caveat:
        'This simulation does not claim the City of Chicago source outline is invalid or that this crossing arose in the archive.',
      options: {mapView: 'repair', inputVariant: 'simulated-crossing', repairView: 'input'},
      controls: ['inputVariant', 'repairView'],
      readouts: ['validityCrossing'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 17.4, transitionMs: 700}
    },
    {
      id: 'make-valid',
      title: 'Rebuild bounded faces from the crossing',
      headline: 'GPU repair applies odd-parity linework',
      textAlternative:
        'The amber line shows the deliberately simulated crossing and the GPU-produced repaired face outlines appear over the unchanged teal source. Readouts report output polygons, required ring slots, diagnostic remnants, source IDs, and validity of the returned faces.',
      body: '`GPUMakeValid` nodes the simulated edges and keeps faces covered an odd number of times by the input rings, matching an even-odd linework parity interpretation. The displayed repaired boundaries come from its GPU output buffers; the real source outline remains a visual reference.',
      evidence:
        '**{{repairCounts}}**. Diagnostics: **{{repairDiagnostics}}**. Returned face validity: **{{validityRepaired}}**.',
      caveat:
        'The result demonstrates repair behavior on a constructed bow-tie. Bounded output is withheld when the GPU reports overflow or uncertain predicates.',
      options: {mapView: 'repair', inputVariant: 'simulated-crossing', repairView: 'repaired'},
      controls: ['inputVariant', 'repairView'],
      readouts: ['repairCounts', 'repairDiagnostics', 'repairSources', 'validityRepaired'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 17.4, transitionMs: 700}
    },
    {
      id: 'work-area-intersection',
      title: 'Overlay a hypothetical work area',
      headline: 'Intersection keeps the shared planar region',
      textAlternative:
        'Teal outlines show the selected real building footprints; a violet dashed rectangle shows the explicitly hypothetical work area. The green result paths come from GPU intersection output, and the provenance readout lists its source feature IDs.',
      body: 'The overlay uses a local metre frame and a synthetic rectangle that partially crosses building 814558. `GPUPolygonOverlay` computes the selected operation and returns polygon rings plus boundary provenance. The rectangle has scenario ID 900001 only; it is not part of the City dataset.',
      evidence: '**{{overlayStatus}}**. Output source IDs: **{{overlaySources}}**.',
      caveat:
        'This is a geometric comparison, not a surveyed work boundary, parcel decision, or construction recommendation. The overlay uses its default even-odd fill rule for these ordinary polygon inputs.',
      options: {mapView: 'overlay', overlayOperation: 'intersection'},
      controls: ['mapView', 'overlayOperation'],
      readouts: ['overlayStatus', 'overlaySources', 'overlayCounts'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 17.4, transitionMs: 700}
    },
    {
      id: 'compare-overlay',
      title: 'Compare set operations and dissolve',
      headline: 'Different operations keep different boundary pieces',
      textAlternative:
        'The source buildings and dashed synthetic rectangle remain visible while GPU output outlines and the written-versus-required chart respond to the selected overlay operation. The status and source IDs identify actual returned geometry and its provenance.',
      body: 'Compare intersection with union, difference, symmetric difference, and dissolve. The first four compare one real building footprint against the hypothetical rectangle. Dissolve unions the selected real footprints alone. Each operation builds a graph variant on first use, and its output is shown only when capacity and predicate status are complete.',
      evidence: '**{{overlayStatus}}**. Output source IDs: **{{overlaySources}}**.',
      caveat:
        'Overlay source IDs identify contributing input features; they do not encode legal ownership or imply that the scenario rectangle exists on the ground.',
      options: {mapView: 'overlay', overlayOperation: 'dissolve'},
      controls: ['mapView', 'overlayOperation'],
      readouts: ['overlayStatus', 'overlaySources', 'overlayCounts'],
      camera: {longitude: -87.6365, latitude: 41.882, zoom: 16.8, transitionMs: 700}
    }
  ],
  readouts: [
    {
      id: 'validitySource',
      label: 'Real source validity mask',
      help: 'GPU_GEOMETRY_VALIDITY_BIT names detected conditions. Zero means no listed issue was detected.',
      unit: 'status'
    },
    {id: 'validityCrossing', label: 'Simulated crossing mask', unit: 'status'},
    {id: 'validityRepaired', label: 'GPU repaired-face validity', unit: 'status'},
    {id: 'repairCounts', label: 'Repair polygons / required ring slots', unit: 'status'},
    {
      id: 'repairDiagnostics',
      label: 'Repair diagnostics and bounded status',
      unit: 'status',
      hood: true
    },
    {id: 'repairSources', label: 'GPU repair source IDs', unit: 'IDs'},
    {id: 'overlayStatus', label: 'GPU overlay and boundary status', unit: 'status'},
    {id: 'overlaySources', label: 'GPU overlay source IDs', unit: 'IDs'},
    {
      id: 'overlayCounts',
      label: 'Overlay polygons, ring requirement, and boundary edges',
      kind: 'chart',
      help: 'Polygon count, required ring capacity, written boundary edges, and required boundary edges are distinct GPU status values.'
    }
  ],
  legends: state => [
    {
      id: 'repair-overlay',
      title: state.mapView === 'repair' ? 'Repair linework' : 'Overlay inputs and output',
      kind: 'categories',
      entries:
        state.mapView === 'repair'
          ? [
              {color: [40, 116, 122, 220], label: 'unchanged source outline'},
              {color: [202, 111, 60, 230], label: 'selected input boundary'},
              {color: [220, 143, 70, 240], label: 'GPU repaired faces'}
            ]
          : [
              {color: [40, 116, 122, 200], label: 'real source footprints'},
              ...(state.overlayOperation === 'dissolve'
                ? []
                : [{color: [125, 102, 157, 210] as const, label: 'synthetic scenario rectangle'}]),
              {color: [77, 148, 139, 240], label: 'GPU overlay result'}
            ],
      note: 'The simulated bow-tie and work area are scenario inputs; the teal outline preserves the real source geometry.'
    }
  ],
  snippet: state =>
    `graph.add(new GPUGeometryValidity({polygons: source, intersectionCapacity, output: sourceValidity}));\n` +
    `graph.add(new GPUMakeValid({polygons: simulatedCrossing, intersectionCapacity, precision, output: repaired}));\n` +
    `graph.add(new GPUPolygonOverlay({left, right, operation: '${state.overlayOperation}', vertexTolerance: 0.01, capacity, output: overlay}));`,
  about: {
    what: '`GPUGeometryValidity` emits per-feature issue bits. `GPUMakeValid` nodes and reconstructs polygon faces from linework using odd-parity coverage. `GPUPolygonOverlay` computes bounded planar set operations and emits boundary provenance.',
    why: 'Topology repair and set overlay answer different questions: one interprets a boundary arrangement as faces, while the other combines already represented polygon regions.',
    howToRead:
      'Projected input boundaries and GPU output rings are shown together. The repair is calculated from a deliberately simulated crossing ring; the overlay rectangle is a synthetic partial-overlap scenario. Output paths and source IDs come from GPU readbacks, and output is hidden when capacity, uncertainty, or candidate status is incomplete.'
  },
  create: async ctx => (await import('./repair-overlay.compute')).createRepairOverlay(ctx)
});
