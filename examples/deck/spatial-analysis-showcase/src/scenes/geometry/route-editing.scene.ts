// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RouteEditingOptions} from './route-editing.types';

const ROUTE_VIEW = {longitude: -87.65, latitude: 41.88, zoom: 10.2};

export default defineScene<RouteEditingOptions>({
  id: 'route-editing',
  create: async ctx => {
    const compute = await import('./route-editing.compute');
    return compute.createRouteEditing(ctx);
  },
  title: 'Which edits preserve a route?',
  chapter: 'geometry',
  order: 12,
  summary:
    'Follow Chicago’s Red Line through reversal, coordinate reduction, station snapping and a deliberately simulated revisit, comparing each output with the stored route.',
  contributors: [
    'GPUGeometryOrientation',
    'GPUNormalizeGeometry',
    'GPUCanonicalGeometryEquality',
    'GPUGeometryPredicates',
    'GPUGeometryCleanup',
    'GPULinePrecisionReduction',
    'GPUVertexSnap',
    'GPUSegmentSnap',
    'GPULineLoopRemoval',
    'GPUAffineTransform',
    'GPUGeometryMeasures'
  ],
  datasets: [
    {id: 'cta-transit', role: 'Red Line shape, route metadata and station reference points'}
  ],
  initialView: ROUTE_VIEW,
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Editing a transit line',
      subtitle: 'Storage order, precision and reference points'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cta),
    caveat:
      'A route shape is stored geometry; it does not assert current service or track position.'
  },
  options: [
    {
      kind: 'select',
      id: 'edit',
      label: 'Geometry operation',
      group: 'Edit',
      apply: 'compile',
      default: 'reverse',
      options: [
        {value: 'reverse', label: 'Reverse stored vertex order'},
        {value: 'cleanup', label: 'Geometry cleanup'},
        {value: 'precision', label: 'Grid precision reduction'},
        {value: 'vertex-snap', label: 'Snap vertices to stations'},
        {value: 'segment-snap', label: 'Insert stations near segments'},
        {value: 'loop-revisit', label: 'Simulated revisit stress test'},
        {value: 'translation', label: 'Declared translation and inverse'}
      ],
      help: 'Each operation is compiled against the same retained Red Line source. The revisit option uses a short derived path and never replaces the source.'
    },
    {
      kind: 'slider',
      id: 'tolerance',
      label: 'Tolerance',
      group: 'Edit',
      apply: 'compile',
      min: 0,
      max: 100,
      step: 1,
      default: 20,
      unit: 'm',
      help: 'Local metric distance for cleanup and station snapping. Segment snapping inserts eligible station references near segments; it does not project them onto an edge.'
    },
    {
      kind: 'slider',
      id: 'gridSize',
      label: 'Grid size',
      group: 'Edit',
      apply: 'compile',
      min: 0.1,
      max: 100,
      step: 0.1,
      default: 1,
      format: value => `${value.toFixed(1)} m`,
      help: 'Grid spacing for cleanup and precision reduction. Large values up to 100 m are coarse stress settings that round local metric coordinates and may collapse vertices; they are not recommended positional accuracy.'
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Map extent',
      group: 'Display',
      apply: 'param',
      default: 'route',
      options: [
        {value: 'route', label: 'Retained route extent'},
        {value: 'downtown', label: 'Downtown detail'},
        {value: 'revisit', label: 'Derived revisit detail'}
      ],
      help: 'Changes only the map camera; output geometry stays the same.'
    }
  ],
  story: [
    {
      id: 'stored-order',
      title: 'A line has a stored direction',
      headline: 'Reversal changes order, not the traced path.',
      textAlternative:
        'The muted CTA Red Line shape remains beneath an amber line with its first and last labels exchanged.',
      body: 'This CTA source record stores 96 coordinate rows for the Red Line. Reverse the row order on the GPU, then normalize the original and reversed line before comparing canonical geometry.',
      evidence:
        'Canonical comparison: {{canonicalStatus}} Exact stored order: {{exactOrderIdentical}}; exact coordinates: {{exactCoordinateEquality}}. Source simplicity: {{sourceSimple}}; closure: {{sourceClosed}}.',
      caveat:
        'This directional shape row is retained from a city-bounded export and omits geometry outside its prepared extent. Stored order does not claim train direction or current track position.',
      options: {edit: 'reverse', focus: 'route'},
      controls: ['edit', 'focus'],
      readouts: [
        'sourceVertexCount',
        'canonicalStatus',
        'exactOrderIdentical',
        'exactCoordinateEquality'
      ],
      stage: 'edit'
    },
    {
      id: 'cleanup',
      title: 'Cleanup changes coordinates',
      headline: 'A tolerance can move or collapse vertices.',
      textAlternative:
        'A bold edited route overlays the muted stored source shape; the vertex count and same-index coordinate differences are listed beside it.',
      body: 'Geometry cleanup applies a local metric grid and distance tolerance. Increase either setting and inspect the output count and status before interpreting the visible line.',
      evidence:
        'The output has {{editedVertexCount}} rows, {{removedVertexCount}} fewer than the input; same-index coordinate differences: {{sameIndexDifferenceCount}}. {{editStatus}}',
      caveat: 'A tidy-looking line does not prove that every moved point is more accurate.',
      options: {edit: 'cleanup', tolerance: 20, gridSize: 1, focus: 'downtown'},
      controls: ['edit', 'tolerance', 'gridSize', 'focus'],
      readouts: [
        'editedVertexCount',
        'removedVertexCount',
        'sameIndexDifferenceCount',
        'editStatus'
      ],
      stage: 'edit'
    },
    {
      id: 'precision',
      title: 'Precision reduction is a separate operation',
      headline: 'A grid can merge nearby positions.',
      textAlternative:
        'The precision-reduced path is shown over the retained Red Line source with its resulting vertex count.',
      body: 'GPU line precision reduction writes coordinates onto a selected grid and reports validity. Compare its output with cleanup: the methods expose different controls and statuses.',
      evidence:
        'The GPU output contains {{editedVertexCount}} vertices, {{removedVertexCount}} fewer than the input. Same-index differences: {{sameIndexDifferenceCount}}. {{editStatus}}',
      caveat:
        'The chosen grid is an analysis setting, not a documented accuracy of the source coordinates.',
      options: {edit: 'precision', gridSize: 1, focus: 'downtown'},
      controls: ['edit', 'gridSize', 'focus'],
      readouts: ['editedVertexCount', 'removedVertexCount', 'requiredVertexCount', 'editStatus'],
      stage: 'edit'
    },
    {
      id: 'vertex-snap',
      title: 'Snap stored vertices to station points',
      headline: 'A reference set can pull vertices across space.',
      textAlternative:
        'The edited line and CTA rail station reference points are displayed together in downtown Chicago.',
      body: 'GPU vertex snap moves each route vertex to a qualifying station reference within tolerance. The station set is selected from CTA stop-route links for the Red Line.',
      evidence:
        '{{stationReferenceCount}} linked rail stations are available; {{snappedVertexCount}} route vertices report a matched reference. Same-index differences: {{sameIndexDifferenceCount}}.',
      caveat:
        'Stations are point references, not surveyed track centerlines. Snapping to them is an exploratory geometry edit.',
      options: {edit: 'vertex-snap', tolerance: 20, focus: 'downtown'},
      controls: ['edit', 'tolerance', 'focus'],
      readouts: [
        'stationReferenceCount',
        'snappedVertexCount',
        'sameIndexDifferenceCount',
        'editStatus'
      ],
      stage: 'snap'
    },
    {
      id: 'segment-snap',
      title: 'Insert nearby stations along segments',
      headline: 'Insertion changes the route’s vertex sequence.',
      textAlternative:
        'Station references appear beside an edited line that may include inserted station coordinates.',
      body: 'GPU segment snap checks station references against each segment and inserts qualifying reference points in projection order. Unlike vertex snap, this can add rows to the output.',
      evidence:
        'The output contains {{editedVertexCount}} of {{requiredVertexCount}} required vertices; same-index differences: {{sameIndexDifferenceCount}}. {{editStatus}}',
      caveat:
        'This operation does not project a station onto an edge, infer a connection, or certify route topology.',
      options: {edit: 'segment-snap', tolerance: 20, focus: 'downtown'},
      controls: ['edit', 'tolerance', 'focus'],
      readouts: [
        'editedVertexCount',
        'requiredVertexCount',
        'sameIndexDifferenceCount',
        'editStatus'
      ],
      stage: 'snap'
    },
    {
      id: 'simulated-revisit',
      title: 'Try loop removal on a deliberate revisit',
      headline: 'A stress case is not evidence of source error.',
      textAlternative:
        'A short derived path doubles back over itself; the loop-removal output is highlighted beside the unchanged retained Red Line shape.',
      body: 'The loop tool receives an explicit short sequence derived from the first Red Line coordinates, with a segment repeated to create a revisit. The retained source shape remains a muted reference while the derived input and output are shown.',
      evidence:
        'The derived input contains {{sourceVertexCount}} rows; the contributor reports {{editStatus}}.',
      caveat:
        'This constructed revisit is only a method demonstration. It says nothing about defects in the published Red Line shape.',
      options: {edit: 'loop-revisit', focus: 'revisit'},
      controls: ['edit', 'focus'],
      readouts: ['sourceVertexCount', 'editedVertexCount', 'editStatus'],
      stage: 'loop'
    },
    {
      id: 'declared-translation',
      title: 'Translate the line and reverse the shift',
      headline: 'A declared frame change preserves the measured line length.',
      textAlternative:
        'The original CTA line is gray, a copy shifted east and south is amber, and a GPU inverse-transformed copy is cyan.',
      body: 'Apply a declared local-metric translation of 64 metres east and 32 metres south to every retained Red Line coordinate, then apply the inverse translation on the GPU. Read both line lengths and the maximum coordinate recovery error from GPU readback.',
      evidence:
        'Declared shift: {{declaredTranslation}}. Source length {{sourceLineLength}}; translated length {{translatedLineLength}}. {{translationRecovery}}.',
      caveat:
        'This is a controlled coordinate-frame scenario, not observed movement of a train, track or source feature. Float32 storage can leave a small recovery error.',
      options: {edit: 'translation', focus: 'downtown'},
      controls: ['edit', 'focus'],
      readouts: [
        'declaredTranslation',
        'sourceLineLength',
        'translatedLineLength',
        'translationRecovery'
      ],
      stage: 'edit'
    }
  ],
  readouts: [
    {id: 'sourceVertexCount', label: 'Input vertices', format: 'integer'},
    {id: 'editedVertexCount', label: 'Output vertices', format: 'integer'},
    {
      id: 'requiredVertexCount',
      label: 'Required vertices',
      format: 'integer',
      help: 'Exact required count where the contributor reports one; output capacity may truncate inserted rows.'
    },
    {
      id: 'sameIndexDifferenceCount',
      label: 'Same-index coordinate differences',
      help: 'An ordinal comparison of output rows to input rows. Insertions and removals shift row alignment, so this is not a correspondence or a moved-vertex count.'
    },
    {
      id: 'removedVertexCount',
      label: 'Fewer output rows',
      format: 'integer',
      help: 'Positive when the output contains fewer rows than its input. This does not identify which source vertices were removed.'
    },
    {id: 'stationReferenceCount', label: 'Linked rail stations', format: 'integer'},
    {
      id: 'snappedVertexCount',
      label: 'Matched route vertices',
      format: 'integer',
      help: 'GPUVertexSnap reference row markers; unmatched vertices are not counted.'
    },
    {
      id: 'canonicalStatus',
      label: 'Canonical comparison',
      help: 'GPU normalization and canonical geometry equality compare original and reversed line structure.'
    },
    {
      id: 'exactOrderIdentical',
      label: 'Same ordered coordinates',
      help: 'GPUGeometryPredicates.equalsIdentical compares the original and reversed stored coordinate sequence.'
    },
    {
      id: 'exactCoordinateEquality',
      label: 'Exact coordinate sequence equality',
      help: 'GPUGeometryPredicates.equalsExact compares structure and coordinates at zero tolerance.'
    },
    {
      id: 'sourceSimple',
      label: 'Source line is simple',
      help: 'GPU self-intersection predicate; an overflow is reported separately.'
    },
    {
      id: 'sourceClosed',
      label: 'Source line is closed',
      help: 'GPU predicate checks whether the first and last stored coordinates coincide.'
    },
    {
      id: 'predicateStatus',
      label: 'Predicate completeness',
      help: 'Intersection-list overflow and detected non-neighbor intersection count.'
    },
    {
      id: 'editStatus',
      label: 'Contributor status',
      help: 'Status, overflow or completion text from GPU output readback.'
    },
    {id: 'declaredTranslation', label: 'Declared translation'},
    {
      id: 'sourceLineLength',
      label: 'Source line length',
      format: 'decimal',
      unit: 'm',
      help: 'GPU-measured length of the retained source line in its local metric frame.'
    },
    {
      id: 'translatedLineLength',
      label: 'Translated line length',
      format: 'decimal',
      unit: 'm',
      help: 'GPU-measured length after the declared translation. Translation preserves pairwise distances mathematically; float32 storage may introduce small differences.'
    },
    {
      id: 'translationRecovery',
      label: 'Inverse recovery',
      help: 'Maximum coordinate error after the GPU inverse transform, with input row count and line offsets preserved.'
    }
  ],
  legends: state => [
    {
      kind: 'categories',
      title:
        state.edit === 'loop-revisit'
          ? 'Derived stress test'
          : state.edit === 'translation'
            ? 'Declared frame scenario'
            : 'Route editing',
      entries: [
        {color: [161, 174, 184, 165] as const, label: 'Stored Red Line source'},
        {
          color: [255, 186, 86, 245] as const,
          label:
            state.edit === 'loop-revisit'
              ? 'Loop-removal output'
              : state.edit === 'translation'
                ? 'Translated copy'
                : 'GPU edited geometry'
        },
        ...(state.edit === 'translation'
          ? [{color: [73, 192, 204, 240] as const, label: 'Inverse recovered copy'}]
          : []),
        ...(state.edit === 'loop-revisit'
          ? [{color: [217, 133, 94, 235] as const, label: 'Derived revisit input'}]
          : []),
        ...(['vertex-snap', 'segment-snap'].includes(state.edit)
          ? [{color: [91, 208, 221, 230] as const, label: 'Linked station reference'}]
          : [])
      ]
    }
  ],
  about: {
    what: 'The CTA Red Line shape is projected with a local metric frame around downtown Chicago. GPU contributors reverse, normalize, simplify precision or snap the retained coordinates to rail station references.',
    howToRead:
      'This directional shape row is prepared from a city-bounded CTA GTFS export and is not a complete uncut route. Station points come from linked rail stops in the same feed. The loop test is a constructed revisit derived from the first six source vertices, not a source correction. Source: [CTA GTFS feed](https://www.transitchicago.com/downloads/sch_data/google_transit.zip), September 11–19, 2026; see the [feed license terms](https://www.transitchicago.com/developers/).',
    why: 'Geometry edits can alter order, coordinate values, vertex count and topology in different ways. Read the contributor status alongside each output.'
  },
  snippet: state => {
    if (state.edit === 'reverse')
      return `graph.add(new GPUGeometryOrientation({
  mode: 'reverse', positions, lineOffsets,
  output: {positions: reversedPositions}
}));

// Compare normalized original and reversed line trees with GPUCanonicalGeometryEquality.`;
    if (state.edit === 'cleanup')
      return `graph.add(new GPUGeometryCleanup({
  geometryType: 'lines', positions, lineOffsets,
  parameters: cleanupParameters.importToGraph(graph),
  output: {positions: cleanedPositions, lineOffsets: cleanedOffsets,
    status: {count, overflow, requiredCount}}
}));`;
    if (state.edit === 'precision')
      return `graph.add(new GPULinePrecisionReduction({
  positions, lineOffsets, gridSize: ${Number(state.gridSize)}, mode: 'valid-output',
  output: {positions: reducedPositions, lineOffsets: reducedOffsets, status}
}));`;
    if (state.edit === 'vertex-snap')
      return `graph.add(new GPUVertexSnap({
  positions, referencePositions: stationPositions,
  parameters: snapParameters.importToGraph(graph),
  output: {positions: snappedPositions, referenceRows}
}));`;
    if (state.edit === 'segment-snap')
      return `graph.add(new GPUSegmentSnap({
  positions, lineOffsets, referencePositions: stationPositions,
  tolerance: ${Number(state.tolerance)}, maximumVerticesPerLine,
  output: {positions: insertedPositions, sourceEdges, counts, validity}
}));`;
    if (state.edit === 'translation')
      return `graph.add(new GPUAffineTransform({
  geometryType: 'lines', spatialContext: localMetricContext,
  positions, lineOffsets,
  parameters: translationParameters.importToGraph(graph),
  output: {positions: translatedPositions}
}));
graph.add(new GPUAffineTransform({
  geometryType: 'lines', spatialContext: localMetricContext,
  positions: translatedPositions, lineOffsets,
  parameters: inverseParameters.importToGraph(graph),
  output: {positions: recoveredPositions}
}));
// The declared (+64, -32) m translation is a scenario, not observed movement.`;
    return `graph.add(new GPULineLoopRemoval({
  lines: {kind: 'lines', positions: revisitPositions, lineOffsets: revisitOffsets},
  intersectionCapacity, output: {positions: loopFreePositions, lineOffsets: outputOffsets,
    sourceEdges, status}
}));

// revisitPositions is an explicit derived stress case, not the published route.`;
  }
});
