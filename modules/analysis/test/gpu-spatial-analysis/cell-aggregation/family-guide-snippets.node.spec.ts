// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {computeS2CapCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {computePointToCellOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {decodeGPUTrackSimilarityPairStatus} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {
  decodeGPUSphericalArcOverlayCounts,
  decodeGPUSphericalConvexIntersectionCounts
} from '../../../src/gpu-spatial-analysis/global-coordinates/spherical-status';
import {GPUCellCompaction} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline';
import {getScaledValue, isScaledValueSaturated} from './cell-aggregation-oracle';
import {GPUParameterBuffer} from '../../../src/analysis-core/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {loadGuideFunction, readGuide} from './family-guide-execution.test-utils';
import {computeTrackSimilarityOnCPU} from '../../../src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {DOCS_ROOT} from '../../analysis-core/doc-program';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('DOC-110 family guides declare public imports and typed caller inputs', () => {
  const pages = [
    'operators-cells',
    'operators-lines',
    'movement-semantics',
    'global-coordinates',
    'track-similarity-measures'
  ];
  // The lines guide adds withLocalFrameProjection's compiled-graph/projection cleanup example.
  const fenceCounts = [8, 16, 1, 2, 1];
  const snippets = pages.flatMap((name, index) => {
    const page = `gpu-spatial-analysis/${name}.md`;
    const extracted = extractSnippets(page, readFileSync(join(DOCS_ROOT, page), 'utf8'));
    expect(extracted.length, name).toBe(fenceCounts[index]);
    return extracted;
  });
  const failures = getSnippetDiagnostics(snippets);
  expect(
    failures,
    failures.map(failure => `${failure.page}:${failure.line} ${failure.message}`).join('\n')
  ).toEqual([]);
}, 120_000);

it('UCT-12 final parked window needs includeLast and lasts 600000 ms', () => {
  // Three coincident fixes span exactly 600000 - 0 milliseconds.
  for (const includeLast of [0, 1]) {
    const result = computeTrajectoryStaypointsOnCPU(
      {
        positions: new Float32Array(6),
        timestamps: Float32Array.of(0, 300000, 600000),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(100, 300000, 900000, includeLast)
      },
      {capacity: 1}
    );
    expect(Array.from(result.count)).toEqual([includeLast]);
    expect(Number.isFinite(result.durations[0])).toBe(true);
    expect(Array.from(result.durations)).toEqual([includeLast ? 600000 : 0]);
  }
});

it('UCT-16 inserting the permitted ERP gap vertex costs zero', () => {
  // Independent edit script: retain (1,0), insert gap (0,0) at cost |gap-gap|=0,
  // retain (2,0). Nonnegative costs establish the exact minimum zero.
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {positions: Float32Array.of(1, 0, 2, 0), lineOffsets: Uint32Array.of(0, 2)},
      linesB: {positions: Float32Array.of(1, 0, 0, 0, 2, 0), lineOffsets: Uint32Array.of(0, 3)},
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(0)
    },
    {measures: ['editDistanceWithRealPenalty'], editDistanceGapPoint: [0, 0]}
  );
  expect(Array.from(result.pairStatus)).toEqual([0]);
  expect(Number.isFinite(result.editDistanceWithRealPenalty[0])).toBe(true);
  expect(Array.from(result.editDistanceWithRealPenalty)).toEqual([0]);
});

it('B1 declares the actual ring fence with separate producer and ring status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const inputs: Record<string, unknown> = {};
  for (const name of [
    'count',
    'ringCount',
    'overflow',
    'candidateOverflow',
    'requiredCount',
    'openSegmentCount',
    'touchingSegmentCount'
  ]) {
    inputs[name] = createTransientView(graph, name, 'uint32', 1);
  }
  for (const name of ['groups', 'segmentRings', 'segmentVertices', 'segmentFlags'])
    inputs[name] = createTransientView(graph, name, 'uint32', 3);
  for (const name of ['ringIsHole', 'ringShells', 'ringGroups'])
    inputs[name] = createTransientView(graph, name, 'uint32', 1);
  inputs['ringAreas'] = createTransientView(graph, 'areas', 'float32', 1);
  inputs['endpoints'] = createTransientView(graph, 'endpoints', 'float32x4', 3);
  for (const name of ['positions', 'polygonPositions'])
    inputs[name] = createTransientView(graph, name, 'float32x2', 4);
  for (const name of ['ringOffsets', 'polygonRingOffsets', 'polygonOffsets', 'featureOffsets'])
    inputs[name] = createTransientView(graph, name, 'uint32', 2);
  inputs['segmentStatus'] = {
    count: inputs['count'],
    overflow: createTransientView(graph, 'source-overflow', 'uint32', 1),
    candidateOverflow: createTransientView(graph, 'source-candidate', 'uint32', 1)
  };
  const declare = loadGuideFunction<
    (graph: GPUCommandGraph, inputs: Record<string, unknown>) => void
  >('operators-cells', 'addOperatorsCellsExample10');
  expect(() => declare(graph, inputs)).not.toThrow();
});

it('B2 parked example uploads includeLast into the buffer it imports', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'stay-parameters',
    format: 'float32',
    length: 4
  });
  const write = vi.spyOn(parameters, 'write');
  const imported = vi.spyOn(parameters, 'importToGraph');
  const inputs: Record<string, unknown> = {parameterBuffer: parameters};
  inputs['positions'] = createTransientView(graph, 'positions', 'float32x2', 3);
  inputs['timestamps'] = createTransientView(graph, 'times', 'float32', 3);
  inputs['trackOffsets'] = createTransientView(graph, 'tracks', 'uint32', 2);
  for (const name of ['ids', 'count', 'overflow', 'candidateOverflow', 'startRows', 'finishRows'])
    inputs[name] = createTransientView(graph, name, 'uint32', 1);
  inputs['centroids'] = createTransientView(graph, 'centroids', 'float32x2', 1);
  inputs['durations'] = createTransientView(graph, 'durations', 'float32', 1);
  inputs['staypointIds'] = createTransientView(graph, 'stay-ids', 'uint32', 3);
  const declare = loadGuideFunction<
    (graph: GPUCommandGraph, inputs: Record<string, unknown>) => void
  >('movement-semantics', 'addMovementSemanticsExample0');
  try {
    expect(() => declare(graph, inputs)).not.toThrow();
    expect(write).toHaveBeenCalledWith(Float32Array.of(100, 300000, 900000, 1));
    expect(imported).toHaveBeenCalledWith(graph);
  } finally {
    parameters.destroy();
  }
});

for (const [page, phrases] of [
  [
    'operators-cells',
    [
      'row clamp',
      'decodeGPUGroupSum',
      'GPUCellPyramid.getExecutionPlan',
      'through level 22',
      "longitudePolicy: 'wrap'",
      "longitudePolicy: 'family-default'",
      "precision: 'double-single'"
    ]
  ],
  [
    'operators-lines',
    ['decodeGPUTrackSimilarityPairStatus', 'invalidDecisionThreshold', "longitudePolicy: 'wrap'"]
  ],
  [
    'movement-semantics',
    [
      'staypointStatus.overflow',
      'staypointStatus.candidateOverflow',
      'output.status.complete',
      'sourceRows',
      'sourceSelectionMask',
      'missingFixCount',
      'Euclidean'
    ]
  ],
  [
    'global-coordinates',
    [
      '4294967295',
      'input-ring validation',
      '#hemisphere-polygon-overlay-contract',
      '#intrinsic-arc-overlay-contract',
      'decodeGPUSphericalArcOverlayCounts(new Uint8Array'
    ]
  ],
  [
    'track-similarity-measures',
    ['output.similarity', 'output.upperBound', 'output.distance', 'output.lowerBound']
  ]
] as const) {
  it(`review contracts remain explicit on ${page}`, () => {
    for (const phrase of phrases)
      expect(readGuide(page).includes(phrase), `${page}: ${phrase}`).toBe(true);
  });
}

it('B4 a scaled row can saturate before a representable final sum', () => {
  // The predecessor of 2^62 in float32 is 2^62 - 2^(62-24).
  expect((1n << 62n) - (1n << 38n)).toBe(4611685743549480960n);
  expect(getScaledValue(2 ** 62, 1)).toBe(4611685743549480960n);
  expect(isScaledValueSaturated(2 ** 62, 1)).toBe(true);
});

it('B5 level-30 float32 coordinates can skip more than one adjacent S2 column', () => {
  // Published float64 face-I fixture at (7 degrees, 0 degrees). float32's local ULP is 64.
  expect(Math.fround(628013611)).toBe(628013632);
  expect(Math.fround(628013611) - 628013611).toBe(21);
});

it('B6 an invalid decision threshold leaves a valid zero distance unavailable as a decision', () => {
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)},
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(0),
      frechetEpsilon: Float32Array.of(-1)
    },
    {measures: ['frechet', 'frechetWithin']}
  );
  expect(Array.from(result.frechet)).toEqual([0]);
  expect(Array.from(result.frechetWithin)).toEqual([0]);
  expect(Array.from(result.pairStatus)).toEqual([16]);
  expect(decodeGPUTrackSimilarityPairStatus(result.pairStatus)).toEqual([
    expect.objectContaining({
      geometryValid: true,
      decisionValid: false,
      invalidDecisionThreshold: true
    })
  ]);
});

it('B7 converged location labels do not certify a lossy staypoint input', () => {
  const result = computeTrajectoryLocationsOnCPU({
    centroids: Float32Array.of(0, 0),
    tracks: Uint32Array.of(0),
    count: Uint32Array.of(1),
    parameters: Float32Array.of(1, 0, 0, 0),
    staypointStatus: {candidateOverflow: Uint32Array.of(1)}
  });
  expect(Array.from(result.converged)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
  expect(Array.from(result.complete)).toEqual([0]);
  expect(Array.from(result.locationCount)).toEqual([0]);
  expect(Array.from(result.locationIds)).toEqual([0xffffffff]);
});

it('B9 a complete CPU cover requirement can exceed the uint32 count layout', () => {
  // Six faces, 4^16 descendants per face: 25769803776 cannot fit uint32.
  expect(6 * 4 ** 16).toBe(25769803776);
  expect(() =>
    computeS2CapCoveringOnCPU(Float32Array.of(1, 0, 0, Math.PI), {level: 16, capacity: 1})
  ).toThrow(/requiredCount must be at most 4294967295/);
});

it('S1 wrap aligns geohash keys for equivalent unwrapped longitudes', () => {
  const positions = Float32Array.of(190, 5, -170, 5);
  const defaults = computePointToCellOnCPU(positions, {family: 'geohash', resolution: 6});
  const wrapped = computePointToCellOnCPU(positions, {
    family: 'geohash',
    resolution: 6,
    longitudePolicy: 'wrap'
  });
  expect(Array.from(defaults.cells.slice(0, 2))).not.toEqual(
    Array.from(defaults.cells.slice(2, 4))
  );
  expect(Array.from(wrapped.cells.slice(0, 2))).toEqual(Array.from(wrapped.cells.slice(2, 4)));
});

it('S4 spherical count decoders honor byte windows and reject missing words', () => {
  const words = Uint32Array.of(99, 4, 1, 4, 0, 88);
  expect(decodeGPUSphericalArcOverlayCounts(new Uint8Array(words.buffer, 4, 16))).toEqual({
    writtenVertexCount: 4,
    writtenRingCount: 1,
    requiredSegmentCount: 4,
    referenceInside: false
  });
  expect(decodeGPUSphericalConvexIntersectionCounts(Uint32Array.of(3, 3))).toEqual({
    writtenVertexCount: 3,
    requiredVertexCount: 3
  });
  expect(() => decodeGPUSphericalArcOverlayCounts(new Uint8Array(2))).toThrow();
  const decoded = loadGuideFunction<
    () => {convex: {writtenVertexCount: number}; arc: {writtenRingCount: number}}
  >('global-coordinates', 'readSphericalCountExamples')();
  expect(decoded.convex.writtenVertexCount).toBe(3);
  expect(decoded.arc.writtenRingCount).toBe(1);
});

it('B2 all line parameter examples write their packed settings to imported buffers', () => {
  const guide = readGuide('operators-lines');
  for (const buffer of [
    'segmentizeParameters',
    'arcParameters',
    'chunkParameters',
    'locateParameters',
    'toleranceParameters',
    'toleranceBuffer',
    'stopParameters',
    'playheadParameters'
  ]) {
    expect(guide.includes(`${buffer}.write(`), buffer).toBe(true);
    expect(guide.includes(`${buffer}.importToGraph(`), buffer).toBe(true);
  }
});

it('B2 the actual track-similarity guide adds its contributor', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const lines = {
    kind: 'lines',
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2)
  };
  const inputs = {
    linesA: lines,
    linesB: lines,
    pairA: createTransientView(graph, 'pair-a', 'uint32', 1),
    pairB: createTransientView(graph, 'pair-b', 'uint32', 1),
    edr: createTransientView(graph, 'edr', 'float32', 1),
    erp: createTransientView(graph, 'erp', 'float32', 1),
    sspd: createTransientView(graph, 'sspd', 'float32', 1)
  };
  const add = vi.spyOn(graph, 'add');
  loadGuideFunction<(graph: GPUCommandGraph, inputs: Record<string, unknown>) => void>(
    'track-similarity-measures',
    'addTrackSimilarityMeasuresExample0'
  )(graph, inputs);
  expect(add).toHaveBeenCalled();
});

it('S2 the source selection mask precedes compact output clamping', () => {
  // Both endpoints are selected, but capacity one stores only source row zero.
  const result = computeTrajectoryThinningOnCPU(
    {
      positions: Float32Array.of(0, 0, 2, 0),
      timestamps: Float32Array.of(0, 1),
      trackOffsets: Uint32Array.of(0, 2),
      parameters: Float32Array.of(1, 0, 0, 0)
    },
    {mode: 'minimum-distance', capacity: 1}
  );
  expect(Array.from(result.sourceRows)).toEqual([0]);
  expect(Array.from(result.sourceSelectionMask)).toEqual([1, 1]);
  expect(Array.from(result.count)).toEqual([1]);
  expect(Array.from(result.overflow)).toEqual([1]);
});

it('B8 actual compaction and outline fences carry counted sources and both loss outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const scalar = () => createTransientView(graph, `status-${serial++}`, 'uint32', 1);
  const source = {
    cells: createTransientView(graph, 'source-cells', 'uint32x2', 2),
    status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
  };
  const status = {count: scalar(), overflow: scalar(), candidateOverflow: scalar()};
  const add = vi.spyOn(graph, 'add');
  loadGuideFunction<(graph: GPUCommandGraph, inputs: Record<string, unknown>) => void>(
    'operators-cells',
    'addOperatorsCellsExample4'
  )(graph, {
    cells: source.cells,
    diskCells: createTransientView(graph, 'disk', 'uint32x2', 38),
    distances: createTransientView(graph, 'distances', 'uint32', 38),
    counts: createTransientView(graph, 'disk-counts', 'uint32', 2),
    source,
    compacted: createTransientView(graph, 'compacted', 'uint32x2', 2),
    ...status
  });
  const compaction = add.mock.calls
    .map(([producer]) => producer)
    .find(producer => producer instanceof GPUCellCompaction);
  if (!(compaction instanceof GPUCellCompaction)) throw new Error('Missing compaction');
  expect(compaction.props.source).toMatchObject(source);
  expect(compaction.props.output.status.candidateOverflow).toBe(status.candidateOverflow);
  const outlineStatus = {
    count: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar(),
    requiredCount: scalar()
  };
  const ringOutput = {
    positions: createTransientView(graph, 'ring-positions', 'float32x2', 32),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 5),
    status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
  };
  loadGuideFunction<(graph: GPUCommandGraph, inputs: Record<string, unknown>) => void>(
    'operators-cells',
    'addOperatorsCellsExample8'
  )(graph, {
    source,
    groups: createTransientView(graph, 'source-groups', 'uint32', 2),
    rows: createTransientView(graph, 'outline-rows', 'uint32', 12),
    segmentCells: createTransientView(graph, 'outline-cells', 'uint32x2', 12),
    edgeIndices: createTransientView(graph, 'edge-indices', 'uint32', 12),
    endpoints: createTransientView(graph, 'endpoints', 'float32x4', 12),
    segmentGroups: createTransientView(graph, 'segment-groups', 'uint32', 12),
    ringOutput,
    ...outlineStatus
  });
  const outline = add.mock.calls
    .map(([producer]) => producer)
    .find(producer => producer instanceof GPUCellSetOutline);
  if (!(outline instanceof GPUCellSetOutline)) throw new Error('Missing outline');
  expect(outline.props.source).toMatchObject(source);
  expect(outline.props.output.status.candidateOverflow).toBe(outlineStatus.candidateOverflow);
});
