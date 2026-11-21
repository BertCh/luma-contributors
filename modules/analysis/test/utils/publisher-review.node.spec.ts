// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {formatZoneEventsStatus} from '../../../../examples/deck/spatial-analysis-explorer/modes/zone-events-status';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createPublishNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {createSpatialJoinFinalizeNodes} from '../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {GPUBufferSelection} from '../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {GPUBoundsFilter} from '../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {GPUPairGather} from '../../src/gpu-spatial-analysis/pair-gather/gpu-pair-gather';
import {GPUFlowAggregation} from '../../src/gpu-network/flow-aggregation/gpu-flow-aggregation';
import {
  GPURegionStatistics,
  GPU_REGION_STATISTICS_FLAGS,
  GPU_REGION_STATISTICS_SUMMARY_LAYOUT
} from '../../src/gpu-spatial-analysis/region-statistics/index';
import {GPUNetworkPathExtraction} from '../../src/gpu-network/network-analysis/gpu-network-path-extraction';
import {computeNetworkPathExtractionOnCPU} from '../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {GPURandomPointsOnLine} from '../../src/gpu-spatial-analysis/dot-density/gpu-random-points-on-line';
import {GPURandomPointsInPolygon} from '../../src/gpu-spatial-analysis/dot-density/gpu-random-points-in-polygon';
import {GPUZoneEvents} from '../../src/gpu-spatial-analysis/trajectory-zones/gpu-zone-events';
import {addFleetDwellZoneEventsRecipe} from '../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {GPUTrajectoryEncounters} from '../../src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-encounters';
import {GPUTrajectoryClosestApproach} from '../../src/gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach';
import {GPUTrajectoryStaypoints} from '../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let counter = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length = 1
  ) => createTransientView(graph, `review-${counter++}`, format, length);
  const scalar = () => view('uint32');
  const compact = (capacity = 2) => ({
    ids: view('uint32', capacity),
    status: {
      count: scalar(),
      overflow: scalar(),
      candidateOverflow: scalar(),
      requiredCount: scalar()
    }
  });
  return {device, graph, view, scalar, compact};
}

function executeScalars(nodes: readonly GPUCommandNode[], values: Record<string, Uint32Array>) {
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      const bindings = Object.fromEntries(
        new WgslReflect(kernel.source).storage.map(declaration => {
          values[declaration.name] ??= new Uint32Array([37]);
          return [declaration.binding, values[declaration.name]];
        })
      );
      new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
    }
  } finally {
    errors.mockRestore();
  }
}

test('R1 anti finalization separates candidate loss and clears the caller candidate word', () => {
  const {device, graph, view, compact} = createFixture();
  try {
    const nodes = new GPUSpatialPredicateJoin({
      left: {kind: 'points', positions: view('float32x2')},
      right: {kind: 'points', positions: view('float32x2', 2)},
      predicate: 'intersects',
      how: 'anti',
      candidateCapacity: 1,
      output: {unmatched: compact()}
    })
      .getCommandNodes(graph)
      .filter(node => node.id.includes('-finalize'));
    const values = {
      state: new Uint32Array([2, 0, 0, 0, 0]),
      bvhOverflow: new Uint32Array([0]),
      unmatchedOverflow: new Uint32Array([37]),
      unmatchedCandidateOverflow: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.unmatchedOverflow[0], values.unmatchedCandidateOverflow[0]]).toEqual([0, 1]);
    values.state.set([1, 0, 0, 0, 0]);
    executeScalars(nodes, values);
    expect([values.unmatchedOverflow[0], values.unmatchedCandidateOverflow[0]]).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

test.each([
  {cause: 'truncated lasso', flag: GPU_REGION_STATISTICS_FLAGS.regionTruncated},
  {cause: 'truncated grid candidates', flag: GPU_REGION_STATISTICS_FLAGS.candidatesTruncated}
])('R5 compact region output treats $cause as candidate loss', ({flag}) => {
  const {device, graph, view, compact} = createFixture();
  try {
    const nodes = new GPURegionStatistics({
      positions: view('float32x2'),
      values: view('float32'),
      region: {kind: 'polygon', vertices: view('float32x2', 5), vertexCount: view('uint32')},
      output: {summary: view('uint32', 8), selectedRows: compact(4)}
    })
      .getCommandNodes(graph)
      .filter(node => /selection-finalize|selection-candidate-status/.test(node.id));
    const summary = new Uint32Array(16);
    summary[GPU_REGION_STATISTICS_SUMMARY_LAYOUT.flags] = flag;
    const values = {
      totalIn: new Uint32Array([1]),
      overflowIn0: new Uint32Array([1]),
      summary,
      overflowOut: new Uint32Array([37]),
      candidateOverflow: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.overflowOut[0], values.candidateOverflow[0]]).toEqual([0, 1]);
    summary[GPU_REGION_STATISTICS_SUMMARY_LAYOUT.flags] = 0;
    executeScalars(nodes, values);
    expect([values.overflowOut[0], values.candidateOverflow[0]]).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

test('staypoint publication preserves bounded-search loss and clears it on reuse', () => {
  const {device, graph, view, compact} = createFixture();
  try {
    const nodes = new GPUTrajectoryStaypoints({
      id: 'review-staypoints',
      positions: view('float32x2', 2),
      timestamps: view('float32', 2),
      trackOffsets: view('uint32', 2),
      parameters: view('float32', 4),
      maximumSearchRows: 256,
      output: compact(8)
    })
      .getCommandNodes(graph)
      .filter(node => /-candidate-reset$|-candidate-overflow$|-publish$/.test(node.id));
    const candidate = new Uint32Array([37]);
    const values = {
      fillTarget: candidate,
      candidateOverflow: candidate,
      candidateOverflowOut: candidate,
      reach: new Uint32Array([1, 0]),
      flags: new Uint32Array([3, 0]),
      totalIn: new Uint32Array([1]),
      overflowOut: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.overflowOut[0], candidate[0]]).toEqual([0, 1]);
    values.flags[0] = 1;
    executeScalars(nodes, values);
    expect([values.overflowOut[0], candidate[0]]).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

test.each([
  0, 4, 256
])('staypoint status validates separate candidate publication at offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const words = view('uint32', 65);
    const output = compact(8);
    output.status.count = graph.createDataView(words.buffer, {format: 'uint32', length: 1});
    output.status.candidateOverflow = graph.createDataView(words.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUTrajectoryStaypoints({
        id: 'review-staypoint-layout',
        positions: view('float32x2', 2),
        timestamps: view('float32', 2),
        trackOffsets: view('uint32', 2),
        parameters: view('float32', 4),
        output
      }).getCommandNodes(graph);
    if (byteOffset === 0) expect(construct).toThrow(/overlap/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});

test('R5 bounded paths report earlier walk loss independently on GPU and CPU forms', () => {
  const {device, graph, view, compact} = createFixture();
  try {
    const nodes = new GPUNetworkPathExtraction({
      predecessors: view('uint32', 3),
      costs: view('float32', 3),
      targets: view('uint32'),
      maximumPathLength: 1,
      output: compact(4)
    })
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-publish'));
    const values = {
      totalIn: new Uint32Array([0]),
      overflowIn0: new Uint32Array([1]),
      candidateOverflowIn0: new Uint32Array([1]),
      overflowOut: new Uint32Array([37]),
      candidateOverflowOut: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.overflowOut[0], values.candidateOverflowOut[0]]).toEqual([0, 1]);
    const result = computeNetworkPathExtractionOnCPU({
      predecessors: new Uint32Array([0xffffffff, 0, 1]),
      costs: new Float32Array([0, 1, 2]),
      targets: new Uint32Array([2]),
      capacity: 4,
      maximumPathLength: 1
    });
    expect(Array.from(result.count)).toEqual([0]);
    expect(Array.from(result.candidateOverflow)).toEqual([1]);
  } finally {
    device.destroy();
  }
});

test.each([
  'line',
  'polygon'
] as const)('R5 %s sampling slot clamping is candidate loss with ample final capacity', kind => {
  const {device, graph, view, compact} = createFixture();
  try {
    const counts = view('uint32', 65536);
    const parameters = view('uint32', 8);
    const output = {
      positions: view('float32x2', 65536),
      points: compact(65536),
      dots: compact(65536)
    };
    const producer =
      kind === 'line'
        ? new GPURandomPointsOnLine({
            positions: view('float32x2', 2),
            lineOffsets: view('uint32', 65537),
            counts,
            parameters,
            output
          })
        : new GPURandomPointsInPolygon({
            positions: view('float32x2', 4),
            featureOffsets: view('uint32', 65537),
            polygonOffsets: view('uint32', 2),
            ringOffsets: view('uint32', 2),
            counts,
            parameters,
            output
          });
    const nodes = producer.getCommandNodes(graph).filter(node => node.id.endsWith('-publish'));
    const values = {
      totalIn: new Uint32Array([65535]),
      overflowIn0: new Uint32Array([1]),
      candidateOverflowIn0: new Uint32Array([1]),
      overflowOut: new Uint32Array([37]),
      candidateOverflowOut: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.overflowOut[0], values.candidateOverflowOut[0]]).toEqual([0, 1]);
  } finally {
    device.destroy();
  }
});

test('R6 fleet dwell returns the caller compact candidate view and attributes event-stage loss to it', () => {
  const {device, graph, view, scalar, compact} = createFixture();
  try {
    const events = compact();
    const result = addFleetDwellZoneEventsRecipe(graph, {
      trajectories: {
        kind: 'trajectories',
        positions: view('float32x2', 3),
        timestamps: view('float32', 3),
        trackOffsets: view('uint32', 2)
      },
      zones: {
        kind: 'polygons',
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2)
      },
      candidateCapacity: 8,
      trackEventCapacity: 1,
      eventCapacity: 2,
      output: {events, status: {candidateOverflow: scalar()}}
    });
    expect(result.outputs.events.status.candidateOverflow === events.status.candidateOverflow).toBe(
      true
    );
    expect(
      result.status.stages.find(stage => stage.stage === 'zone-events')?.status
        .candidateOverflow === events.status.candidateOverflow
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

test('R7 visit table inherits scratch loss independently from the event-list track cap', () => {
  const {device, graph, view, compact, scalar} = createFixture();
  try {
    const producer = new GPUZoneEvents({
      tracks: {
        kind: 'trajectories',
        positions: view('float32x2', 3),
        timestamps: view('float32', 3),
        trackOffsets: view('uint32', 2)
      },
      zones: {
        kind: 'polygons',
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2)
      },
      candidateCapacity: 1,
      trackEventCapacity: 1,
      output: {events: compact(), visitTable: compact(), status: {candidateOverflow: scalar()}}
    });
    const nodes = producer
      .getCommandNodes(graph)
      .filter(node => node.id.endsWith('-table-publish'));
    const values = {
      totalIn: new Uint32Array([1]),
      candidateOverflowIn0: new Uint32Array([1]),
      candidateOverflowOut: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect(values.candidateOverflowOut[0]).toBe(1);
    values.candidateOverflowIn0[0] = 0;
    executeScalars(nodes, values);
    expect(values.candidateOverflowOut[0]).toBe(0);
  } finally {
    device.destroy();
  }
});

test.each([
  'encounters',
  'closest'
] as const)('R7 %s compact pairs inherit the candidate status', kind => {
  const {device, graph, view, scalar, compact} = createFixture();
  try {
    const pairs = compact();
    const producer =
      kind === 'encounters'
        ? new GPUTrajectoryEncounters({
            samples: view('float32x2', 2),
            trackCount: 2,
            bucketCount: 1,
            distance: view('float32'),
            cellSize: 1,
            bounds: [0, 0, 1, 1],
            hitCapacity: 1,
            output: {pairs, status: {candidateOverflow: scalar()}}
          })
        : new GPUTrajectoryClosestApproach({
            positions: view('float32x2', 4),
            timestamps: view('float32', 4),
            trackOffsets: view('uint32', 3),
            distance: view('float32'),
            cellSize: 1,
            timeCellSize: 1,
            entryCapacity: 4,
            hitCapacity: 1,
            output: {pairs, status: {entryOverflow: scalar(), hitOverflow: scalar()}}
          });
    const nodes = producer.getCommandNodes(graph).filter(node => node.id.endsWith('-publish'));
    const values = {
      totalIn: new Uint32Array([1]),
      candidateOverflowIn0: new Uint32Array([1]),
      candidateOverflowOut: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect(values.candidateOverflowOut[0]).toBe(1);
    values.candidateOverflowIn0[0] = 0;
    executeScalars(nodes, values);
    expect(values.candidateOverflowOut[0]).toBe(0);
  } finally {
    device.destroy();
  }
});

test.each([
  1, 4, 5
])('R1 compact matches separate candidate loss and clear it (capacity %i)', candidateCapacity => {
  const {device, graph, view, scalar, compact} = createFixture();
  try {
    const nodes = createSpatialJoinFinalizeNodes(graph, {
      id: 'review-finalize',
      operation: 'Review',
      state: view('uint32', 4),
      bvhOverflow: scalar(),
      candidateCapacity,
      overflow: scalar(),
      matches: compact(20)
    });
    const values = {
      state: new Uint32Array([candidateCapacity === 1 ? 2 : 10, 0, candidateCapacity, 0]),
      bvhOverflow: new Uint32Array([0]),
      matchCount: new Uint32Array([37]),
      matchTotalCount: new Uint32Array([37]),
      overflow: new Uint32Array([37]),
      matchOverflow: new Uint32Array([37]),
      matchCandidateOverflow: new Uint32Array([37])
    };
    executeScalars(nodes, values);
    expect([values.matchOverflow[0], values.matchCandidateOverflow[0]]).toEqual([0, 1]);
    expect([values.matchCount[0], values.matchTotalCount[0], values.overflow[0]]).toEqual([
      candidateCapacity,
      candidateCapacity,
      1
    ]);
    const cleanCount = Math.min(candidateCapacity, 2);
    values.state.set([cleanCount, 0, cleanCount, 0]);
    executeScalars(nodes, values);
    expect([values.matchOverflow[0], values.matchCandidateOverflow[0]]).toEqual([0, 0]);
    expect([values.matchCount[0], values.matchTotalCount[0], values.overflow[0]]).toEqual([
      cleanCount,
      cleanCount,
      0
    ]);
  } finally {
    device.destroy();
  }
});

test('R2 publisher refuses explicitly supplied candidate sources without a destination', () => {
  const {device, graph, view, scalar} = createFixture();
  try {
    expect(() =>
      createPublishNode(graph, {
        id: 'review-publisher',
        operation: 'Review',
        requiredCount: scalar(),
        candidateOverflowSources: [scalar()],
        output: {ids: view('uint32', 2), status: {count: scalar(), overflow: scalar()}}
      })
    ).toThrow(/review-publisher.*candidateOverflow/);
  } finally {
    device.destroy();
  }
});

test('R8 explorer reports earlier losses even when the final event capacity fits', () => {
  expect(formatZoneEventsStatus(0, 1, 1, 0)).toContain('per-track');
  expect(formatZoneEventsStatus(0, 1, 0, 1)).toContain('scratch');
  expect(formatZoneEventsStatus(0, 0, 0, 0)).toBe('no overflow');
});

test('R9 pair-gather lens allocates and captures both ends of candidate status', () => {
  const source = readFileSync(
    resolve(
      'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.lens.ts'
    ),
    'utf8'
  );
  expect(/uploads\.view\(\s*'pairCandidateOverflow'/.test(source)).toBe(true);
  expect(source).toContain('view: result.candidateOverflow');
});

test('R10 coarsening example wires both flags and names the excluded-label condition', () => {
  const source = readFileSync(resolve('docs/api/gpu-network/operators.md'), 'utf8');
  const guide = source.slice(
    source.indexOf('### `GPUNetworkCoarsening`'),
    source.indexOf('#### How it computes', source.indexOf('### `GPUNetworkCoarsening`'))
  );
  expect(guide).toContain('status: {count, overflow, candidateOverflow}');
  expect(guide).toContain('label >= groupCapacity');
  expect(guide).not.toContain('excluded and sets `edges.status.overflow`');
});

test('R3 buffer selection supports all compact words and draw count on eight bindings', () => {
  const {device, graph, view, scalar, compact} = createFixture();
  try {
    const producer = new GPUBufferSelection({
      points: {kind: 'points', positions: view('float32x2')},
      features: {kind: 'points', positions: view('float32x2')},
      radius: view('float32'),
      candidateCapacity: 1,
      output: {matches: compact(), drawInstanceCount: scalar()}
    });
    expect(() => producer.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

test('R4 bounds filter rejects an ID word reused for the candidate destination', () => {
  const {device, view, compact} = createFixture();
  try {
    const output = compact(1);
    output.status.candidateOverflow = output.ids;
    expect(
      () =>
        new GPUBoundsFilter({bounds: view('float32x4', 2), parameters: view('float32', 8), output})
    ).toThrow(/overlap/);
  } finally {
    device.destroy();
  }
});

test.each([
  0, 4, 256
])('R4 gather validates separate-pass status words at byte offset %i', byteOffset => {
  const {device, graph, view, scalar} = createFixture();
  try {
    const storage = view('uint32', 65);
    const count = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    const candidate = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUPairGather({
        pairs: {
          leftRows: scalar(),
          rightRows: scalar(),
          status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
        },
        output: {
          leftRows: scalar(),
          rightRows: scalar(),
          status: {count, overflow: scalar(), candidateOverflow: candidate}
        }
      }).getCommandNodes(graph);
    if (byteOffset === 0) expect(construct).toThrow(/overlap/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  0, 4, 256
])('R4 flow validates status words sharing a buffer at byte offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const storage = view('uint32', 65);
    const output = compact();
    output.status.count = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    output.status.candidateOverflow = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUFlowAggregation({
        zones: {kind: 'ids', zoneCount: 2},
        originZoneIds: view('uint32'),
        destinationZoneIds: view('uint32'),
        pairCapacity: 2,
        output
      });
    if (byteOffset < 256) expect(construct).toThrow(/overlap/);
    else expect(() => construct().getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  0, 4, 256
])('R4 buffer selection validates separate-pass candidate word at offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const storage = view('uint32', 65),
      matches = compact();
    matches.status.count = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    matches.status.candidateOverflow = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUBufferSelection({
        points: {kind: 'points', positions: view('float32x2')},
        features: {kind: 'points', positions: view('float32x2')},
        radius: view('float32'),
        candidateCapacity: 1,
        output: {matches}
      }).getCommandNodes(graph);
    if (byteOffset === 0) expect(construct).toThrow(/overlap/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  4, 256
])('R4 buffer final pass accepts count beside a separate-pass aggregate at offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const storage = view('uint32', 65);
    const matches = compact();
    const aggregate = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    matches.status.count = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    expect(() =>
      new GPUBufferSelection({
        points: {kind: 'points', positions: view('float32x2')},
        features: {kind: 'points', positions: view('float32x2')},
        radius: view('float32'),
        candidateCapacity: 1,
        output: {matches, status: {overflow: aggregate}}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

test.each([
  4, 256
])('R4 buffer candidate pass rejects mixed access to its aggregate buffer at offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const storage = view('uint32', 65);
    const matches = compact();
    const aggregate = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    matches.status.candidateOverflow = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUBufferSelection({
        points: {kind: 'points', positions: view('float32x2')},
        features: {kind: 'points', positions: view('float32x2')},
        radius: view('float32'),
        candidateCapacity: 1,
        output: {matches, status: {overflow: aggregate}}
      }).getCommandNodes(graph);
    expect(construct).toThrow(/read-only and writable in the same dispatch/);
  } finally {
    device.destroy();
  }
});

test.each([
  0, 4, 256
])('R4 bounds filter validates same-pass candidate word at offset %i', byteOffset => {
  const {device, graph, view, compact} = createFixture();
  try {
    const storage = view('uint32', 65),
      output = compact();
    output.status.count = graph.createDataView(storage.buffer, {format: 'uint32', length: 1});
    output.status.candidateOverflow = graph.createDataView(storage.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset
    });
    const construct = () =>
      new GPUBoundsFilter({
        bounds: view('float32x4', 2),
        parameters: view('float32', 8),
        output
      }).getCommandNodes(graph);
    if (byteOffset < 256) expect(construct).toThrow(/overlap/);
    else expect(construct).not.toThrow();
  } finally {
    device.destroy();
  }
});
