// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {getSnapRoundingGridCell} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  COMPONENTS,
  createOracleScene,
  POISON,
  STATUS_DESTINATIONS,
  UNSUPPORTED_STATUS_FIELDS,
  NONE,
  SQUARE_SEGMENTS,
  type ViewFactory,
  type Scene,
  type StatusDestination
} from './make-valid-output-oracles-fixtures';

useCPUExecutorTestHooks();

function createCPUScene(
  scene: Scene = 'squares',
  snapped = false,
  uncertainty?: StatusDestination
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: ViewFactory = (format, length, values) => {
    const result = createTransientView(graph, `oracle-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * COMPONENTS[format]).fill(POISON);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  const fixture = createOracleScene(view, scene, uncertainty, snapped);
  return {graph, memory, fixture, view};
}

it('production make-valid edge-row and directed publishers match labeled square literals on CPU', () => {
  const {graph, memory, fixture} = createCPUScene();
  recordWGSLKernelNodes(graph);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const {nodedSegments, directedEdges} = fixture.output;
  // Stage boundary fixture, derived by hand: ring 1 gained one closure row, hence closed
  // edge rows 5..8 must be restored to original polygon rows 4..7. This does not execute
  // noding or ring assembly; their full output is asserted by the headless companion.
  memory.write(
    nodedSegments.endpoints,
    Float32Array.from([...SQUARE_SEGMENTS.flat(), ...new Array(16).fill(0)])
  );
  memory.write(
    nodedSegments.sourceRingIds,
    Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1, NONE, NONE, NONE, NONE)
  );
  memory.write(
    nodedSegments.sourceFeatureIds,
    Uint32Array.of(70, 70, 70, 70, 80, 80, 80, 80, NONE, NONE, NONE, NONE)
  );
  memory.write(
    nodedSegments.sourceEdgeIds,
    Uint32Array.of(0, 1, 2, 3, 5, 6, 7, 8, NONE, NONE, NONE, NONE)
  );
  memory.write(nodedSegments.status.count, Uint32Array.of(8));
  memory.write(nodedSegments.status.requiredCount, Uint32Array.of(8));
  memory.write(nodedSegments.status.candidateOverflow, Uint32Array.of(0));
  executeCommandNodesOnCPU(
    nodes.filter(
      node =>
        node.id === 'make-valid-oracle-source-edge-rows' ||
        node.id.startsWith('make-valid-oracle-polygonize-directed-edge-')
    ),
    memory
  );
  expect(Array.from(memory.readUint32(nodedSegments.sourceEdgeIds))).toEqual([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    NONE,
    NONE,
    NONE,
    NONE
  ]);
  expect(Array.from(memory.readFloat32(directedEdges.endpoints))).toEqual([
    1,
    2,
    3,
    2,
    3,
    2,
    1,
    2,
    3,
    2,
    3,
    4,
    3,
    4,
    3,
    2,
    3,
    4,
    1,
    4,
    1,
    4,
    3,
    4,
    1,
    4,
    1,
    2,
    1,
    2,
    1,
    4,
    6,
    1,
    9,
    1,
    9,
    1,
    6,
    1,
    9,
    1,
    9,
    3,
    9,
    3,
    9,
    1,
    9,
    3,
    6,
    3,
    6,
    3,
    9,
    3,
    6,
    3,
    6,
    1,
    6,
    1,
    6,
    3,
    ...new Array(32).fill(0)
  ]);
  expect(Array.from(memory.readUint32(directedEdges.sourceSegmentIds))).toEqual([
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    ...new Array(8).fill(NONE)
  ]);
  expect(Array.from(memory.readUint32(directedEdges.sourceFeatureIds))).toEqual([
    70,
    70,
    70,
    70,
    70,
    70,
    70,
    70,
    80,
    80,
    80,
    80,
    80,
    80,
    80,
    80,
    ...new Array(8).fill(NONE)
  ]);
  expect(Array.from(memory.readUint32(directedEdges.twinIds))).toEqual([
    1,
    0,
    3,
    2,
    5,
    4,
    7,
    6,
    9,
    8,
    11,
    10,
    13,
    12,
    15,
    14,
    ...new Array(8).fill(NONE)
  ]);
  expect(Array.from(memory.readUint32(directedEdges.status.count))).toEqual([16]);
  expect(Array.from(memory.readUint32(directedEdges.status.requiredCount))).toEqual([16]);
  expect(Array.from(memory.readUint32(directedEdges.status.overflow!))).toEqual([0]);
  expect(Array.from(memory.readUint32(directedEdges.status.candidateOverflow))).toEqual([0]);
});

it('snap grid lower bound is the independently derived power of two', () => {
  expect(getSnapRoundingGridCell(0.3)).toBe(0.5);
});

it.each([
  false,
  true
])('retraced closing-edge provenance keeps its source row (snapped=%s)', snapped => {
  const {graph, memory, fixture} = createCPUScene('dangle', snapped);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const firstDirected = nodes.findIndex(
    node => node.id === 'make-valid-oracle-polygonize-directed-edge-geometry'
  );
  expect(firstDirected).toBeGreaterThan(0);
  executeCommandNodesOnCPU(nodes.slice(0, firstDirected), memory);
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === 'make-valid-oracle-source-edge-rows'),
    memory
  );
  const segments = fixture.output.nodedSegments;
  // A->B is original row 0 and its implicit closing traversal B->A is row 1.
  // Both traverse their own edge from t=0 to t=1, even though their midpoints coincide.
  expect(Array.from(memory.readUint32(segments.status.count))).toEqual([2]);
  expect(Array.from(memory.readUint32(segments.sourceEdgeIds)).slice(0, 2)).toEqual([0, 1]);
  expect(Array.from(memory.readFloat32(segments.sourceStartParameters)).slice(0, 2)).toEqual([
    0, 0
  ]);
  expect(Array.from(memory.readFloat32(segments.sourceEndParameters)).slice(0, 2)).toEqual([1, 1]);
});

it.each([
  undefined,
  ...STATUS_DESTINATIONS
])('uncertifiable input reaches the %s uncertainty destination and all candidate statuses', destination => {
  const {graph, memory, fixture} = createCPUScene('uncertain', false, destination);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const firstDirected = nodes.findIndex(
    node => node.id === 'make-valid-oracle-polygonize-directed-edge-geometry'
  );
  executeCommandNodesOnCPU(nodes.slice(0, firstDirected), memory);
  executeCommandNodesOnCPU(
    nodes.filter(
      node =>
        node.id === 'make-valid-oracle-polygonize-directed-edge-status' ||
        node.id === 'make-valid-oracle-polygonize-publish-candidate-overflow' ||
        node.id.startsWith('make-valid-oracle-polygonize-publish-uncertain-')
    ),
    memory
  );
  // Two source traversals per ring give 2*2=4 uncertifiable cross-ring pairs. This runs
  // the real intersection/noding producer, not a manually injected publisher input.
  for (const status of Object.values(fixture.statuses)) {
    expect(Array.from(memory.readUint32(status.candidateOverflow))).toEqual([1]);
    if (status.uncertainCount)
      expect(Array.from(memory.readUint32(status.uncertainCount))).toEqual([4]);
  }
  expect(Array.from(memory.readUint32(fixture.statuses.noded.count))).toEqual([4]);
  expect(Array.from(memory.readUint32(fixture.statuses.directed.count))).toEqual([8]);
});

it('the four-pair oracle distinguishes a publisher wired to a fresh zero buffer', () => {
  const {graph, memory, fixture, view} = createCPUScene('uncertain', false, 'top');
  recordWGSLKernelNodes(graph);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const firstDirected = nodes.findIndex(
    node => node.id === 'make-valid-oracle-polygonize-directed-edge-geometry'
  );
  executeCommandNodesOnCPU(nodes.slice(0, firstDirected), memory);
  const publisher = getWGSLKernelNodeRecord(
    graph,
    'make-valid-oracle-polygonize-publish-uncertain-0'
  )!;
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === publisher.id),
    memory
  );
  expect(Array.from(memory.readUint32(fixture.statuses.top.uncertainCount!))).toEqual([4]);
  const disconnected = view('uint32', 1, [0]);
  const mutant = createWGSLKernelNode(graph, {
    id: 'disconnected-uncertainty-publisher',
    operation: 'test',
    invocationCount: 1,
    bindings: publisher.bindings.map(binding =>
      binding.name === 'uncertainCount' ? {...binding, view: disconnected} : binding
    ),
    body: publisher.body
  });
  executeCommandNodesOnCPU([mutant], memory);
  // The former zero-only oracle would accept this wrong connection; the new input expects 4.
  expect(Array.from(memory.readUint32(fixture.statuses.top.uncertainCount!))).toEqual([0]);
  expect(Array.from(memory.readUint32(fixture.statuses.top.uncertainCount!))).not.toEqual([4]);
});

it('holed layout distinguishes feature offsets from polygon offsets including padded tails', () => {
  const {graph, memory, fixture} = createCPUScene('hole');
  recordWGSLKernelNodes(graph);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const layout = getWGSLKernelNodeRecord(graph, 'make-valid-oracle-polygon-layout')!;
  const write = (name: string, values: number[]) =>
    memory.write(
      layout.bindings.find(binding => binding.name === name)!.view,
      Uint32Array.from(values)
    );
  // Independent stage fixture: one retained feature, containing two rings. All other
  // candidates are dropped. The inclusive kept-feature and kept-ring scans end at 1 and 2.
  write('keepScan', [1, 1, 1, 1, 1, 1, 1, 1]);
  write('ringScan', [2, 2, 2, 2, 2, 2, 2, 2]);
  write('ringCounts', [2, 0, 0, 0, 0, 0, 0, 0]);
  write('keptIds', [0, NONE, NONE, NONE, NONE, NONE, NONE, NONE]);
  write('faceSourceIds', [0, NONE, NONE, NONE, NONE, NONE, NONE, NONE]);
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === layout.id),
    memory
  );
  const features = Array.from(memory.readUint32(fixture.output.polygons.featureOffsets));
  const polygons = Array.from(memory.readUint32(fixture.output.polygons.polygonOffsets));
  expect(features).toEqual([0, 1, 1, 1, 1, 1, 1, 1, 1]);
  expect(polygons).toEqual([0, 2, 2, 2, 2, 2, 2, 2, 2]);
  // Reproduce the proposed wrong implementation: copying polygon offsets would have
  // passed the former one-ring-per-feature fixture but fails the new feature expectation.
  memory.write(fixture.output.polygons.featureOffsets, Uint32Array.from(polygons));
  expect(Array.from(memory.readUint32(fixture.output.polygons.featureOffsets))).not.toEqual(
    features
  );
});

it('make-valid initializes duplicate flags before propagating retraced dangle ownership', () => {
  const {graph, memory, fixture} = createCPUScene('dangle');
  recordWGSLKernelNodes(graph);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const marker = getWGSLKernelNodeRecord(
    graph,
    'make-valid-oracle-polygonize-faces-mark-duplicate-owners'
  )!;
  const propagation = getWGSLKernelNodeRecord(
    graph,
    'make-valid-oracle-polygonize-faces-propagate-duplicate-owners'
  )!;
  // A two-vertex implicitly closed ring has A->B and B->A source segments. Expanding
  // each into forward/reverse sides gives owners [0,1,1,0]; remaining rows are inactive.
  // Marking must initialize flags, rather than preserve any bit from poison 777 (0x309),
  // which includes the cancellation bit 8. Only duplicate rows receive internal bit 31.
  const owners = Uint32Array.from([
    0,
    1,
    1,
    0,
    ...Array.from({length: 20}, (_, index) => index + 4)
  ]);
  memory.write(marker.bindings.find(binding => binding.name === 'duplicateOwner')!.view, owners);
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === marker.id),
    memory
  );
  const flags = memory.readUint32(fixture.output.directedEdges.flags);
  expect(Array.from(flags)).toEqual([0, 0, 0x80000000, 0x80000000, ...new Array(20).fill(0)]);

  // Matching produces dangling bit 2 on the two unique orientations. Duplicate sides
  // must inherit exactly that bit; the artificial cancellation bit must not survive.
  flags[0] = 2;
  flags[1] = 2;
  memory.write(fixture.output.directedEdges.flags, flags);
  for (const name of ['leader', 'segmentRings', 'segmentVertices']) {
    const binding = propagation.bindings.find(value => value.name === name)!;
    memory.write(binding.view, new Uint32Array(24).fill(NONE));
  }
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === propagation.id),
    memory
  );
  expect(Array.from(memory.readUint32(fixture.output.directedEdges.flags))).toEqual([
    2,
    2,
    2,
    2,
    ...new Array(20).fill(0)
  ]);
});

it('duplicate marking preserves flags initialized by the optional cancellation pass', () => {
  const {graph, memory, fixture, view} = createCPUScene('dangle');
  recordWGSLKernelNodes(graph);
  const edges = fixture.output.directedEdges;
  const nodes = new GPUSegmentRingAssembly({
    id: 'cancelled-marker',
    endpoints: edges.endpoints,
    pairedOpposites: true,
    cancelOpposingSegments: true,
    output: {
      positions: view('float32x2', 64),
      ringOffsets: view('uint32', 9),
      segmentFlags: edges.flags,
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  const marker = getWGSLKernelNodeRecord(graph, 'cancelled-marker-mark-duplicate-owners')!;
  memory.write(
    marker.bindings.find(binding => binding.name === 'duplicateOwner')!.view,
    Uint32Array.from([0, 1, 1, 0, ...Array.from({length: 20}, (_, index) => index + 4)])
  );
  // The cancellation pass has already initialized every row. Its public cancelled bit 8
  // must survive on both owners and duplicates; only an uncancelled duplicate gains bit 31.
  memory.write(edges.flags, Uint32Array.from([8, 0, 8, 0, ...new Array(20).fill(0)]));
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === marker.id),
    memory
  );
  expect(Array.from(memory.readUint32(edges.flags))).toEqual([
    8,
    0,
    8,
    0x80000000,
    ...new Array(20).fill(0)
  ]);
});

it.each(
  STATUS_DESTINATIONS
)('make-valid publishes a nonzero isolated %s uncertainty counter on CPU', destination => {
  const {graph, memory, fixture} = createCPUScene('squares', false, destination);
  recordWGSLKernelNodes(graph);
  const nodes = fixture.contributor.getCommandNodes(graph);
  const publisher = getWGSLKernelNodeRecord(
    graph,
    'make-valid-oracle-polygonize-publish-uncertain-0'
  )!;
  // A stage-boundary counter of three uncertifiable pairs must be copied verbatim, even
  // when this is the only destination. This checks publication; the full finite-input
  // zero-count expectation is independently checked in the headless graph.
  memory.write(
    publisher.bindings.find(binding => binding.name === 'uncertainCount')!.view,
    Uint32Array.of(3)
  );
  executeCommandNodesOnCPU(
    nodes.filter(node => node.id === publisher.id),
    memory
  );
  expect(Array.from(memory.readUint32(fixture.statuses[destination].uncertainCount!))).toEqual([3]);
});

it('make-valid refuses inherited status destinations that no kernel publishes', () => {
  for (const destination of STATUS_DESTINATIONS) {
    for (const field of UNSUPPORTED_STATUS_FIELDS) {
      const {fixture, view} = createCPUScene();
      Object.assign(fixture.statuses[destination], {[field]: view('uint32', 1)});
      expect(() => new GPUMakeValid(fixture.callerProps), `${destination}.${field}`).toThrow(
        new RegExp(`make-valid-oracle.*${field}.*unsupported.*use count`)
      );
    }
  }
});

it('make-valid snapshots caller output/status maps while preserving original view bindings', () => {
  const {graph, fixture, view} = createCPUScene();
  recordWGSLKernelNodes(graph);
  const originalCount = fixture.statuses.directed.count;
  const originalEndpoints = fixture.output.directedEdges.endpoints;
  const originalInput = fixture.callerProps.polygons.positions;
  const originalFeatureOffsets = fixture.output.polygons.featureOffsets;
  Object.assign(fixture.statuses.directed, {invalidCount: view('uint32', 1)});
  fixture.statuses.directed.count = view('uint32', 1);
  fixture.output.directedEdges.endpoints = view('float32x4', 24);
  fixture.output.polygons.featureOffsets = view('uint32', 9);
  fixture.callerProps.polygons.positions = view('float32x2', 8);
  fixture.callerProps.precision.vertexTolerance = 2;
  // Mutating caller maps must not revalidate under a delegated ID or switch bound views.
  expect(() => fixture.contributor.getCommandNodes(graph)).not.toThrow();
  expect(fixture.contributor.props.output.directedEdges.status.count).toBe(originalCount);
  expect(fixture.contributor.props.output.directedEdges.endpoints).toBe(originalEndpoints);
  expect(fixture.contributor.props.polygons.positions).toBe(originalInput);
  expect(fixture.contributor.props.precision.vertexTolerance).toBe(1 / 1024);
  const status = getWGSLKernelNodeRecord(
    graph,
    'make-valid-oracle-polygonize-directed-edge-status'
  )!;
  expect(status.bindings.find(binding => binding.name === 'directedCount')!.view).toBe(
    originalCount
  );
  const layout = getWGSLKernelNodeRecord(graph, 'make-valid-oracle-polygon-layout')!;
  expect(layout.bindings.find(binding => binding.name === 'featureOffsets')!.view).toBe(
    originalFeatureOffsets
  );
});

it('make-valid accepts default capacities and rejects invalid forwarded capacities at construction', () => {
  const {fixture} = createCPUScene();
  const props = fixture.contributor.props;
  expect(new GPUMakeValid(props).id).toBe('make-valid-oracle');
  // Preserve the make-valid case from the mixed forwarded-capacity spec here so this stream
  // runs its own boundary checks independently of unrelated contributors' input fixtures.
  // Capacities are positive integers; leaf capacity must additionally be a power of two.
  for (const name of ['intersectionCapacity', 'leafCapacity'] as const) {
    for (const value of [0, 1.5, Number.NaN]) {
      expect(() => new GPUMakeValid({...props, [name]: value}), `${name}=${value}`).toThrow(
        new RegExp(`^make-valid-oracle ${name} must be`)
      );
    }
  }
});

it.each([
  'make-valid',
  'polygonize'
] as const)('%s refuses a retired final requiredCount alongside requiredRingCount', kind => {
  const {fixture, view} = createCPUScene();
  const requiredCount = view('uint32', 1);
  expect(requiredCount).not.toBe(fixture.output.status.requiredRingCount);
  const status = {...fixture.output.status, requiredCount};
  const output = {...fixture.output, status};
  const props = {...fixture.contributor.props, output};
  expect(() =>
    kind === 'make-valid'
      ? new GPUMakeValid(props)
      : new GPUPolygonize({
          ...props,
          lines: {
            kind: 'lines',
            positions: props.polygons.positions,
            lineOffsets: props.polygons.ringOffsets
          }
        })
  ).toThrow(/make-valid-oracle.*output.status.requiredCount.*requiredRingCount/);
  // Intermediate ports still require their segment/directed cardinality destinations.
  expect(fixture.output.nodedSegments.status.requiredCount).toBeDefined();
  expect(fixture.output.directedEdges.status.requiredCount).toBeDefined();
});
