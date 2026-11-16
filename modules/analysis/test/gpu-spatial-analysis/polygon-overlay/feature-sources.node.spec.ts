// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {createOverlayFeatureSourceNodes} from '../../../src/gpu-spatial-analysis/polygon-overlay/overlay-feature-sources';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
it('reduces holes into their shell, keeps shell order and flags mixed source rows', () => {
  const graph = new GPUCommandGraph(device);
  const column = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
  const props = {
    id: 'provenance',
    ringShells: column('shells', 3),
    ringAreas: createTransientView(graph, 'areas', 'float32', 3),
    ringCount: column('rings', 1),
    segmentRings: column('segment-rings', 5),
    boundaryCount: column('boundaries', 1),
    sourceRows: column('source-rows', 5),
    operandIds: column('operand-ids', 5),
    output: {
      sourceRows: column('output-rows', 3),
      operandIds: column('output-operands', 3),
      mixed: column('mixed', 3)
    }
  };
  const nodes = createOverlayFeatureSourceNodes(graph, props);
  for (const {source} of getGPUCommandNodeKernelSources(nodes))
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  const memory = createCPUGraphMemory();
  memory.write(props.ringShells, new Uint32Array([0, 0, 2]));
  memory.write(props.ringAreas, new Float32Array([4, -1, 4]));
  memory.write(props.ringCount, new Uint32Array([3]));
  memory.write(props.segmentRings, new Uint32Array([0, 1, 2, 2, 0xffffffff]));
  memory.write(props.boundaryCount, new Uint32Array([5]));
  memory.write(props.sourceRows, new Uint32Array([4, 4, 9, 2, 99]));
  memory.write(props.operandIds, new Uint32Array([0, 0, 1, 1, 0]));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([4, 2, 0xffffffff]);
  expect(Array.from(memory.readUint32(props.output.operandIds))).toEqual([0, 1, 0xffffffff]);
  expect(Array.from(memory.readUint32(props.output.mixed))).toEqual([0, 1, 0]);
  // All words are rewritten on another encoding, including stale mixed bits and unused capacity.
  memory.write(props.boundaryCount, new Uint32Array([0]));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual(Array(3).fill(0xffffffff));
  expect(Array.from(memory.readUint32(props.output.mixed))).toEqual([0, 0, 0]);
});

function createContributor(
  mode: 'overlay' | 'buffer',
  options: {
    storage?: boolean;
    rings?: number;
    packed?: boolean;
    unusedSource?: boolean;
    vertexTolerance?: number;
  } = {}
) {
  const graph = new GPUCommandGraph(device);
  const rings = options.rings ?? 8;
  const column = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
  const packed = createTransientView(graph, 'packed-provenance', 'uint32', 256);
  const sourceRows =
    options.storage === false
      ? graph.createDataView(
          graph.createTransientBuffer({
            id: 'nonstorage-source',
            byteLength: rings * 4,
            usage: Buffer.COPY_DST
          }),
          {format: 'uint32', length: rings}
        )
      : options.packed
        ? graph.createDataView(packed.buffer, {format: 'uint32', length: rings})
        : column('source-rows', rings);
  const operandIds = options.packed
    ? graph.createDataView(packed.buffer, {format: 'uint32', length: rings, byteOffset: 256})
    : column('operand-ids', rings);
  const mixed = options.packed
    ? graph.createDataView(packed.buffer, {format: 'uint32', length: rings, byteOffset: 512})
    : column('mixed', rings);
  const status = (id: string) => ({
    count: column(`${id}-count`, 1),
    overflow: column(`${id}-overflow`, 1),
    candidateOverflow: column(`${id}-candidates`, 1)
  });
  const output = {
    geometry: {
      kind: 'polygons' as const,
      positions: createTransientView(graph, 'output-positions', 'float32x2', 64),
      featureOffsets: column('output-features', rings + 1),
      polygonOffsets: column('output-polygons', rings + 1),
      ringOffsets: column('output-rings', rings + 1)
    },
    featureSources: {sourceRows, operandIds, mixed},
    status: status('output'),
    boundary: {
      endpoints: createTransientView(graph, 'boundary', 'float32x4', 64),
      operandIds: column('boundary-operands', 64),
      sourceFeatureIds: column('boundary-sources', 64),
      status: {...status('boundary'), requiredCount: column('boundary-required', 1)}
    }
  };
  const foreign = new GPUCommandGraph(device);
  const unusedSource = foreign.createDataView(
    foreign.createTransientBuffer({id: 'unused-source', byteLength: 4, usage: Buffer.COPY_DST}),
    {format: 'uint32', length: 1}
  );
  const geometry = {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    featureOffsets: column('features', 2),
    polygonOffsets: column('polygons', 2),
    ringOffsets: column('rings', 2),
    ...(options.unusedSource ? {sourceIds: unusedSource} : {})
  };
  const common = {
    vertexTolerance: options.vertexTolerance ?? 1e-5,
    capacity: {intersections: 8, nodedSegments: 64},
    output
  };
  const contributor =
    mode === 'overlay'
      ? new GPUPolygonOverlay({...common, left: geometry, operation: 'dissolve'})
      : new GPUBufferSurface({
          ...common,
          geometry,
          parameters: createTransientView(graph, 'parameters', 'float32', 4),
          quadSegments: 1
        });
  return {graph, contributor};
}

it.each([
  'overlay',
  'buffer'
] as const)('%s checks optional source STORAGE usage before allocating any nested scratch', mode => {
  const {graph, contributor} = createContributor(mode, {storage: false});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each([
  'overlay',
  'buffer'
] as const)('%s checks downstream ring bounds before allocating any resources', mode => {
  const {graph, contributor} = createContributor(mode, {rings: 10_000_000});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/ringBounds.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});
it.each([
  'overlay',
  'buffer'
] as const)('%s accepts packed source columns and ignores unused source-ID bindings', mode => {
  const {graph, contributor} = createContributor(mode, {packed: true, unusedSource: true});
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

it.each([
  'overlay',
  'buffer'
] as const)('%s validates the downstream power-of-two grid limit at construction', mode => {
  expect(() => createContributor(mode, {vertexTolerance: 2e38})).toThrow(/vertexTolerance/);
  expect(() => createContributor(mode, {vertexTolerance: 2 ** 127})).not.toThrow();
  expect(() => createContributor(mode, {vertexTolerance: 2 ** 127 * (1 + 2 ** -23)})).toThrow(
    /vertexTolerance/
  );
});
