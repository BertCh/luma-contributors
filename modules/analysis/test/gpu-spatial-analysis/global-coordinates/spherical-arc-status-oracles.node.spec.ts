// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {
  computeSphericalArcOverlayOnCPU,
  CPUSphericalArcOverlayOperation
} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {
  GPUSphericalArcOverlayOperation,
  type SphericalArcOverlayOperationProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {getGPUSpatialAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology/index';
import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {
  GPUSphericalArcOverlay,
  type GPUSphericalArcOverlayProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createArcStatusScenes} from './spherical-arc-status-oracles-fixture';

it('both arc operation forms capture nested configuration before deferred execution', async () => {
  const scene = createArcStatusScenes().find(
    entry => entry.name === 'coincident union at capacity four'
  )!;
  const referencePoint: [number, number, number] = [0, 0, -1];
  const props: SphericalArcOverlayOperationProps = {
    ...scene.options,
    referencePoint,
    left: {positions: 'leftPositions', ringOffsets: 'leftOffsets', referenceInside: false},
    right: {positions: 'rightPositions', ringOffsets: 'rightOffsets', referenceInside: false},
    output: {positions: 'positions', ringOffsets: 'offsets', counts: 'counts', status: 'status'}
  };
  const gpu = new GPUSphericalArcOverlayOperation(props);
  const cpu = new CPUSphericalArcOverlayOperation(props);
  props.left.referenceInside = true;
  props.right.referenceInside = true;
  props.capacity = 3;
  props.operation = 'intersection';
  props.output.counts = 'replaced-counts';
  referencePoint[2] = 1;
  const contributor = constructNetworkOperationContributor(gpu, {
    leftPositions: {format: 'float32x3', length: 4},
    rightPositions: {format: 'float32x3', length: 4},
    leftRingOffsets: {format: 'uint32', length: 2},
    rightRingOffsets: {format: 'uint32', length: 2}
  }) as GPUSphericalArcOverlay;
  expect.soft(contributor.referenceInside).toBe(false);
  expect.soft(contributor.props.operation).toBe('union');
  expect.soft(contributor.props.referencePoint).toEqual([0, 0, -1]);
  const vector = (values: Float32Array | Uint32Array): GPUCPUVector => {
    const format = values instanceof Float32Array ? 'float32x3' : 'uint32';
    const length = values.length / (format === 'float32x3' ? 3 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const result = await cpu.execute({
    inputs: {
      leftPositions: vector(scene.input.left.positions),
      leftRingOffsets: vector(scene.input.left.ringOffsets),
      rightPositions: vector(scene.input.right.positions),
      rightRingOffsets: vector(scene.input.right.ringOffsets)
    }
  });
  // Union of the originally configured identical squares has four unique edges,
  // fits four slots, and excludes the original south-pole reference.
  expect(Array.from(result['counts'] as Uint32Array)).toEqual([4, 1, 4, 0]);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([0]);
});

it('bounds input-ring validation and dispatches spare output offsets in parallel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const positions = createTransientView(graph, 'square', 'float32x3', 4);
  const offsets = createTransientView(graph, 'impossible-offsets', 'uint32', 5);
  new GPUSphericalArcOverlay({
    id: 'ring-bounds',
    left: {positions, ringOffsets: offsets, referenceInside: false},
    right: {positions, ringOffsets: offsets, referenceInside: false},
    referencePoint: [0, 0, -1],
    operation: 'union',
    output: {
      positions: createTransientView(graph, 'output', 'float32x3', 4),
      ringOffsets: createTransientView(graph, 'output-offsets', 'uint32', 1000001),
      counts: createTransientView(graph, 'counts', 'uint32', 4),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  }).getCommandNodes(graph);
  try {
    const body = getWGSLKernelNodeRecord(graph, 'ring-bounds')!.body;
    const validation = body.slice(
      body.indexOf('for(var operand=0u;'),
      body.indexOf('  for(var edge=0u;')
    );
    const evaluate = new Function(
      'arcRingCount',
      'arcVertexCount',
      'arcOffset',
      ('let failure=0;' + validation + '}\nreturn failure;')
        .replace(/\bvar\b/g, 'let')
        .replace(/\b(\d+)u\b/g, '$1')
    );
    let offsetReads = 0;
    const failure = evaluate(
      () => 4,
      () => 4,
      (_operand: number, row: number) => {
        offsetReads++;
        return [0, 0, 0, 0, 4][row];
      }
    );
    // Four rings cannot use just four vertices: each ring needs at least three.
    // The actual shader gate must return invalid before reading any offset slot.
    expect(failure).toBe(1);
    expect(offsetReads).toBe(0);
    const padding = getWGSLKernelNodeRecord(graph, 'ring-bounds-pad-ring-offsets');
    expect(padding?.invocationCount).toBe(1000001);
  } finally {
    device.destroy();
  }
});

it('captures validated arc views and scalars before caller replacement', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const referencePoint: [number, number, number] = [0, 0, -1];
  const props: GPUSphericalArcOverlayProps = {
    id: 'captured-arc',
    left: {
      positions: createTransientView(graph, 'left', 'float32x3', 64),
      ringOffsets: createTransientView(graph, 'left-offsets', 'uint32', 2),
      referenceInside: false
    },
    right: {
      positions: createTransientView(graph, 'right', 'float32x3', 64),
      ringOffsets: createTransientView(graph, 'right-offsets', 'uint32', 2),
      referenceInside: false
    },
    referencePoint,
    operation: 'union',
    tolerance: 1e-6,
    output: {
      positions: createTransientView(graph, 'output', 'float32x3', 1024),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      counts: createTransientView(graph, 'counts', 'uint32', 4),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  };
  const initialLeft = props.left.positions;
  const initialOutput = props.output.positions;
  const contributor = new GPUSphericalArcOverlay(props);
  props.left.positions = createTransientView(graph, 'replacement-left', 'float32x3', 65);
  props.output.positions = createTransientView(graph, 'replacement-output', 'float32x3', 1025);
  props.tolerance = 1e-3;
  props.operation = 'intersection';
  referencePoint[2] = 1;
  try {
    contributor.getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, 'captured-arc')!;
    // L=R=64 gives 128+4*4096=16512 records, seven floats each.
    expect(record.bindings.find(binding => binding.name === 'scratch')!.view.length).toBe(115584);
    expect(record.bindings.find(binding => binding.name === 'leftPositions')!.view).toBe(
      initialLeft
    );
    expect(record.bindings.find(binding => binding.name === 'output')!.view).toBe(initialOutput);
    expect(record.variant).toBe('union');
    expect(record.declarations).toContain('ARC_TOLERANCE:f32=0.000001;');
    expect(record.declarations).toContain('ARC_REFERENCE=vec3f(0,0,-1);');
  } finally {
    device.destroy();
  }
});

it('refuses mixed physical counts/offset bindings before mutation and permits same-ID retry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const packed = createTransientView(graph, 'packed-arc-status', 'uint32', 128);
  const positions = createTransientView(graph, 'triangle', 'float32x3', 3);
  const offsets = createTransientView(graph, 'triangle-offsets', 'uint32', 2);
  const props: GPUSphericalArcOverlayProps = {
    id: 'physical-arc',
    left: {positions, ringOffsets: offsets, referenceInside: false},
    right: {positions, ringOffsets: offsets, referenceInside: false},
    referencePoint: [0, 0, -1],
    operation: 'union',
    output: {
      positions: createTransientView(graph, 'arc-output', 'float32x3', 6),
      counts: graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 0, length: 4}),
      ringOffsets: graph.createDataView(packed.buffer, {
        format: 'uint32',
        byteOffset: 256,
        length: 9
      }),
      status: createTransientView(graph, 'arc-status', 'uint32', 1)
    }
  };
  try {
    // WebGPU storage usage is physical-buffer scoped even though [0,16) and [256,292)
    // do not overlap. Refusal must leave the contributor ID available for a valid retry.
    expect(() => new GPUSphericalArcOverlay(props)).toThrow(
      /physical-arc.*separate physical buffers/
    );
    expect(getWGSLKernelNodeRecord(graph, 'physical-arc')).toBeUndefined();
    const separate = createTransientView(graph, 'separate-arc-state', 'uint32', 128);
    props.output.counts = graph.createDataView(separate.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 4
    });
    // Counts/status are only bound together as writable storage in the arrangement;
    // their disjoint packing remains safe in this unrelated dispatch.
    props.output.status = graph.createDataView(separate.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    expect(() => new GPUSphericalArcOverlay(props).getCommandNodes(graph)).not.toThrow();
    const padding = getWGSLKernelNodeRecord(graph, 'physical-arc-pad-ring-offsets')!;
    const pad = new Function(
      'index',
      'counts',
      'rings',
      'const countsOffset=0,ringsOffset=0;' + padding.body.replace(/\b(\d+)u\b/g, '$1')
    );
    // Preserve the live prefix and pad every spare slot, including the last slot.
    for (const [vertexCount, ringCount, expected] of [
      [4, 1, [0, 4, 4, 4, 4, 4, 4, 4, 4]],
      [0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0]],
      [6, 2, [0, 3, 6, 6, 6, 6, 6, 6, 6]]
    ] as const) {
      // The multi-ring prefix represents two valid three-vertex rings.
      const rings = Uint32Array.from([99, 3, 6, 99, 99, 99, 99, 99, 99]);
      if (ringCount === 1) rings[1] = 4;
      for (let index = 0; index < 9; index++) pad(index, [vertexCount, ringCount], rings);
      expect(Array.from(rings)).toEqual(expected);
      for (let ring = 0; ring < ringCount; ring++)
        expect(
          rings[ring + 1] - rings[ring],
          'each live ring has at least three vertices'
        ).toBeGreaterThanOrEqual(3);
    }
    expect(padding.invocationCount).toBe(9);
  } finally {
    device.destroy();
  }
});

it('public arc descriptor declares its bounded reference execution and scratch', () => {
  const operator = getGPUSpatialAnalysisOperator('GPUSphericalArcOverlay');
  expect(operator?.tier).toBe('reference');
  expect(operator?.limits?.map(limit => [limit.name, limit.maximum])).toEqual([
    ['left.positions.length + right.positions.length', 128],
    ['output.positions.length', 1024],
    ['segmentCapacity', 16512]
  ]);
  expect(operator?.evidence.dispatch).toBe('multi-pass');
  expect(operator?.evidence.transientMemory).toBe('quadratic-source');
  expect(operator?.limitation).toContain('one arrangement invocation');
  expect(operator?.limitation).toContain('max(1,L+R+4LR)*7');
});

it('enforces the bounded serial reference route at 128 inputs and 1024 output slots', () => {
  const construct = (leftCount: number, rightCount: number, capacity: number) =>
    constructNetworkOperationContributor(
      new GPUSphericalArcOverlayOperation({
        id: 'bounded-arc-status',
        left: {positions: 'leftPositions', ringOffsets: 'leftOffsets', referenceInside: false},
        right: {positions: 'rightPositions', ringOffsets: 'rightOffsets', referenceInside: false},
        referencePoint: [0, 0, -1],
        operation: 'union',
        capacity,
        ringCapacity: 1,
        output: {positions: 'positions', ringOffsets: 'offsets', counts: 'counts', status: 'status'}
      }),
      {
        leftPositions: {format: 'float32x3', length: leftCount},
        rightPositions: {format: 'float32x3', length: rightCount},
        leftRingOffsets: {format: 'uint32', length: 2},
        rightRingOffsets: {format: 'uint32', length: 2}
      }
    );
  // The limit is on the sum, not on each operand separately. These literal
  // boundaries justify the serial route without executing any GPU commands.
  expect(() => construct(64, 64, 1024)).not.toThrow();
  expect(() => construct(65, 64, 1024)).toThrow(
    /bounded-arc-status.*left plus right positions.*128/
  );
  expect(() => construct(64, 64, 1025)).toThrow(/bounded-arc-status.*output.positions.*1024/);
});

it.each(createArcStatusScenes())('CPU arc status and all four counts: $name', scene => {
  const result = computeSphericalArcOverlayOnCPU(scene.input, scene.options);
  expect(Array.from(result.counts)).toEqual(scene.counts);
  expect(Array.from(result.status)).toEqual(scene.status);
  if (scene.ringOffsets) expect(Array.from(result.ringOffsets)).toEqual(scene.ringOffsets);
});

it('the actual arc WGSL duplicate publisher counts four coincident edges even at capacity three', () => {
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts',
      import.meta.url
    ),
    'utf8'
  );
  const block = source.match(
    /for\(var previous=0u;previous<(?:min\(required,[^)]*\)|required);previous\+\+\)\{duplicate=duplicate\|\|[^]*?if\(!duplicate\)\{[^]*?required\+=1u;\}/
  )?.[0];
  expect(block).toBeDefined();
  // Translate this actual WGSL publication block; inputs are already-classified
  // oriented boundary segments, isolating duplicate detection from arc predicates.
  const publish = new Function(
    'required',
    'capacity',
    'startPoint',
    'endPoint',
    'arcStart',
    'arcEnd',
    'arcStore',
    'distance',
    'min',
    'ARC_TOLERANCE',
    'let duplicate=false;' +
      block!
        .replace(/\$\{capacity\}u/g, 'capacity')
        .replace(/\b(\d+)u\b/g, '$1')
        .replace(/\bvar\b/g, 'let') +
      ';return required;'
  );
  const corners = [
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
  ].map(point => point.map(value => value / Math.hypot(...point)));
  for (const capacity of [3, 4]) {
    const stored: {start: number[]; end: number[]}[] = [];
    let required = 0;
    for (let operand = 0; operand < 2; operand++)
      for (let edge = 0; edge < 4; edge++)
        required = publish(
          required,
          capacity,
          corners[edge],
          corners[(edge + 1) % 4],
          (row: number) => stored[row].start,
          (row: number) => stored[row].end,
          (row: number, start: number[], end: number[]) => {
            stored[row] = {start, end};
          },
          (first: number[], second: number[]) =>
            Math.hypot(...first.map((value, index) => value - second[index])),
          Math.min,
          1e-6
        );
    // Four distinct oriented edges, each presented twice: a hand-sized set union.
    expect(required, `capacity ${capacity}`).toBe(4);
  }
});
