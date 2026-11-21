// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeS2CapCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {computeS2PolygonCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {CPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {GPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {GPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';

it('captures S2 GPU operation budgets and hemisphere options before lowering', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const props = {
    caps: 'caps',
    capCount: 1,
    level: 1,
    capacity: 6,
    frontierCapacity: 6,
    refinement: 'level-synchronous' as const,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  const cap = new GPUS2CapCoveringOperation(props);
  props.frontierCapacity = 100;
  const hemisphere: [number, number, number] = [1, 0, 0];
  const polygon = new GPUS2PolygonCoveringOperation({
    positions: 'positions',
    ringOffsets: 'rings',
    positionCount: 3,
    ringOffsetCount: 2,
    capacity: 6,
    maximumLevel: 1,
    maximumVisitedCells: 8,
    hemisphereCenter: hemisphere,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  });
  hemisphere[0] = 0;
  const views = {
    caps: createTransientView(graph, 'caps', 'float32x4', 1),
    positions: createTransientView(graph, 'positions', 'float32x3', 3),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    cells: createTransientView(graph, 'cells', 'uint32x2', 6),
    counts: createTransientView(graph, 'counts', 'uint32x2', 1),
    status: createTransientView(graph, 'status', 'uint32', 1)
  };
  try {
    const contributor = cap.createContributor(
      port => views[port as keyof typeof views] as never
    ) as GPUS2CapCovering;
    expect(contributor.frontierCapacity).toBe(6);
    expect(() =>
      polygon.createContributor(port => views[port as keyof typeof views] as never)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('captures the CPU polygon hemisphere before deferred execution', async () => {
  const hemisphere: [number, number, number] = [1, 0, 0];
  const operation = new CPUS2PolygonCoveringOperation({
    positions: 'positions',
    ringOffsets: 'rings',
    positionCount: 3,
    ringOffsetCount: 2,
    maximumLevel: 0,
    capacity: 2,
    hemisphereCenter: hemisphere,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  });
  hemisphere[0] = 0;
  const positions = Float32Array.from(
    [
      [1, -0.1, -0.1],
      [1, 0.1, -0.1],
      [1, 0, 0.1]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  const offsets = new Uint32Array([0, 3]);
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x3',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      },
      ringOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: offsets, length: 2}],
        pack: () => offsets
      }
    }
  });
  expect(Array.from(result['counts'] as Uint32Array)).toEqual([1, 1]);
  expect(Array.from(result['cells'] as Uint32Array)).toEqual([0, 0x10000000, 0, 0]);
});
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from './output-oracle-test-utils';
import {GPULocalFrameProjectionOperation} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection-operation';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('short circuits a negative effective interior radius before GPU frontier expansion', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const caps = createTransientView(graph, 'caps', 'float32x4', 1);
  const nodes = new GPUS2CapCovering({
    caps,
    level: 1,
    capacity: 6,
    frontierCapacity: 6,
    mode: 'interior',
    tolerance: 1e-5,
    refinement: 'level-synchronous',
    output: {
      cells: createTransientView(graph, 'cells', 'uint32x2', 6),
      counts: createTransientView(graph, 'counts', 'uint32x2', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  }).getCommandNodes(graph);
  const classify = nodes.find(node => node.id.endsWith('-classify-0'))!;
  const record = getWGSLKernelNodeRecord(graph, classify.id)!;
  const view = (name: string) => record.bindings.find(binding => binding.name === name)!.view;
  const memory = createCPUGraphMemory();
  memory.write(caps, new Float32Array([Math.SQRT1_2, Math.SQRT1_2, 0, 9.5e-6]));
  memory.write(view('roundParameters'), new Uint32Array([0]));
  memory.write(view('listCounts'), new Uint32Array([6]));
  memory.write(
    view('list'),
    new Uint32Array([
      0, 0x10000000, 0, 0x30000000, 0, 0x50000000, 0, 0x70000000, 0, 0x90000000, 0, 0xb0000000
    ])
  );
  try {
    executeCommandNodesOnCPU([classify], memory);
    expect(Array.from(memory.readUint32(view('counts'))).slice(0, 6)).toEqual([0, 0, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

const FACE_TRIANGLE = Float32Array.from(
  [
    [1, -0.1, -0.1],
    [1, 0.1, -0.1],
    [1, 0, 0.1]
  ].flatMap(point => point.map(value => value / Math.hypot(...point)))
);

it('preserves complete CPU covers when budgets are omitted and binds explicit limits', async () => {
  const caps = new Float32Array([1, 0, 0, Math.PI]);
  const options = {level: 2, capacity: 1, refinement: 'level-synchronous' as const};
  const complete = computeS2CapCoveringOnCPU(caps, options);
  // Six faces, each with 4^2 descendants.
  expect(Array.from(complete.counts)).toEqual([1, 96]);
  expect(Array.from(complete.status)).toEqual([2]);
  const bounded = computeS2CapCoveringOnCPU(caps, {...options, frontierCapacity: 72});
  expect(Array.from(bounded.counts)).toEqual([0, 0]);
  expect(Array.from(bounded.status)).toEqual([4]);
  const polygon = {positions: FACE_TRIANGLE, ringOffsets: new Uint32Array([0, 3])};
  const polygonOptions = {minimumLevel: 7, maximumLevel: 7, complement: true, capacity: 1};
  // Independent S2 cell enumeration oracle beside this spec: 98,304 leaves minus 126 strictly excluded leaves.
  const full = computeS2PolygonCoveringOnCPU(polygon, polygonOptions);
  expect(Array.from(full.counts)).toEqual([1, 98178]);
  expect(full.status[0]).toBe(2);
  const limited = computeS2PolygonCoveringOnCPU(polygon, {
    ...polygonOptions,
    maximumVisitedCells: 65536
  });
  expect(limited.counts[0]).toBe(1);
  expect(limited.counts[1]).toBeLessThan(98178);
  expect(limited.status[0]).toBe(6);
  expect(limited.visitedCount[0]).toBe(65536);
  for (const explicit of [false, true]) {
    const operation = new CPUS2PolygonCoveringOperation({
      positions: 'positions',
      ringOffsets: 'rings',
      positionCount: 3,
      ringOffsetCount: 2,
      ...polygonOptions,
      ...(explicit ? {maximumVisitedCells: 65536} : {}),
      output: {cells: 'cells', counts: 'counts', status: 'status'}
    });
    const positions = polygon.positions,
      offsets = polygon.ringOffsets;
    const result = await operation.execute({
      inputs: {
        positions: {
          format: 'float32x3',
          length: 3,
          chunks: [{values: positions, length: 3}],
          pack: () => positions
        },
        ringOffsets: {
          format: 'uint32',
          length: 2,
          chunks: [{values: offsets, length: 2}],
          pack: () => offsets
        }
      }
    });
    expect((result['counts'] as Uint32Array)[0]).toBe(1);
    expect((result['status'] as Uint32Array)[0]).toBe(explicit ? 6 : 2);
    if (!explicit) expect(Array.from(result['counts'] as Uint32Array)).toEqual([1, 98178]);
  }
});

it('reports six visits when the next polygon frontier already exceeds eight slots', () => {
  const actual = computeS2PolygonCoveringOnCPU(
    {positions: FACE_TRIANGLE, ringOffsets: new Uint32Array([0, 3])},
    {
      minimumLevel: 1,
      maximumLevel: 1,
      capacity: 1,
      complement: true,
      refinement: 'level-synchronous',
      maximumVisitedCells: 8
    }
  );
  expect(Array.from(actual.counts)).toEqual([0, 0]);
  expect(actual.status[0]).toBe(4);
  expect(actual.visitedCount[0]).toBe(6);
});

it('honors explicit level-one polygon visit budgets in each traversal mode', () => {
  const positions = Float32Array.from(
    [
      [1, -0.1, -0.1],
      [1, 0.1, -0.1],
      [1, 0, 0.1]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  for (const refinement of ['serial', 'level-synchronous'] as const) {
    const actual = computeS2PolygonCoveringOnCPU(
      {positions, ringOffsets: new Uint32Array([0, 3])},
      {maximumLevel: 1, capacity: 6, maximumVisitedCells: 6, refinement}
    );
    expect(Array.from(actual.counts)).toEqual(refinement === 'serial' ? [4, 4] : [0, 0]);
    expect(Array.from(actual.status)).toEqual([4]);
    expect(Array.from(actual.visitedCount)).toEqual([6]);
    expect(Array.from(actual.cells)).toEqual(
      refinement === 'serial'
        ? [0, 0x04000000, 0, 0x0c000000, 0, 0x14000000, 0, 0x1c000000, 0, 0, 0, 0]
        : Array(12).fill(0)
    );
  }
});

it('refuses an exhausted level-one cap frontier rather than publishing a complete count', () => {
  const actual = computeS2CapCoveringOnCPU(new Float32Array([1, 0, 0, Math.PI]), {
    level: 1,
    capacity: 6,
    frontierCapacity: 6,
    refinement: 'level-synchronous'
  });
  expect(Array.from(actual.counts)).toEqual([0, 0]);
  expect(Array.from(actual.status)).toEqual([4]);
  expect(Array.from(actual.cells)).toEqual(Array(12).fill(0));
});

function createMemoryBackedDevice() {
  const device = createNullWebGPUDevice();
  // NullBuffer discards writes. Retain bytes here to test the fixture's real
  // upload/readback path without a GPU or production geometry implementation.
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = props => {
    const buffer = createBuffer(props);
    const bytes = new Uint8Array(buffer.byteLength);
    buffer.write = (data, offset = 0) => {
      const source = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);
      bytes.set(source, offset);
    };
    buffer.readAsync = async (offset = 0, length = bytes.byteLength - offset) =>
      bytes.slice(offset, offset + length);
    return buffer;
  };
  return device;
}

it('detects a publisher that skips its second write', async () => {
  const device = createMemoryBackedDevice();
  const fixture = createOutputOracleFixture(device);
  const target = fixture.output('target', 'uint32', 1);
  let submissions = 0;
  const compiled = {
    id: 'skip-second-write',
    encode: () => {
      if (submissions++ === 0)
        getCoreDefaultBuffer(target.buffer.defaultBuffer!).write(
          new Uint32Array([37]),
          target.byteOffset
        );
      return {};
    }
  } as unknown as ReturnType<typeof fixture.graph.compile>;
  try {
    await expect(
      fixture.submitTwice(compiled, async () => {
        const values = await fixture.readRows([target]);
        expect(values).toEqual([37]);
        return values;
      })
    ).rejects.toThrow();
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('re-poisons produced words without changing inputs, guards, or preservation-only metadata', async () => {
  const device = createMemoryBackedDevice();
  const fixture = createOutputOracleFixture(device);
  const source = fixture.input('source', 'uint32', new Uint32Array([13]));
  const metadata = fixture.output('metadata', 'uint32', 1, {preserveOnReplay: true});
  getCoreDefaultBuffer(metadata.buffer.defaultBuffer!).write(
    new Uint32Array([17]),
    metadata.byteOffset
  );
  const target = fixture.output('target', 'uint32', 1);
  let submissions = 0;
  const compiled = {
    id: 'write-every-time',
    encode: () => {
      submissions++;
      getCoreDefaultBuffer(target.buffer.defaultBuffer!).write(
        new Uint32Array([37]),
        target.byteOffset
      );
      return {};
    }
  } as unknown as ReturnType<typeof fixture.graph.compile>;
  try {
    const actual = await fixture.submitTwice(compiled, async () => ({
      target: await fixture.readRows([target]),
      input: await fixture.readRows([source]),
      metadata: await fixture.readRows([metadata])
    }));
    expect(actual).toEqual({target: [37], input: [13], metadata: [17]});
    expect(submissions).toBe(2);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('reads integer vector IDs as integer words and supplies naturally aligned projection views', async () => {
  const device = createMemoryBackedDevice();
  const fixture = createOutputOracleFixture(device);
  const keys = fixture.input('keys', 'uint32x2', new Uint32Array([0xffffffff, 0x4823ffff]));
  const triangles = fixture.input('triangles', 'uint32x3', new Uint32Array([0, 1, 2]));
  const poison = fixture.output('poison', 'uint32x2', 1);
  const packed = fixture.input(
    'packed',
    'uint32x4',
    new Uint32Array([0, 0x10000000, 0xffffffff, 37])
  );
  const positions = fixture.input('positions', 'float32x2', new Float32Array([0.5, 0.25, 1, 1]));
  const output = fixture.output('projected', 'float32x2', 2);
  const operation = new GPULocalFrameProjectionOperation({
    positions: 'positions',
    rowCount: 2,
    output: 'projected',
    bounds: [-1, -1, 1, 1],
    frame: createGPULocalFrame({origin: [0, 0]})
  });
  try {
    // These independent words include a float-NaN bit pattern and tiny subnormals
    // when misread as f32, exactly discriminating the lead's v1-1 failures.
    expect(await fixture.readRows([keys])).toEqual([0xffffffff, 0x4823ffff]);
    expect(await fixture.readRows([triangles])).toEqual([0, 1, 2]);
    expect(await fixture.readRows([poison])).toEqual([0xa5a5a5a5, 0xa5a5a5a5]);
    expect(await fixture.readRows([packed])).toEqual([0, 0x10000000, 0xffffffff, 37]);
    expect(await fixture.readRows([positions])).toEqual([0.5, 0.25, 1, 1]);
    for (const view of [keys, triangles, poison, packed, positions, output]) {
      expect(view.byteOffset).toBeGreaterThan(256);
      expect(view.byteOffset % view.rowByteLength).toBe(0);
    }
    // Execute the upstream projection constructor's actual alignment validation.
    const views = {positions, output};
    expect(() =>
      operation.createContributor(port => views[port as keyof typeof views] as never)
    ).not.toThrow();
  } finally {
    operation.destroy();
    fixture.destroy();
    device.destroy();
  }
});

it('visits exactly six roots for a triangle strictly within the positive X face', () => {
  // Gnomonic triangle (y/x,z/x)=(-.1,-.1),(.1,-.1),(0,.1)
  // lies strictly inside face zero. At maximumLevel zero there are six root
  // visits, one intersecting face and no descendants.
  const positions = Float32Array.from(
    [
      [1, -0.1, -0.1],
      [1, 0.1, -0.1],
      [1, 0, 0.1]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  const actual = computeS2PolygonCoveringOnCPU(
    {positions, ringOffsets: new Uint32Array([0, 3])},
    {maximumLevel: 0, capacity: 2}
  );
  expect(Array.from(actual.cells)).toEqual([0, 0x10000000, 0, 0]);
  expect(Array.from(actual.counts)).toEqual([1, 1]);
  expect(Array.from(actual.visitedCount)).toEqual([6]);
  expect(Array.from(actual.status)).toEqual([0]);
});

it('covers a face-axis point and the whole globe by the six level-zero face IDs', () => {
  // S2 level-zero IDs are (2*face+1)<<60. A radius-zero point at +X
  // belongs only to face 0; the whole sphere contains all six faces.
  for (const radius of [0, Math.PI]) {
    const actual = computeS2CapCoveringOnCPU(new Float32Array([1, 0, 0, radius]), {
      level: 0,
      capacity: 6
    });
    expect(Array.from(actual.counts)).toEqual(radius === 0 ? [1, 1] : [6, 6]);
    expect(Array.from(actual.cells)).toEqual(
      radius === 0
        ? [0, 0x10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        : [0, 0x10000000, 0, 0x30000000, 0, 0x50000000, 0, 0x70000000, 0, 0x90000000, 0, 0xb0000000]
    );
    expect(Array.from(actual.status)).toEqual([0]);
  }
});
