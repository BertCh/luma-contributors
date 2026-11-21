// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeDelaunayTessellationOnCPU,
  CPUDelaunayTessellationOperation
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  GPUDelaunayTessellationOperation,
  type DelaunayTessellationOperationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';

useCPUExecutorTestHooks();

it('keeps validated CPU Delaunay capacity and group count after caller mutation', async () => {
  const props: DelaunayTessellationOperationProps = {
    positions: 'positions',
    capacity: 1,
    groups: {ids: 'groupIds', count: 2},
    output: {
      triangles: 'triangles',
      status: {count: 'count', requiredCount: 'required', overflow: 'overflow'},
      groups: {triangleCounts: 'counts', status: {converged: 'converged'}}
    }
  };
  const operation = new CPUDelaunayTessellationOperation(props);
  props.capacity = 4;
  props.groups!.count = 3;
  const positions = new Float32Array([0, 0, 4, 0, 3, 2, 0, 3]),
    groupIds = new Uint32Array(4);
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 4,
        chunks: [{values: positions, length: 4}],
        pack: () => positions
      },
      groupIds: {
        format: 'uint32',
        length: 4,
        chunks: [{values: groupIds, length: 4}],
        pack: () => groupIds
      }
    }
  });
  // SciPy supplies two triangles; only the first fits the declared one-row output.
  expect(Array.from(result['triangles'] as Uint32Array)).toEqual([0, 1, 2]);
  expect(Array.from(result['count'] as Uint32Array)).toEqual([1]);
  expect(Array.from(result['counts'] as Uint32Array)).toEqual([2, 0]);
});

it('captures Voronoi accessor ports once, including stable inherited convergence', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const packed = createTransientView(graph, 'packed-accessors', 'uint32', 65);
  const invalidCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const conflicting = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required'),
    overflow: scalar('overflow'),
    invalidCount
  };
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
    triangleCount: scalar('requested'),
    clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 3),
      siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
      status
    }
  };
  let reads = 0;
  const safe = scalar('safe');
  Object.defineProperty(status, 'converged', {
    enumerable: true,
    configurable: true,
    get: () => (reads++ === 0 ? conflicting : safe)
  });
  try {
    expect(() => new GPUVoronoiDiagram(props)).toThrow(/converged.*invalidCount/);
    expect(reads).toBe(1);
    delete (status as {converged?: unknown}).converged;
    Object.setPrototypeOf(status, {
      get converged() {
        return safe;
      }
    });
    const nodes = new GPUVoronoiDiagram(props).getCommandNodes(graph);
    expect(
      nodes.some(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
          binding => binding.view === safe && binding.access === 'read_write'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('suppresses Voronoi segments after upstream Delaunay truncation or exhaustion', () => {
  for (const scenario of ['capacity', 'exhausted', 'complete'] as const) {
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const positions = createTransientView(graph, 'positions', 'float32x2', 4);
    const triangles = createTransientView(
      graph,
      'triangles',
      'uint32x3',
      scenario === 'capacity' ? 1 : 2
    );
    const upstream = {
      count: scalar('triangle-count'),
      requiredCount: scalar('triangle-required'),
      overflow: scalar('triangle-overflow'),
      converged: scalar('triangle-converged')
    };
    const downstream = {
      count: scalar('count'),
      requiredCount: scalar('required'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidate'),
      converged: scalar('converged')
    };
    const bounds = createTransientView(graph, 'bounds', 'float32x4', 1);
    let nodes = new GPUDelaunayTessellation({
      positions,
      output: {triangles, status: upstream}
    }).getCommandNodes(graph);
    if (scenario === 'exhausted')
      nodes = nodes.filter(
        node =>
          !node.id.includes('-continue-') &&
          (!/-step-(\d+)-/.test(node.id) || Number(/-step-(\d+)-/.exec(node.id)![1]) < 3)
      );
    const segments = createTransientView(graph, 'segments', 'float32x4', 6);
    const dual = new GPUVoronoiDiagram({
      positions,
      triangles,
      triangleCount: upstream.count,
      triangleStatus: upstream,
      clipBounds: bounds,
      output: {
        segments,
        siteIds: createTransientView(graph, 'sites', 'uint32x2', 6),
        status: downstream
      }
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(positions, new Float32Array([0, 0, 4, 0, 3, 2, 0, 3]));
    memory.write(bounds, new Float32Array([-2, -2, 6, 5]));
    try {
      executeCommandNodesOnCPU([...nodes, ...dual], memory);
      expect(memory.readUint32(upstream.overflow)[0]).toBe(scenario === 'capacity' ? 1 : 0);
      expect(memory.readUint32(upstream.converged)[0]).toBe(scenario === 'exhausted' ? 0 : 1);
      // Four hull edges plus the SciPy diagonal give five Voronoi segments.
      expect(memory.readUint32(downstream.count)[0]).toBe(scenario === 'complete' ? 5 : 0);
      expect(memory.readUint32(downstream.overflow)[0]).toBe(scenario === 'capacity' ? 1 : 0);
      expect(memory.readUint32(downstream.candidateOverflow)[0]).toBe(
        scenario === 'capacity' ? 1 : 0
      );
      expect(memory.readUint32(downstream.converged)[0]).toBe(scenario === 'complete' ? 1 : 0);
      if (scenario === 'complete') {
        const expected = [
          2,
          0.25,
          2,
          -2,
          2,
          0.25,
          6,
          2.25,
          2,
          0.25,
          7 / 6,
          1.5,
          7 / 6,
          1.5,
          7 / 3,
          5,
          7 / 6,
          1.5,
          -2,
          1.5
        ];
        Array.from(memory.readFloat32(segments))
          .slice(0, 20)
          .forEach((value, index) => {
            expect(Number.isFinite(value)).toBe(true);
            expect(Math.abs(value - expected[index])).toBeLessThan(2e-5);
          });
      }
    } finally {
      device.destroy();
    }
  }
});

it('defines capacity-one grouped ranges without reading past stored triangles', () => {
  const actual = computeDelaunayTessellationOnCPU(
    {
      positions: new Float32Array([0, 0, 4, 0, 3, 2, 0, 3, 10, 0, 14, 0, 13, 2, 10, 3]),
      groupIds: new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1])
    },
    {capacity: 1, groupCount: 3}
  );
  expect(Array.from(actual.count)).toEqual([1]);
  expect(Array.from(actual.requiredCount)).toEqual([4]);
  expect(Array.from(actual.groupTriangleCounts)).toEqual([2, 2, 0]);
  expect(Array.from(actual.groupTriangleOffsets)).toEqual([0, 2, 4]);
  const readable = Array.from(actual.groupTriangleOffsets, (offset, group) =>
    Math.max(0, Math.min(actual.groupTriangleCounts[group], actual.count[0] - offset))
  );
  expect(readable).toEqual([1, 0, 0]);
});

it('reports oversized Voronoi input counts and clears incompleteness on replay', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3),
    triangles = createTransientView(graph, 'triangles', 'uint32x3', 1),
    triangleCount = createTransientView(graph, 'triangle-count', 'uint32', 1),
    clipBounds = createTransientView(graph, 'bounds', 'float32x4', 1);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required'),
    overflow: scalar('overflow'),
    invalidCount: scalar('invalid'),
    candidateOverflow: scalar('candidate'),
    converged: scalar('converged')
  };
  const nodes = new GPUVoronoiDiagram({
    positions,
    triangles,
    triangleCount,
    clipBounds,
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 3),
      siteIds: createTransientView(graph, 'site-ids', 'uint32x2', 3),
      status
    }
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, new Float32Array([0, 0, 2, 0, 0, 2]));
  memory.write(triangles, new Uint32Array([0, 1, 2]));
  memory.write(clipBounds, new Float32Array([-2, -2, 3, 3]));
  try {
    for (const requested of [1, 2, 1]) {
      memory.write(triangleCount, new Uint32Array([requested]));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(status.candidateOverflow))).toEqual([
        requested === 2 ? 1 : 0
      ]);
      expect(Array.from(memory.readUint32(status.converged))).toEqual([requested === 2 ? 0 : 1]);
      expect(Array.from(memory.readUint32(status.count))).toEqual([3]);
    }
  } finally {
    device.destroy();
  }
});

it('rejects physical buffer sharing between Voronoi invalidCount and converged at offsets 0 and 256', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const packed = createTransientView(graph, 'packed', 'uint32', 65);
  const invalidCount = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const converged = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  try {
    expect(
      () =>
        new GPUVoronoiDiagram({
          positions: createTransientView(graph, 'positions', 'float32x2', 3),
          triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
          triangleCount: scalar('triangle-count'),
          clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
          output: {
            segments: createTransientView(graph, 'segments', 'float32x4', 3),
            siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
            status: {
              count: scalar('count'),
              requiredCount: scalar('required'),
              overflow: scalar('overflow'),
              invalidCount,
              converged
            }
          }
        })
    ).toThrow(/converged.*invalidCount|invalidCount.*converged/);
  } finally {
    device.destroy();
  }
});

it('preserves a declared Delaunay diagnostic after caller status mutation', () => {
  const status: DelaunayTessellationOperationProps['output']['status'] = {
    count: 'count',
    requiredCount: 'required',
    overflow: 'overflow',
    approximation: 'accuracy'
  };
  const operation = new GPUDelaunayTessellationOperation({
    positions: 'positions',
    capacity: 1,
    output: {triangles: 'triangles', status}
  });
  delete status.approximation;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const views = Object.fromEntries(
    Object.entries(operation.outputs).map(([port, descriptor]) => [
      port,
      createTransientView(graph, port, descriptor.format, descriptor.length)
    ])
  );
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  try {
    expect(operation.outputs['approximation'].value).toBe('accuracy');
    const nodes = operation
      .createContributor(port => (port === 'positions' ? positions : views[port]) as never)
      .getCommandNodes(graph);
    expect(
      nodes.some(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
          binding => binding.view === views['approximation'] && binding.access === 'read_write'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('counts a duplicate and a rejected nonfinite site independently', () => {
  for (const invalid of [false, true]) {
    const actual = computeDelaunayTessellationOnCPU(
      {positions: new Float32Array([0, 0, 3, 0, 0, 4, 0, 0, ...(invalid ? [NaN, 0] : [])])},
      {capacity: 1}
    );
    expect(Array.from(actual.duplicateCount)).toEqual([1]);
    expect(Array.from(actual.invalidCount)).toEqual([invalid ? 1 : 0]);
    expect(Array.from(actual.converged)).toEqual([invalid ? 0 : 1]);
    expect(Array.from(actual.requiredCount)).toEqual([invalid ? 0 : 1]);
  }
});

it('matches independent SciPy topology and literal grouped prefix counts', () => {
  const actual = computeDelaunayTessellationOnCPU(
    {
      positions: new Float32Array([0, 0, 4, 0, 3, 2, 0, 3, 10, 0, 14, 0, 13, 2, 10, 3]),
      groupIds: new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1])
    },
    {capacity: 4, groupCount: 3}
  );
  expect(
    Array.from(actual.triangles)
      .reduce<number[][]>((rows, _, index, values) => {
        if (index % 3 === 0)
          rows.push(values.slice(index, index + 3).sort((left, right) => left - right));
        return rows;
      }, [])
      .sort((left, right) => left.join(',').localeCompare(right.join(',')))
  ).toEqual([
    [0, 1, 2],
    [0, 2, 3],
    [4, 5, 6],
    [4, 6, 7]
  ]);
  expect(Array.from(actual.requiredCount)).toEqual([4]);
  expect(Array.from(actual.invalidCount)).toEqual([0]);
  expect(Array.from(actual.duplicateCount)).toEqual([0]);
  expect(Array.from(actual.converged)).toEqual([1]);
  expect(Array.from(actual.iterationLimitReached)).toEqual([0]);
  expect(Array.from(actual.candidateOverflow)).toEqual([0]);
  expect(Array.from(actual.uncertainCount)).toEqual([0]);
  expect(Array.from(actual.approximation)).toEqual([0]);
  expect(Array.from(actual.groupTriangleCounts)).toEqual([2, 2, 0]);
  expect(Array.from(actual.groupTriangleOffsets)).toEqual([0, 2, 4]);
  expect(Array.from(actual.groupConverged)).toEqual([1, 1, 1]);
});

for (const model of ['delaunay', 'voronoi'] as const) {
  for (const field of model === 'delaunay'
    ? (['candidateOverflow', 'uncertainCount', 'approximation'] as const)
    : ([
        'complete',
        'candidateOverflow',
        'uncertainCount',
        'iterationCount',
        'converged',
        'iterationLimitReached',
        'approximation'
      ] as const)) {
    it(`overwrites poisoned ${model} ${field} with the complete noniterative diagnostic`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const positions = createTransientView(graph, 'positions', 'float32x2', 3);
      const triangles = createTransientView(graph, 'triangles', 'uint32x3', 1);
      const target = createTransientView(graph, 'diagnostic', 'uint32', 1);
      const status = {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        [field]: target
      };
      const contributor =
        model === 'delaunay'
          ? new GPUDelaunayTessellation({positions, output: {triangles, status}})
          : new GPUVoronoiDiagram({
              positions,
              triangles,
              triangleCount: createTransientView(graph, 'triangle-count', 'uint32', 1),
              clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
              output: {
                segments: createTransientView(graph, 'segments', 'float32x4', 3),
                siteIds: createTransientView(graph, 'site-ids', 'uint32x2', 3),
                status
              }
            });
      const nodes = contributor.getCommandNodes(graph);
      const writers = nodes.filter(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
          binding => binding.view === target && binding.access === 'read_write'
        )
      );
      const memory = createCPUGraphMemory();
      memory.write(target, new Uint32Array([0xa5a5a5a5]));
      try {
        // Execute the actual generated producer slice. Other input state is zero:
        // valid completed work with no rejected triangles, capacity or uncertainty loss.
        executeCommandNodesOnCPU(writers, memory);
        expect(Array.from(memory.readUint32(target))).toEqual([
          field === 'converged' || field === 'complete' ? 1 : 0
        ]);
        if (model === 'voronoi' && field === 'converged') {
          const input = getWGSLKernelNodeRecord(graph, writers[0].id)!.bindings.find(
            binding => binding.name === 'invalidCount'
          )!.view;
          memory.write(input, new Uint32Array([1]));
          executeCommandNodesOnCPU(writers, memory);
          expect(Array.from(memory.readUint32(target))).toEqual([0]);
        }
      } finally {
        device.destroy();
      }
    });
  }
}

it.each([
  2, 3
])('keeps Voronoi capacity and solver reasons separate and rewrites completeness on reuse (capacity=%s)', capacity => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const triangles = createTransientView(graph, 'triangles', 'uint32x3', 1);
  const triangleCount = scalar('triangle-count');
  const upstream = {overflow: scalar('upstream-overflow'), converged: scalar('upstream-converged')};
  const clipBounds = createTransientView(graph, 'bounds', 'float32x4', 1);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required'),
    overflow: scalar('overflow'),
    candidateOverflow: scalar('candidate'),
    converged: scalar('converged'),
    complete: scalar('complete')
  };
  let reads = 0;
  const producer = new GPUVoronoiDiagram({
    positions,
    triangles,
    triangleCount,
    triangleStatus: upstream,
    clipBounds,
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', capacity),
      siteIds: createTransientView(graph, 'site-ids', 'uint32x2', capacity),
      status: {
        ...status,
        get complete() {
          reads++;
          return status.complete;
        }
      }
    }
  });
  expect(reads).toBe(1);
  const nodes = producer.getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.of(0, 0, 2, 0, 0, 2));
  memory.write(triangles, Uint32Array.of(0, 1, 2));
  memory.write(triangleCount, Uint32Array.of(1));
  memory.write(clipBounds, Float32Array.of(-2, -2, 3, 3));
  try {
    // A right triangle has three clipped hull rays. Two fit locally; upstream capacity
    // loss and solver failure instead suppress all segments, with distinct reason flags.
    for (const [overflow, converged, expected] of [
      [0, 0, [0, 0, 0, 0, 0, 0]],
      [1, 1, [0, 0, 1, 1, 0, 0]],
      [0, 1, capacity === 2 ? [2, 3, 1, 0, 1, 0] : [3, 3, 0, 0, 1, 1]],
      [0, 0, [0, 0, 0, 0, 0, 0]],
      [0, 1, capacity === 2 ? [2, 3, 1, 0, 1, 0] : [3, 3, 0, 0, 1, 1]]
    ] as const) {
      for (const view of Object.values(status)) memory.write(view, Uint32Array.of(913));
      memory.write(upstream.overflow, Uint32Array.of(overflow));
      memory.write(upstream.converged, Uint32Array.of(converged));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Object.values(status).map(view => memory.readUint32(view)[0])).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

for (const model of ['delaunay', 'voronoi'] as const) {
  for (const field of model === 'delaunay'
    ? (['complete', 'unresolvedCount'] as const)
    : (['unresolvedCount'] as const)) {
    it(`rejects inherited unwritten ${model} ${field} before dropping its destination`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      let reads = 0;
      const target = scalar('unsupported');
      const status = Object.setPrototypeOf(
        {count: scalar('count'), requiredCount: scalar('required'), overflow: scalar('overflow')},
        {
          get [field]() {
            reads++;
            return target;
          }
        }
      );
      const positions = createTransientView(graph, 'positions', 'float32x2', 3);
      const triangles = createTransientView(graph, 'triangles', 'uint32x3', 1);
      try {
        expect(() =>
          model === 'delaunay'
            ? new GPUDelaunayTessellation({positions, output: {triangles, status}})
            : new GPUVoronoiDiagram({
                positions,
                triangles,
                triangleCount: scalar('triangle-count'),
                clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
                output: {
                  segments: createTransientView(graph, 'segments', 'float32x4', 3),
                  siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
                  status
                }
              })
        ).toThrow(new RegExp(`output.status.${field}.*unsupported`));
        expect(reads).toBe(1);
      } finally {
        device.destroy();
      }
    });
  }
}

it('captures the CPU Delaunay nested frame before deferred execution', async () => {
  const frame = {
    id: 'construction-frame',
    kind: 'local-azimuthal-equidistant' as const,
    origin: [0, 0] as [number, number]
  };
  const props: DelaunayTessellationOperationProps = {
    positions: 'positions',
    capacity: 1,
    spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native', frame},
    output: {
      triangles: 'triangles',
      status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
    }
  };
  const operation = new CPUDelaunayTessellationOperation(props);
  frame.id = '';
  frame.origin[0] = 99;
  const positions = Float32Array.of(0, 0, 2, 0, 0, 2);
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      }
    }
  });
  // Three noncollinear sites have exactly the sole CCW triangle, independently of frame metadata.
  expect(Array.from(result['triangles'] as Uint32Array)).toEqual([0, 1, 2]);
  expect(Array.from(result['count'] as Uint32Array)).toEqual([1]);
});

it('retains packed requiredCount/converged storage when no completeness writer reads the count', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'packed', 'uint32', 65);
  const requiredCount = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  const converged = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
    triangleCount: scalar('triangle-count'),
    clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 3),
      siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
      status: {count: scalar('count'), overflow: scalar('overflow'), requiredCount, converged}
    }
  };
  try {
    expect(() => new GPUVoronoiDiagram(props).getCommandNodes(graph)).not.toThrow();
    expect(
      () =>
        new GPUVoronoiDiagram({
          ...props,
          output: {
            ...props.output,
            status: {...props.output.status, complete: converged, converged: undefined}
          }
        })
    ).toThrow(/complete.*requiredCount|must not share buffers/);
  } finally {
    device.destroy();
  }
});

it('S2 invalid triangles prevent Voronoi completeness and clear on valid-only replay', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const triangles = createTransientView(graph, 'triangles', 'uint32x3', 2);
  const triangleCount = createTransientView(graph, 'requested', 'uint32', 1);
  const clipBounds = createTransientView(graph, 'bounds', 'float32x4', 1);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required'),
    overflow: scalar('overflow'),
    invalidCount: scalar('invalid'),
    complete: scalar('complete')
  };
  const nodes = new GPUVoronoiDiagram({
    positions,
    triangles,
    triangleCount,
    clipBounds,
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 6),
      siteIds: createTransientView(graph, 'sites', 'uint32x2', 6),
      status
    }
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.of(0, 0, 2, 0, 0, 2));
  memory.write(triangles, Uint32Array.of(0, 1, 2, 99, 100, 101));
  memory.write(clipBounds, Float32Array.of(-2, -2, 3, 3));
  try {
    for (const requested of [2, 1, 2, 1]) {
      for (const view of Object.values(status)) memory.write(view, Uint32Array.of(913));
      memory.write(triangleCount, Uint32Array.of(requested));
      executeCommandNodesOnCPU(nodes, memory);
      // One right triangle has three hull rays. The out-of-range vertex row 99
      // contributes no ray, counts as one invalid triangle and forbids completeness.
      expect(Array.from(memory.readUint32(status.count))).toEqual([3]);
      expect(Array.from(memory.readUint32(status.requiredCount))).toEqual([3]);
      expect(Array.from(memory.readUint32(status.invalidCount))).toEqual([requested === 2 ? 1 : 0]);
      expect(Array.from(memory.readUint32(status.complete))).toEqual([requested === 2 ? 0 : 1]);
    }
  } finally {
    device.destroy();
  }
});
