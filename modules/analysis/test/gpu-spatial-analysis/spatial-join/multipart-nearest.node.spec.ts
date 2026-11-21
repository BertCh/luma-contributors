// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeNearestFeatureJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  MULTIPART_NEAREST_CASES,
  getMultipartNearestGeometry,
  createMultipartNearestFixture,
  assertMultipartNearestFixture
} from './multipart-nearest-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {adaptMultipartNearestUniforms} from './multipart-nearest-executor';
import {
  NEAREST_REVIEW_CASES,
  createReviewNearestFixture,
  assertReviewNearestFixture
} from './multipart-nearest-review-fixture';
import {NEAREST_COMMON_WGSL} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

useCPUExecutorTestHooks();

it('R1 determinant magnitude retains the original-coordinate residual', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const result = createTransientView(graph, 'determinant-result', 'float32', 2);
  try {
    const node = createWGSLKernelNode(graph, {
      id: 'determinant-probe',
      operation: 'nearest-review',
      invocationCount: 1,
      bindings: [{name: 'result', view: result, type: 'f32', access: 'read_write'}],
      declarations: NEAREST_COMMON_WGSL,
      body: `let point = vec2f(0.25, 0.2500002384185791);
let start = vec2f(10.0); let end = vec2f(0.0);
let cross = nearestPointSegmentCross(point, start, end, getPlanarDistanceVector(point, start), getPlanarDistanceVector(end, start));
result[resultOffset] = cross.significand;
result[resultOffset + 1u] = f32(cross.exponent);`
    });
    executeCommandNodesOnCPU([node], memory);
    const values = memory.readFloat32(result);
    expect(Array.from(values)).toEqual([0.625, -18]);
  } finally {
    device.destroy();
  }
});

for (const definition of NEAREST_REVIEW_CASES) {
  it(`${definition.name} CPU regression`, () => {
    const result = computeNearestFeatureJoinOnCPU(definition.queries, definition.features);
    const queryCount = definition.queries.featureOffsets!.length - 1;
    const stride = definition.expectedRows.length / queryCount;
    const expectedRows = Array.from(
      {length: queryCount},
      (_, row) => definition.expectedRows[row * stride]
    );
    expect(Array.from(result.neighborRows)).toEqual(expectedRows);
    expect(Array.from(result.neighborIds)).toEqual(expectedRows);
    expect(Array.from(result.neighborCounts)).toEqual(
      expectedRows.map(row => Number(row !== 0xffffffff))
    );
    expect(result.status).toEqual({overflow: 0, candidateOverflow: 0});
    result.neighborDistances.forEach((value, row) => {
      const expected = definition.expectedDistances[row * stride];
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected)).toBeLessThanOrEqual(
        Math.max(1e-14, Math.abs(expected) * 2e-5)
      );
    });
  });
  it(`${definition.name} interpreted regression`, async () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    try {
      const fixture = createReviewNearestFixture(
        graph,
        (name, format, length, values) => {
          const view = createTransientView(graph, name, format, length);
          memory.write(view, values);
          return view;
        },
        definition
      );
      executeCommandNodesOnCPU(
        adaptMultipartNearestUniforms(fixture.join.getCommandNodes(graph)),
        memory
      );
      await assertReviewNearestFixture(fixture, async view =>
        Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view))
      );
    } finally {
      device.destroy();
    }
  });
}

for (const definition of MULTIPART_NEAREST_CASES) {
  it(`CPU multipart ${definition.queryKind}/${definition.featureKind} agrees with exhaustive Shapely ranking`, () => {
    const features = getMultipartNearestGeometry('feature', definition.featureKind);
    const result = computeNearestFeatureJoinOnCPU(
      getMultipartNearestGeometry('query', definition.queryKind),
      features
    );
    expect(Array.from(result.neighborRows)).toEqual(definition.ranked.map(row => row[0].row));
    expect(Array.from(result.neighborIds)).toEqual(
      definition.ranked.map(row => features.sourceIds![row[0].row])
    );
    expect(Array.from(result.neighborCounts)).toEqual(Array(30).fill(1));
    result.neighborDistances.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - definition.ranked[index][0].distance)).toBeLessThan(2e-4);
    });
    expect(result.status).toEqual({overflow: 0, candidateOverflow: 0});
  });
}

for (const [caseIndex, definition] of MULTIPART_NEAREST_CASES.entries()) {
  for (const options of [
    {mode: 'nearest' as const},
    {mode: 'nearest' as const, querySort: true},
    {neighborCount: 4, radius: 2.25},
    {neighborCount: 32},
    {neighborCount: 1, neighborCapacity: 2, ties: 'all' as const},
    {neighborCount: 1, ties: 'all' as const}
  ]) {
    it(`interpreted multipart ${definition.queryKind}/${definition.featureKind} ${JSON.stringify(options)} writes Shapely results`, async () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      try {
        const fixture = createMultipartNearestFixture(
          graph,
          (name, format, length, values) => {
            const view = createTransientView(graph, name, format, length);
            if (values) memory.write(view, values);
            return view;
          },
          caseIndex,
          options
        );
        executeCommandNodesOnCPU(
          adaptMultipartNearestUniforms(fixture.join.getCommandNodes(graph)),
          memory
        );
        if (fixture.weights)
          executeCommandNodesOnCPU(
            adaptMultipartNearestUniforms(fixture.weights.getCommandNodes(graph)),
            memory
          );
        await assertMultipartNearestFixture(fixture, async view =>
          Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view))
        );
      } finally {
        device.destroy();
      }
    });
  }
}

it('multipart nearest-feature mode accepts one output row per logical query', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const join = new GPUNearestFeatureJoin({
      queries: {
        kind: 'points',
        positions: createTransientView(graph, 'queries', 'float32x2', 2),
        featureOffsets: createTransientView(graph, 'query-offsets', 'uint32', 2)
      },
      features: {
        kind: 'points',
        positions: createTransientView(graph, 'features', 'float32x2', 3),
        featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 3)
      },
      radius: createTransientView(graph, 'radius', 'float32', 1),
      output: {
        nearestFeatureIds: createTransientView(graph, 'ids', 'uint32', 1),
        status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    });
    expect(join.queryCount).toBe(1);
    expect(join.featureCount).toBe(2);
    expect(join.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('R5 padding oracle rejects a kernel that leaves coordinate padding unwritten', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const initial = new Map<object, number[]>();
  try {
    const fixture = createMultipartNearestFixture(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        if (values) {
          memory.write(view, values);
          initial.set(view, Array.from(values));
        }
        return view;
      },
      0,
      {neighborCount: 4, radius: 2.25}
    );
    executeCommandNodesOnCPU(
      adaptMultipartNearestUniforms(fixture.join.getCommandNodes(graph)),
      memory
    );
    executeCommandNodesOnCPU(
      adaptMultipartNearestUniforms(fixture.weights!.getCommandNodes(graph)),
      memory
    );
    await assertMultipartNearestFixture(fixture, async view =>
      Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view))
    );
    const rows = memory.readUint32(fixture.results['neighborRows']);
    expect(Array.from(rows).includes(0xffffffff)).toBe(true);
    // Controlled mutation: valid outputs stay correct, but padding writes are removed from
    // both coordinate outputs. The oracle must reject their original finite sentinel bytes.
    await expect(
      assertMultipartNearestFixture(fixture, async view => {
        const values = Array.from(
          view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view)
        );
        if (
          view === fixture.results['neighborFootPoints'] ||
          view === fixture.results['neighborQueryPoints']
        ) {
          rows.forEach((row, slot) => {
            if (row === 0xffffffff) {
              values[slot * 2] = initial.get(view)![slot * 2];
              values[slot * 2 + 1] = initial.get(view)![slot * 2 + 1];
            }
          });
        }
        return values;
      })
    ).rejects.toThrow(/NaN/);
  } finally {
    device.destroy();
  }
});

it('R5 reused graph clears both point coordinates when the radius empties occupied slots', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let radius: ReturnType<typeof createTransientView> | undefined;
  try {
    const fixture = createMultipartNearestFixture(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        if (values) memory.write(view, values);
        if (name === 'radius') radius = view;
        return view;
      },
      0,
      {neighborCount: 4, radius: 10000}
    );
    const nodes = adaptMultipartNearestUniforms([
      ...fixture.join.getCommandNodes(graph),
      ...fixture.weights!.getCommandNodes(graph)
    ]);
    const read = async (view: import('@luma.gl/graph/gpu-core').GraphDataView) =>
      Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view));
    executeCommandNodesOnCPU(nodes, memory);
    await assertMultipartNearestFixture(fixture, read);
    for (const port of ['neighborFootPoints', 'neighborQueryPoints'])
      expect((await read(fixture.results[port])).slice(0, 8).every(Number.isFinite)).toBe(true);
    memory.write(radius!, Float32Array.of(0));
    fixture.radiusValue = 0;
    executeCommandNodesOnCPU(nodes, memory);
    await assertMultipartNearestFixture(fixture, read);
    for (const port of ['neighborFootPoints', 'neighborQueryPoints'])
      expect((await read(fixture.results[port])).slice(0, 8).every(Number.isNaN)).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const port of [
  'neighborCounts',
  'neighborDistances',
  'neighborFootPoints',
  'neighborSegmentIndices',
  'neighborRows',
  'neighborQueryPoints'
] as const) {
  it(`R3 nearest-feature mode refuses output.${port} without neighborIds`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const format =
        port === 'neighborDistances'
          ? 'float32'
          : port === 'neighborFootPoints' || port === 'neighborQueryPoints'
            ? 'float32x2'
            : 'uint32';
      expect(
        () =>
          new GPUNearestFeatureJoin({
            queries: {
              kind: 'points',
              positions: createTransientView(graph, 'positions', 'float32x2', 1),
              featureOffsets: createTransientView(graph, 'offsets', 'uint32', 2)
            },
            features: {
              kind: 'points',
              positions: createTransientView(graph, 'features', 'float32x2', 1)
            },
            radius: createTransientView(graph, 'radius', 'float32', 1),
            output: {
              nearestFeatureIds: createTransientView(graph, 'ids', 'uint32', 1),
              [port]: createTransientView(graph, 'extra', format, 1),
              status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
            }
          })
      ).toThrow(new RegExp(`output\\.${port} requires output\\.neighborIds`));
    } finally {
      device.destroy();
    }
  });
}

for (const mode of ['nearest', 'neighbors'] as const) {
  it(`multipart ${mode} propagates lost BVH features to every query and dependent weights`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    try {
      const fixture = createMultipartNearestFixture(
        graph,
        (name, format, length, values) => {
          const view = createTransientView(graph, name, format, length);
          if (values) memory.write(view, values);
          return view;
        },
        0,
        {mode, leafCapacity: 16}
      );
      executeCommandNodesOnCPU(
        adaptMultipartNearestUniforms(fixture.join.getCommandNodes(graph)),
        memory
      );
      if (fixture.weights)
        executeCommandNodesOnCPU(
          adaptMultipartNearestUniforms(fixture.weights.getCommandNodes(graph)),
          memory
        );
      const read = (name: string) => Array.from(memory.readUint32(fixture.results[name]));
      expect(read('overflow')).toEqual([1]);
      expect(read('candidateOverflow')).toEqual([1]);
      expect(read('candidateCount')).toEqual([0]);
      expect(read('candidateTruncated')).toEqual(Array(30).fill(1));
      if (fixture.weights) {
        expect(read('weightOverflow')).toEqual([0]);
        expect(read('weightCandidateOverflow')).toEqual([1]);
      } else {
        expect(read('matchCandidateOverflow')).toEqual([1]);
      }
    } finally {
      device.destroy();
    }
  });
}

for (const {multipartQuery, filter} of [false, true].flatMap(multipartQuery =>
  (['none', 'exclusive', 'attribute'] as const).map(filter => ({multipartQuery, filter}))
)) {
  it(`nearest-feature mode keeps later target parts with multipart query=${multipartQuery}, filter=${filter}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    let serial = 0;
    const positions = (values: number[]) => {
      const view = createTransientView(
        graph,
        `positions-${serial++}`,
        'float32x2',
        values.length / 2
      );
      memory.write(view, Float32Array.from(values));
      return view;
    };
    const integers = (values: number[]) => {
      const view = createTransientView(graph, `integers-${serial++}`, 'uint32', values.length);
      memory.write(view, Uint32Array.from(values));
      return view;
    };
    try {
      const distances = createTransientView(graph, 'distances', 'float32', 1);
      const radius = createTransientView(graph, 'radius', 'float32', 1);
      memory.write(radius, Float32Array.of(filter === 'none' ? 2 : 20));
      const ids = integers([123]);
      // By hand: query (2,0) projects to (3,0) on the SECOND line of feature zero,
      // distance 1; feature one's only part is at x=20, distance 18. Grouping never
      // makes the first feature's far line at x=99 its only candidate.
      const join = new GPUNearestFeatureJoin({
        queries: {
          kind: 'points',
          positions: positions(multipartQuery ? [-100, 0, 2, 0] : [2, 0]),
          ...(multipartQuery ? {featureOffsets: integers([0, 2])} : {})
        },
        features: {
          kind: 'lines',
          positions: positions([99, -1, 99, 1, 3, -1, 3, 1, 20, -1, 20, 1]),
          lineOffsets: integers([0, 2, 4, 6]),
          featureOffsets: integers([0, 2, 3]),
          sourceIds: integers([42, 7])
        },
        radius,
        ...(filter === 'exclusive' ? {exclusive: true, queryIds: integers([42])} : {}),
        ...(filter === 'attribute'
          ? {onAttribute: {left: integers([9]), right: integers([8, 9])}}
          : {}),
        spatialSort: false,
        output: {
          nearestFeatureIds: ids,
          nearestDistances: distances,
          status: {overflow: integers([0])}
        }
      });
      executeCommandNodesOnCPU(adaptMultipartNearestUniforms(join.getCommandNodes(graph)), memory);
      expect(Array.from(memory.readUint32(ids))).toEqual([filter === 'none' ? 42 : 7]);
      expect(Array.from(memory.readFloat32(distances))).toEqual([filter === 'none' ? 1 : 18]);
    } finally {
      device.destroy();
    }
  });
}
