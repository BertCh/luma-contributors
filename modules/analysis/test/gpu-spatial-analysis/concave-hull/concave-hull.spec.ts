// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUConcaveHull,
  GPUGroupConcaveHull
} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createNullWebGPUDevice,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createConcaveHullTestOutput,
  createConcaveHullValidationProps
} from './concave-hull-validation-cases';

async function setup() {
  const testDevice = await getWebGPUTestDevice();
  requireTestDevice(testDevice);
  // Rebinding keeps the narrowed type inside hoisted function declarations.
  const device = testDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const result = createConcaveHullTestOutput(
      graph,
      device,
      `output-${identifier++}`,
      format,
      length
    );
    buffers.push(result.buffer);
    return result;
  }
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

function createRandomPositions(count: number, extent: number): number[] {
  let state = 12345;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  return Array.from({length: count * 2}, () => Math.fround(next() * extent));
}

/** CPU reference: directed triangle edges whose reverse is not an edge of another retained triangle. */
function getReferenceBoundary(triangles: number[], validity: number[], count: number): Set<string> {
  const directed = new Set<string>();
  for (let triangle = 0; triangle < count; triangle++) {
    if (!validity[triangle]) continue;
    for (let corner = 0; corner < 3; corner++) {
      directed.add(
        `${triangles[triangle * 3 + corner]},${triangles[triangle * 3 + ((corner + 1) % 3)]}`
      );
    }
  }
  const boundary = new Set<string>();
  for (const edge of directed) {
    const [from, to] = edge.split(',');
    if (!directed.has(`${to},${from}`)) boundary.add(edge);
  }
  return boundary;
}

async function runHull(pointCount: number, positionsValues: number[], radius: number) {
  const {graph, input, output, run} = await setup();
  const capacity = getDelaunayMaximumTriangleCount(pointCount);
  const triangles = output(capacity, 'uint32x3');
  const triangleValidity = output(capacity, 'uint32');
  const edges = output(capacity * 3, 'uint32x2');
  const edgeValidity = output(capacity * 3, 'uint32');
  const converged = output(1, 'uint32');
  graph.add(
    new GPUConcaveHull({
      positions: input(positionsValues, 'float32x2'),
      maximumCircumradius: radius,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  const result = {
    converged: (await readUint32(converged.buffer, 1))[0],
    triangles: await readUint32(triangles.buffer, capacity * 3),
    triangleValidity: await readUint32(triangleValidity.buffer, capacity),
    edges: await readUint32(edges.buffer, capacity * 6),
    edgeValidity: await readUint32(edgeValidity.buffer, capacity * 3),
    capacity
  };
  destroy();
  return result;
}

for (const exponent of [45, -52, 100, -100]) {
  it(`alpha hull retains triangles across the analytic circumradius at scale 2^${exponent}`, async () => {
    const scale = 2 ** exponent;
    for (const [factor, expected] of [
      [0.6, 0],
      [0.8, 1]
    ]) {
      const result = await runHull(3, [0, 0, scale, 0, 0, scale], factor * scale);
      expect(result.converged).toBe(1);
      expect(result.triangleValidity).toEqual([expected]);
      expect(result.edgeValidity).toEqual([expected, expected, expected]);
    }
  });
}

it('packed grouped alpha hull compacts groups and snapshots caller maps before execution', async () => {
  for (const maximumCircumradius of [1, 4]) {
    const {graph, input, output, run} = await setup();
    const groupCount = 4;
    const capacity = getDelaunayMaximumTriangleCount(7);
    const triangles = output(capacity, 'uint32x3');
    const triangleValidity = output(capacity, 'uint32');
    const edges = output(3 * capacity, 'uint32x2');
    const edgeValidity = output(3 * capacity, 'uint32');
    const converged = output(groupCount, 'uint32');
    const count = output(1, 'uint32');
    const requiredCount = output(1, 'uint32');
    const overflow = output(1, 'uint32');
    const triangleOffsets = output(groupCount, 'uint32');
    const triangleCounts = output(groupCount, 'uint32');
    const edgeOffsets = output(groupCount, 'uint32');
    const edgeCounts = output(groupCount, 'uint32');
    const props = {
      positions: input([0, 0, 1, 0, 0, 1, 10, 0, 14, 0, 10, 4, 30, 30], 'float32x2'),
      groupIds: input([0, 0, 0, 2, 2, 2, 3], 'uint32'),
      groupCount,
      maximumCircumradius,
      layout: 'packed' as const,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {
          converged: converged.view,
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view
        },
        groups: {
          triangleOffsets: triangleOffsets.view,
          triangleCounts: triangleCounts.view,
          edgeOffsets: edgeOffsets.view,
          edgeCounts: edgeCounts.view
        }
      }
    };
    const contributor = new GPUGroupConcaveHull(props);
    // R6: none of these caller-side edits may change the validated packed publication or targets.
    Reflect.set(props, 'layout', 'fixed');
    props.output.groups.edgeCounts = triangleCounts.view;
    props.output.status.count = overflow.view;
    props.output.triangles = output(capacity, 'uint32x3').view;
    graph.add(contributor);
    const destroy = run();
    try {
      const retainedSecondGroup = maximumCircumradius === 4 ? 1 : 0;
      const retainedCount = 1 + retainedSecondGroup;
      expect(await readUint32(converged.buffer, groupCount)).toEqual([1, 1, 1, 1]);
      expect(await readUint32(triangleOffsets.buffer, groupCount)).toEqual([
        0,
        1,
        1,
        retainedCount
      ]);
      expect(await readUint32(triangleCounts.buffer, groupCount)).toEqual([
        1,
        0,
        retainedSecondGroup,
        0
      ]);
      expect(await readUint32(edgeOffsets.buffer, groupCount)).toEqual([
        0,
        3,
        3,
        3 * retainedCount
      ]);
      expect(await readUint32(edgeCounts.buffer, groupCount)).toEqual([
        3,
        0,
        3 * retainedSecondGroup,
        0
      ]);
      expect(await readUint32(count.buffer, 1)).toEqual([3 * retainedCount]);
      expect(await readUint32(requiredCount.buffer, 1)).toEqual([3 * retainedCount]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      const triangleRows = await readUint32(triangles.buffer, 3 * capacity);
      expect(triangleRows.slice(0, 3).sort()).toEqual([0, 1, 2]);
      if (retainedSecondGroup) expect(triangleRows.slice(3, 6).sort()).toEqual([3, 4, 5]);
      expect(triangleRows.slice(3 * retainedCount)).toEqual(
        new Array(3 * (capacity - retainedCount)).fill(0)
      );
      expect(await readUint32(triangleValidity.buffer, capacity)).toEqual([
        ...new Array(retainedCount).fill(1),
        ...new Array(capacity - retainedCount).fill(0)
      ]);
      expect(await readUint32(edgeValidity.buffer, 3 * capacity)).toEqual([
        ...new Array(3 * retainedCount).fill(1),
        ...new Array(3 * (capacity - retainedCount)).fill(0)
      ]);
      const edgeRows = await readUint32(edges.buffer, 6 * capacity);
      const actualEdges = Array.from({length: 3 * retainedCount}, (_, row) =>
        edgeRows.slice(2 * row, 2 * row + 2).join(',')
      ).sort();
      expect(actualEdges).toEqual(
        ['0,1', '1,2', '2,0', ...(retainedSecondGroup ? ['3,4', '4,5', '5,3'] : [])].sort()
      );
    } finally {
      destroy();
    }
  }
});

it('packed grouped alpha hull publishes zero counts with no reserved geometry rows', async () => {
  const {graph, input, output, run} = await setup();
  const groupCount = 3;
  const triangles = output(0, 'uint32x3');
  const triangleValidity = output(0, 'uint32');
  const edges = output(0, 'uint32x2');
  const edgeValidity = output(0, 'uint32');
  const converged = output(groupCount, 'uint32');
  const count = output(1, 'uint32');
  const requiredCount = output(1, 'uint32');
  const overflow = output(1, 'uint32');
  const triangleOffsets = output(groupCount, 'uint32');
  const triangleCounts = output(groupCount, 'uint32');
  const edgeOffsets = output(groupCount, 'uint32');
  const edgeCounts = output(groupCount, 'uint32');
  // Zero is the analytic result for two isolated sites. Seeding caller outputs distinguishes
  // publication from an implementation that skips the packed branch when triangle capacity is 0.
  for (const field of [
    count,
    requiredCount,
    overflow,
    converged,
    triangleOffsets,
    triangleCounts,
    edgeOffsets,
    edgeCounts
  ]) {
    field.buffer.write(new Uint32Array(field.view.length).fill(99));
  }
  graph.add(
    new GPUGroupConcaveHull({
      positions: input([0, 0, 2, 2], 'float32x2'),
      groupIds: input([0, 2], 'uint32'),
      groupCount,
      maximumCircumradius: 1,
      layout: 'packed',
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {
          converged: converged.view,
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view
        },
        groups: {
          triangleOffsets: triangleOffsets.view,
          triangleCounts: triangleCounts.view,
          edgeOffsets: edgeOffsets.view,
          edgeCounts: edgeCounts.view
        }
      }
    })
  );
  const destroy = run();
  try {
    expect(await readUint32(converged.buffer, groupCount)).toEqual([1, 1, 1]);
    for (const field of [count, requiredCount, overflow]) {
      expect(await readUint32(field.buffer, 1)).toEqual([0]);
    }
    for (const field of [triangleOffsets, triangleCounts, edgeOffsets, edgeCounts]) {
      expect(await readUint32(field.buffer, groupCount)).toEqual([0, 0, 0]);
    }
  } finally {
    destroy();
  }
});

function expectBoundaryMatchesReference(result: Awaited<ReturnType<typeof runHull>>) {
  const reference = getReferenceBoundary(
    result.triangles,
    result.triangleValidity,
    result.capacity
  );
  const actual = new Set<string>();
  result.edgeValidity.forEach((valid, index) => {
    if (valid) actual.add(`${result.edges[index * 2]},${result.edges[index * 2 + 1]}`);
  });
  expect(actual).toEqual(reference);
  return actual.size;
}

it('alpha hull keeps disconnected components as separate oriented boundaries', async () => {
  const result = await runHull(6, [0, 0, 1, 0, 0, 1, 10, 0, 11, 0, 10, 1], 1);
  expect(result.converged).toBe(1);
  expect(result.triangleValidity.reduce((sum, value) => sum + value, 0)).toBe(2);
  expect(expectBoundaryMatchesReference(result)).toBe(6);
});

it('alpha hull boundary matches the reference on random points, including holes', async () => {
  const positions = createRandomPositions(300, 100);
  const result = await runHull(300, positions, 6);
  expect(result.converged).toBe(1);
  const boundaryCount = expectBoundaryMatchesReference(result);
  expect(boundaryCount).toBeGreaterThan(3);
  // A radius filter on random points leaves at least one hole: edges exceed the outer hull count.
  expect(result.triangleValidity.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(10);
});

it('alpha hull rejects a radius beyond f32 and degenerate collinear input retains nothing', async () => {
  const device = createNullWebGPUDevice();
  try {
    const props = createConcaveHullValidationProps(new GPUCommandGraph(device), 3);
    expect(() => new GPUConcaveHull({...props, maximumCircumradius: 1e39})).toThrow(
      /^concave-hull maximumCircumradius must be a finite number in \[0, Infinity\]/
    );
    expect(() => new GPUConcaveHull({...props, maximumCircumradius: 0})).toThrow(
      /^concave-hull maximumCircumradius must be greater than 0/
    );
  } finally {
    device.destroy();
  }
  const result = await runHull(3, [0, 0, 1, 1, 2, 2], 100);
  expect(result.triangleValidity.reduce((sum, value) => sum + value, 0)).toBe(0);
  expect(result.edgeValidity.reduce((sum, value) => sum + value, 0)).toBe(0);
});

it('alpha hull scales to 20000 points without quadratic boundary extraction', async () => {
  const positions = createRandomPositions(20000, 1000);
  const start = performance.now();
  const result = await runHull(20000, positions, 12);
  const elapsed = performance.now() - start;
  expect(result.converged).toBe(1);
  expectBoundaryMatchesReference(result);
  // The former boundary kernel did about 6e5 * 4e4 reads here and ran for tens of seconds.
  expect(elapsed).toBeLessThan(20000);
}, 60000);

it('group alpha hulls match per-group references and keep empty groups empty', async () => {
  const {graph, input, output, run} = await setup();
  const positions = input([0, 0, 1, 0, 0, 1, 10, 0, 11, 0, 10, 1], 'float32x2');
  const capacity = getDelaunayMaximumTriangleCount(6);
  const triangles = output(capacity * 3, 'uint32x3');
  const triangleValidity = output(capacity * 3, 'uint32');
  const boundaryEdges = output(capacity * 9, 'uint32x2');
  const edgeValidity = output(capacity * 9, 'uint32');
  const converged = output(3, 'uint32');
  graph.add(
    new GPUGroupConcaveHull({
      positions,
      groupIds: input([0, 0, 0, 1, 1, 1], 'uint32'),
      groupCount: 3,
      maximumCircumradius: 1,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: boundaryEdges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  const validity = await readUint32(edgeValidity.buffer, capacity * 9);
  const perGroup = [0, 1, 2].map(group =>
    validity.slice(group * capacity * 3, (group + 1) * capacity * 3).reduce((a, b) => a + b, 0)
  );
  expect(perGroup).toEqual([3, 3, 0]);
  const edges = await readUint32(boundaryEdges.buffer, capacity * 18);
  // Group 1 edges only reference its own points (3..5) or the first member used for masking.
  validity.slice(capacity * 3, capacity * 6).forEach((valid, index) => {
    if (valid) {
      expect([3, 4, 5]).toContain(edges[(capacity * 3 + index) * 2]);
    }
  });
  expect(await readUint32(converged.buffer, 3)).toEqual([1, 1, 1]);
  destroy();
});

it('group alpha hulls of spatially interleaved groups equal the hull of each group alone', async () => {
  const pointCount = 160;
  const groupCount = 4;
  const positionsValues = createRandomPositions(pointCount, 30);
  const groupValues = Array.from({length: pointCount}, (_, row) => row % groupCount);
  const capacity = getDelaunayMaximumTriangleCount(pointCount);
  const {graph, input, output, run} = await setup();
  const triangles = output(capacity * groupCount, 'uint32x3');
  const triangleValidity = output(capacity * groupCount, 'uint32');
  const boundaryEdges = output(capacity * groupCount * 3, 'uint32x2');
  const edgeValidity = output(capacity * groupCount * 3, 'uint32');
  const converged = output(groupCount, 'uint32');
  graph.add(
    new GPUGroupConcaveHull({
      positions: input(positionsValues, 'float32x2'),
      groupIds: input(groupValues, 'uint32'),
      groupCount,
      maximumCircumradius: 4,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: boundaryEdges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  const edgeRows = await readUint32(boundaryEdges.buffer, capacity * groupCount * 3 * 2);
  const edgeFlags = await readUint32(edgeValidity.buffer, capacity * groupCount * 3);
  expect(await readUint32(converged.buffer, groupCount)).toEqual([1, 1, 1, 1]);
  destroy();
  for (let group = 0; group < groupCount; group++) {
    const rows = groupValues.flatMap((value, row) => (value === group ? [row] : []));
    const alone = await runHull(
      rows.length,
      rows.flatMap(row => [positionsValues[2 * row], positionsValues[2 * row + 1]]),
      4
    );
    const expected = new Set<string>();
    alone.edgeValidity.forEach((valid, index) => {
      if (valid) {
        expected.add(`${rows[alone.edges[2 * index]]},${rows[alone.edges[2 * index + 1]]}`);
      }
    });
    const actual = new Set<string>();
    for (let slot = 0; slot < capacity * 3; slot++) {
      const global = group * capacity * 3 + slot;
      if (edgeFlags[global]) {
        actual.add(`${edgeRows[2 * global]},${edgeRows[2 * global + 1]}`);
      }
    }
    expect(expected.size).toBeGreaterThan(3);
    expect(actual, `group ${group}`).toEqual(expected);
  }
});

it('alpha hulls reject undersized outputs and invalid group counts with named limits', () => {
  const device = createNullWebGPUDevice();
  try {
    const capacity = getDelaunayMaximumTriangleCount(6);
    const undersized = createConcaveHullValidationProps(
      new GPUCommandGraph(device),
      6,
      1,
      capacity - 1
    );
    expect(() => new GPUConcaveHull(undersized)).toThrow(
      new RegExp(`^concave-hull output.triangles must have ${capacity} rows`)
    );
    const graph = new GPUCommandGraph(device);
    const props = createConcaveHullValidationProps(graph, 6, 2);
    const groupIds = createTransientView(graph, 'groups', 'uint32', 6);
    expect(() => new GPUGroupConcaveHull({...props, groupIds, groupCount: 0})).toThrow(
      /^group-concave-hull groupCount must be an integer in \[1, 4294967295\]/
    );
    const converged = graph.createDataView(props.output.status.converged.buffer, {
      format: 'uint32',
      length: 1
    });
    expect(
      () =>
        new GPUGroupConcaveHull({
          ...props,
          groupIds,
          groupCount: 2,
          output: {...props.output, status: {converged}}
        })
    ).toThrow(/^group-concave-hull output.status.converged must have at least 2 rows/);
  } finally {
    device.destroy();
  }
});
