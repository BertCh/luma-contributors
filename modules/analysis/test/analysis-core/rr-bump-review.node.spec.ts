// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPU_GRID_COORDINATE_WGSL
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  createEncounterScanNode,
  type EncounterShape
} from '../../src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels';
import {GPUPointDensity} from '../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {GPUClassBreaks} from '../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {computeEncountersOracle} from '../gpu-spatial-analysis/trajectory-encounters/encounters-oracle';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';

function createReviewEncounterKernel(shape: EncounterShape) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32x3' | 'uint32' | 'float32' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const node = createEncounterScanNode(graph, {
    id: 'review-scan',
    shape,
    positions: view('positions', 'float32x3', shape.trackCount * shape.bucketCount),
    cellOffsets: view('offsets', 'uint32', shape.columns * shape.rows * shape.bucketCount + 1),
    objectIds: view('objects', 'uint32', shape.trackCount * shape.bucketCount),
    distance: view('distance', 'float32', 1),
    state: view('state', 'uint32', 1),
    hitPairs: view('pairs', 'uint32x2', shape.hitCapacity),
    hitBuckets: view('buckets', 'uint32', shape.hitCapacity),
    hitDistances: view('distances', 'float32', shape.hitCapacity)
  });
  return {device, kernel: getGPUCommandNodeKernelSources([node])[0]};
}

for (const [centerX, centerY] of [
  [0, 0],
  [500000, 6000000]
]) {
  it.each([
    0, 0.0001
  ])(`R9 range visits stay bounded on a translated 2048-square grid (${centerX},${centerY}, distance=%s)`, distance => {
    const {device, kernel} = createReviewEncounterKernel({
      trackCount: 2,
      bucketCount: 1,
      hitCapacity: 2,
      pairCapacity: 2,
      columns: 41,
      rows: 1,
      bounds: [-20.5, 0, 20.5, 1],
      cellSize: 1
    });
    try {
      const range = kernel.source.match(/fn getEncounterAxisRange\([\s\S]*?\n\}/)![0];
      const source = `${GPU_GRID_COORDINATE_WGSL}\n${range}
@group(0) @binding(0) var<storage, read_write> results: array<u32>;
@compute @workgroup_size(1) fn main() {
  let x = getEncounterAxisRange(${centerX}.0, ${distance}, ${centerX - 1}.0, ${centerX + 1}.0, 2048u);
  let y = getEncounterAxisRange(${centerY}.0, ${distance}, ${centerY - 1}.0, ${centerY + 1}.0, 2048u);
  results[0] = x.y - x.x + 1u; results[1] = y.y - y.x + 1u;
}`;
      const results = new Uint32Array(2);
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
        0: {0: results}
      });
      // A zero-distance query needs its cell plus one cell of rounding padding per side.
      // A positive distance below one cell adds at most one cell per side: 9 / 25 visits.
      for (const width of results) {
        expect(width).toBeGreaterThanOrEqual(1);
        expect(width).toBeLessThanOrEqual(distance === 0 ? 3 : 5);
      }
      expect(results[0] * results[1]).toBeLessThanOrEqual(distance === 0 ? 9 : 25);
    } finally {
      device.destroy();
    }
  });
}

for (const wrapColumns of [false, true]) {
  it.each([
    [-140, 170],
    [170, -140]
  ])(
    'R8 great-circle seam ownership is independent of track order (wrap=' +
      wrapColumns +
      ', longitudes=%j)',
    (first, second) => {
      const columns = wrapColumns ? 4 : 3;
      const {device, kernel} = createReviewEncounterKernel({
        trackCount: 2,
        bucketCount: 1,
        hitCapacity: 2,
        pairCapacity: 2,
        columns,
        rows: 1,
        bounds: wrapColumns ? [-180, -1, 180, 1] : [-170, -1, 170, 1],
        cellSize: 86,
        geodesic: {radius: 1, maximumDistance: 1.4, wrapColumns}
      });
      // Hand-assigned builder cells: smaller longitude occupies 0; larger occupies the last.
      const occupied = first < second ? [0, columns - 1] : [columns - 1, 0];
      const values: Record<string, Float32Array | Uint32Array> = {
        positions: Float32Array.of(first, 0, 0.5, second, 0, 0.5),
        cellOffsets: Uint32Array.from(
          {length: columns + 1},
          (_, boundary) => occupied.filter(column => column < boundary).length
        ),
        objectIds: first < second ? Uint32Array.of(0, 1) : Uint32Array.of(1, 0),
        distance: Float32Array.of(1),
        state: new Uint32Array(1),
        hitPairs: new Uint32Array(4).fill(99),
        hitBuckets: new Uint32Array(2).fill(99),
        hitDistances: new Float32Array(2).fill(99)
      };
      const errors = vi.spyOn(console, 'error').mockImplementation(message => {
        if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
      });
      try {
        const bindings = Object.fromEntries(
          new WgslReflect(kernel.source).storage.map(binding => [
            binding.binding,
            values[binding.name]
          ])
        );
        // Same non-saturating serial reservation substitution as the R1 interpreter probe.
        const source = kernel.source.replace(
          /fn reserveTrajectoryCount\(\) -> u32 \{[\s\S]*?\n\}/,
          'fn reserveTrajectoryCount() -> u32 { return atomicAdd(&state[stateOffset], 1u); }'
        );
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [1, 1, 1],
          {0: bindings},
          {constants: kernel.constants}
        );
        expect(Array.from(values['state'])).toEqual([wrapColumns ? 1 : 0]);
        if (wrapColumns) {
          expect(Array.from(values['hitPairs'].slice(0, 2))).toEqual([0, 1]);
          // Equatorial shortest arc: (360 - 310) degrees at radius 1 = 50*pi/180.
          expect(values['hitDistances'][0]).toBeCloseTo(0.8726646259971648, 5);
          expect(values['hitBuckets'][0]).toBe(0);
        }
      } finally {
        errors.mockRestore();
        device.destroy();
      }
    }
  );
}

it('R5 default class-break methods accept aligned chunks while maximum-breaks requires packed values', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createVectorView('values', 'float32', [
    createTransientView(graph, 'first', 'float32', 2),
    createTransientView(graph, 'second', 'float32', 2)
  ]);
  const props = {
    values,
    parameters: createTransientView(graph, 'parameters', 'float32', 32),
    maximumClassCount: 6,
    output: {
      breaks: createTransientView(graph, 'breaks', 'float32', 7),
      classCount: createTransientView(graph, 'class-count', 'uint32', 1)
    }
  };
  try {
    const contributor = new GPUClassBreaks(props);
    expect(contributor.props.values).toBe(values);
    expect(contributor.methods).toContain('equal-interval');
    expect(contributor.methods).toContain('natural-breaks');
    expect(contributor.methods).not.toContain('maximum-breaks');
    expect(() => new GPUClassBreaks({...props, methods: ['maximum-breaks']})).toThrow(
      /maximum-breaks.*one packed view/
    );
    expect(new GPUClassBreaks({...props, methods: ['equal-interval']}).props.values).toBe(values);
  } finally {
    device.destroy();
  }
});

it('R1 encounter scan finds the exact-distance pair across rounded columns 1 and 3 once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32x3' | 'uint32' | 'float32' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const samples = Float32Array.of(-18.5, 0.5, -17.5, 0.5);
  expect(computeEncountersOracle(samples, 2, 1, 1, undefined, [-20.5, 0, 20.5, 1])).toEqual([
    {track: 0, partner: 1, firstBucket: 0, minimumDistance: 1, bucketCount: 1}
  ]);
  const node = createEncounterScanNode(graph, {
    id: 'boundary-scan',
    shape: {
      trackCount: 2,
      bucketCount: 1,
      hitCapacity: 2,
      pairCapacity: 2,
      columns: 41,
      rows: 1,
      bounds: [-20.5, 0, 20.5, 1],
      cellSize: 1
    },
    positions: view('positions', 'float32x3', 2),
    cellOffsets: view('offsets', 'uint32', 42),
    objectIds: view('objects', 'uint32', 2),
    distance: view('distance', 'float32', 1),
    state: view('state', 'uint32', 1),
    hitPairs: view('pairs', 'uint32x2', 2),
    hitBuckets: view('buckets', 'uint32', 2),
    hitDistances: view('distances', 'float32', 2)
  });
  const [kernel] = getGPUCommandNodeKernelSources([node]);
  // NumPy binary32 published formula: columns [1,3], from normalized products
  // [1.9999998807907104,3]. Offsets below are those independently assigned cells.
  const values: Record<string, Float32Array | Uint32Array> = {
    positions: Float32Array.of(-18.5, 0.5, 0.5, -17.5, 0.5, 0.5),
    cellOffsets: Uint32Array.from({length: 42}, (_, boundary) =>
      boundary <= 1 ? 0 : boundary <= 3 ? 1 : 2
    ),
    objectIds: Uint32Array.of(0, 1),
    distance: Float32Array.of(1),
    state: new Uint32Array(1),
    hitPairs: new Uint32Array(4).fill(99),
    hitBuckets: new Uint32Array(2).fill(99),
    hitDistances: new Float32Array(2).fill(99)
  };
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => [binding.binding, values[binding.name]])
    );
    // The interpreter lacks compare-exchange. Two serial invocations cannot saturate,
    // so substitute only slot reservation; the complete production candidate scan is executed.
    const source = kernel.source.replace(
      /fn reserveTrajectoryCount\(\) -> u32 \{[\s\S]*?\n\}/,
      'fn reserveTrajectoryCount() -> u32 { return atomicAdd(&state[stateOffset], 1u); }'
    );
    expect(source).not.toContain('atomicCompareExchangeWeak');
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: kernel.constants}
    );
    expect(Array.from(values['state'])).toEqual([1]);
    expect(Array.from(values['hitPairs'].slice(0, 2))).toEqual([0, 1]);
    expect(values['hitBuckets'][0]).toBe(0);
    expect(values['hitDistances'][0]).toBe(1);
  } finally {
    errors.mockRestore();
    device.destroy();
  }
});

it.each([
  'float32',
  'float32x4'
] as const)('R10 captures a %s bounds getter once and preserves view identity', format => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const bounds = createTransientView(graph, 'bounds', format, format === 'float32' ? 4 : 1);
  let boundsReads = 0;
  try {
    const contributor = new GPUPointDensity({
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      gridSize: [2, 1],
      get bounds() {
        boundsReads++;
        return bounds;
      },
      output: {values: createTransientView(graph, 'values', 'float32', 2)}
    });
    expect(contributor.props.bounds).toBe(bounds);
    getParsedKernelSources(contributor.getCommandNodes(graph));
    expect(boundsReads).toBe(1);
  } finally {
    device.destroy();
  }
});

it.each([
  'atomic',
  'workgroup'
] as const)('R2 public point density generates parseable extreme-bound %s kernels', sumAccumulation => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const callerBounds: [number, number, number, number] = [-3e38, 0, 3e38, 1];
  try {
    const contributor = new GPUPointDensity({
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      weights: createTransientView(graph, 'weights', 'float32', 1),
      bounds: callerBounds,
      gridSize: [2, 1],
      statistic: 'sum',
      sumAccumulation,
      output: {
        values: createTransientView(graph, 'values', 'float32', 2),
        counts: createTransientView(graph, 'counts', 'uint32', 2),
        sums: createTransientView(graph, 'sums', 'float32', 2)
      }
    });
    // R10: the validated literal domain must survive mutation of the caller's tuple.
    callerBounds[2] = 0;
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(1);
    const initializer = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.source.includes('boundsValues[boundsValuesOffset + index] =')
    )!;
    const bounds = new Float32Array(4).fill(99);
    const bindings = Object.fromEntries(
      new WgslReflect(initializer.source).storage.map(binding => [binding.binding, bounds])
    );
    new WgslExec(new WgslParser().parse(initializer.source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: initializer.constants}
    );
    // Independent binary32 rounding of the four accepted literal bounds.
    expect(Array.from(bounds)).toEqual([-3.0000000054977558e38, 0, 3.0000000054977558e38, 1]);
  } finally {
    device.destroy();
  }
});
