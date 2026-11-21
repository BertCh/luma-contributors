// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {expect, it} from 'vitest';
import {
  CPUIsolinesOperation,
  computeIsolinesOnCPU
} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {
  GPUIsolinesOperation,
  type IsolinesOperationProps
} from '../../../src/raster-analysis/isolines/gpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {createTransientView, GPUCommandGraph, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const grid = makeGPURasterMetadata({width: 3, height: 3});
const parameters = getGPUIsolinesParameterValues({levelCount: 2, extent: [0, 0, 3, 3]}, grid);
const input = {
  values: Float32Array.of(0, 1, 2, 0, 1, 2, 0, 1, 2),
  levels: Float32Array.of(0.5, 1.5),
  parameters
};

it('extracts the two vertical lines of the affine plane x, with high values to the left', () => {
  const result = computeIsolinesOnCPU(input, {grid, segmentCapacity: 8, vertexCapacity: 12});
  expect(Array.from(result.segments.subarray(0, 16))).toEqual([
    1, 1.5, 1, 0.5, 2, 1.5, 2, 0.5, 1, 2.5, 1, 1.5, 2, 2.5, 2, 1.5
  ]);
  expect(Array.from(result.segmentLevels.subarray(0, 4))).toEqual([0, 1, 0, 1]);
  expect(Array.from(result.vertices.subarray(0, 12))).toEqual([
    1, 2.5, 1, 1.5, 1, 0.5, 2, 2.5, 2, 1.5, 2, 0.5
  ]);
  expect(Array.from(result.polylineOffsets.subarray(0, 3))).toEqual([0, 3, 6]);
  expect(result.polylineCount[0]).toBe(2);
  expect(result.vertexCount[0]).toBe(6);
  expect(result.overflow[0]).toBe(0);
  expect(result.polylineOverflow[0]).toBe(0);
});

it('keeps only complete chains and suppresses stitching after segment overflow', () => {
  const result = computeIsolinesOnCPU(input, {grid, segmentCapacity: 8, vertexCapacity: 3});
  expect(result.polylineCount[0]).toBe(1);
  expect(result.vertexCount[0]).toBe(3);
  expect(result.polylineOverflow[0]).toBe(1);
  const overflow = computeIsolinesOnCPU(input, {grid, segmentCapacity: 1, vertexCapacity: 12});
  expect(overflow.requiredCount[0]).toBe(4);
  expect(overflow.count[0]).toBe(1);
  expect(overflow.overflow[0]).toBe(1);
  expect(overflow.polylineCount[0]).toBe(0);
  expect(overflow.polylineOverflow[0]).toBe(1);
});

it('resolves a bilinear saddle tie by connecting the high diagonal and walks a closed hill', () => {
  const saddleGrid = makeGPURasterMetadata({width: 2, height: 2});
  const saddle = computeIsolinesOnCPU(
    {
      values: Float32Array.of(1, -1, -1, 1),
      levels: Float32Array.of(0),
      parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 2]}, saddleGrid)
    },
    {grid: saddleGrid, segmentCapacity: 2}
  );
  expect(Array.from(saddle.segmentEdges)).toEqual([0, 3, 1, 2]);
  const hill = computeIsolinesOnCPU(
    {...input, values: Float32Array.of(0, 0, 0, 0, 2, 0, 0, 0, 0), levels: Float32Array.of(1)},
    {grid, segmentCapacity: 8, vertexCapacity: 8}
  );
  expect(hill.count[0]).toBe(4);
  expect(hill.polylineCount[0]).toBe(1);
  expect(hill.polylineClosed[0]).toBe(1);
  expect(hill.vertexCount[0]).toBe(5);
  expect(Array.from(hill.vertices.subarray(0, 2))).toEqual(
    Array.from(hill.vertices.subarray(8, 10))
  );
});

it('validates malformed arrays and operation construction', () => {
  expect(() =>
    computeIsolinesOnCPU({...input, mask: new Uint32Array(8)}, {grid, segmentCapacity: 8})
  ).toThrow(/mask/);
  const props: IsolinesOperationProps = {
    id: 'isoline-test',
    grid,
    values: 'values',
    levels: 'levels',
    parameters: 'parameters',

    segmentCapacity: 8,
    output: {
      segments: 'segments',
      segmentLevels: 'segmentLevels',
      status: {count: 'count', overflow: 'overflow'}
    }
  };
  expect(() => new CPUIsolinesOperation({...props, noDataValue: Infinity})).toThrow(
    /isoline-test.*noDataValue/
  );
  for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation])
    expect(() => new Operation({...props, segmentCapacity: 0})).toThrow(/isoline-test/);
});

it('honours rounded nodata, the input mask, active level clamping and zero-length bridges', () => {
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 2});
  const scene = {
    values: Float32Array.of(0.1, 2, 0, 0),
    levels: Float32Array.of(1),
    parameters: getGPUIsolinesParameterValues({levelCount: 100, extent: [0, 0, 2, 2]}, sceneGrid)
  };
  expect(
    computeIsolinesOnCPU(scene, {grid: sceneGrid, segmentCapacity: 2, noDataValue: 0.1}).count[0]
  ).toBe(0);
  expect(
    computeIsolinesOnCPU(
      {...scene, mask: Uint32Array.of(0, 1, 1, 1)},
      {grid: sceneGrid, segmentCapacity: 2}
    ).count[0]
  ).toBe(0);
  const bridge = computeIsolinesOnCPU(
    {...scene, values: Float32Array.of(1, 0, 0, 0)},
    {grid: sceneGrid, segmentCapacity: 2, vertexCapacity: 2}
  );
  expect(bridge.count[0]).toBe(1);
  expect(Array.from(bridge.segments.subarray(0, 4))).toEqual([0.5, 0.5, 0.5, 0.5]);
  expect(bridge.vertexCount[0]).toBe(1);
  // Capacity is tested before duplicate collapse: this one-point chain requires two slots.
  const dropped = computeIsolinesOnCPU(
    {...scene, values: Float32Array.of(1, 0, 0, 0)},
    {grid: sceneGrid, segmentCapacity: 2, vertexCapacity: 1}
  );
  expect(dropped.polylineCount[0]).toBe(0);
  expect(dropped.polylineOverflow[0]).toBe(1);
});

it('derives active levels from the input view rather than a stale maximumLevelCount prop', async () => {
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 2});
  const scene = {
    values: Float32Array.of(0, 2, 0, 2),
    levels: Float32Array.of(0.5, 1.5),
    parameters: getGPUIsolinesParameterValues({levelCount: 2, extent: [0, 0, 2, 2]}, sceneGrid)
  };
  const expected = [0.75, 1.5, 0.75, 0.5, 1.25, 1.5, 1.25, 0.5];
  expect(
    Array.from(computeIsolinesOnCPU(scene, {grid: sceneGrid, segmentCapacity: 2}).segments)
  ).toEqual(expected);
  const props: IsolinesOperationProps = {
    ...{maximumLevelCount: 1},
    grid: sceneGrid,
    segmentCapacity: 2,
    values: 'values',
    levels: 'levels',
    parameters: 'parameters',
    output: {
      segments: 'segments',
      segmentLevels: 'segmentLevels',
      status: {count: 'count', overflow: 'overflow'}
    }
  };
  const operation = new CPUIsolinesOperation(props);
  const rows = (values: Float32Array): GPUCPUVector => ({
    format: 'float32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await operation.execute({
    inputs: {
      values: rows(scene.values),
      levels: rows(scene.levels),
      parameters: rows(scene.parameters)
    }
  });
  expect(Array.from(result['segments'] as Float32Array)).toEqual(expected);
  expect(Array.from(result['count'] as Uint32Array)).toEqual([2]);
});

it('rejects malformed or nonfinite grid.affine in pure and operation validation', () => {
  const props: IsolinesOperationProps = {
    id: 'invalid-affine',
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    segmentCapacity: 1,
    values: 'values',
    levels: 'levels',
    parameters: 'parameters',
    output: {
      segments: 'segments',
      segmentLevels: 'segmentLevels',
      status: {count: 'count', overflow: 'overflow'}
    }
  };
  const scene = {
    values: Float32Array.of(0, 2, 0, 2),
    levels: Float32Array.of(1),
    parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 2]}, props.grid)
  };
  for (const affine of [
    [1, 0, NaN, 0, -1, 0],
    [1, 0, 0, 0, -1]
  ]) {
    const invalidGrid = {...props.grid, affine: affine as unknown as typeof props.grid.affine};
    expect(() => new CPUIsolinesOperation({...props, grid: invalidGrid})).toThrow(
      /invalid-affine.*grid.affine/
    );
    expect(() => computeIsolinesOnCPU(scene, {grid: invalidGrid, segmentCapacity: 1})).toThrow(
      /grid.affine/
    );
  }
  // Constructor-only check with the published null device; no GPU commands are executed.
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(
      () =>
        new GPUIsolines({
          id: 'invalid-affine',
          grid: {...props.grid, affine: [1, 0, NaN, 0, -1, 0]},
          values: createTransientView(graph, 'values', 'float32', 4),
          levels: createTransientView(graph, 'levels', 'float32', 1),
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          output: {
            segments: createTransientView(graph, 'segments', 'float32x4', 1),
            segmentLevels: createTransientView(graph, 'segmentLevels', 'uint32', 1),
            status: {
              count: createTransientView(graph, 'count', 'uint32', 1),
              overflow: createTransientView(graph, 'overflow', 'uint32', 1)
            }
          }
        })
    ).toThrow(/invalid-affine.*affine/);
  } finally {
    device.destroy();
  }
});

it('retains the mathematical midpoint when opposing finite samples overflow a float32 difference', () => {
  const maximum = 3.4028234663852886e38;
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 2});
  const result = computeIsolinesOnCPU(
    {
      values: Float32Array.of(-maximum, maximum, -maximum, maximum),
      levels: Float32Array.of(0),
      parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 2]}, sceneGrid)
    },
    {grid: sceneGrid, segmentCapacity: 1}
  );
  expect(Array.from(result.segments)).toEqual([1, 1.5, 1, 0.5]);
});

it('WRT-01 reports segment loss without stitching in the CPU helper and named operation', async () => {
  const sceneGrid = makeGPURasterMetadata({width: 2, height: 2});
  const scene = {
    values: Float32Array.of(0, 1, 1, 0),
    levels: Float32Array.of(0.5),
    parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 2]}, sceneGrid)
  };
  // A checkerboard cell has two crossings/segments, so capacity one loses one.
  for (const vertexCapacity of [undefined, 4]) {
    const result = computeIsolinesOnCPU(scene, {
      grid: sceneGrid,
      segmentCapacity: 1,
      vertexCapacity
    });
    expect([
      result.requiredCount[0],
      result.count[0],
      result.overflow[0],
      result.polylineOverflow[0]
    ]).toEqual([2, 1, 1, 1]);
  }
  const operation = new CPUIsolinesOperation({
    grid: sceneGrid,
    segmentCapacity: 1,
    values: 'values',
    levels: 'levels',
    parameters: 'parameters',
    output: {
      segments: 'segments',
      segmentLevels: 'segmentLevels',
      status: {count: 'count', overflow: 'overflow', polylineOverflow: 'polylineOverflow'}
    }
  });
  const rows = (values: Float32Array): GPUCPUVector => ({
    format: 'float32',
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const retainedWord = Uint32Array.of(99);
  const observed = [retainedWord[0]];
  for (const values of [scene.values, new Float32Array(4)]) {
    const result = await operation.execute({
      inputs: {values: rows(values), levels: rows(scene.levels), parameters: rows(scene.parameters)}
    });
    retainedWord.set(result['polylineOverflow'] as Uint32Array);
    observed.push(retainedWord[0]);
  }
  expect(observed).toEqual([99, 1, 0]);
});

it('EGR-13 decodes only retained CPU polyline rows and rejects incomplete layouts', async () => {
  const {decodeGPUIsolinesPolylines} = await import(
    '../../../src/raster-analysis/isolines/gpu-isolines'
  );
  const result = computeIsolinesOnCPU(input, {grid, segmentCapacity: 8, vertexCapacity: 12});
  const readback = {polylines: result, status: result};
  const decoded = decodeGPUIsolinesPolylines(readback, {
    levels: input.levels,
    localOrigin: [100, 200]
  });
  expect(decoded).toEqual({
    segmentComplete: true,
    polylineComplete: true,
    complete: true,
    polylines: [
      {
        levelIndex: 0,
        levelValue: 0.5,
        closed: false,
        coordinates: [
          [101, 202.5],
          [101, 201.5],
          [101, 200.5]
        ]
      },
      {
        levelIndex: 1,
        levelValue: 1.5,
        closed: false,
        coordinates: [
          [102, 202.5],
          [102, 201.5],
          [102, 200.5]
        ]
      }
    ]
  });
  const bytes = (values: Float32Array | Uint32Array) => {
    const storage = new Uint8Array(values.byteLength + 3);
    storage.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), 1);
    return storage.subarray(1, 1 + values.byteLength);
  };
  const packed = {
    polylines: {
      ...result,
      vertices: bytes(result.vertices),
      polylineOffsets: bytes(result.polylineOffsets),
      polylineCount: result.polylineCount.slice().buffer
    },
    status: result
  };
  expect(
    decodeGPUIsolinesPolylines(packed, {levels: bytes(input.levels)}).polylines[0].coordinates
  ).toEqual([
    [1, 2.5],
    [1, 1.5],
    [1, 0.5]
  ]);
  for (const invalid of [
    {...readback, status: {...result, overflow: new Uint32Array(0)}},
    {...readback, polylines: {...result, vertices: new Uint8Array(3)}},
    {...readback, polylines: {...result, polylineCount: Uint32Array.of(20)}},
    {...readback, polylines: {...result, polylineOffsets: Uint32Array.of(0, 0, 6)}},
    {...readback, polylines: {...result, polylineLevels: Uint32Array.of(0, 9)}},
    {...readback, polylines: {...result, vertexCount: Uint32Array.of(25)}}
  ])
    expect(() => decodeGPUIsolinesPolylines(invalid, {levels: input.levels})).toThrow(
      /decodeGPUIsolinesPolylines/
    );
  const segmentOverflow = computeIsolinesOnCPU(input, {
    grid,
    segmentCapacity: 1,
    vertexCapacity: 12
  });
  expect(
    decodeGPUIsolinesPolylines(
      {polylines: segmentOverflow, status: segmentOverflow},
      {levels: input.levels}
    )
  ).toMatchObject({
    segmentComplete: false,
    polylineComplete: false,
    complete: false,
    polylines: []
  });
  const truncated = computeIsolinesOnCPU(input, {grid, segmentCapacity: 8, vertexCapacity: 3});
  expect(
    decodeGPUIsolinesPolylines({polylines: truncated, status: truncated}, {levels: input.levels})
  ).toMatchObject({segmentComplete: true, polylineComplete: false, complete: false});
});

it('EGR-13 retains closure and bounds empty results without reading stale coordinate tails', async () => {
  const {decodeGPUIsolinesPolylines} = await import(
    '../../../src/raster-analysis/isolines/gpu-isolines'
  );
  const polylines = {
    vertices: Float32Array.of(0, 0, 1, 0, 0, 0, NaN, NaN),
    polylineOffsets: Uint32Array.of(0, 3, 999),
    polylineLevels: Uint32Array.of(0, 999),
    polylineClosed: Uint32Array.of(1, 999),
    polylineCount: Uint32Array.of(1),
    vertexCount: Uint32Array.of(3)
  };
  const status = {overflow: Uint32Array.of(0), polylineOverflow: Uint32Array.of(0)};
  expect(
    decodeGPUIsolinesPolylines({polylines, status}, {levels: Float32Array.of(7)}).polylines
  ).toEqual([
    {
      coordinates: [
        [0, 0],
        [1, 0],
        [0, 0]
      ],
      levelIndex: 0,
      levelValue: 7,
      closed: true
    }
  ]);
  expect(
    decodeGPUIsolinesPolylines(
      {
        polylines: {...polylines, polylineCount: Uint32Array.of(0), vertexCount: Uint32Array.of(0)},
        status
      },
      {levels: new Float32Array(0)}
    ).polylines
  ).toEqual([]);
  expect(() =>
    decodeGPUIsolinesPolylines(
      {polylines, status: {...status, polylineOverflow: new Uint32Array(0)}},
      {levels: Float32Array.of(7)}
    )
  ).toThrow(/polylineOverflow/);
});
