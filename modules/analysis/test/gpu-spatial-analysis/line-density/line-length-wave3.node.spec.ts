// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLineLengthPerPolygonOnCPU} from '../../../src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation';
import type {CPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';

const polygons: CPUSpatialJoinGeometry = {
  kind: 'polygons',
  positions: Float32Array.of(0, 0, 0.5, 0, 0.5, 1, 0, 1),
  ringOffsets: Uint32Array.of(0, 4),
  polygonOffsets: Uint32Array.of(0, 1),
  featureOffsets: Uint32Array.of(0, 1)
};
it('review multipart control weights both parts once by their feature row', () => {
  // Two horizontal parts each intersect [0,2] x [0,2] in length 2:
  // total 4, weighted total 4*7=28, contributing segment count 2.
  const result = computeLineLengthPerPolygonOnCPU(
    {
      lines: {
        kind: 'lines',
        positions: Float32Array.of(-1, 0.5, 3, 0.5, -1, 1.5, 3, 1.5),
        lineOffsets: Uint32Array.of(0, 2, 4),
        featureOffsets: Uint32Array.of(0, 2)
      },
      polygons: {...polygons, positions: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2)},
      lineWeights: Float32Array.of(7)
    },
    {pairCapacity: 4}
  );
  expect([...result.clippedLengths]).toEqual([4]);
  expect([...result.weightedLengths]).toEqual([28]);
  expect([...result.segmentCounts]).toEqual([2]);
  expect([...result.overflow]).toEqual([0]);
});
it('UGE-07 retains the half-unit clip in both directions at 2^56', () => {
  // Horizontal intersection with [0,.5] x [0,1] has length .5, weight 3 gives 1.5.
  for (const positions of [
    Float32Array.of(-(2 ** 56), 0.5, 1, 0.5),
    Float32Array.of(1, 0.5, -(2 ** 56), 0.5)
  ]) {
    const result = computeLineLengthPerPolygonOnCPU(
      {
        lines: {kind: 'lines', positions, lineOffsets: Uint32Array.of(0, 2)},
        polygons,
        lineWeights: Float32Array.of(3)
      },
      {pairCapacity: 4}
    );
    expect([...result.clippedLengths]).toEqual([0.5]);
    expect([...result.weightedLengths]).toEqual([1.5]);
    expect([...result.segmentCounts]).toEqual([1]);
    expect([...result.overflow]).toEqual([0]);
    expect([...result.candidateOverflow]).toEqual([0]);
  }
});
it('UGE-10 preserves an empty MultiPolygon row and rejects incomplete terminal offsets', () => {
  const lines: CPUSpatialJoinGeometry = {
    kind: 'lines',
    positions: Float32Array.of(0, 0.5, 1, 0.5),
    lineOffsets: Uint32Array.of(0, 2)
  };
  const result = computeLineLengthPerPolygonOnCPU(
    {
      lines,
      polygons: {
        kind: 'polygons',
        positions: new Float32Array(),
        ringOffsets: Uint32Array.of(0),
        polygonOffsets: Uint32Array.of(0),
        featureOffsets: Uint32Array.of(0, 0)
      }
    },
    {pairCapacity: 4}
  );
  expect([...result.clippedLengths]).toEqual([0]);
  expect([...result.segmentCounts]).toEqual([0]);
  for (const ringOffsets of [Uint32Array.of(0, 3), Uint32Array.of(0, 999)]) {
    expect(() =>
      computeLineLengthPerPolygonOnCPU(
        {lines, polygons: {...polygons, ringOffsets}},
        {pairCapacity: 4}
      )
    ).toThrow(/ringOffsets/);
  }
});

it('UGE-07 maps clippedLengths in both operation backends and rejects the retired output name', async () => {
  const {createCPULineLengthPerPolygonOperation} = await import(
    '../../../src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation'
  );
  const {createLineLengthPerPolygonOperation} = await import(
    '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon-operation'
  );
  const {createSamplingCPUInputs} = await import(
    '../point-process/validation-sampling-operation-test-utils'
  );
  const props = {
    lines: {
      kind: 'lines' as const,
      positions: {value: 'linePositions', length: 2},
      lineOffsets: {value: 'lineOffsets', length: 2}
    },
    polygons: {
      kind: 'polygons' as const,
      positions: {value: 'polygonPositions', length: 4},
      ringOffsets: {value: 'rings', length: 2},
      polygonOffsets: {value: 'parts', length: 2},
      featureOffsets: {value: 'features', length: 2}
    },
    pairCapacity: 4,
    output: {
      clippedLengths: 'clip',
      segmentCounts: 'count',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    }
  };
  expect(createLineLengthPerPolygonOperation(props).outputs['clippedLengths'].format).toBe(
    'float32'
  );
  const legacy = {...props, output: {...props.output, lengths: 'clip'}};
  expect(() => createLineLengthPerPolygonOperation(legacy)).toThrow(/lengths.*clippedLengths/);
  const inputs = createSamplingCPUInputs({
    linespositions: {format: 'float32x2', values: Float32Array.of(-1, 0.5, 1, 0.5)},
    lineslineOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
    polygonspositions: {format: 'float32x2', values: Float32Array.from(polygons.positions)},
    polygonsringOffsets: {format: 'uint32', values: Uint32Array.of(0, 4)},
    polygonspolygonOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)},
    polygonsfeatureOffsets: {format: 'uint32', values: Uint32Array.of(0, 1)}
  });
  const operationProps = {...props, output: {...props.output}};
  const gpuOperation = createLineLengthPerPolygonOperation(operationProps);
  const cpuOperation = createCPULineLengthPerPolygonOperation(operationProps);
  const {output} = operationProps;
  expect(gpuOperation.outputs['clippedLengths'].value).toBe('clip');
  // Mutating the caller's routing after construction must not redirect either backend.
  output.segmentCounts = 'changed';
  expect(gpuOperation.outputs['segmentCounts'].value).toBe('count');
  const result = await cpuOperation.execute({inputs});
  expect(result['clip']).toEqual(Float32Array.of(0.5));
  expect(result['count']).toEqual(Uint32Array.of(1));
  expect(result['changed']).toBeUndefined();
  let outputReadCount = 0;
  class GetterProps {
    readonly lines = props.lines;
    readonly polygons = props.polygons;
    readonly pairCapacity = props.pairCapacity;
    get output() {
      outputReadCount++;
      return props.output;
    }
  }
  const getterProps = new GetterProps();
  expect(createLineLengthPerPolygonOperation(getterProps).outputs['clippedLengths'].value).toBe(
    'clip'
  );
  outputReadCount = 0;
  const getterOperation = createCPULineLengthPerPolygonOperation(getterProps);
  expect(outputReadCount).toBe(1);
  const getterResult = await getterOperation.execute({inputs});
  expect(getterResult['clip']).toEqual(Float32Array.of(0.5));
  expect(getterResult['count']).toEqual(Uint32Array.of(1));
  expect(() => createCPULineLengthPerPolygonOperation(legacy)).toThrow(/lengths.*clippedLengths/);
});

it('UGE-07 generates valid planar and spherical clipping kernels with clippedLengths', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice, getParsedKernelSources} = await import(
    '../../utils/gpu-contributor-test-utils'
  );
  const {auditWGSLModule} = await import('../../analysis-core/kernel-audit/kernel-audit-wgsl');
  const {GPULineLengthPerPolygon} = await import(
    '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon'
  );
  for (const spherical of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const sample = new GPULineLengthPerPolygon({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      lineOffsets: createTransientView(graph, 'lines', 'uint32', 2),
      polygons: {
        kind: 'polygons',
        positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
        polygonOffsets: createTransientView(graph, 'parts', 'uint32', 2),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
      },
      pairCapacity: 4,
      ...(spherical
        ? {
            spatialContext: {
              coordinateSpace: 'longitude-latitude' as const,
              metric: 'great-circle' as const,
              units: 'meters' as const
            }
          }
        : {}),
      output: {
        clippedLengths: createTransientView(graph, 'lengths', 'float32', 1),
        status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    });
    for (const {id, source, constants} of getParsedKernelSources(sample.getCommandNodes(graph))
      .configurations)
      expect(auditWGSLModule(source, constants), id).toEqual([]);
  }
});

it('B1 preserves a shared tangent vertex without inventing a positive clipped interval', () => {
  const triangle: CPUSpatialJoinGeometry = {
    kind: 'polygons',
    positions: Float32Array.of(0.2, 1.2, -0.8, 2.2, 2.3, 3.3),
    ringOffsets: Uint32Array.of(0, 3),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
  for (const positions of [Float32Array.of(-5, 1.2, 15, 1.2), Float32Array.of(15, 1.2, -5, 1.2)]) {
    // Every triangle vertex except (.2,1.2) is strictly above the line: intersection is a point.
    const result = computeLineLengthPerPolygonOnCPU(
      {
        lines: {kind: 'lines', positions, lineOffsets: Uint32Array.of(0, 2)},
        polygons: triangle,
        lineWeights: Float32Array.of(3)
      },
      {pairCapacity: 4}
    );
    expect([...result.clippedLengths]).toEqual([0]);
    expect([...result.weightedLengths]).toEqual([0]);
    expect([...result.segmentCounts]).toEqual([0]);
    expect([...result.overflow]).toEqual([0]);
    expect([...result.candidateOverflow]).toEqual([0]);
  }
});

it('B1 retains a genuine interval as small as the false tangent sliver', () => {
  // A 2^-52-wide rectangle has positive intersection length 2^-52, despite being
  // smaller than the spurious tangent length. An epsilon length cutoff is incorrect.
  const width = 2 ** -52;
  const narrow: CPUSpatialJoinGeometry = {
    kind: 'polygons',
    positions: Float32Array.of(0, 0, width, 0, width, 1, 0, 1),
    ringOffsets: Uint32Array.of(0, 4),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  };
  for (const positions of [Float32Array.of(-1, 0.5, 1, 0.5), Float32Array.of(1, 0.5, -1, 0.5)]) {
    const result = computeLineLengthPerPolygonOnCPU(
      {
        lines: {kind: 'lines', positions, lineOffsets: Uint32Array.of(0, 2)},
        polygons: narrow,
        lineWeights: Float32Array.of(3)
      },
      {pairCapacity: 4}
    );
    expect([...result.clippedLengths]).toEqual([width]);
    expect([...result.weightedLengths]).toEqual([3 * width]);
    expect([...result.segmentCounts]).toEqual([1]);
    expect([...result.overflow]).toEqual([0]);
    expect([...result.candidateOverflow]).toEqual([0]);
  }
});
