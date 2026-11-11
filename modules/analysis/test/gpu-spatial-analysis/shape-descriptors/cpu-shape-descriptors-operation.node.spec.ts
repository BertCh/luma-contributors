// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeShapeDescriptorsOnCPU,
  CPUShapeDescriptorsOperation
} from '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation';
import {
  GPUShapeDescriptorsOperation,
  type ShapeDescriptorsOperationProps
} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {getGPUShapeDescriptorsParameterValues} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';

const PROPS: ShapeDescriptorsOperationProps = {
  id: 'shape-test',
  positions: 'positions',
  ringOffsets: 'ringOffsets',
  polygonOffsets: 'polygonOffsets',
  featureOffsets: 'featureOffsets',
  parameters: 'parameters',
  featureCount: 1,
  output: {areas: 'areas', clockwise: 'clockwise'}
};

it('matches the closed-form descriptors of a translated 4-by-2 rectangle in both windings', () => {
  for (const reverse of [false, true]) {
    const ring = [
      [100, 200],
      [104, 200],
      [104, 202],
      [100, 202]
    ];
    const result = computeShapeDescriptorsOnCPU(
      flattenPolygonFeatures([[reverse ? ring.reverse() : ring]]),
      getGPUShapeDescriptorsParameterValues({sliverThreshold: 0.8})
    );
    expect(result.areas[0]).toBe(8);
    expect(result.perimeters[0]).toBe(12);
    expect(result.polsbyPopper[0]).toBeCloseTo((2 * Math.PI) / 9, 6);
    expect(result.schwartzberg[0]).toBeCloseTo(Math.sqrt((2 * Math.PI) / 9), 6);
    expect(result.elongation[0]).toBe(0.5);
    expect(result.orientation[0]).toBe(0);
    expect(result.convexity[0]).toBe(1);
    expect(result.clockwise[0]).toBe(reverse ? 1 : 0);
    expect(result.sliver[0]).toBe(1);
  }
});

it('weights same-winding holes negatively and reads the packed per-frame threshold', () => {
  const input = flattenPolygonFeatures([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ],
      [
        [1, 1],
        [3, 1],
        [3, 3],
        [1, 3]
      ]
    ]
  ]);
  const result = computeShapeDescriptorsOnCPU(input, getGPUShapeDescriptorsParameterValues());
  expect(result.areas[0]).toBe(12);
  expect(result.perimeters[0]).toBe(24);
  expect(result.convexity[0]).toBe(0.75);
  expect(result.polsbyPopper[0]).toBeCloseTo(Math.PI / 12, 6);
  expect(result.elongation[0]).toBe(0);
  expect(result.orientation[0]).toBe(0);
  expect(result.sliver[0]).toBe(0);
  expect(
    computeShapeDescriptorsOnCPU(
      input,
      getGPUShapeDescriptorsParameterValues({sliverThreshold: 0.3})
    ).sliver[0]
  ).toBe(1);
});

it('uses every multipolygon shell: parallel-axis closed form of two separated squares', () => {
  const result = computeShapeDescriptorsOnCPU(
    {
      positions: new Float32Array([0, 0, 2, 0, 2, 2, 0, 2, 4, 0, 6, 0, 6, 2, 4, 2]),
      ringOffsets: new Uint32Array([0, 4, 8]),
      polygonOffsets: new Uint32Array([0, 1, 2]),
      featureOffsets: new Uint32Array([0, 2])
    },
    getGPUShapeDescriptorsParameterValues()
  );
  expect(result.areas[0]).toBe(8);
  expect(result.perimeters[0]).toBe(16);
  expect(result.convexity[0]).toBeCloseTo(2 / 3, 6);
  expect(result.elongation[0]).toBeCloseTo(1 - Math.sqrt(1 / 13), 6);
  expect(result.orientation[0]).toBe(0);
});

it('handles empty and zero-area features and rejects inconsistent arrays by name', () => {
  const input = flattenPolygonFeatures([
    [],
    [
      [
        [0, 0],
        [2, 0]
      ]
    ]
  ]);
  const parameters = getGPUShapeDescriptorsParameterValues();
  const result = computeShapeDescriptorsOnCPU(input, parameters);
  expect(Array.from(result.areas)).toEqual([0, 0]);
  expect(Array.from(result.perimeters)).toEqual([0, 4]);
  for (const column of [
    'polsbyPopper',
    'schwartzberg',
    'elongation',
    'orientation',
    'convexity'
  ] as const) {
    expect(result[column].every(Number.isNaN)).toBe(true);
  }
  expect(Array.from(result.clockwise)).toEqual([0, 0]);
  expect(Array.from(result.sliver)).toEqual([0, 1]);
  expect(() => computeShapeDescriptorsOnCPU(input, new Float32Array(3))).toThrow(/parameters/);
  expect(() =>
    computeShapeDescriptorsOnCPU({...input, ringOffsets: new Uint32Array(0)}, parameters)
  ).toThrow(/ringOffsets/);
});

it('validates operation shapes and aliases at construction', () => {
  for (const Operation of [GPUShapeDescriptorsOperation, CPUShapeDescriptorsOperation]) {
    expect(() => new Operation(PROPS)).not.toThrow();
    expect(() => new Operation({...PROPS, featureCount: 0})).toThrow(/^shape-test/);
    expect(() => new Operation({...PROPS, output: {areas: 'positions'}})).toThrow(/^shape-test/);
  }
  expect(() => new CPUShapeDescriptorsOperation({...PROPS, output: {}})).toThrow(
    /^shape-test.*output/
  );
});

it('retains a zero-area bow-tie hole numerator in the measured centroid', () => {
  const result = computeShapeDescriptorsOnCPU(
    {
      positions: new Float32Array([0, 0, 8, 0, 8, 8, 0, 8, 1, 2, 7, 6, 1, 6, 7, 2]),
      ringOffsets: new Uint32Array([0, 4, 8]),
      polygonOffsets: new Uint32Array([0, 2]),
      featureOffsets: new Uint32Array([0, 1])
    },
    getGPUShapeDescriptorsParameterValues()
  );
  // The shell y numerator is 1536; the zero-area hole's 96 is subtracted: centroid y = 3.75.
  // Only the shell contributes central moments. Its xx = 1024/3, yy = 1024/3 + 4, xy = 0.
  const perimeter = 44 + 4 * Math.sqrt(13);
  expect(result.areas[0]).toBe(64);
  expect(result.perimeters[0]).toBeCloseTo(perimeter, 5);
  expect(result.polsbyPopper[0]).toBeCloseTo((256 * Math.PI) / perimeter ** 2, 6);
  expect(result.schwartzberg[0]).toBeCloseTo((16 * Math.sqrt(Math.PI)) / perimeter, 6);
  expect(result.orientation[0]).toBeCloseTo(Math.PI / 2, 6);
  expect(result.elongation[0]).toBeCloseTo(1 - Math.sqrt(256 / 259), 7);
  expect(result.convexity[0]).toBe(1);
  expect(result.clockwise[0]).toBe(0);
  expect(result.sliver[0]).toBe(0);
});

it('uses the feature row interval for hull membership with descending ring offsets', () => {
  const result = computeShapeDescriptorsOnCPU(
    {
      positions: new Float32Array([0, 0, 4, 0, 4, 4, 0, 4]),
      ringOffsets: new Uint32Array([0, 4, 0]),
      polygonOffsets: new Uint32Array([0, 2]),
      featureOffsets: new Uint32Array([0, 1])
    },
    getGPUShapeDescriptorsParameterValues()
  );
  // The independent ring walk still measures the square, but the hull interval is [0, 0).
  expect(result.areas[0]).toBe(16);
  expect(result.perimeters[0]).toBe(16);
  expect(result.polsbyPopper[0]).toBeCloseTo(Math.PI / 4, 6);
  expect(result.schwartzberg[0]).toBeCloseTo(Math.sqrt(Math.PI) / 2, 6);
  expect(result.orientation[0]).toBe(0);
  expect(result.elongation[0]).toBe(0);
  expect(Number.isNaN(result.convexity[0])).toBe(true);
  expect(result.clockwise[0]).toBe(0);
  expect(result.sliver[0]).toBe(0);
});
