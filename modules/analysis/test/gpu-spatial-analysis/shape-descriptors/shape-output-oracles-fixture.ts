// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {ShapeDescriptorsOperationProps} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export const SHAPE_INPUTS = {
  positions: {
    format: 'float32x2' as const,
    values: new Float32Array([0, 0, 4, 0, 4, 2, 0, 2, 10, 0, 10, 2, 12, 2, 12, 0])
  },
  ringOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 4, 8])},
  polygonOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 1, 2])},
  featureOffsets: {format: 'uint32' as const, values: new Uint32Array([0, 1, 2])},
  parameters: {format: 'float32' as const, values: new Float32Array([0.75, 0, 0, 0])}
};
// Rectangle A=8, P=12, PP=4*pi*A/P²=2*pi/9; square A=4,P=8,PP=pi/4.
// Schwartzberg=sqrt(PP); only the first PP is below .75. Second ring winds clockwise.
const EXPECTED: Record<string, number[]> = {
  areas: [8, 4],
  perimeters: [12, 8],
  polsbyPopper: [0.6981317007977318, 0.7853981633974483],
  schwartzberg: [0.8355427582103335, 0.8862269254527579],
  clockwise: [0, 1],
  sliver: [1, 0],
  elongation: [0.5, 0],
  orientation: [0, 0],
  convexity: [1, 1]
};
const all = Object.fromEntries(Object.keys(EXPECTED).map(name => [name, name]));
export const SHAPE_OUTPUT_CASES = [
  all,
  ...['areas', 'perimeters', 'polsbyPopper', 'schwartzberg', 'clockwise', 'sliver'].flatMap(
    name => [
      {[name]: name},
      Object.fromEntries(Object.entries(all).filter(([column]) => column !== name))
    ]
  )
];
export function getShapeProps(
  output: ShapeDescriptorsOperationProps['output']
): ShapeDescriptorsOperationProps {
  return {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    parameters: 'parameters',
    featureCount: 2,
    output
  };
}
export function expectShapeOracle(
  result: Record<string, OperationGraphRows>,
  requestedOutputs: readonly string[]
): void {
  expect(Object.keys(result).sort()).toEqual([...requestedOutputs].sort());
  for (const [name, actual] of Object.entries(result)) {
    expect(actual.length).toBe(2);
    for (const [index, expected] of EXPECTED[name].entries()) {
      expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
      // Areas, perimeters and codes are exact for these integer-coordinate rectangles.
      // Compactness and elongation use float32 division/sqrt (elongation also subtracts
      // rounded eigenvalues). Allow 2e-6 absolute for that arithmetic; Metal returned
      // 0.4999999701976776 for the closed-form 0.5 elongation, one float32 step below it.
      if (name === 'polsbyPopper' || name === 'schwartzberg' || name === 'elongation')
        expect(Math.abs(actual[index] - expected)).toBeLessThanOrEqual(2e-6);
      else expect(actual[index]).toBe(expected);
    }
  }
}
