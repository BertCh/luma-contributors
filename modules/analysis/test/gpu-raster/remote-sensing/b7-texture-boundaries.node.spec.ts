// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPURasterTexture} from '../../../src/raster-analysis/remote-sensing/gpu-raster-texture';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

// /usr/bin/python3: for ordinal in range(1, levels), use Fraction on the float32 endpoints
// to compute ((levels - ordinal) * low + ordinal * high) / levels exactly. Start with the
// nearest float32 (struct.pack('>f', float(exact))) and step one float32 at a time until
// Fraction(boundary) >= exact and Fraction(previous_float32(boundary)) < exact.
const cases: {
  label: string;
  levels: number;
  range: readonly [number, number];
  boundaries: readonly number[];
}[] = [
  {
    label: 'two levels with a midpoint just above 0.5',
    levels: 2,
    range: [2 ** -56, 1],
    boundaries: [0.5000000596046448]
  },
  {
    label: 'three levels from a subnormal endpoint to a wide upper endpoint',
    levels: 3,
    range: [2 ** -149, 3 * 2 ** 100],
    boundaries: [1.2676507513439569e30, 2.5353015026879137e30]
  },
  {
    label: 'three levels with negative fractional boundaries',
    levels: 3,
    range: [-1, -(2 ** -56)],
    boundaries: [-0.6666666269302368, -0.3333333134651184]
  },
  {
    label: 'three levels across zero with subnormal endpoints',
    levels: 3,
    range: [-(2 ** -149), 2 ** -149],
    boundaries: [0, 1.401298464324817e-45]
  },
  {
    label: 'eleven levels with exact integer boundaries',
    levels: 11,
    range: [0, 77],
    boundaries: [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
  },
  {
    label: 'eleven levels with boundaries just above integers',
    levels: 11,
    range: [2 ** -56, 77],
    boundaries: [
      7.000000476837158, 14.000000953674316, 21.000001907348633, 28.000001907348633,
      35.000003814697266, 42.000003814697266, 49.000003814697266, 56.000003814697266,
      63.000003814697266, 70.00000762939453
    ]
  }
];

it.each(cases)('B7-TEX-01 rounds exact texture boundaries upward for $label', fixture => {
  const {device, graph, view} = createKernelAuditContext('texture-exact-boundaries');
  try {
    const texture = new GPURasterTexture({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      values: view('float32', 2),
      levels: fixture.levels,
      range: fixture.range,
      radius: 1,
      offset: [1, 0],
      features: view('float32', 14),
      pairCount: view('uint32', 2)
    });
    const source = getGPUCommandNodeKernelSources(texture.getCommandNodes(graph))[0].source;
    // Both signs of zero give the same comparison threshold.
    const boundaries = Array.from(
      source.matchAll(/value >= ([\d.eE+-]+)/g),
      match => Number(match[1]) + 0
    );
    expect(boundaries).toEqual(fixture.boundaries);
  } finally {
    device.destroy();
  }
});
