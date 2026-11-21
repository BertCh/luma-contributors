// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {createUnrepairableCrossingOracle} from './coverage-simplification-output-oracles';

it('measures the crossing still present after the final repair round', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createUnrepairableCrossingOracle(device);
  try {
    const result = await fixture.runGPU();
    expect(result['statistics']).toEqual([1, 1, 0, 0]);
    expect(result['positions']).toEqual([0, 0, 4, 0, 0, 4, 4, 4]);
    expect(result['offsets']).toEqual([0, 4]);
    expect(result['overflow']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('counts one swept span and one restored coverage point with statistics alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The shared arc (0,0)-(0,4)-(3,5)-(0,6)-(0,10) straightens to x=0 at tolerance
  // 3.5. It sweeps the triangular island inside its bulge without crossing any simplified edge.
  // There is ONE owned span (the other polygon holds its reverse); restoring its unique farthest
  // point (3,5) fixes the sweep. Shared copies count as ONE restored coverage point: [1,0,1,0].
  const rings = [
    [-10, 0, 0, 0, 0, 4, 3, 5, 0, 6, 0, 10, -10, 10],
    [0.5, 5.2, 1.5, 5, 0.5, 4.8],
    [0, 0, 10, 0, 10, 10, 0, 10, 0, 6, 3, 5, 0, 4],
    [0.5, 4.8, 1.5, 5, 0.5, 5.2]
  ];
  for (const statistics of [true, false]) {
    const fixture = createOutputOracleFixture(device);
    const {column, output, graph} = fixture;
    graph.add(
      new GPUCoverageSimplification({
        positions: column('input', 'float32x2', new Float32Array(rings.flat())),
        ringOffsets: column('rings', 'uint32', new Uint32Array([0, 7, 10, 17, 20])),
        polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 2, 3, 4])),
        featureOffsets: column('features', 'uint32', new Uint32Array([0, 1, 2, 3])),
        parameters: column('parameters', 'float32', new Float32Array([3.5, 0, 0, 0])),
        maximumIterations: 32,
        topologyIterationCount: 4,
        output: {
          positions: output('positions', 'float32x2', 20),
          ringOffsets: output('offsets', 'uint32', 5),
          status: {overflow: output('overflow', 'uint32', 1)},
          ...(statistics ? {topologyStatistics: output('statistics', 'uint32', 4)} : {})
        }
      })
    );
    try {
      const result = await fixture.runGPU();
      if (statistics) expect(result['statistics']).toEqual([1, 0, 1, 0]);
      expect(result['overflow']).toEqual([0]);
      expect(result['offsets']).toEqual([0, 5, 8, 13, 16]);
      // Retained coordinates are copied from float32 input. IEEE-754 nearest float32 values
      // of decimal 5.2 and 4.8 are 5.199999809265137 and 4.800000190734863 respectively.
      // Independently verified with /usr/bin/python3:
      // import struct; print([struct.unpack('<f', struct.pack('<f', x))[0] for x in (5.2, 4.8)])
      expect(result['positions'].slice(0, 32)).toEqual([
        -10, 0, 0, 0, 3, 5, 0, 10, -10, 10, 0.5, 5.199999809265137, 1.5, 5, 0.5, 4.800000190734863,
        0, 0, 10, 0, 10, 10, 0, 10, 3, 5, 0.5, 4.800000190734863, 1.5, 5, 0.5, 5.199999809265137
      ]);
    } finally {
      fixture.destroy();
    }
  }
});

it('flags a truncated topology candidate list even when every retained pair is legitimate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {column, output, graph} = fixture;
  // Two triangles meet only at (0,0): four cross-ring touching edge pairs all share that
  // coverage point, hence zero offending crossings/restorations, but capacity ONE drops pairs.
  graph.add(
    new GPUCoverageSimplification({
      positions: column(
        'input',
        'float32x2',
        new Float32Array([0, 0, 2, 0, 0, 2, 0, 0, -2, 0, 0, -2])
      ),
      ringOffsets: column('rings', 'uint32', new Uint32Array([0, 3, 6])),
      polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 1, 2])),
      featureOffsets: column('features', 'uint32', new Uint32Array([0, 1, 2])),
      parameters: column('parameters', 'float32', new Float32Array(4)),
      topologyIterationCount: 1,
      topologyPairCapacity: 1,
      output: {
        positions: output('positions', 'float32x2', 6),
        ringOffsets: output('offsets', 'uint32', 3),
        topologyStatistics: output('statistics', 'uint32', 4),
        status: {overflow: output('overflow', 'uint32', 1)}
      }
    })
  );
  try {
    expect((await fixture.runGPU())['statistics']).toEqual([0, 0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});
