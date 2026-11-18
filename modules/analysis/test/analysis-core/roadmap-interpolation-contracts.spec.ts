// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULocalRBF} from '../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {createPaddedQueryFixture} from '../gpu-spatial-analysis/linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it('R2 skips the nonfinite nearest sample before selecting the usable RBF neighbor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0));
  const values = input('observations', 'float32', Float32Array.of(NaN, 7));
  const queries = input('queries', 'float32x2', Float32Array.of(0, 0));
  const scanValues = output('scan-values', 'float32x4', 1);
  const scanStatus = output('scan-status', 'uint32', 1);
  const candidateValues = output('candidate-values', 'float32x4', 1);
  const candidateStatus = output('candidate-status', 'uint32', 1);
  const settings = {
    positions,
    values,
    queries,
    neighborCount: 1,
    kernel: 'linear' as const,
    degree: 0 as const
  };
  graph.add(
    new GPULocalRBF({
      ...settings,
      id: 'usable-scan',
      output: {values: scanValues, status: scanStatus}
    })
  );
  // A raw-position top-1 index returns row zero and loses the farther usable observation.
  graph.add(
    new GPULocalRBF({
      ...settings,
      id: 'raw-top-one',
      neighborhoods: input('raw-nearest-row', 'uint32', Uint32Array.of(0)),
      neighborhoodCapacity: 1,
      output: {values: candidateValues, status: candidateStatus}
    })
  );
  const compiled = graph.compile();
  try {
    const actual = await fixture.submitTwice(compiled, async () => ({
      scanValues: await fixture.readRows([scanValues]),
      scanStatus: await fixture.readRows([scanStatus]),
      candidateStatus: await fixture.readRows([candidateStatus])
    }));
    expect(actual.scanStatus).toEqual([0]);
    expect(Number.isFinite(actual.scanValues[0])).toBe(true);
    // With degree zero and one usable observation, the polynomial coefficient is exactly seven.
    expect(actual.scanValues[0]).toBeCloseTo(7, 5);
    expect(actual.scanValues[2]).toBe(1);
    expect(actual.candidateStatus).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
