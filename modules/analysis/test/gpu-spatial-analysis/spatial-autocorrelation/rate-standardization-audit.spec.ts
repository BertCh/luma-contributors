// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAssuncaoReisStandardization} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  RATE_AUDIT_REFERENCE,
  RATE_AUDIT_MASK,
  RATE_AUDIT_POPULATIONS
} from './rate-standardization-audit-fixture';

function compare(actual: readonly number[], expected: readonly number[], label: string) {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) expect(actual[index], `${label}[${index}]`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `${label}[${index}]`).toBe(true);
      // f32 b, s2 and a round at each pass. Fixtures avoid v=0 cancellation boundaries;
      // this bound admits a few ulps through sqrt/division and rejects globally clamping a.
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(
        2e-6 + 5e-6 * Math.abs(value)
      );
    }
  }
}

it('Assuncao-Reis publishes every output against exact rational expectations on reused buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const events = rig.input(new Float32Array(10), 'float32', 10);
  const populations = rig.input(new Float32Array(10), 'float32', 10);
  const mask = rig.input(Uint32Array.from(RATE_AUDIT_MASK), 'uint32', 10);
  const rates = rig.output('float32', 10);
  const statistics = rig.output('float32', 5);
  const invalidCount = rig.output('uint32', 1);
  rig.graph.add(
    new GPUAssuncaoReisStandardization({
      events,
      populations,
      mask,
      output: {
        standardizedRates: rates.view,
        statistics: statistics.view,
        status: {invalidCount: invalidCount.view}
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const reference of RATE_AUDIT_REFERENCE) {
      for (const reversed of [false, true]) {
        const eventValues = reference.events.map(value => reference.scale * value);
        const populationValues = RATE_AUDIT_POPULATIONS.map(value => reference.scale * value);
        rig
          .bufferOf(events)
          .write(Float32Array.from(reversed ? eventValues.slice().reverse() : eventValues));
        rig
          .bufferOf(populations)
          .write(
            Float32Array.from(reversed ? populationValues.slice().reverse() : populationValues)
          );
        rig
          .bufferOf(mask)
          .write(Uint32Array.from(reversed ? RATE_AUDIT_MASK.slice().reverse() : RATE_AUDIT_MASK));
        submitGraph(device, compiled, undefined);
        compare(
          await readFloat32(rates.buffer, 10),
          reversed ? reference.rates.slice().reverse() : reference.rates,
          reference.name
        );
        compare(
          await readFloat32(statistics.buffer, 5),
          reference.statistics,
          `${reference.name} statistics`
        );
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([3]);
      }
    }
    rig.bufferOf(mask).write(new Uint32Array(10));
    submitGraph(device, compiled, undefined);
    compare(await readFloat32(rates.buffer, 10), new Array(10).fill(NaN), 'empty rates');
    compare(await readFloat32(statistics.buffer, 5), [0, NaN, NaN, NaN, NaN], 'empty statistics');
    expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
