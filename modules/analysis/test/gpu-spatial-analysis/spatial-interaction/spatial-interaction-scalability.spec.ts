// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  computeSpatialInteractionOnCPU,
  type CPUSpatialInteractionInput
} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';
import {
  GPUSpatialInteractionOperation,
  type SpatialInteractionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {AnalysisRig} from '../catchment-accessibility/rig';

async function runInteraction(
  input: CPUSpatialInteractionInput,
  options: Pick<
    SpatialInteractionOperationProps,
    'method' | 'originCount' | 'destinationCount' | 'coefficients' | 'maximumIterations'
  >
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const output = {
    predictions: 'predictions',
    summary: 'summary',
    status: 'status',
    candidates: 'candidates',
    originResiduals: 'originResiduals',
    destinationResiduals: 'destinationResiduals'
  };
  const props: SpatialInteractionOperationProps = {
    ...options,
    constraint: 'doubly',
    tolerance: 1e-5,
    origins: 'origins',
    destinations: 'destinations',
    flows: 'flows',
    covariate: 'covariate',
    pairState: input.pairState ? 'pairState' : undefined,
    pairCount: input.flows.length,
    output
  };
  const inputs = {
    origins: {format: 'uint32' as const, values: input.origins},
    destinations: {format: 'uint32' as const, values: input.destinations},
    flows: {format: 'float32' as const, values: input.flows},
    covariate: {format: 'float32' as const, values: input.covariate},
    ...(input.pairState ? {pairState: {format: 'uint32' as const, values: input.pairState}} : {})
  };
  return runOperationGraph(device, {
    inputs,
    operations: [new GPUSpatialInteractionOperation(props)],
    outputs: Object.values(output)
  });
}

it('sparse and dense IPF match a float64 CPU oracle with structural, missing, duplicate and zero-margin cells', async () => {
  const input: CPUSpatialInteractionInput = {
    origins: Uint32Array.of(0, 0, 1, 1, 0, 99, 0),
    destinations: Uint32Array.of(0, 1, 0, 1, 2, 99, 0),
    flows: Float32Array.of(10, 20, 30, 40, 0, NaN, 5),
    covariate: Float32Array.of(0, 1, 2, 0, NaN, NaN, 1),
    pairState: Uint32Array.of(1, 1, 1, 1, 0, 2, 1)
  };
  const options = {
    originCount: 3,
    destinationCount: 3,
    coefficients: [-1, 0, 1],
    maximumIterations: 256
  };
  const expected = computeSpatialInteractionOnCPU(input, {
    ...options,
    constraint: 'doubly',
    tolerance: 1e-5
  });
  for (const method of ['dense', 'sparse'] as const) {
    const actual = await runInteraction(input, {...options, method});
    expect(actual['status'][0]).toBe(0);
    expect(actual['summary'][0]).toBe(expected.summary[0]);
    for (let pair = 0; pair < input.flows.length; pair++) {
      const value = expected.predictions[pair];
      if (Number.isNaN(value)) expect(actual['predictions'][pair]).toBeNaN();
      else
        expect(Math.abs(actual['predictions'][pair] - value)).toBeLessThanOrEqual(
          5e-4 * Math.max(1, value)
        );
    }
    expect(actual['predictions'][4]).toBe(0);
    expect(actual['predictions'][5]).toBeNaN();
    expect(actual['originResiduals'][2]).toBe(0);
    expect(actual['destinationResiduals'][2]).toBe(0);
  }
});

for (const method of ['dense', 'sparse'] as const) {
  it(`${method} avoids an overflowing IPF ratio when the finite solution is known`, async () => {
    const input = {
      origins: Uint32Array.of(0, 1),
      destinations: Uint32Array.of(0, 1),
      flows: Float32Array.of(1e30, 1),
      covariate: Float32Array.of(-80, 0)
    };
    const actual = await runInteraction(input, {
      method,
      originCount: 2,
      destinationCount: 2,
      coefficients: [1],
      maximumIterations: 4
    });
    expect(actual['status'][0]).toBe(0);
    expect(actual['predictions'][0]).toBe(input.flows[0]);
    expect(actual['predictions'][1]).toBe(1);
  });
  for (const scene of ['no-mass', 'invalid', 'span', 'overflow-mass'] as const) {
    it(`${method} reports ${scene} without selecting an invalid candidate`, async () => {
      const input = {
        origins: Uint32Array.of(0, 1),
        destinations: Uint32Array.of(0, 1),
        flows:
          scene === 'overflow-mass'
            ? Float32Array.of(3e38, 3e38)
            : Float32Array.of(scene === 'no-mass' ? 0 : scene === 'invalid' ? -1 : 1, 0),
        covariate: Float32Array.of(scene === 'span' ? 81 : 0, 0)
      };
      const actual = await runInteraction(input, {
        method,
        originCount: 2,
        destinationCount: 2,
        coefficients: [1],
        maximumIterations: 2
      });
      expect(actual['status'][0]).toBe(scene === 'no-mass' ? 1 : scene === 'span' ? 3 : 2);
      expect(actual['candidates'][3]).toBe(scene === 'no-mass' ? 2 : scene === 'span' ? 4 : 3);
      expect([...actual['predictions']].every(Number.isNaN)).toBe(true);
    });
  }
}

for (const method of ['dense', 'sparse'] as const) {
  it(`${method} writes disjoint aligned writable outputs imported once from one physical buffer`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const storage = rig.output('uint32', 129);
      const predictions = rig.graph.createDataView(storage.buffer, {format: 'float32', length: 4});
      const summary = rig.graph.createDataView(storage.buffer, {
        format: 'float32',
        length: 4,
        byteOffset: 256
      });
      const status = rig.graph.createDataView(storage.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 512
      });
      rig.run(
        new GPUSpatialInteraction({
          method,
          constraint: 'doubly',
          originCount: 2,
          destinationCount: 2,
          origins: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32'),
          destinations: rig.input(Uint32Array.of(0, 1, 0, 1), 'uint32'),
          flows: rig.input(Float32Array.of(10, 20, 30, 40), 'float32'),
          covariate: rig.input(new Float32Array(4), 'float32'),
          coefficients: [0],
          maximumIterations: 4,
          tolerance: 1e-5,
          output: {predictions, summary, status, candidates: rig.output('float32', 4)}
        })
      );
      const words = Uint32Array.from(await rig.readUint(storage));
      const floats = new Float32Array(words.buffer);
      expect(words[128]).toBe(0);
      // Independent margins: [30,70] x [40,60] / 100.
      for (const [index, expected] of [12, 18, 28, 42].entries())
        expect(floats[index]).toBeCloseTo(expected, 4);
      expect(floats[64]).toBe(0);
    } finally {
      rig.destroy();
    }
  });
}

it('sparse IPF handles one million explicit cells and 1,024 origins/destinations with closed-form means', async () => {
  const groups = 1024;
  const pairs = groups * groups;
  const input = {
    origins: Uint32Array.from({length: pairs}, (_, index) => Math.floor(index / groups)),
    destinations: Uint32Array.from({length: pairs}, (_, index) => index % groups),
    flows: new Float32Array(pairs).fill(2),
    covariate: new Float32Array(pairs)
  };
  const actual = await runInteraction(input, {
    originCount: groups,
    destinationCount: groups,
    coefficients: [0],
    maximumIterations: 2
  });
  expect(actual['status'][0]).toBe(0);
  expect(actual['summary'][1]).toBe(0);
  expect(actual['summary'][2]).toBe(0);
  expect(actual['summary'][3]).toBe(1);
  expect([...actual['predictions']].every(value => value === 2)).toBe(true);
  expect([...actual['originResiduals']].every(value => value === 0)).toBe(true);
  expect([...actual['destinationResiduals']].every(value => value === 0)).toBe(true);
});

for (const method of ['dense', 'sparse'] as const) {
  it(`${method} reports the exhausted balancing budget without selecting the unfinished fit`, async () => {
    const input = {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(10, 20, 30, 40),
      covariate: Float32Array.of(0, 1, 2, 0)
    };
    const actual = await runInteraction(input, {
      method,
      originCount: 2,
      destinationCount: 2,
      coefficients: [-1],
      maximumIterations: 1
    });
    expect(actual['status'][0]).toBe(3);
    expect(actual['candidates'][2]).toBe(1);
    expect(actual['candidates'][3]).toBe(1);
    expect(actual['candidates'][1]).toBeGreaterThan(1e-5);
    expect([...actual['predictions']].every(Number.isNaN)).toBe(true);
  });
}

it('keeps a four-million-cell hub marginal accurate against the constant-flow oracle', async () => {
  const pairs = 2 ** 22;
  const flow = Math.fround(0.1);
  const input = {
    origins: new Uint32Array(pairs),
    destinations: new Uint32Array(pairs),
    flows: new Float32Array(pairs).fill(flow),
    covariate: new Float32Array(pairs)
  };
  const actual = await runInteraction(input, {
    originCount: 1,
    destinationCount: 1,
    coefficients: [0],
    maximumIterations: 2
  });
  expect(actual['status'][0]).toBe(0);
  // Every independent duplicate cell has the same exponent and mean. A naive strided f32
  // sum of the observed marginal drifts by about 1.5e-4 relative at this size.
  expect([...actual['predictions']].every(value => Math.abs(value - flow) <= 1e-6 * flow)).toBe(
    true
  );
});

for (const method of ['dense', 'sparse'] as const) {
  it(`${method} preserves positive representable predictions across fifty orders of marginal mass`, async () => {
    const input = {
      origins: Uint32Array.of(0, 0, 1, 1),
      destinations: Uint32Array.of(0, 1, 0, 1),
      flows: Float32Array.of(1e-20, 1e-20, 1e30, 1e30),
      covariate: new Float32Array(4)
    };
    const actual = await runInteraction(input, {
      method,
      originCount: 2,
      destinationCount: 2,
      coefficients: [0],
      maximumIterations: 4
    });
    expect(actual['status'][0]).toBe(0);
    for (let pair = 0; pair < 4; pair++) {
      expect(actual['predictions'][pair]).toBeGreaterThan(0);
      expect(Math.abs(actual['predictions'][pair] / input.flows[pair] - 1)).toBeLessThan(5e-6);
    }
  });
}

it('preserves an early sparse winner through invalid, span-rejected and exhausted candidates', async () => {
  const input = {
    origins: Uint32Array.of(0, 0, 1, 1),
    destinations: Uint32Array.of(0, 1, 0, 1),
    flows: Float32Array.of(10, 20, 30, 40),
    covariate: Float32Array.of(0, 1, 2, 0)
  };
  // Zero converges in one round to the independent table. Huge finite coefficient overflows
  // one exponent; 41 exceeds the supported span; -1 needs more than this one-round budget.
  const actual = await runInteraction(input, {
    method: 'sparse',
    originCount: 2,
    destinationCount: 2,
    coefficients: [0, Math.fround(3e38), 41, -1],
    maximumIterations: 1
  });
  expect(actual['status'][0]).toBe(0);
  expect(actual['summary'][0]).toBe(0);
  expect(actual['summary'][2]).toBeLessThanOrEqual(1e-5);
  expect(actual['summary'][3]).toBe(1);
  for (const [pair, expected] of [12, 18, 28, 42].entries())
    expect(actual['predictions'][pair]).toBeCloseTo(expected, 4);
  expect([3, 7, 11, 15].map(index => actual['candidates'][index])).toEqual([0, 3, 4, 1]);
  const expectedDeviance =
    2 *
    [12, 18, 28, 42].reduce(
      (sum, prediction, pair) =>
        sum +
        input.flows[pair] * Math.log(input.flows[pair] / prediction) -
        input.flows[pair] +
        prediction,
      0
    );
  expect(actual['summary'][1]).toBeCloseTo(expectedDeviance, 3);
  expect([...actual['originResiduals']].every(value => Math.abs(value) < 1e-4)).toBe(true);
  expect([...actual['destinationResiduals']].every(value => Math.abs(value) < 1e-4)).toBe(true);
});
