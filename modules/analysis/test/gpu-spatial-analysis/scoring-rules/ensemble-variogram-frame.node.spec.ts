// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, ScalarData} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUEnsembleScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores';
import {computeEnsembleScoresOnCPU} from '../../../src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function evaluatePair(
  ensembleValues: Float32Array,
  observedValues: Float32Array,
  weight = 1,
  variogramOrder = 0.5,
  flushSmallComparisons = false
): number {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const ensemble = createTransientView(graph, 'ensemble', 'float32', ensembleValues.length);
    const observations = createTransientView(graph, 'observations', 'float32', 2);
    const pairs = createTransientView(graph, 'pairs', 'uint32x2', 1);
    const weights = createTransientView(graph, 'weights', 'float32', 1);
    const output = createTransientView(graph, 'variogram', 'float32', 1);
    recordWGSLKernelNodes(graph);
    const nodes = new GPUEnsembleScores({
      ensemble,
      observations,
      variogramPairs: pairs,
      variogramWeights: weights,
      variogramOrder,
      memberCount: ensembleValues.length / 2,
      dimension: 2,
      output: {variogramScore: output}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(ensemble, ensembleValues);
    memory.write(observations, observedValues);
    memory.write(pairs, Uint32Array.of(0, 1));
    memory.write(weights, Float32Array.of(weight));
    // The terms view is the sole write of the pair kernel. Read it without running a reduction.
    const term = nodes.find(node => node.id.endsWith('-variogram-terms'))!;
    const record = getWGSLKernelNodeRecord(graph, term.id)!;
    const original = WgslExec.prototype._evalBinaryOp;
    if (flushSmallComparisons) {
      WgslExec.prototype._evalBinaryOp = function (...args: Parameters<typeof original>) {
        const [expression, context] = args;
        const result = original.apply(this, args);
        if (
          (expression.operator === '<' || expression.operator === '==') &&
          result instanceof ScalarData
        ) {
          const left = this.evalExpression(expression.left, context);
          const right = this.evalExpression(expression.right, context);
          if (
            left instanceof ScalarData &&
            right instanceof ScalarData &&
            (left.typeInfo.name === 'f32' || right.typeInfo.name === 'f32')
          ) {
            const first = Math.abs(left.value) < 2 ** -126 ? 0 : left.value;
            const second = Math.abs(right.value) < 2 ** -126 ? 0 : right.value;
            result.value = Number(expression.operator === '<' ? first < second : first === second);
          }
        }
        return result;
      };
    }
    try {
      executeCommandNodesOnCPU([term], memory);
    } finally {
      WgslExec.prototype._evalBinaryOp = original;
    }
    return memory.readFloat32(record.bindings.find(binding => binding.name === 'terms')!.view)[0];
  } finally {
    device.destroy();
  }
}

it('WGS-09 keeps an extreme perfect forecast finite before fractional powers', () => {
  // Identical fields have zero variogram discrepancy, even though 2^128 overflows f32.
  const values = Float32Array.of(-(2 ** 127), 2 ** 127);
  const cpu = computeEnsembleScoresOnCPU(
    {ensemble: values, observations: values, variogramPairs: Uint32Array.of(0, 1)},
    {memberCount: 1, dimension: 2, scores: ['variogramScore']}
  );
  expect(cpu.variogramScore[0]).toBe(0);
  for (const ensemble of [values, Float32Array.of(...values, ...values, ...values)]) {
    const value = evaluatePair(ensemble, values);
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBe(0);
  }
});

it('WGS-09 restores a representable weighted square after the common exponent frame', () => {
  // sqrt(2^128)^2 * 1/4 = 2^126, despite overflowing the unpowered endpoint difference.
  const value = evaluatePair(Float32Array.of(-(2 ** 127), 2 ** 127), Float32Array.of(0, 0), 0.25);
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value / 2 ** 126 - 1)).toBeLessThan(3e-6);
});

for (const weight of [2, 0, -1, Infinity, -Infinity, NaN]) {
  it(`NUM-07 checks resident pair weight ${weight} in the emitted pair kernel`, () => {
    const value = evaluatePair(Float32Array.of(0, 1), Float32Array.of(0, 0), weight);
    if (Number.isFinite(weight) && weight >= 0) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBe(weight);
    } else expect(value).toBeNaN();
  });
}

it('WGS-09 restores a fractional exponent near the finite float32 ceiling', () => {
  // Python: p=struct.unpack('f',struct.pack('f',.497))[0]; f32(2**(256*p)).
  // The pair gap is 2^128, so its squared p-power is 2^(256*p), still finite.
  const value = evaluatePair(
    Float32Array.of(-(2 ** 127), 2 ** 127),
    Float32Array.of(0, 0),
    1,
    0.497
  );
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value / 1.998246724570999e38 - 1)).toBeLessThan(3e-5);
});

it('WGS-09 retains a representable higher-order score at a large common offset', () => {
  // The f32 pair gap is 64 = 2^6. At order 10, its squared power is 2^120.
  const value = evaluatePair(Float32Array.of(2 ** 29, 2 ** 29 + 64), Float32Array.of(0, 0), 1, 10);
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value / 1.329227995784916e36 - 1)).toBeLessThan(3e-5);
});

it('WGS-09 preserves an ordinary multi-member fractional variogram', () => {
  // sqrt gaps are 1 and 2, their mean is 1.5, observed gap is zero: 2 * 1.5^2 = 4.5.
  const value = evaluatePair(Float32Array.of(0, 1, 0, 4), Float32Array.of(0, 0), 2);
  expect(Number.isFinite(value)).toBe(true);
  expect(value).toBeCloseTo(4.5, 6);
});

it('B1 preserves a unit order-one discrepancy between exactly representable large gaps', () => {
  // (10,000,001 - mean(10,000,000, 10,000,000))^2 = 1 exactly.
  const value = evaluatePair(
    Float32Array.of(0, 10000000, 0, 10000000),
    Float32Array.of(0, 10000001),
    1,
    1
  );
  expect(Number.isFinite(value)).toBe(true);
  expect(value).toBe(1);
});

it('B2 divides a constant ensemble discrepancy once after summing', () => {
  const ensemble = new Float32Array(20000);
  for (let member = 0; member < 10000; member++) ensemble[member * 2 + 1] = 1;
  // Every powered forecast gap is 1 and the observation gap is zero, so the score is 1.
  const value = evaluatePair(ensemble, Float32Array.of(0, 0));
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value - 1)).toBeLessThan(3e-5);
}, 60000);

for (const exponent of [-149, -126]) {
  it(`S2 retains the recoverable fractional power of gap 2^${exponent}`, () => {
    // (|2^exponent|^(1/4))^2 = 2^(exponent/2), normal in float32.
    const value = evaluatePair(Float32Array.of(0, 2 ** exponent), Float32Array.of(0, 0), 1, 0.25);
    const expected = exponent === -149 ? 3.743392066509216e-23 : 1.0842021724855044e-19;
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value / expected - 1)).toBeLessThan(3e-5);
  });
}

it('S2 recovers a normal gap before a flush-to-zero comparison can erase its half', () => {
  const value = evaluatePair(Float32Array.of(0, 2 ** -126), Float32Array.of(0, 0), 1, 0.25, true);
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value / 1.0842021724855044e-19 - 1)).toBeLessThan(3e-5);
});

it('S3 rejects a negative subnormal weight by bits while retaining signed zero', () => {
  const invalid = evaluatePair(
    Float32Array.of(0, 1),
    Float32Array.of(0, 0),
    -(2 ** -149),
    0.5,
    true
  );
  expect(invalid).toBeNaN();
  for (const weight of [-0, 0]) {
    expect(evaluatePair(Float32Array.of(0, 1), Float32Array.of(0, 0), weight, 0.5, true)).toBe(0);
  }
});

it('B1 preserves the default-order discrepancy between close large gaps', () => {
  // Python f32((1/(sqrt(10000001)+sqrt(10000000)))**2), using the rationalized difference.
  const value = evaluatePair(
    Float32Array.of(0, 10000000, 0, 10000000),
    Float32Array.of(0, 10000001)
  );
  expect(Number.isFinite(value)).toBe(true);
  expect(Math.abs(value / 2.4999998515795596e-8 - 1)).toBeLessThan(3e-5);
});
