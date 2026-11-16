// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {ScalarData, WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expect, it, vi} from 'vitest';
import {GPUEnsembleScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (length: number) =>
    createTransientView(graph, `scores-${serial++}`, 'float32', length);
  const props = {
    id: 'score-validation',
    memberCount: 2,
    dimension: 2,
    ensemble: view(4),
    observations: view(2),
    output: {crps: view(2), energyScore: view(1)}
  };
  return {device, graph, view, props};
}

it('checks aligned inputs before declaring score or reduction scratch', () => {
  const fixture = createFixture();
  try {
    const storage = fixture.view(5);
    const ensemble = fixture.graph.createDataView(storage.buffer, {
      format: 'float32',
      byteOffset: 4,
      length: 4
    });
    const contributor = new GPUEnsembleScores({...fixture.props, ensemble});
    Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {value: 16});
    const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
    const createView = vi.spyOn(fixture.graph, 'createDataView');
    expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(/ensemble.*20 bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    fixture.device.destroy();
  }
});

it('checks writable logical ranges without rejecting valid separate-dispatch packing', () => {
  const fixture = createFixture();
  try {
    const shared = fixture.view(3);
    const crps = fixture.graph.createDataView(shared.buffer, {format: 'float32', length: 2});
    const energyScore = fixture.graph.createDataView(shared.buffer, {
      format: 'float32',
      byteOffset: 8,
      length: 1
    });
    expect(
      () => new GPUEnsembleScores({...fixture.props, output: {crps, energyScore}})
    ).not.toThrow();
    expect(() =>
      new GPUEnsembleScores({...fixture.props, output: {crps, energyScore}}).getCommandNodes(
        fixture.graph
      )
    ).not.toThrow();
    const overlapping = fixture.graph.createDataView(shared.buffer, {
      format: 'float32',
      byteOffset: 4,
      length: 1
    });
    expect(
      () => new GPUEnsembleScores({...fixture.props, output: {crps, energyScore: overlapping}})
    ).toThrow(/logical writable.*overlap/);
  } finally {
    fixture.device.destroy();
  }
});

it('validates float32 scoring exponents and audits identifiers in the stable norm kernel', () => {
  const fixture = createFixture();
  try {
    for (const variogramOrder of [1e-100, 1e100, Infinity, NaN])
      expect(() => new GPUEnsembleScores({...fixture.props, variogramOrder})).toThrow(
        /variogramOrder/
      );
    const kernels = getGPUCommandNodeKernelSources(
      new GPUEnsembleScores({
        ...fixture.props,
        variogramPairs: createTransientView(fixture.graph, 'audit-pairs', 'uint32x2', 1),
        variogramWeights: fixture.view(1),
        output: {...fixture.props.output, variogramScore: fixture.view(1)}
      }).getCommandNodes(fixture.graph)
    );
    for (const kernel of kernels) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const words = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    }
    const energy = kernels.find(kernel => kernel.id.includes('energy-terms'))!;
    expect(energy.source).toContain('accumulateEnsembleMagnitude');
    expect(energy.source).not.toContain('observedSquares += difference * difference');
    expect(energy.source).not.toMatch(/(?:state\.x\s*\/\s*magnitude|magnitude\s*\/\s*state\.x)/);
    expect(energy.source).toContain('getScaledQuotient(magnitude, state.x, 1.0, 0)');
    expect(auditWGSLModule(energy.source, energy.constants)).toEqual([]);
    const identifiers = stripWGSLComments(energy.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
  } finally {
    fixture.device.destroy();
  }
});

it('the emitted energy terms retain the closed-form score under reciprocal flushing', () => {
  const fixture = createFixture();
  try {
    recordWGSLKernelNodes(fixture.graph);
    let caseIndex = 0;
    for (const ensemble of [
      Float32Array.of(3e20, 4e20, 0, 0),
      Float32Array.of(3e38, 3e38, 0, 0),
      Float32Array.of(3e38, 1.5e38, 0, 0),
      Float32Array.of(1.5e38, 3e38, 0, 0),
      Float32Array.of(3e38, -3e38),
      Float32Array.of(3e38, -2e38, 1e38, 0, 0, 0),
      Float32Array.of(NaN, 1, 0, 0)
    ]) {
      const dimension = ensemble.length / 2;
      const nodes = new GPUEnsembleScores({
        ...fixture.props,
        id: `score-profile-${caseIndex++}`,
        dimension,
        ensemble: fixture.view(ensemble.length),
        observations: fixture.view(dimension),
        output: {energyScore: fixture.props.output.energyScore}
      }).getCommandNodes(fixture.graph);
      const energy = nodes.find(node => node.id.includes('energy-terms'))!;
      const record = getWGSLKernelNodeRecord(fixture.graph, energy.id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const values = new Float32Array(getViewBindingRange(binding.view).size / 4);
          if (binding.name === 'ensemble') values.set(ensemble);
          if (binding.name === 'observations') values.set(ensemble.slice(dimension));
          return [index, values];
        })
      );
      createReciprocalFlushExecutor(record.source).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
      const terms = bindings[record.bindings.findIndex(binding => binding.name === 'terms')];
      const differences = Array.from(
        {length: dimension},
        (_, site) => ensemble[site] - ensemble[dimension + site]
      );
      const expected = Math.hypot(...differences) / 4;
      if (Number.isNaN(expected)) expect(terms[0] + terms[1]).toBeNaN();
      else expect(Math.abs((terms[0] + terms[1]) / expected - 1)).toBeLessThan(2e-5);
    }
  } finally {
    fixture.device.destroy();
  }
});

/** Model a permitted GPU division lowering: f32 multiplication by a reciprocal flushed to zero.
 * The ordinary interpreter uses IEEE division and missed the real Metal failure. */
function createReciprocalFlushExecutor(source: string): WgslExec {
  // wgsl_reflect does not implement frexp/ldexp. Supply scalar IEEE bit decomposition and
  // exponent scaling; leave the emitted quotient, norm and score arithmetic intact.
  const executableSource = `${source.replace(/^\s*_\s*=[^;]*;/gm, '').replace(/\bfrexp\(/g, 'scoreProbeFrexp(')}
struct ScoreProbeParts { fract: f32, exp: i32 }
fn scoreProbeFrexp(value: f32) -> ScoreProbeParts {
  if (value == 0.0) { return ScoreProbeParts(0.0, 0); }
  let probeBits = bitcast<u32>(value);
  return ScoreProbeParts(bitcast<f32>((probeBits & 0x807fffffu) | 0x3f000000u), i32((probeBits >> 23u) & 255u) - 126);
}`;
  const identifiers = stripWGSLComments(executableSource).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
  expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
  const executor = new WgslExec(new WgslParser().parse(executableSource));
  executor.builtins.Ldexp = (call, context) => {
    const [valueArgument, exponentArgument] = call.args ?? [];
    if (!valueArgument || !exponentArgument) throw new Error('ldexp requires two arguments');
    const value = executor.evalExpression(valueArgument, context);
    const exponent = executor.evalExpression(exponentArgument, context);
    if (!(value instanceof ScalarData) || !(exponent instanceof ScalarData))
      throw new Error('score probe requires scalar ldexp arguments');
    return new ScalarData(value.value * 2 ** exponent.value, value.typeInfo);
  };
  const evaluateBinary = executor._evalBinaryOp.bind(executor);
  executor._evalBinaryOp = (node, context) => {
    const result = evaluateBinary(node, context);
    if (node.operator === '/' && result instanceof ScalarData && result.typeInfo.name === 'f32') {
      const numerator = executor.evalExpression(node.left, context);
      const denominator = executor.evalExpression(node.right, context);
      if (numerator instanceof ScalarData && denominator instanceof ScalarData) {
        const reciprocal = Math.fround(1 / denominator.value);
        const normalReciprocal = Math.abs(reciprocal) < 2 ** -126 ? 0 : reciprocal;
        result.value = Math.fround(numerator.value * normalReciprocal);
      }
    }
    return result;
  };
  return executor;
}

it('the division profile reproduces the near-maximum reciprocal flush missed by IEEE emulation', () => {
  const executor = createReciprocalFlushExecutor(`
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(1) fn main() { values[1] = values[0] / values[0]; }`);
  const values = Float32Array.of(1.5e38, 0);
  executor.dispatchWorkgroups('main', [1], {0: {0: values}});
  expect(values[1]).toBe(0);
  // One of the two scaled norm contributions disappears, exactly explaining the GPU log.
  const oldScore = 1.5e38 - 2 * Math.SQRT2 * 3.75e37;
  const expected = (Math.SQRT2 * 3e38) / 4;
  expect(Math.abs(oldScore / expected - 1)).toBeCloseTo(0.5857864080164626, 6);
});
