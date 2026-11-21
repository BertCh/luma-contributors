// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {GPUEnsembleScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getGPUCommandNodeKernelSources,
  createNullWebGPUDevice
} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {expect, it} from 'vitest';
import {GPUEnsembleScoresOperation} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-operation';
import {CPUEnsembleScoresOperation} from '../../../src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation';

it('TYP-01 captures score destinations and optional input selections at construction', async () => {
  const output = {energyScore: 'energy', variogramScore: 'variogram'};
  const props = {
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 2,
    variogramPairs: 'pairs' as string | undefined,
    variogramWeights: 'weights' as string | undefined,
    output
  };
  const operation = new CPUEnsembleScoresOperation(props);
  output.energyScore = 'variogram';
  output.variogramScore = 'energy';
  props.variogramPairs = undefined;
  props.variogramWeights = undefined;
  props.memberCount = 1;
  const vector = (values: Float32Array) => ({
    format: 'float32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const pairs = Uint32Array.of(0, 1);
  const result = await operation.execute({
    inputs: {
      ensemble: vector(Float32Array.of(1, 1, 3, 3)),
      observations: vector(Float32Array.of(2, 2)),
      variogramPairs: {
        format: 'uint32x2',
        length: 1,
        chunks: [{values: pairs, length: 1}],
        pack: () => pairs
      },
      variogramWeights: vector(Float32Array.of(1))
    }
  });
  // E||X-y|| = sqrt(2); empirical pair correction = sqrt(2)/2.
  expect(Array.from(result['energy'] as Float32Array)).toEqual([0.7071067690849304]);
  expect(Array.from(result['variogram'] as Float32Array)).toEqual([0]);
});

it.each([
  ['CPU', CPUEnsembleScoresOperation],
  ['GPU', GPUEnsembleScoresOperation]
] as const)('TYP-03 rejects unknown output names before %s publishes a port', (_name, Operation) => {
  const output = {crps: 'crps', energyScroe: 'energy'};
  const props = {
    ensemble: 'members',
    observations: 'observed',
    memberCount: 2,
    dimension: 1,
    output
  };
  expect(() => new Operation(props)).toThrow('ensemble-scores output.energyScroe is unknown');
});

it('TYP-07 asks the core getter to check every input and declared output format', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const calls: [string, unknown][] = [];
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 4},
      observations: {format: 'float32', length: 2},
      variogramPairs: {format: 'uint32x2', length: 1},
      variogramWeights: {format: 'float32', length: 1},
      energyScore: {format: 'float32', length: 1},
      variogramScore: {format: 'float32', length: 1}
    });
    const checkedGetter: GPUContributorOperationViewGetter = <Format extends GPUGraphVectorFormat>(
      port: string,
      format?: Format | readonly Format[]
    ) => {
      calls.push([port, format]);
      return getter<Format>(port);
    };
    new GPUEnsembleScoresOperation({
      ensemble: 'members',
      observations: 'observed',
      memberCount: 2,
      dimension: 2,
      variogramPairs: 'pairs',
      variogramWeights: 'weights',
      output: {energyScore: 'energy', variogramScore: 'variogram'}
    }).createContributor(checkedGetter);
    expect(calls).toEqual([
      ['ensemble', 'float32'],
      ['observations', 'float32'],
      ['variogramPairs', 'uint32x2'],
      ['variogramWeights', 'float32'],
      ['energyScore', 'float32'],
      ['variogramScore', 'float32']
    ]);
  } finally {
    device.destroy();
  }
});

it('captures GPU settings and nested output selection while preserving resource identities', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 4},
      observations: {format: 'float32', length: 2},
      energyScore: {format: 'float32', length: 1}
    });
    const props = {
      ensemble: 'members',
      observations: 'observed',
      memberCount: 2,
      dimension: 2,
      output: {energyScore: 'energy'}
    };
    const operation = new GPUEnsembleScoresOperation(props);
    props.output.energyScore = 'changed';
    props.memberCount = 99;
    props.dimension = 99;
    const contributor = operation.createContributor(getView) as GPUEnsembleScores;
    expect(operation.getOutput('energyScore').value).toBe('energy');
    expect(contributor.props.memberCount).toBe(2);
    expect(contributor.props.dimension).toBe(2);
    expect(contributor.props.ensemble).toBe(getView('ensemble', 'float32'));
    const direct = {...contributor.props, output: {...contributor.props.output}};
    const captured = new GPUEnsembleScores(direct);
    direct.output.energyScore = undefined;
    expect(captured.props.output.energyScore).toBe(contributor.props.output.energyScore);
  } finally {
    device.destroy();
  }
});

it('NUM-12 estimator validation is consistent in both operations and the contributor', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 2},
      observations: {format: 'float32', length: 1},
      crps: {format: 'float32', length: 1}
    });
    for (const invalid of [
      {estimator: 'fair' as const, fair: false},
      {estimator: 'unknown' as 'fair'}
    ]) {
      const props = {
        ensemble: 'members',
        observations: 'observed',
        memberCount: 2,
        dimension: 1,
        output: {crps: 'crpsValue'},
        ...invalid
      };
      expect(() => new GPUEnsembleScoresOperation(props)).toThrow(/estimator/);
      expect(() => new CPUEnsembleScoresOperation(props)).toThrow(/estimator/);
      expect(
        () =>
          new GPUEnsembleScores({
            ...props,
            ensemble: getter('ensemble', 'float32'),
            observations: getter('observations', 'float32'),
            output: {crps: getter('crps', 'float32')}
          })
      ).toThrow(/estimator/);
    }
  } finally {
    device.destroy();
  }
});

it('audits fair scalar and scaled variogram shaders for WGSL declarations and identifiers', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 2},
      observations: {format: 'float32', length: 1},
      variogramPairs: {format: 'uint32x2', length: 1},
      variogramWeights: {format: 'float32', length: 1},
      crps: {format: 'float32', length: 1},
      energyScore: {format: 'float32', length: 1},
      variogramScore: {format: 'float32', length: 1}
    });
    const contributor = new GPUEnsembleScoresOperation({
      ensemble: 'members',
      observations: 'observed',
      memberCount: 2,
      dimension: 1,
      estimator: 'fair',
      variogramPairs: 'pairs',
      variogramWeights: 'weights',
      output: {crps: 'crpsValue', energyScore: 'energy', variogramScore: 'variogram'}
    }).createContributor(getter) as GPUEnsembleScores;
    for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('B6 captures a prototype estimator getter in the GPU operation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 2},
      observations: {format: 'float32', length: 1},
      crps: {format: 'float32', length: 1}
    });
    class GetterProps {
      ensemble = 'members';
      observations = 'observed';
      memberCount = 2;
      dimension = 1;
      output = {crps: 'score'};
      get estimator(): 'fair' {
        return 'fair';
      }
    }
    const contributor = new GPUEnsembleScoresOperation(new GetterProps()).createContributor(
      getter
    ) as GPUEnsembleScores;
    expect(contributor.props.estimator).toBe('fair');
  } finally {
    device.destroy();
  }
});

it('B6 captures a prototype output getter in the direct contributor', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 2},
      observations: {format: 'float32', length: 1},
      crps: {format: 'float32', length: 1}
    });
    class GetterOutput {
      get crps() {
        return getter('crps', 'float32');
      }
    }
    const contributor = new GPUEnsembleScores({
      ensemble: getter('ensemble', 'float32'),
      observations: getter('observations', 'float32'),
      memberCount: 2,
      dimension: 1,
      output: new GetterOutput()
    });
    expect(contributor.props.output.crps).toBe(getter('crps', 'float32'));
  } finally {
    device.destroy();
  }
});

it('B7 preserves the supplied estimator when copying public contributor props', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getter = createOperationTestViewGetter(graph, {
      ensemble: {format: 'float32', length: 2},
      observations: {format: 'float32', length: 1},
      crps: {format: 'float32', length: 1}
    });
    const original = new GPUEnsembleScores({
      ensemble: getter('ensemble', 'float32'),
      observations: getter('observations', 'float32'),
      memberCount: 2,
      dimension: 1,
      output: {crps: getter('crps', 'float32')}
    });
    expect(() => new GPUEnsembleScores({...original.props, ...{fair: true}})).toThrow(
      /fair was removed.*estimator/
    );
    expect(() => new GPUEnsembleScores({...original.props, estimator: 'fair'})).not.toThrow();
    expect(original.props.estimator).toBeUndefined();
  } finally {
    device.destroy();
  }
});

it.each([
  ['variogram without pairs', {output: {variogramScore: 'score'}}, /variogramPairs/],
  ['fair with one member', {memberCount: 1, estimator: 'fair' as const}, /memberCount/],
  ['weights without pairs', {variogramWeights: 'weights'}, /variogramWeights/],
  ['empty selection', {output: {}}, /output needs/]
])('S1 rejects %s before publishing the GPU operation contract', (_name, replacement, message) => {
  expect(
    () =>
      new GPUEnsembleScoresOperation({
        ensemble: 'members',
        observations: 'observed',
        memberCount: 2,
        dimension: 1,
        output: {crps: 'score'},
        ...replacement
      })
  ).toThrow(message);
});

it('S5 captures a nonunit weight selection on a nonzero variogram discrepancy', async () => {
  const props = {
    ensemble: 'members',
    observations: 'observed',
    memberCount: 1,
    dimension: 2,
    variogramPairs: 'pairs',
    variogramWeights: 'weights' as string | undefined,
    output: {energyScore: 'energy', variogramScore: 'variogram'}
  };
  const operation = new CPUEnsembleScoresOperation(props);
  props.variogramWeights = undefined;
  const vector = (values: Float32Array) => ({
    format: 'float32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const pairs = Uint32Array.of(0, 1);
  const result = await operation.execute({
    inputs: {
      ensemble: vector(Float32Array.of(0, 1)),
      observations: vector(Float32Array.of(0, 0)),
      variogramPairs: {
        format: 'uint32x2',
        length: 1,
        chunks: [{values: pairs, length: 1}],
        pack: () => pairs
      },
      variogramWeights: vector(Float32Array.of(7))
    }
  });
  // Gap discrepancy is one, weight is seven. Rereading the removed selection would return one.
  expect(Array.from(result['variogram'] as Float32Array)).toEqual([7]);
  expect(Array.from(result['energy'] as Float32Array)).toEqual([1]);
});
