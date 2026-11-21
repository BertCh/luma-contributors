// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {CPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {GPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import type {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {constructSamplingContributor} from './validation-sampling-operation-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('extreme-rank inference reduces parallel curve decisions before constant-work publication', () => {
  const operation = new GPUPointPatternEnvelopeOperation(getEnvelopeProps('extreme-rank', true));
  const factory = operation.createContributor.bind(operation);
  let sources: ReturnType<typeof getGPUCommandNodeKernelSources> = [];
  operation.createContributor = getView => {
    const contributor = factory(getView) as GPUPointPatternEnvelope;
    const graph = getView('observed').buffer.graph;
    if (!(graph instanceof GPUCommandGraph))
      throw new Error('test views must belong to a GPUCommandGraph');
    sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    return contributor;
  };
  constructSamplingContributor(operation, {
    observed: {format: 'float32', length: 3},
    simulations: {format: 'float32', length: 12},
    reference: {format: 'float32', length: 3},
    integrationWeights: {format: 'float32', length: 3}
  });
  for (const name of ['strict', 'inclusive', 'invalid', 'critical'])
    expect(
      sources.some(({id}) => id.includes(`-rank-${name}-reduction-`)),
      name
    ).toBe(true);
  const publication = sources.find(({id}) => id.endsWith('-rank-test'))!;
  for (const {id, source} of sources)
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  expect(publication.source.slice(publication.source.indexOf('fn main('))).not.toMatch(
    /\b(?:for|while)\s*\(|\bloop\s*\{/
  );
});
import {
  ENVELOPE_INPUTS,
  getHierarchicalRankEnvelopeCase,
  expectHierarchicalRankEnvelopeOracle,
  expectEnvelopeOracle,
  getEnvelopeProps,
  getTiedEnvelopeProps,
  TIED_ENVELOPE_INPUTS,
  expectTiedEnvelopeOracle,
  getHalfRankEnvelopeCase,
  expectHalfRankEnvelopeOracle
} from './envelope-output-oracles-fixture';

it('CPU rank envelope matches the hand-ranked 300-curve reduction fixture including late invalidation', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const invalid of [false, true]) {
      const {props, inputs} = getHierarchicalRankEnvelopeCase(invalid);
      expectHierarchicalRankEnvelopeOracle(
        await runOperationGraph(device, {
          inputs,
          operations: [new CPUPointPatternEnvelopeOperation(props)],
          outputs: ['lower', 'upper', 'statistics', 'status', 'interval']
        }),
        invalid
      );
    }
  } finally {
    device.destroy();
  }
});

it('CPU extreme-rank cutoff separates attainable half-integer tie groups', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const defaultLevel of [false, true]) {
      const {props, inputs} = getHalfRankEnvelopeCase(defaultLevel);
      expectHalfRankEnvelopeOracle(
        await runOperationGraph(device, {
          inputs,
          operations: [new CPUPointPatternEnvelopeOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        defaultLevel
      );
    }
  } finally {
    device.destroy();
  }
});

it('envelope oracle refuses a missing requested extreme-rank interval', async () => {
  const device = createNullWebGPUDevice();
  const props = getEnvelopeProps('extreme-rank', true);
  try {
    const result = await runOperationGraph(device, {
      inputs: ENVELOPE_INPUTS,
      operations: [new CPUPointPatternEnvelopeOperation(props)],
      outputs: Object.values(props.output) as string[]
    });
    delete result['interval'];
    expect(() => expectEnvelopeOracle(result, 'extreme-rank', false, true)).toThrow();
  } finally {
    device.destroy();
  }
});

for (const backend of ['CPU', 'GPU factory'] as const)
  it(`${backend} envelope captures statistic and output map before caller reuse`, async () => {
    const props = getEnvelopeProps('maximum-deviation');
    const operation =
      backend === 'CPU'
        ? new CPUPointPatternEnvelopeOperation(props)
        : new GPUPointPatternEnvelopeOperation(props);
    const requested = Object.values(props.output) as string[];
    props.statistic = 'integrated-squared-deviation';
    props.output.statistics = 'replacement';
    props.output.pValueInterval = 'late-interval';
    if (operation instanceof GPUPointPatternEnvelopeOperation) {
      let statistic: unknown;
      const createContributor = operation.createContributor.bind(operation);
      operation.createContributor = getView => {
        const contributor = createContributor(getView) as GPUPointPatternEnvelope;
        statistic = contributor.props.statistic;
        return contributor;
      };
      constructSamplingContributor(operation, {
        observed: {format: 'float32', length: 3},
        simulations: {format: 'float32', length: 12},
        reference: {format: 'float32', length: 3},
        integrationWeights: {format: 'float32', length: 3}
      });
      expect(statistic).toBe('maximum-deviation');
    } else {
      const device = createNullWebGPUDevice();
      try {
        expectEnvelopeOracle(
          await runOperationGraph(device, {
            inputs: ENVELOPE_INPUTS,
            operations: [operation],
            outputs: requested
          }),
          'maximum-deviation'
        );
      } finally {
        device.destroy();
      }
    }
  });

it('CPU extreme-rank bounds exclude the entire rejected tie group', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const withInterval of [true, false]) {
      const props = getTiedEnvelopeProps(withInterval);
      expectTiedEnvelopeOracle(
        await runOperationGraph(device, {
          inputs: TIED_ENVELOPE_INPUTS,
          operations: [new CPUPointPatternEnvelopeOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        withInterval
      );
    }
  } finally {
    device.destroy();
  }
});

for (const statistic of [
  'maximum-deviation',
  'integrated-squared-deviation',
  'extreme-rank'
] as const)
  for (const withInterval of statistic === 'extreme-rank' ? [true, false] : [false])
    it(`CPU ${statistic} interval=${withInterval} matches hand-sorted envelopes and inference`, async () => {
      const device = createNullWebGPUDevice();
      const props = getEnvelopeProps(statistic, withInterval);
      try {
        for (const invalid of [false, true]) {
          const inputs = {
            ...ENVELOPE_INPUTS,
            observed: {
              ...ENVELOPE_INPUTS.observed,
              values: new Float32Array(invalid ? [NaN, 1, -1] : [2, 1, -1])
            }
          };
          expectEnvelopeOracle(
            await runOperationGraph(device, {
              inputs,
              operations: [new CPUPointPatternEnvelopeOperation(props)],
              outputs: Object.values(props.output) as string[]
            }),
            statistic,
            invalid,
            withInterval
          );
        }
      } finally {
        device.destroy();
      }
    });
