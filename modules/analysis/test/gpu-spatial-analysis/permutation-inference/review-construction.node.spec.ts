// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalPermutationTest,
  type GPUGlobalPermutationTestProps
} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test';
import {GPULocalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {
  GPUGlobalSpatialStatistics,
  type GPUGlobalSpatialStatisticsProps
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 6),
    weights: view('float32', 6)
  };
  const common = {
    weights,
    values: view('float32', 4),
    parameters: view('uint32', 4),
    maximumPermutations: 9
  };
  const global: GPUGlobalPermutationTestProps = {
    ...common,
    statistic: 'moran',
    output: {summary: view('float32', 12)}
  };
  const quantile = {
    ...common,
    alternative: 'greater' as const,
    quantileParameters: view('uint32', 2),
    output: {
      membership: view('uint32', 4),
      boundaries: view('float32', 2),
      validCount: view('uint32', 1),
      classes: view('uint32', 4),
      invalidQuantile: view('uint32', 1),
      exceedances: view('uint32', 4),
      pseudoPValues: view('float32', 4),
      status: {overflow: view('uint32', 1)}
    }
  };
  return {device, graph, view, weights, common, global, quantile};
}

it('B4 permutation consumers retain and invert complete certificates while the analytic consumer refuses them', () => {
  for (const location of ['outer', 'chunk'] as const) {
    const fixture = createFixture();
    const {weights, view, common, global} = fixture;
    const status = {complete: view('uint32', 1)};
    const input =
      location === 'outer'
        ? {...weights, status}
        : {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights: {...weights, status}}]};
    try {
      const contributors = [
        new GPUGlobalPermutationTest({...global, weights: input}),
        new GPULocalPermutationTest({
          ...common,
          weights: input,
          statistic: 'localMoran',
          output: {
            exceedances: view('uint32', 4),
            pseudoPValues: view('float32', 4),
            status: {overflow: view('uint32', 1)}
          }
        })
      ];
      for (const contributor of contributors) {
        const captured = contributor.props.weights;
        const capturedStatus =
          'chunks' in captured ? captured.chunks[0].weights.status : captured.status;
        expect(capturedStatus?.complete).toBe(status.complete);
        const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph));
        // A complete certificate is positive: zero, rather than one, suppresses inference.
        expect(
          sources.some(({source}) =>
            source.includes(
              'if (loss[lossOffset] == 0u) { analysisFlags[analysisFlagsOffset + index] = 0u; }'
            )
          )
        ).toBe(true);
      }
      expect(
        () =>
          new GPUGlobalSpatialStatistics({
            weights: input,
            values: common.values,
            statistics: ['moran'],
            output: {summary: view('float32', 52)}
          })
      ).toThrow(/complete.*candidateOverflow/);
    } finally {
      fixture.device.destroy();
    }
  }
});

it('B5 captures inherited global options and scalar status ports by name', () => {
  const fixture = createFixture();
  const {global, view, graph} = fixture;
  const status = Object.create({invalidCount: view('uint32', 1)}) as NonNullable<
    GPUGlobalPermutationTestProps['output']['status']
  >;
  const props = Object.assign(Object.create({alternative: 'lesser'}), global, {
    output: {...global.output, status}
  }) as GPUGlobalPermutationTestProps;
  const contributor = new GPUGlobalPermutationTest(props);
  expect(contributor.getInferenceMetadata({seed: 7, permutationCount: 9}).alternative).toBe(
    'lesser'
  );
  expect(contributor.props.output.status?.invalidCount).toBe(status.invalidCount);
  expect(
    getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).some(({id}) =>
      id.endsWith('-invalid-clear')
    )
  ).toBe(true);
  fixture.device.destroy();
});

it('B5 captures inherited Quantile LISA overflow without dropping its writer', () => {
  const fixture = createFixture();
  const inherited = Object.create(
    fixture.quantile.output.status
  ) as typeof fixture.quantile.output.status;
  const contributor = new GPUQuantileLISA({
    ...fixture.quantile,
    output: {...fixture.quantile.output, status: inherited}
  });
  expect(contributor.props.output.status.overflow).toBe(inherited.overflow);
  expect(() => contributor.getCommandNodes(fixture.graph)).not.toThrow();
  fixture.device.destroy();
});

it('B8 retains the supported fixed-count Moran graph when added passes are unused', () => {
  const fixture = createFixture();
  const {view, global} = fixture;
  const weights = {
    rowCount: 5,
    chunks: Array.from({length: 5}, (_, rowStart) => ({
      rowStart,
      rowCount: 1,
      weights: {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 1),
        weights: view('float32', 1)
      }
    }))
  };
  expect(
    () =>
      new GPUGlobalPermutationTest({
        ...global,
        weights,
        values: view('float32', 5),
        permutationBatchSize: 1,
        maximumPermutations: 4073
      })
  ).not.toThrow();
  fixture.device.destroy();
});

it('S1 counts original included signs before normalization can underflow', () => {
  const fixture = createFixture();
  const contributor = new GPUGlobalPermutationTest({...fixture.global, statistic: 'getisOrdG'});
  const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(fixture.graph)).find(
    ({id}) => id.endsWith('-negative-indicators')
  )!.source;
  expect(source).toContain('sourceValues[sourceValuesOffset + index] < 0.0');
  fixture.device.destroy();
});

it('S5 rejects an explicitly supplied null analyticAlternative', () => {
  const fixture = createFixture();
  expect(
    () =>
      new GPUGlobalSpatialStatistics({
        weights: fixture.weights,
        values: fixture.common.values,
        statistics: ['moran'],
        analyticAlternative: null as unknown as NonNullable<
          GPUGlobalSpatialStatisticsProps['analyticAlternative']
        >,
        output: {summary: fixture.view('float32', 52)}
      })
  ).toThrow(/analyticAlternative.*two-sided.*greater.*lesser/);
  fixture.device.destroy();
});
