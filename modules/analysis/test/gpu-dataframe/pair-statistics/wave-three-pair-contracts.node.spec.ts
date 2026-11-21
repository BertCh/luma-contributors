// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUCoregionalizationFitInput} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import {fitLinearModelOfCoregionalization} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {GPURipleyEnvelope} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope';
import {snapshotSpaceTimePairProps} from '../../../src/dataframe-analysis/pair-statistics/space-time-pair-construction';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUSpaceTimeVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {GPUVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {GPUSpatialCorrelogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {QUADRAT_INDEX_WGSL} from '../../../src/dataframe-analysis/pair-statistics/quadrat-index-wgsl';
import {computeRipleyEnvelopeOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {CPUCoregionalizationFitOperation} from '../../../src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation';
import type {CoregionalizationFitOperationProps} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation';
import {
  fitVariogramModel,
  fitVariogramModelWithDiagnostics,
  type VariogramModel,
  type VariogramModelFitDiagnostics
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  getGPUPointPatternIndicesParameterValues,
  type GPUPointPatternIndicesSettings
} from '../../../src/dataframe-analysis/pair-statistics/point-pattern-indices-parameters';
import {getGPUVariogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/variogram-parameters';

it('PCK-01 uses scalar positions at the eight-byte prefix on the space-time route', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const storage = createTransientView(graph, 'storage', 'float32x2', 4);
    const positions = graph.createDataView(storage.buffer, {
      format: 'float32x2',
      byteOffset: 8,
      length: 3
    });
    const contributor = new GPUSpaceTimeVariogram({
      positions,
      values: createTransientView(graph, 'values', 'float32', 3),
      times: createTransientView(graph, 'times', 'float32', 3),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [2, 2],
      lagCount: 1,
      temporalLagCount: 1,
      output: {
        semivariances: createTransientView(graph, 'semivariances', 'float32', 4),
        pairCounts: createTransientView(graph, 'pair-counts', 'uint32', 4),
        status: createTransientView(graph, 'status', 'uint32', 4)
      }
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-complete-case')
    )!.source;
    expect(source).toContain('positions: array<f32>');
    expect(source).toContain('const positionsOffset: u32 = 2u');
    expect(source).toContain('positionsOffset + 2u * index');
  } finally {
    device.destroy();
  }
});

it('W2-11 restricts IQR to cell inclusion and reduces variogram overflow in parallel', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const correlogram of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const props = {
        positions: createTransientView(graph, 'positions', 'float32x2', 20),
        values: createTransientView(graph, 'values', 'float32', 20),
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        gridSize: [2, 2] as const
      };
      const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
      const result = createTransientView(graph, 'result', 'float32', 8);
      const contributor = correlogram
        ? new GPUSpatialCorrelogram({
            ...props,
            bandCount: 8,
            output: {moransI: result, status: {overflow}}
          })
        : new GPUVariogram({
            ...props,
            lagCount: 8,
            output: {semivariances: result, status: {overflow}}
          });
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(
        sources.some(kernel => kernel.source.includes('cellKeys[cellKeysOffset + index] < 4u'))
      ).toBe(true);
      if (!correlogram) {
        const source = sources.find(kernel => kernel.id.endsWith('-overflow'))!.source;
        expect(source).not.toContain('for (var bin');
        expect(source).toContain('readPairOverflow()');
      }
    }
  } finally {
    device.destroy();
  }
});

it('WGS-07 recovers column six even if the quotient seed flushes to zero', () => {
  // 3*2^125 / 2^127 * 8 = 6 exactly. Model reciprocal flushing only at the old seed.
  const helper = QUADRAT_INDEX_WGSL.replace('offset / extent', '0.0');
  const result = new Uint32Array(1);
  const source = `${helper}\n@group(0) @binding(0) var<storage,read_write> result:array<u32>;\n@compute @workgroup_size(1) fn main(){result[0]=getQuadratIndex(${3 * 2 ** 125}, ${2 ** 127}, 8u);}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
  expect([...result]).toEqual([6]);
});

it('NUM-04 preserves deviation but has no probability without usable simulations', () => {
  const result = computeRipleyEnvelopeOnCPU(
    {
      positions: Float32Array.of(0.5, 0.5, 0.51, 0.5),
      parameters: getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 0.45,
        edgeCorrection: 'border',
        seed: 0
      }),
      observedK: Float32Array.of(0.5)
    },
    {
      radiusCount: 1,
      simulationCount: 2
    }
  );
  expect(result.statistics[1]).toBe(0);
  // Observed maximum |sqrt(K/pi)-r| with K=1/2, r encoded as float32.
  expect(Number.isFinite(result.statistics[2])).toBe(true);
  expect(result.statistics[2]).toBeCloseTo(0.0510577075, 7);
  expect(Number.isNaN(result.statistics[4])).toBe(true);
});

it('PCK-13 rejects unusable float32 area, duration and volume', () => {
  const settings = {
    bounds: [0, 0, 1, 1] as const,
    maximumDistance: 1,
    maximumTimeLag: 1,
    timeWindow: [0, 1] as const
  };
  for (const [options, quantity] of [
    [{...settings, bounds: [0, 0, 1e-25, 1e-25] as const}, /area/],
    [{...settings, timeWindow: [-3e38, 3e38] as const}, /duration/],
    [{...settings, bounds: [0, 0, 1e10, 1e10] as const, timeWindow: [0, 1e20] as const}, /volume/]
  ] as const)
    expect(() => getGPUSpaceTimeKFunctionParameterValues(options)).toThrow(quantity);
});

it('FRM-01 validates original destinations with a custom operation ID', () => {
  const props: CoregionalizationFitOperationProps = {
    id: 'custom-fit',
    variableCount: 2,
    structureCount: 1,
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairs',
    parameters: 'parameters',
    output: {
      matrices: 'matrices',
      loadings: 'loadings',
      diagnostics: 'diagnostics',
      status: {
        validity: 'validity',
        iterationCount: 'iterations',
        converged: 'converged',
        iterationLimitReached: 'limit',
        invalidCount: 'invalid'
      }
    }
  };
  for (const matrices of ['', 'loadings', 'distances'])
    expect(
      () => new CPUCoregionalizationFitOperation({...props, output: {...props.output, matrices}})
    ).toThrow(/^custom-fit /);
});

it('TYP-02 returns fixed variogram fit shapes and rejects the removed flag', () => {
  const input = {distances: [1, 2, 3], semivariances: [1, 2, 3], pairCounts: [1, 1, 1]};
  const model = fitVariogramModel(input, {model: 'spherical'});
  const diagnostics = fitVariogramModelWithDiagnostics(input, {model: 'spherical'});
  expectTypeOf(model).toEqualTypeOf<VariogramModel>();
  expectTypeOf(diagnostics).toEqualTypeOf<VariogramModelFitDiagnostics>();
  expect(model).toEqual(diagnostics.model);
  const legacy = {model: 'spherical' as const, diagnostics: true};
  expect(() => fitVariogramModel(input, legacy)).toThrow(
    /diagnostics was removed.*WithDiagnostics/
  );
});

it('NAM-10 and NAM-11 preserve words four and six using the canonical settings', () => {
  const settings: GPUPointPatternIndicesSettings = {bounds: [0, 0, 10, 10], indexCellSizeHint: 1};
  expect(
    getGPUPointPatternIndicesParameterValues({...settings, analyticAlternative: 'lesser'})[6]
  ).toBe(2);
  expect(getGPUPointPatternIndicesParameterValues({...settings, indexCellSizeHint: 3})[4]).toBe(3);
  expect(() =>
    getGPUPointPatternIndicesParameterValues({...settings, ...{alternative: 'greater'}})
  ).toThrow(/alternative was removed.*analyticAlternative/);
  expect(() =>
    getGPUPointPatternIndicesParameterValues({...settings, ...{maximumDistance: 2}})
  ).toThrow(/maximumDistance was removed.*indexCellSizeHint/);
  for (const invalid of ['folded', 'directed'])
    expect(() =>
      getGPUPointPatternIndicesParameterValues({...settings, analyticAlternative: invalid as never})
    ).toThrow(/two-sided.*greater.*lesser/);
});

it('UNI-19 packs sectorOffsetRadians and rejects the old offset key', () => {
  const settings = {bounds: [0, 0, 1, 1] as const, maximumDistance: 1};
  expect(getGPUVariogramParameterValues({...settings, sectorOffsetRadians: 0.25})[5]).toBe(0.25);
  expect(() => getGPUVariogramParameterValues({...settings, ...{azimuthOffset: 0.5}})).toThrow(
    /azimuthOffset was removed.*sectorOffsetRadians/
  );
});

it('NUM-04 generates a publication guard for the actual GPU summary', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPURipleyEnvelope({
      positions: createTransientView(graph, 'points', 'float32x2', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [2, 2],
      radiusCount: 1,
      simulationCount: 2,
      output: {statistics: createTransientView(graph, 'statistics', 'float32', 6)}
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-probability-validity')
    )!.source;
    expect(() => new WgslReflect(source)).not.toThrow();
    expect(source).toContain('if (summary[summaryOffset + 1u] == 0.0)');
    expect(source).toContain('summary[summaryOffset + 4u] = getQuietNaN(index)');
  } finally {
    device.destroy();
  }
});

it('TSC-02 preserves each concrete output map and GPU resource identity', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const first = createTransientView(graph, 'first', 'float32', 1);
    const replacement = createTransientView(graph, 'replacement', 'float32', 1);
    const props = {gridSize: [2, 3] as const, output: {k: first}};
    const captured = snapshotSpaceTimePairProps(props, ['gridSize', 'output'], ['k']);
    expectTypeOf(captured).toEqualTypeOf<typeof props>();
    props.output.k = replacement;
    expect(captured.output.k).toBe(first);
    expect(Object.isFrozen(captured.output)).toBe(true);
    const diggle = {gridSize: [2, 3] as const, output: {d: first, summary: replacement}};
    const second = snapshotSpaceTimePairProps(diggle, ['gridSize', 'output'], ['d', 'summary']);
    expectTypeOf(second).toEqualTypeOf<typeof diggle>();
    expect(second.output.d).toBe(first);
    expect(second.output.summary).toBe(replacement);
  } finally {
    device.destroy();
  }
});

it('WGS-07 exact quadrat comparisons retain edge ties and adjacent float32 values', () => {
  for (const [offset, extent, count, expected] of [
    [0, 1, 8, 0],
    [0.5, 1, 8, 4],
    [0.4999999701976776, 1, 8, 3],
    [1, 1, 8, 7],
    [0.5, 1, 65536, 32768]
  ]) {
    const result = new Uint32Array(1);
    const source = `${QUADRAT_INDEX_WGSL}\n@group(0) @binding(0) var<storage,read_write> result:array<u32>;\n@compute @workgroup_size(1) fn main(){result[0]=getQuadratIndex(${offset.toExponential(16)}, ${extent.toExponential(16)}, ${count}u);}`;
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
    expect(result[0]).toBe(expected);
  }
});

it('DOC-110 coregionalization example has typed complete inputs and a usable supplied model', () => {
  const input: Omit<CPUCoregionalizationFitInput, 'parameters'> = {
    distances: Float32Array.of(50, 100, 200),
    semivariances: Float32Array.of(1, -0.2, 2, 2, -0.5, 3, 3, -0.8, 4),
    pairCounts: Uint32Array.of(20, 30, 40)
  };
  const fit = fitLinearModelOfCoregionalization(input, {
    variableCount: 2,
    structures: [
      {model: 'spherical', range: 1, sill: 0, nugget: 1},
      {model: 'exponential', range: 200, sill: 1}
    ],
    maximumIterations: 512
  });
  expect(fit.validity[0]).toBe(1);
  expect(fit.coregionalizationModels).toHaveLength(2);
  expect(fit.coregionalizationModels[0].loadings.every(Number.isFinite)).toBe(true);
});
