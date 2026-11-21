// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  createFinalTemporalCase,
  type FinalTemporalScenario
} from './temporal-regression-w5-fixtures';
import {getSerialAuditBody} from './local-regression-serial-audit';
import {getGPUGeographicallyTemporallyWeightedRegressionModelResult as readModel} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUGTWRParameterValues,
  getGPUGTWRBandwidthSearchParameterValues,
  snapshotGTWRProps,
  GPUGeographicallyTemporallyWeightedRegression,
  type GPUGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {
  getGPUMGTWRParameterValues,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  type GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-temporally-weighted-regression';
import {
  TEMPORAL_AUDIT_MODELS,
  createTemporalAuditCase,
  type TemporalAuditModel,
  type TemporalAuditOptions
} from './temporal-gwr-audit-fixtures';

it('temporal parameter packers preserve documented layouts and caller target tails', () => {
  const target = new Float32Array(6).fill(777);
  expect(getGPUGTWRParameterValues({bandwidth: 5, timeScale: 1.25}, target)).toBe(target);
  expect(Array.from(target)).toEqual([5, 1.25, 0, 0, 777, 777]);
  expect(
    Array.from(
      getGPUMGTWRParameterValues({
        initial: {bandwidth: 6, timeScale: 0.5},
        components: [
          {bandwidth: 5, timeScale: 1.25},
          {bandwidth: 6, timeScale: 2}
        ]
      })
    )
  ).toEqual([6, 0.5, 5, 1.25, 6, 2]);
  expect(
    Array.from(
      getGPUGTWRBandwidthSearchParameterValues({
        bandwidthBounds: [5, 5],
        timeScaleBounds: [1.25, 1.25],
        absoluteTolerance: 0.125,
        relativeTolerance: 0.25
      })
    )
  ).toEqual([5, 5, 1.25, 1.25, 0.125, 0.25]);
});

function runCase(
  model: TemporalAuditModel,
  options: TemporalAuditOptions,
  scenario?: FinalTemporalScenario,
  route: 'dense' | 'search' = 'dense'
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const memory = new Map<GraphBufferHandle, ArrayBuffer>();
  let serial = 0;
  const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    values: number[]
  ) => {
    const view = createTransientView(
      graph,
      `temporal-audit-${serial++}`,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
    memory.set(
      view.buffer,
      (format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)).buffer
    );
    return view;
  };
  const getMemory = (buffer: GraphBufferHandle) => {
    if (!memory.has(buffer)) memory.set(buffer, new ArrayBuffer(buffer.byteLength));
    return memory.get(buffer)!;
  };
  try {
    const {nodes, outputs, expected} = scenario
      ? createFinalTemporalCase(graph, port, scenario, route, options.multiscale)
      : createTemporalAuditCase(graph, port, model, options);
    for (const node of nodes) {
      if (
        node.condition?.source === 'gpu' &&
        new Uint32Array(getMemory(node.condition.buffer), node.condition.byteOffset, 3)[0] === 0
      )
        continue;
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const range = getViewBindingRange(binding.view);
          return [index, new Uint8Array(getMemory(binding.view.buffer), range.offset, range.size)];
        })
      );
      // Array-copy syntax adaptations only: solver and model arithmetic remain emitted WGSL.
      const serialBody = getSerialAuditBody(record);
      const source =
        `fn executorLdexp(value: f32, power: i32) -> f32 {
        let first = power / 2i;
        return (value * exp2(f32(first))) * exp2(f32(power - first));
      }\n` +
        record.source
          .replace(record.body, serialBody)
          .replace(/@workgroup_size\(256\)/, '@workgroup_size(1)')
          .replace(/\bldexp\s*\(/g, 'executorLdexp(')
          .replace(/^\s*_\s*=.*;$/gm, '')
          .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
          .replace(
            '*scale = factorization.scale;',
            'for (var copied = 0u; copied < 2u; copied++) { (*scale)[copied] = factorization.scale[copied]; }'
          )
          .replace(
            'var factorization = DenseCholesky_2(*scale, 2u, 0u, 1.0);',
            'var factorization: DenseCholesky_2; factorization.size = 2u; factorization.smallestPivotRatio = 1.0; for (var copied = 0u; copied < 2u; copied++) { factorization.scale[copied] = (*scale)[copied]; }'
          )
          .replace(
            'return DenseCholesky_2(array<f32, 2>(1.0, 1.0), 2u, 0u, 1.0);',
            'var result: DenseCholesky_2; result.scale[0] = 1.0; result.scale[1] = 1.0; result.size = 2u; result.smallestPivotRatio = 1.0; return result;'
          );
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
    }
    const read = (view: GraphDataView) =>
      Array.from(
        view.format === 'uint32'
          ? new Uint32Array(getMemory(view.buffer))
          : new Float32Array(getMemory(view.buffer))
      );
    return {
      expected,
      actual: Object.fromEntries(Object.entries(outputs).map(([name, view]) => [name, read(view)]))
    };
  } finally {
    device.destroy();
  }
}

it('W5-53 identical mixed-sign designs share the well-conditioned response origin', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {}, 'mixed-sign');
  // Category means: negative design 1/6, positive design 1; b0=7/12, b1=5/12.
  for (const row of [0, 1]) {
    expect(actual['localStatus'][row]).toBe(0);
    expect(actual['coefficients'][2 * row]).toBeCloseTo(7 / 12, 5);
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(5 / 12, 5);
    expect(actual['fitted'][row]).toBeCloseTo(1 / 6, 5);
  }
});

it('B2 negligible Gaussian weight cannot erase the high-weight local response signal', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {}, 'negligible');
  // Hand WLS: symmetry makes slope 1, intercept 2+(1e8-2)*exp(-32)/(4+exp(-32)).
  for (let row = 0; row < 4; row++) {
    expect(actual['localStatus'][row]).toBe(0);
    expect(actual['coefficients'][2 * row]).toBeCloseTo(2.0000003166, 5);
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(1, 5);
    expect(Number.isFinite(actual['residuals'][row])).toBe(true);
    // Two float32 ulps of the unit-scale local prediction; still excludes zero residual.
    expect(Math.abs(actual['residuals'][row] + 0.0000003166041)).toBeLessThan(2.5e-7);
  }
});

it('B3 multiscale coincident outliers retain the common 0.75 intercept', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {multiscale: true}, 'component-outliers');
  for (let row = 0; row < 8; row++) {
    expect(actual['localStatus'][row]).toBe(0);
    expect(actual['coefficients'][2 * row]).toBeCloseTo(0.75, 6);
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(0, 6);
  }
});

it('B5 backfitting retains initialized sub-ulp residuals at the orthogonal solution', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {multiscale: true}, 'orthogonal-rounded');
  // X is orthogonal to the constant: b1=sum(x*y)/sum(x*x)=1/96.
  const residuals = [1, 0, -1, -2, 2, 1, 0, -1].map(value => value / 96);
  for (let row = 0; row < 8; row++) {
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(1 / 96, 7);
    expect(Number.isFinite(actual['residuals'][row])).toBe(true);
    expect(Math.abs(actual['residuals'][row] - residuals[row])).toBeLessThan(1e-7);
  }
});

for (const route of ['dense', 'search'] as const) {
  it(`W5-48 ${route} retains finite fit-time RSS and criteria through model readback`, () => {
    const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {}, 'rounded', route);
    const model = readModel({summary: actual['summary']});
    expect(model.rowCount).toBe(8);
    expect(model.status.code).toBe(0);
    // Independent rational least squares: residual numerator /1376; RSS=5/11008.
    for (const [row, numerator] of [11, 2, -7, -16, 16, 7, -2, -11].entries()) {
      expect(Number.isFinite(actual['residuals'][row])).toBe(true);
      expect(Math.abs(actual['residuals'][row] - numerator / 1376)).toBeLessThan(2e-5);
      expect(actual['fitted'][row]).toBe(1e6 + (row >= 4 ? 1 / 16 : 0));
    }
    expect(Number.isFinite(model.residualSumOfSquares)).toBe(true);
    expect(model.residualSumOfSquares).toBeCloseTo(5 / 11008, 8);
    expect(Number.isFinite(model.logLikelihood)).toBe(true);
    expect(model.logLikelihood).toBeCloseTo(27.75401649203806, 4);
    expect(Number.isFinite(model.correctedAkaikeInformationCriterion)).toBe(true);
    expect(model.correctedAkaikeInformationCriterion).toBeCloseTo(-43.50803298407612, 4);
  });
}

it('W5-48 genuine exact fit retains zero residuals and unbounded likelihood', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {}, 'exact');
  const model = readModel({summary: actual['summary']});
  expect(actual['residuals']).toEqual(Array(8).fill(0));
  expect(model.status.code).toBe(0);
  expect(model.residualSumOfSquares).toBe(0);
  expect(model.logLikelihood).toBe(Infinity);
  expect(model.correctedAkaikeInformationCriterion).toBe(-Infinity);
});

it('W5-54 two MGTWR sweeps preserve an exact million-intercept line', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {straightLine: true, multiscale: true});
  for (let row = 0; row < 12; row++) {
    expect(actual['localStatus'][row]).toBe(0);
    expect(actual['coefficients'][2 * row]).toBe(1000000);
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(1, 5);
    expect(actual['residuals'][row]).toBe(0);
  }
});

it('R2 dense GTWR preserves an exactly representable straight line with a million-unit intercept', () => {
  const {actual} = runCase(TEMPORAL_AUDIT_MODELS[0], {straightLine: true});
  expect(actual['localStatus'].slice(0, 12)).toEqual(Array(12).fill(0));
  for (let row = 0; row < 12; row++) {
    expect(actual['coefficients'][2 * row + 1]).toBeCloseTo(1, 5);
    expect(actual['fitted'][row]).toBe(1000000 + (row % 6 < 3 ? -1 : 1));
    expect(actual['residuals'][row]).toBe(0);
  }
});

for (const multiscale of [true, false]) {
  it(`R8 ${multiscale ? 'MGTWR' : 'GTWR'} snapshots caller props, output records and search budgets`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `snapshot-${serial++}`, format, length);
    const base = {
      positions: view('float32x2', 14),
      predictors: view('float32', 14),
      response: view('float32', 14),
      times: view('float32', 14),
      predictorCount: 1,
      temporalContext: {
        unit: 'second' as const,
        epoch: 0n,
        encoding: 'float32-relative' as const,
        axis: {kind: 'irregular' as const}
      },
      parameters: view('float32', 6),
      bandwidthSearch: {parameters: view('float32', 6), maximumIterations: 1},
      output: {
        status: undefined,
        coefficients: view('float32', 28),
        fitted: view('float32', 14),
        residuals: view('float32', 14),
        localStatus: view('uint32', 14),
        scores: view('float32', 2)
      }
    };
    try {
      const calendarStep = {calendarUnit: 'day' as const, interval: 2};
      const regularContext = {
        ...base.temporalContext,
        axis: {kind: 'regular' as const, step: calendarStep}
      };
      const regularSnapshot = snapshotGTWRProps({...base, temporalContext: regularContext});
      calendarStep.interval = 9;
      expect(regularSnapshot.temporalContext.axis).toEqual({
        kind: 'regular',
        step: {calendarUnit: 'day', interval: 2}
      });
      let cutoffReads = 0;
      const inherited = Object.create({
        kernel: 'bisquare',
        get gaussianCutoff() {
          cutoffReads++;
          return 3;
        }
      });
      Object.assign(inherited, base, {iterationCount: 2});
      if (multiscale) {
        expect(
          () => new GPUMultiscaleGeographicallyTemporallyWeightedRegression(inherited)
        ).toThrow(/gaussianCutoff is not supported/);
        expect(cutoffReads).toBe(1);
      } else {
        const contributor = new GPUGeographicallyTemporallyWeightedRegression({
          ...Object.create(null),
          ...base,
          output: {...base.output, hatDiagonal: view('float32', 14), summary: view('float32', 12)}
        });
        let kernelReads = 0;
        const inheritedProps = Object.assign(
          Object.create({
            get kernel() {
              kernelReads++;
              return 'bisquare';
            }
          }),
          contributor.props,
          {
            bandwidthSearch: undefined,
            output: {...contributor.props.output, status: undefined, selectedParameters: undefined}
          }
        );
        delete inheritedProps.kernel;
        expect(
          new GPUGeographicallyTemporallyWeightedRegression(inheritedProps).searchStructure
        ).toBe('grid');
        expect(kernelReads).toBe(1);
      }
      const missingAxis = {...base, temporalContext: {...base.temporalContext, axis: undefined}};
      const constructMissing = () =>
        multiscale
          ? new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
              ...missingAxis,
              iterationCount: 2
            } as unknown as GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps)
          : new GPUGeographicallyTemporallyWeightedRegression({
              ...missingAxis,
              output: {
                ...base.output,
                hatDiagonal: view('float32', 14),
                summary: view('float32', 12)
              }
            } as unknown as GPUGeographicallyTemporallyWeightedRegressionProps);
      expect(constructMissing).toThrow(/axis must be regular or irregular/);
      if (multiscale) {
        const props: GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps = {
          ...base,
          iterationCount: 2
        };
        const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression(props);
        const originalScores = props.output.scores;
        props.iterationCount = 3;
        props.output.scores = view('float32', 3);
        props.bandwidthSearch!.maximumIterations = 2;
        const nodes = contributor.getCommandNodes(graph);
        const scoreNodes = nodes.filter(node => /-iteration-\d+-score$/.test(node.id));
        expect(scoreNodes).toHaveLength(2);
        expect(contributor.props.bandwidthSearch!.maximumIterations).toBe(1);
        for (const node of scoreNodes)
          expect(
            getWGSLKernelNodeRecord(graph, node.id)!.bindings.find(
              binding => binding.name === 'scores'
            )!.view
          ).toBe(originalScores);
      } else {
        const props: GPUGeographicallyTemporallyWeightedRegressionProps = {
          ...base,
          output: {...base.output, hatDiagonal: view('float32', 14), summary: view('float32', 12)}
        };
        const contributor = new GPUGeographicallyTemporallyWeightedRegression(props);
        const originalFitted = props.output.fitted;
        props.predictorCount = 2;
        props.output.fitted = view('float32', 1);
        props.bandwidthSearch!.maximumIterations = 2;
        expect(() => contributor.getCommandNodes(graph)).not.toThrow();
        expect(contributor.props.predictorCount).toBe(1);
        expect(contributor.props.output.fitted).toBe(originalFitted);
        expect(contributor.props.bandwidthSearch!.maximumIterations).toBe(1);
      }
    } finally {
      device.destroy();
    }
  });
}

for (const multiscale of [false, true]) {
  it.each(
    TEMPORAL_AUDIT_MODELS
  )(`${multiscale ? 'MGTWR' : 'GTWR'} emitted $kernel/adaptive=$adaptive matches every independent QR/block-solve output`, model => {
    const maximumErrors: Record<string, number> = {};
    for (const options of [
      {multiscale},
      {multiscale, rescaled: true},
      {multiscale, search: true}
    ]) {
      const {actual, expected} = runCase(model, options);
      for (const [name, values] of Object.entries(expected)) {
        expect(actual[name].length).toBe(values.length);
        values.forEach((value, index) => {
          if (Number.isNaN(value)) expect(actual[name][index], `${name}[${index}]`).toBeNaN();
          else {
            expect(Number.isFinite(actual[name][index]), `${name}[${index}]`).toBe(true);
            const difference = Math.abs(actual[name][index] - value);
            maximumErrors[name] = Math.max(maximumErrors[name] ?? 0, difference);
            const tolerance = [
              'localStatus',
              'iterationCount',
              'converged',
              'iterationLimitReached',
              'selectedParameters'
            ].includes(name)
              ? 0
              : 1e-4;
            expect(difference, `${name}[${index}]`).toBeLessThanOrEqual(tolerance);
          }
        });
      }
    }
    console.info(
      `${multiscale ? 'MGTWR' : 'GTWR'}/${model.kernel}/${model.adaptive} maximum absolute differences`,
      maximumErrors
    );
  });
}
