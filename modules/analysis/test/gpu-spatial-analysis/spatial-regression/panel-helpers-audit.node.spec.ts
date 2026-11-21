// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getPanelWithinNodes,
  getPanelWithinScratchLengths
} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-within-nodes';
import {getPanelDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostics';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {computeSpatialPanelErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createPanelAuditInput, expectPanelAuditRows} from './panel-audit.test-utils';
import {PANEL_DIAGNOSTIC_AUDIT_EXPECTED} from './panel-regression-audit.test-utils';
import reviewDiagnostics from './panel-review-diagnostics.json';
import {getPanelDiagnosticZeroFitNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostic-zero-fit-nodes';

it('review 2 R2 refuses exactly zero pooled fitted values', () => {
  const input = createPanelAuditInput();
  input.response = Float32Array.of(-2, 5, 0, -3, ...new Array(12).fill(0));
  // Exact integer normal scores: sum y=-2+5-3=0, x'y=4+5-9=0.
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
    periodCount: 4,
    predictorCount: 1
  });
  expect(result.diagnosticStatus[0], JSON.stringify(Array.from(result.diagnosticTests))).toBe(2);
  expect(Array.from(result.diagnosticTests).every(Number.isNaN)).toBe(true);
});

for (const perturb of [false, true])
  it(`review 2 R2 exact pooled-score certificate distinguishes one ULP, perturb=${perturb}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      const input = createPanelAuditInput();
      input.response = Float32Array.of(-2, 5, 0, -3, ...new Array(12).fill(0));
      if (perturb) new Uint32Array(input.response.buffer)[0]--;
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `zero-${serial++}`, format, length);
      const predictors = view('float32', 16),
        response = view('float32', 16),
        diagnostics = {tests: view('float32', 6), status: view('uint32', 1)};
      memory.write(predictors, input.predictors);
      memory.write(response, input.response);
      memory.write(diagnostics.tests, new Float32Array(6).fill(987));
      memory.write(diagnostics.status, Uint32Array.of(0));
      executeCommandNodesOnCPU(
        getPanelDiagnosticZeroFitNodes(graph, {
          id: 'zero',
          predictorCount: 1,
          predictors,
          response,
          diagnostics
        }),
        memory
      );
      expect(Array.from(memory.readUint32(diagnostics.status))).toEqual([perturb ? 0 : 2]);
      const values = Array.from(memory.readFloat32(diagnostics.tests));
      if (perturb) expect(values).toEqual(new Array(6).fill(987));
      else expect(values.every(Number.isNaN)).toBe(true);
    } finally {
      device.destroy();
    }
  });

it('R8 CPU pooled LM is invariant to response units 2^32', () => {
  const input = createPanelAuditInput();
  input.response = Float32Array.from(input.response, value => value * 2 ** 32);
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
    periodCount: 4,
    predictorCount: 1
  });
  expect(result.diagnosticStatus[0]).toBe(0);
  expectPanelAuditRows(result.diagnosticTests, PANEL_DIAGNOSTIC_AUDIT_EXPECTED, 2e-6);
});

for (const scene of ['rankZero', 'nearby', 'nearbyShift', 'nearbyNonuniform'] as const)
  it(`R2 pooled LM CPU recognizes projected-lag rank: ${scene}`, () => {
    const input = createPanelAuditInput();
    input.predictors = Float32Array.from(
      {length: 16},
      (_, row) => (Math.floor(row / 4) + 1) * (row % 2 ? -1 : 1)
    );
    if (scene !== 'rankZero') input.predictors[0] += 0.25;
    if (scene === 'nearbyShift')
      input.response = Float32Array.from(input.response, value => value + 32768);
    if (scene === 'nearbyNonuniform') {
      input.response = Float32Array.from(input.response, value => value + 16);
      input.weights.weights[0] = 0.25;
      input.weights.weights[1] = 0.25;
    }
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
      periodCount: 4,
      predictorCount: 1
    });
    expect(result.diagnosticStatus[0], JSON.stringify(Array.from(result.diagnosticTests))).toBe(
      scene === 'rankZero' ? 2 : 0
    );
    expectPanelAuditRows(
      result.diagnosticTests,
      reviewDiagnostics[scene === 'nearbyShift' ? 'nearby' : scene].tests.map(
        value => value ?? NaN
      ),
      2e-6
    );
  });

it('within scratch counts retain the group and component axes', () => {
  // Three entities, four periods, three components; one tile and two words per mean.
  expect(getPanelWithinScratchLengths(3, 4, 2)).toEqual({
    partialLength: 18,
    meanLength: 19,
    tileCount: 1
  });
  expect(getPanelWithinScratchLengths(3, 4, 2, 'time')).toEqual({
    partialLength: 24,
    meanLength: 25,
    tileCount: 1
  });
});

for (const effects of ['unit', 'time'] as const)
  for (const scene of ['ordinary', 'shifted', 'incomplete'] as const)
    it(`within ${effects} kernels equal hand-computed group means: ${scene}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device),
          memory = createCPUGraphMemory();
        recordWGSLKernelNodes(graph);
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `within-audit-${serial++}`, format, length);
        const predictors = view('float32', 16),
          response = view('float32', 16),
          overflow = view('uint32', 1);
        const output = {predictors: view('float32', 16), response: view('float32', 16)};
        const data = createPanelAuditInput(scene === 'shifted');
        memory.write(predictors, data.predictors);
        memory.write(response, data.response);
        memory.write(overflow, Uint32Array.of(scene === 'incomplete' ? 1 : 0));
        const nodes = getPanelWithinNodes(graph, {
          id: 'within-audit',
          entityCount: 4,
          periodCount: 4,
          predictorCount: 1,
          effects,
          predictors,
          response,
          overflow,
          output
        });
        // wgsl_reflect lacks ldexp; on this small fixture every scale is a normal power of
        // two, so this builtin shim is exact. Keep all panel arithmetic and bindings intact.
        executeCommandNodesOnCPU(
          nodes.map(node => {
            const record = getWGSLKernelNodeRecord(graph, node.id)!;
            return createWGSLKernelNode(graph, {
              id: `${node.id}-interpreter`,
              operation: record.operation,
              bindings: [...record.bindings],
              invocationCount: record.invocationCount,
              workgroupSize: record.workgroupSize[0],
              declarations: `fn auditLdexp(value: f32, power: i32) -> f32 { return value * exp2(f32(power)); }\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'auditLdexp(')}`,
              body: record.body
            });
          }),
          memory
        );
        // These means are the exact sums of four small integers, divided by four.
        // Unit x means [.25,.25,.25,.5], y means [3.75,1.25,2.5,3.5].
        // Period x means [.75,-.25,.5,.25], y means [3.75,2.25,2.25,2.75].
        const predictorMeans =
          effects === 'unit' ? [0.25, 0.25, 0.25, 0.5] : [0.75, -0.25, 0.5, 0.25];
        const responseMeans =
          effects === 'unit' ? [3.75, 1.25, 2.5, 3.5] : [3.75, 2.25, 2.25, 2.75];
        expectPanelAuditRows(
          memory.readFloat32(output.predictors),
          Array.from(
            data.predictors,
            (value, row) =>
              value -
              predictorMeans[effects === 'unit' ? row % 4 : Math.floor(row / 4)] -
              (scene === 'shifted' ? 16 : 0)
          ),
          0
        );
        const actual = memory.readFloat32(output.response);
        if (scene === 'incomplete') expect(Array.from(actual).every(Number.isNaN)).toBe(true);
        else
          expectPanelAuditRows(
            actual,
            Array.from(
              data.response,
              (value, row) =>
                value -
                (scene === 'shifted' ? 2 : 1) *
                  responseMeans[effects === 'unit' ? row % 4 : Math.floor(row / 4)] -
                (scene === 'shifted' ? 16 : 0)
            ),
            0
          );
      } finally {
        device.destroy();
      }
    });

for (const shifted of [false, true])
  it(`pooled panel LM CPU diagnostic outputs match independent OLS/projection: affine=${shifted}`, () => {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(createPanelAuditInput(shifted), {
      periodCount: 4,
      predictorCount: 1
    });
    expect(Array.from(result.diagnosticStatus)).toEqual([0]);
    expectPanelAuditRows(result.diagnosticTests, PANEL_DIAGNOSTIC_AUDIT_EXPECTED, 2e-6);
  });

for (const scene of [
  'ordinary',
  'incomplete',
  'rankZero',
  'nearby',
  'scaled',
  'statisticOverflow'
] as const)
  it(`pooled panel LM generated kernels match independent OLS/projection: ${scene}`, () => {
    const incomplete = scene === 'incomplete';
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `lm-audit-${serial++}`, format, length);
      const data = createPanelAuditInput();
      if (scene === 'scaled')
        data.response = Float32Array.from(data.response, value => value * 2 ** 32);
      if (scene === 'rankZero' || scene === 'nearby') {
        data.predictors = Float32Array.from(
          {length: 16},
          (_, row) => (Math.floor(row / 4) + 1) * (row % 2 ? -1 : 1)
        );
        if (scene === 'nearby') data.predictors[0] += 0.25;
      }
      const props = {
        periodCount: 4,
        predictorCount: 1,
        weights: {
          offsets: view('uint32', 5),
          neighbors: view('uint32', 8),
          weights: view('float32', 8),
          status: {candidateOverflow: view('uint32', 1)}
        },
        predictors: view('float32', 16),
        response: view('float32', 16),
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          residuals: view('float32', 16),
          status: {code: view('uint32', 1), converged: view('uint32', 1)}
        },
        diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
      };
      memory.write(props.predictors, data.predictors);
      memory.write(props.response, data.response);
      memory.write(props.weights.offsets, data.weights.offsets);
      memory.write(props.weights.neighbors, data.weights.neighbors);
      memory.write(props.weights.weights, data.weights.weights);
      memory.write(props.weights.status.candidateOverflow, Uint32Array.of(incomplete ? 1 : 0));
      const contributor = new GPUSpatialPanelErrorMaximumLikelihood(props);
      const nodes = getPanelDiagnosticNodes(graph, contributor);
      const lags = nodes.find(node => node.id.endsWith('-diagnostic-lags'))!;
      const tests = nodes.find(node => node.id.endsWith('-diagnostic-tests'))!;
      const lagRecord = getWGSLKernelNodeRecord(graph, lags.id)!;
      const testRecord = getWGSLKernelNodeRecord(graph, tests.id)!;
      const sumRecord = getWGSLKernelNodeRecord(
        graph,
        nodes.find(node => node.id.endsWith('-diagnostic-sums-tiles'))!.id
      )!;
      // NumPy OLS and projection literals from panel-regression-audit.oracle.py. We execute
      // the panel-owned lag/test kernels, supplying independent results for the shared OLS
      // contributor; wgsl_reflect cannot execute its nested array assignments correctly.
      let fitted = [
        -0.876203576341124, 3.82806052269601, 3.82806052269601, 6.96423658872077, 2.25997248968363,
        0.691884456671254, 5.39614855570839, -0.876203576341124, 3.82806052269601, 6.96423658872077,
        -0.876203576341124, 2.25997248968363, 5.39614855570839, -0.876203576341124,
        2.25997248968363, 3.82806052269601
      ];
      let residuals = [
        1.87620357634112, 0.171939477303989, -1.82806052269601, 1.03576341127923, 0.740027510316367,
        -1.69188445667125, 0.60385144429161, 1.87620357634112, 0.171939477303989, -1.96423658872077,
        -1.12379642365888, -0.259972489683633, 1.60385144429161, -2.12379642365888,
        1.74002751031637, -0.828060522696011
      ];
      let projected = [
        1.32435495497882, -0.881105861740794, 3.03911422079015, 0.262066982133429,
        -3.02078031669029, 0.32785334390355, -1.87760747281606, -0.243733078033562,
        2.25507020428396, 0.262066982133429, 0.540310938472627, -1.45269228367791,
        -0.309519439803683, -0.243733078033562, -1.45269228367791, 1.47102618777777
      ];
      const review =
        scene === 'rankZero' || scene === 'nearby' ? reviewDiagnostics[scene] : undefined;
      if (review) {
        fitted = review.fitted;
        residuals = review.residuals;
        projected = review.projected;
      }
      if (scene === 'scaled') {
        fitted = fitted.map(value => value * 2 ** 32);
        residuals = residuals.map(value => value * 2 ** 32);
        projected = projected.map(value => value * 2 ** 32);
      }
      const getView = (name: string) =>
        [...testRecord.bindings, ...sumRecord.bindings].find(binding => binding.name === name)!
          .view;
      memory.write(lagRecord.bindings.find(binding => binding.name === 'fitted')!.view, fitted);
      memory.write(getView('residuals'), residuals);
      memory.write(getView('projected'), projected);
      const summary = new Float32Array(22);
      summary[0] = 16;
      summary[3] =
        scene === 'statisticOverflow'
          ? 1e-30
          : (review?.variance ?? 2.233837689133425) * (scene === 'scaled' ? 2 ** 64 : 1);
      memory.write(getView('summary'), summary);
      memory.write(getView('fitStatus'), Uint32Array.of(0));
      memory.write(getView('projectionStatus'), Uint32Array.of(0));
      executeCommandNodesOnCPU(
        [
          ...nodes.filter(node => node.id.includes('-diagnostic-weights-')),
          lags,
          ...nodes.filter(node => node.id.includes('-diagnostic-sums-')),
          tests,
          ...nodes.filter(node => node.id.includes('-diagnostic-incomplete-'))
        ].map(node => {
          const record = getWGSLKernelNodeRecord(graph, node.id)!;
          return createWGSLKernelNode(graph, {
            ...record,
            id: `${node.id}-interpreter`,
            workgroupSize: record.workgroupSize[0],
            declarations: `fn auditLdexp(value:f32,power:i32)->f32{return value*exp2(f32(power));}\n${(record.declarations ?? '').replace(/\bldexp\(/g, 'auditLdexp(')}`
          });
        }),
        memory
      );
      expect(Array.from(memory.readUint32(props.diagnostics.status))).toEqual([
        incomplete ? 1 : scene === 'rankZero' || scene === 'statisticOverflow' ? 2 : 0
      ]);
      if (incomplete || scene === 'rankZero' || scene === 'statisticOverflow')
        expect(Array.from(memory.readFloat32(props.diagnostics.tests)).every(Number.isNaN)).toBe(
          true
        );
      else
        expectPanelAuditRows(
          memory.readFloat32(props.diagnostics.tests),
          review ? review.tests.map(value => value ?? NaN) : PANEL_DIAGNOSTIC_AUDIT_EXPECTED,
          5e-4
        );
    } finally {
      device.destroy();
    }
  });

it('W5-38 pooled diagnostics use the same variation scale in either period order', () => {
  const input = createPanelAuditInput();
  input.predictors = Float32Array.from(
    {length: 16},
    (_, row) => (Math.floor(row / 4) + 1) * (row % 2 ? -1 : 1)
  );
  input.predictors[0] += 100 * 2 ** -23;
  const reverse = (values: Float32Array) =>
    Float32Array.from({length: 16}, (_, row) => values[(3 - Math.floor(row / 4)) * 4 + (row % 4)]);
  const options = {periodCount: 4, predictorCount: 1};
  const first = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, options);
  const second = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {...input, predictors: reverse(input.predictors), response: reverse(input.response)},
    options
  );
  expect(first.diagnosticStatus[0]).toBe(0);
  expect(second.diagnosticStatus[0]).toBe(0);
  // NumPy least squares / scipy chi2.sf, scratch diagnostics.py; centered rank ratio 1.480297e-12.
  const expected = [
    2.4832931471755644, 0.11506111315877793, 0.1924145021982694, 0.6609146001913988,
    0.19241497912743916, 0.6609142062220347
  ];
  for (let slot = 0; slot < 6; slot++) {
    expect(Number.isFinite(first.diagnosticTests[slot])).toBe(true);
    expect(Number.isFinite(second.diagnosticTests[slot])).toBe(true);
    expect(second.diagnosticTests[slot]).toBeCloseTo(expected[slot], 4);
    expect(first.diagnosticTests[slot]).toBeCloseTo(expected[slot], 4);
  }
});

it('CCK-02 fixed diagnostic clears preserve both incoming-status branches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `affine-${serial++}`, format, length);
    const diagnostics = {tests: view('float32', 6), status: view('uint32', 1)};
    const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
      periodCount: 4,
      predictorCount: 1,
      predictors: view('float32', 16),
      response: view('float32', 16),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      diagnostics,
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const node = nodes.find(node => node.id.endsWith('-affine-reject-diagnostics'))!;
    const record = getWGSLKernelNodeRecord(graph, node.id)!;
    const flags = record.bindings.find(binding => binding.name === 'flags')!.view;
    expect(record.body).not.toMatch(/for\s*\(/);
    for (const [flag, status] of [
      [0, 0],
      [1, 0],
      [1, 1],
      [1, 2]
    ]) {
      memory.write(flags, Uint32Array.of(flag, 0));
      memory.write(diagnostics.status, Uint32Array.of(status));
      memory.write(diagnostics.tests, new Float32Array(6).fill(7));
      executeCommandNodesOnCPU([node], memory);
      const rejected = flag === 1 && status === 0;
      expect(Array.from(memory.readUint32(diagnostics.status))).toEqual([rejected ? 2 : status]);
      expect(
        Array.from(memory.readFloat32(diagnostics.tests)).every(
          rejected ? Number.isNaN : value => value === 7
        )
      ).toBe(true);
    }
  } finally {
    device.destroy();
  }
});
