// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createGWRAuditScene} from './gwr-mathematical-audit-fixtures';
import {createReadableGWRAuditBuffer} from './gwr-audit-buffer-test-utils';
import {GWR_NONSTATIONARITY_AUDIT_CASES} from './gwr-nonstationarity-audit-fixtures';
import {GWR_NONSTATIONARITY_UNDEFINED_CASES} from './gwr-nonstationarity-audit-fixtures';
import {GWR_NONSTATIONARITY_PAIRED_CASE} from './gwr-nonstationarity-audit-fixtures';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

for (const indexed of [false, true]) {
  it(`suppresses inference after real permuted refit failures despite a defined observed spread: grid=${indexed}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      values: readonly number[],
      format: Format
    ) => {
      const buffer = createReadableGWRAuditBuffer(
        device,
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const fixture = GWR_NONSTATIONARITY_PAIRED_CASE;
    const outputs = {
      table: input('table', Array(12).fill(-999), 'float32'),
      summary: input('summary', Array(4).fill(-999), 'float32'),
      standardDeviations: input('distribution', Array(8).fill(-999), 'float32'),
      approximation: input('approximation', [99], 'uint32')
    };
    graph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions: input('positions', fixture.positions, 'float32x2').view,
        predictors: input('predictors', fixture.predictors, 'float32').view,
        predictorCount: 1,
        response: input('response', fixture.response, 'float32').view,
        coefficients: input('coefficients', fixture.coefficients, 'float32').view,
        selectedBandwidth: input('selection', [0, 1], 'float32').view,
        bandwidthParameters: input('bandwidths', [1, 0, 1, 0, 1], 'float32').view,
        parameters: input(
          'parameters',
          Array.from(getGPUPermutationParameterValues({seed: 12345, permutationCount: 2})),
          'uint32'
        ).view,
        maximumBandwidthCount: 1,
        maximumPermutations: 3,
        tileRowCount: 3,
        indexGridSize: indexed ? [4, 1] : false,
        output: {
          table: outputs.table.view,
          summary: outputs.summary.view,
          standardDeviations: outputs.standardDeviations.view,
          status: {approximation: outputs.approximation.view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      for (const [port, expected] of [
        ['table', fixture.table],
        ['summary', fixture.summary],
        ['standardDeviations', fixture.standardDeviations],
        ['approximation', [0]]
      ] as const) {
        const actual =
          port === 'approximation'
            ? await readUint32(outputs[port].buffer)
            : await readFloat32(outputs[port].buffer);
        for (let index = 0; index < expected.length; index++) {
          if (Number.isNaN(expected[index])) expect(actual[index], `${port}[${index}]`).toBeNaN();
          else {
            expect(Number.isFinite(actual[index]), `${port}[${index}]`).toBe(true);
            // Only anchored population spreads of exact two-point fits round in float32.
            expect(Math.abs(actual[index] - expected[index]), `${port}[${index}]`).toBeLessThan(
              1e-5
            );
          }
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const fixture of GWR_NONSTATIONARITY_UNDEFINED_CASES) {
  it(`checks the real finish kernel against incomplete-distribution literals: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      values: readonly number[],
      format: Format
    ) => {
      const buffer = createReadableGWRAuditBuffer(
        device,
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const table = input('table', Array(12).fill(-999), 'float32');
    const summary = input('summary', Array(4).fill(-999), 'float32');
    const distribution = input('distribution', fixture.standardDeviations, 'float32');
    const parameters = Array.from(
      getGPUPermutationParameterValues({seed: 12345, permutationCount: 2})
    );
    if ('permutationCount' in fixture) parameters[2] = fixture.permutationCount;
    const contributor = new GPUGeographicallyWeightedRegressionNonstationarityTest({
      positions: input('positions', Array(16).fill(0), 'float32x2').view,
      predictors: input('predictors', Array(8).fill(0), 'float32').view,
      predictorCount: 1,
      response: input('response', Array(8).fill(0), 'float32').view,
      coefficients: input('coefficients', Array(16).fill(0), 'float32').view,
      bandwidthParameters: input('bandwidths', [0, 0, 1, 0, 3], 'float32').view,
      selectedBandwidth: input('selection', [0, 3], 'float32').view,
      parameters: input('parameters', parameters, 'uint32').view,
      maximumBandwidthCount: 1,
      maximumPermutations: 3,
      indexGridSize: false,
      output: {table: table.view, summary: summary.view, standardDeviations: distribution.view}
    });
    const finish = contributor.getCommandNodes(graph).find(node => node.id.endsWith('-finish'))!;
    const record = getWGSLKernelNodeRecord(graph, finish.id)!;
    const fitCounts = record.bindings.find(binding => binding.name === 'fitCounts')!.view;
    const rows = record.bindings.find(binding => binding.name === 'rows')!.view;
    const initialCounts = input('initial-counts', fixture.fitCounts, 'float32').view;
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'prepare-finish',
        operation: 'GWRNonstationarityAudit',
        invocationCount: 4,
        bindings: [
          {name: 'initialCounts', view: initialCounts, type: 'f32', access: 'read'},
          {name: 'fitCounts', view: fitCounts, type: 'f32', access: 'read_write'},
          {name: 'rows', view: rows, type: 'f32', access: 'read_write'}
        ],
        body: 'fitCounts[fitCountsOffset + index] = initialCounts[initialCountsOffset + index]; if (index == 0u) { rows[rowsOffset + 40u] = 8.0; }'
      })
    );
    graph.add(finish);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      for (const [name, buffer, expected] of [
        ['table', table.buffer, fixture.table],
        ['summary', summary.buffer, fixture.summary]
      ] as const) {
        const actual = await readFloat32(buffer);
        for (let index = 0; index < expected.length; index++) {
          if (Number.isNaN(expected[index])) expect(actual[index], `${name}[${index}]`).toBeNaN();
          else {
            expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
            // Population moments of two exact integer draws: only final f32 divisions round.
            expect(Math.abs(actual[index] - expected[index]), `${name}[${index}]`).toBeLessThan(
              1e-6
            );
          }
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const fixture of GWR_NONSTATIONARITY_AUDIT_CASES) {
  for (const indexed of [false, true]) {
    it(`checks all conditional nonstationarity ports: adaptive=${fixture.adaptive}, transformed=${fixture.transformed}, grid=${indexed}`, async () => {
      await runNonstationarityAudit(fixture, indexed, false);
    });
  }
}

it('counts equal zero spreads as exceedances and writes every unused distribution row', async () => {
  // y=0 gives beta=0 for every full-rank local design, under every permutation. Population SD=0,
  // all five permutations tie, add-one p=1, simulated SD=0, so the z score is undefined.
  await runNonstationarityAudit(GWR_NONSTATIONARITY_AUDIT_CASES[0], false, true);
});

it('counts every failed refit when the selected bandwidth is invalid', async () => {
  // Eight included locations times five permutations = 40 failures; no spread can be defined.
  // The undefined null distribution must suppress inference as well as count failures.
  await runNonstationarityAudit(GWR_NONSTATIONARITY_AUDIT_CASES[0], false, true, true);
});

async function runNonstationarityAudit(
  fixture: (typeof GWR_NONSTATIONARITY_AUDIT_CASES)[number],
  indexed: boolean,
  zeroResponse: boolean,
  invalidSelectedBandwidth = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length = values.length
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const scene = createGWRAuditScene(fixture.transformed);
  if (zeroResponse) scene.response.fill(0);
  const coefficients = Float32Array.from([
    ...(zeroResponse ? new Array<number>(24).fill(0) : fixture.coefficients),
    NaN,
    NaN,
    NaN
  ]);
  const table = output('table', 'float32', 18);
  const summary = output('summary', 'float32', 4);
  const distribution = output('distribution', 'float32', 24);
  const approximation = output('approximation', 'uint32', 1);
  const props = {
    positions: input('positions', scene.positions, 'float32x2', 9),
    predictors: input('predictors', scene.predictors, 'float32'),
    predictorCount: 2,
    response: input('response', scene.response, 'float32'),
    mask: input('mask', scene.mask, 'uint32'),
    coefficients: input('coefficients', coefficients, 'float32'),
    selectedBandwidth: input(
      'selected',
      Float32Array.of(0, invalidSelectedBandwidth ? 0 : fixture.adaptive ? 8 : 3),
      'float32'
    ),
    bandwidthParameters: input(
      'bandwidths',
      getGPUGeographicallyWeightedRegressionParameterValues(
        {
          kernel: fixture.adaptive ? 'bisquare' : 'gaussian',
          bandwidthMode: fixture.adaptive ? 'adaptive' : 'fixed',
          bandwidths: [fixture.adaptive ? 8 : 3]
        },
        1
      ),
      'float32'
    ),
    parameters: input(
      'permutations',
      getGPUPermutationParameterValues({seed: 12345, permutationCount: 5}),
      'uint32'
    ),
    maximumPermutations: 7,
    maximumBandwidthCount: 1,
    tileRowCount: 3,
    indexGridSize: indexed ? ([2, 2] as const) : (false as const),
    gaussianCutoff: indexed && !fixture.adaptive ? 8 : undefined,
    output: {
      table: table.view,
      summary: summary.view,
      standardDeviations: distribution.view,
      status: {approximation: approximation.view}
    }
  };
  const contributor = new GPUGeographicallyWeightedRegressionNonstationarityTest(props);
  props.predictorCount = 1;
  props.maximumPermutations = 1;
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const expectedTable = invalidSelectedBandwidth
      ? Array.from({length: 3}, () => [0, NaN, NaN, NaN, NaN, NaN]).flat()
      : zeroResponse
        ? Array.from({length: 3}, () => [0, 1, 5, 0, 0, NaN]).flat()
        : fixture.table;
    const expectedDistribution = invalidSelectedBandwidth
      ? [0, 0, 0, ...new Array<number>(21).fill(NaN)]
      : [
          ...(zeroResponse ? new Array<number>(18).fill(0) : fixture.standardDeviations),
          ...new Array<number>(6).fill(NaN)
        ];
    for (const [name, actual, expected] of [
      ['table', await readFloat32(table.buffer), expectedTable],
      [
        'summary',
        await readFloat32(summary.buffer),
        invalidSelectedBandwidth ? [5, 8, 8, 40] : fixture.summary
      ],
      ['standardDeviations', await readFloat32(distribution.buffer), expectedDistribution],
      [
        'approximation',
        await readUint32(approximation.buffer),
        [Number(indexed && !fixture.adaptive)]
      ]
    ] as const) {
      for (let index = 0; index < expected.length; index++) {
        if (Number.isNaN(expected[index])) expect(actual[index], `${name}[${index}]`).toBeNaN();
        else {
          expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
          // Float32 local solves followed by anchored spread reductions; 0.1% allows roundoff
          // in small spatial variation. Counts and add-one p values use a much tighter floor.
          const tolerance =
            name === 'summary' || name === 'approximation'
              ? 0
              : name === 'table' && (index % 6 === 1 || index % 6 === 2)
                ? 1e-7
                : 1e-3 * Math.max(0.01, Math.abs(expected[index]));
          expect(
            Math.abs(actual[index] - expected[index]),
            `${name}[${index}]`
          ).toBeLessThanOrEqual(tolerance);
        }
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
