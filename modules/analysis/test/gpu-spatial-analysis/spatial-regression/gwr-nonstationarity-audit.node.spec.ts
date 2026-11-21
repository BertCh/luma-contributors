// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createGWRAuditScene} from './gwr-mathematical-audit-fixtures';
import {GWR_NONSTATIONARITY_AUDIT_CASES} from './gwr-nonstationarity-audit-fixtures';
import {computeNonstationarityOnCPU} from './geographically-weighted-regression-nonstationarity-oracle';
import {
  createTransientView,
  GPUCommandGraph,
  getViewBindingRange,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {GWR_NONSTATIONARITY_UNDEFINED_CASES} from './gwr-nonstationarity-audit-fixtures';
import {GWR_NONSTATIONARITY_PAIRED_CASE} from './gwr-nonstationarity-audit-fixtures';
import {collectClassParameters, renderClassParameters} from '../../analysis-core/doc-parameters';
import {readDocPage} from '../../analysis-core/doc-program';

it('keeps the nonstationarity generated Parameters block synchronized with its source', () => {
  const entry = collectClassParameters().find(
    item => item.className === 'GPUGeographicallyWeightedRegressionNonstationarityTest'
  );
  expect(entry).toBeDefined();
  const page = readDocPage('gpu-spatial-analysis/operators-regression.md');
  const blocks = Array.from(
    page.matchAll(
      /\{\/\* parameters: GPUGeographicallyWeightedRegressionNonstationarityTest [^\n]+\n([\s\S]*?)\{\/\* \/parameters \*\/\}/g
    )
  );
  expect(blocks).toHaveLength(1);
  expect(blocks[0][1].trim()).toBe(renderClassParameters(entry!, 4).trim());
});

it('withholds CPU inference when the observed pairs fit but both permuted designs are singular', () => {
  const fixture = GWR_NONSTATIONARITY_PAIRED_CASE;
  const actual = computeNonstationarityOnCPU(
    {
      positions: Float32Array.from(fixture.positions),
      predictors: Float32Array.from(fixture.predictors),
      response: Float32Array.from(fixture.response),
      predictorCount: 1
    },
    {kernel: 'bisquare', bandwidths: [1]},
    2,
    12345
  );
  for (let column = 0; column < 2; column++) {
    expect(Number.isFinite(actual.standardDeviations[0][column])).toBe(true);
    expect(
      Math.abs(actual.standardDeviations[0][column] - fixture.standardDeviations[column])
    ).toBeLessThan(1e-10);
  }
  expect(actual.standardDeviations.slice(1).flat().every(Number.isNaN)).toBe(true);
  expect(actual.pseudoPValues).toEqual([NaN, NaN]);
  expect(actual.exceedances).toEqual([NaN, NaN]);
});

it('withholds CPU-reference inference when constant predictors make every local system singular', () => {
  // Eight rows, design [1,1] has rank 1 < 2 at every bandwidth and under every relabelling.
  // All observed/permuted spreads are undefined, so no exceedance or p-value is defined.
  const scene = createGWRAuditScene();
  const actual = computeNonstationarityOnCPU(
    {
      positions: scene.positions.slice(0, 16),
      predictors: new Float32Array(8).fill(1),
      predictorCount: 1,
      response: scene.response.slice(0, 8)
    },
    {kernel: 'gaussian', bandwidths: [3]},
    2,
    12345
  );
  expect(actual.standardDeviations.flat().every(Number.isNaN)).toBe(true);
  expect(actual.pseudoPValues).toEqual([NaN, NaN]);
  expect(actual.exceedances).toEqual([NaN, NaN]);
});

it('rejects an invalid CPU-reference permutation count instead of returning a zero-draw p-value', () => {
  const scene = createGWRAuditScene();
  for (const permutations of [0, -1, 1.5, NaN])
    expect(() =>
      computeNonstationarityOnCPU(scene, {bandwidths: [3]}, permutations, 12345)
    ).toThrow(/computeNonstationarityOnCPU.*permutations/);
});

for (const fixture of GWR_NONSTATIONARITY_UNDEFINED_CASES) {
  it(`publishes independent complete-distribution inference in emitted WGSL: ${fixture.name}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const memory = new Map<GraphBufferHandle, ArrayBuffer>();
    let serial = 0;
    const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      values: readonly number[]
    ) => {
      const view = createTransientView(
        graph,
        `finish-audit-${serial++}`,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      );
      memory.set(
        view.buffer,
        (format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)).buffer
      );
      return view;
    };
    const table = port('float32', Array(12).fill(-999));
    const summary = port('float32', Array(4).fill(-999));
    const distribution = port('float32', fixture.standardDeviations);
    const contributor = new GPUGeographicallyWeightedRegressionNonstationarityTest({
      positions: port('float32x2', Array(16).fill(0)),
      predictors: port('float32', Array(8).fill(0)),
      predictorCount: 1,
      response: port('float32', Array(8).fill(0)),
      coefficients: port('float32', Array(16).fill(0)),
      bandwidthParameters: port('float32', [0, 0, 1, 0, 3]),
      maximumBandwidthCount: 1,
      selectedBandwidth: port('float32', [0, 3]),
      parameters: port(
        'uint32',
        Array.from(getGPUPermutationParameterValues({seed: 12345, permutationCount: 2})).map(
          (value, index) =>
            index === 2 && 'permutationCount' in fixture ? fixture.permutationCount : value
        )
      ),
      maximumPermutations: 3,
      indexGridSize: false,
      output: {table, summary, standardDeviations: distribution}
    });
    const finish = contributor.getCommandNodes(graph).find(node => node.id.endsWith('-finish'))!;
    const record = getWGSLKernelNodeRecord(graph, finish.id)!;
    try {
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          if (!memory.has(binding.view.buffer))
            memory.set(binding.view.buffer, new ArrayBuffer(binding.view.buffer.byteLength));
          if (binding.name === 'fitCounts')
            new Float32Array(memory.get(binding.view.buffer)!).set(fixture.fitCounts);
          if (binding.name === 'rows') new Float32Array(memory.get(binding.view.buffer)!)[40] = 8;
          const range = getViewBindingRange(binding.view);
          return [
            index,
            new Uint8Array(memory.get(binding.view.buffer)!, range.offset, range.size)
          ];
        })
      );
      const source = record.source
        .replace(/^\s*_\s*=.*;$/gm, '')
        .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)');
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: record.constants}
      );
      const maximumDifferences = {table: 0, summary: 0};
      for (const [name, view, expected] of [
        ['table', table, fixture.table],
        ['summary', summary, fixture.summary]
      ] as const) {
        const actual = new Float32Array(memory.get(view.buffer)!);
        for (let index = 0; index < expected.length; index++) {
          if (Number.isNaN(expected[index])) expect(actual[index], `${name}[${index}]`).toBeNaN();
          else {
            expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
            maximumDifferences[name] = Math.max(
              maximumDifferences[name],
              Math.abs(actual[index] - expected[index])
            );
            expect(Math.abs(actual[index] - expected[index]), `${name}[${index}]`).toBeLessThan(
              1e-6
            );
          }
        }
      }
      console.info(`Nonstationarity finish audit ${fixture.name}`, maximumDifferences);
    } finally {
      device.destroy();
    }
  });
}

for (const fixture of GWR_NONSTATIONARITY_AUDIT_CASES) {
  it(`matches independent conditional Monte Carlo fits: adaptive=${fixture.adaptive}, transformed=${fixture.transformed}`, () => {
    const scene = createGWRAuditScene(fixture.transformed);
    const actual = computeNonstationarityOnCPU(
      scene,
      {
        kernel: fixture.adaptive ? 'bisquare' : 'gaussian',
        bandwidthMode: fixture.adaptive ? 'adaptive' : 'fixed',
        bandwidths: [fixture.adaptive ? 8 : 3]
      },
      5,
      12345
    );
    const differences = {standardDeviations: 0, pseudoPValues: 0, exceedances: 0};
    for (const [index, value] of actual.standardDeviations.flat().entries()) {
      expect(Number.isFinite(value)).toBe(true);
      differences.standardDeviations = Math.max(
        differences.standardDeviations,
        Math.abs(value - fixture.standardDeviations[index])
      );
      expect(Math.abs(value - fixture.standardDeviations[index])).toBeLessThan(1e-10);
    }
    expect(actual.pseudoPValues).toEqual([fixture.table[1], fixture.table[7], fixture.table[13]]);
    expect(actual.exceedances).toEqual([fixture.table[2], fixture.table[8], fixture.table[14]]);
    console.info('Nonstationarity audit maximum absolute differences', differences);
  });
}
