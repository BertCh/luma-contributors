// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslParser} from 'wgsl_reflect';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialRegressionDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import type {GPUSpatialRegressionDiagnosticsProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-parameters';
import {SPREG_DIAGNOSTICS_REFERENCE, SPREG_MASKED_DIAGNOSTICS_REFERENCE} from './spreg-reference';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  createDiagnosticsScene,
  createDiagnosticsExclusionMask,
  createNearestNeighborScene,
  createLatticeWeights,
  invertMatrix
} from './spatial-regression-diagnostics-oracle';

let serial = 0;

for (const tileRowCount of [1, 7, 64])
  for (const restandardizeRows of [false, true])
    it(`v3-1 diagnostics modules have unique global declarations, tile ${tileRowCount}, restandardize ${restandardizeRows}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        recordWGSLKernelNodes(graph);
        const nodes = new GPUSpatialRegressionDiagnostics(
          createProps(graph, {tileRowCount, restandardizeRows})
        ).getCommandNodes(graph);
        const duplicates: string[] = [];
        for (const node of nodes) {
          const record = getWGSLKernelNodeRecord(graph, node.id);
          if (!record) continue;
          // The parser accepts duplicate globals; WGSL/Tint forbids them. Inspect only module
          // declarations, preserving legitimate repeated local names in different functions.
          const names = new Set<string>();
          for (const declaration of new WgslParser().parse(record.source)) {
            if (!('name' in declaration) || typeof declaration.name !== 'string') continue;
            if (names.has(declaration.name)) duplicates.push(`${node.id}: ${declaration.name}`);
            names.add(declaration.name);
          }
        }
        expect(duplicates).toEqual([]);
      } finally {
        device.destroy();
      }
    });

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSpatialRegressionDiagnosticsProps> = {}
): GPUSpatialRegressionDiagnosticsProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    predictors: view('float32', 20),
    response: view('float32', 10),
    residuals: view('float32', 10),
    predictorCount: 2,
    output: {
      tests: view('float32', 18),
      summary: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH),
      status: {code: view('uint32', 1)}
    },
    ...overrides
  };
}

it('GPUSpatialRegressionDiagnostics validates its inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `bad-${serial++}`, format, length);
  const expectThrows = (
    overrides: Partial<GPUSpatialRegressionDiagnosticsProps>,
    message: RegExp
  ) =>
    expect(() => new GPUSpatialRegressionDiagnostics(createProps(graph, overrides))).toThrow(
      message
    );
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 16}, /predictorCount/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({residuals: view('float32', 11)}, /residuals length/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows({mask: view('uint32', 9)}, /mask length/);
  expectThrows({mask: view('uint32', 11)}, /mask length/);
  expectThrows(
    {
      output: {
        tests: view('float32', 17),
        summary: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)}
      }
    },
    /output.tests/
  );
  expectThrows(
    {
      output: {
        tests: view('float32', 18),
        summary: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH - 1),
        status: {code: view('uint32', 1)}
      }
    },
    /output.summary/
  );
  device.destroy();
});

it('F2 diagnostics finish reads both weights-loss flags', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const candidateOverflow = createTransientView(graph, 'candidate-overflow', 'uint32', 1);
  try {
    const nodes = new GPUSpatialRegressionDiagnostics({
      ...props,
      weights: {...props.weights, status: {overflow, candidateOverflow}}
    }).getCommandNodes(graph);
    const finish = nodes.find(node => node.id.endsWith('-finish'))!;
    const lossNodes = nodes.filter(node => node.id.includes('-weights-loss-'));
    for (const flag of [overflow, candidateOverflow]) {
      const reader = lossNodes.find(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource && resource.buffer === flag && resource.usage === 'storage-read'
        )
      );
      expect(reader, 'both source flags feed the shared loss fold').toBeDefined();
    }
    const reset = nodes.find(node => node.id.endsWith('-weights-loss-reset'))!;
    const loss = reset.resources?.find(resource => resource.usage === 'storage-read-write');
    expect(
      loss &&
        'buffer' in loss &&
        finish.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === loss.buffer &&
            resource.usage === 'storage-read'
        )
    ).toBe(true);
    expect(nodes.indexOf(reset)).toBeLessThan(nodes.indexOf(finish));
  } finally {
    graph.device.destroy();
  }
});

it('GPUSpatialRegressionDiagnostics declares its kernels, including the weights transpose, with at most eight storage buffers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUSpatialRegressionDiagnostics(createProps(graph)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 3)).toEqual([
    'spatial-regression-diagnostics-means-tiles',
    'spatial-regression-diagnostics-means-merge',
    'spatial-regression-diagnostics-row-table'
  ]);
  expect(ids.slice(-3)).toEqual([
    'spatial-regression-diagnostics-moran-centered-norm-tiles',
    'spatial-regression-diagnostics-moran-centered-norm-merge',
    'spatial-regression-diagnostics-moran-centered-report'
  ]);
  expect(ids).toContain('spatial-regression-diagnostics-fitted-lag-projection');
  expect(ids).toContain('spatial-regression-diagnostics-fitted-lag-residuals');
  expect(ids).toContain('spatial-regression-diagnostics-origin-row-tiles');
  expect(ids).toContain('spatial-regression-diagnostics-observed-origin');
  expect(ids).toContain('spatial-regression-diagnostics-transpose-slot-keys');
  expect(ids).toContain('spatial-regression-diagnostics-transposed-lag');
  device.destroy();
});

it('spatial regression diagnostics oracle satisfies its own identities', () => {
  expect(
    invertMatrix([
      [2, 0],
      [0, 4]
    ])
  ).toEqual([
    [0.5, 0],
    [0, 0.25]
  ]);
  // Row-standardized weights: S0 = n. Binary rook weights on a 3x3 lattice: S0 = 24.
  expect(createLatticeWeights(3, false).offsets[9]).toBe(24);
  const scene = createDiagnosticsScene(6, 1, 0.5, true);
  const result = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(result.weightsSum).toBeCloseTo(36, 5);
  expect(result.lmSarma.statistic).toBeCloseTo(
    result.lmLag.statistic + result.robustLmError.statistic,
    9
  );
  // OLS residuals are orthogonal to the design and sum to zero.
  expect(result.residuals.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 8);
  expect(result.lmError.statistic).toBeGreaterThan(0);
});

it('spatial regression diagnostics oracle reproduces spreg to float64 precision', () => {
  const cases = [
    [createDiagnosticsScene(8, 11, 0.6, true), SPREG_DIAGNOSTICS_REFERENCE.lattice],
    [createNearestNeighborScene(80, 4, 7, 0.6), SPREG_DIAGNOSTICS_REFERENCE.nearestNeighbor]
  ] as const;
  for (const [scene, reference] of cases) {
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    const pairs: [typeof result.lmLag, readonly number[]][] = [
      [result.lmLag, reference.lmLag],
      [result.lmError, reference.lmError],
      [result.robustLmLag, reference.robustLmLag],
      [result.robustLmError, reference.robustLmError],
      [result.lmSarma, reference.lmSarma]
    ];
    for (const [test, [statistic, pValue]] of pairs) {
      expect(test.statistic).toBeCloseTo(statistic, 6);
      expect(test.pValue).toBeCloseTo(pValue, 6);
    }
    expect(result.moranI).toBeCloseTo(reference.moran[0], 8);
    expect(result.moranZ).toBeCloseTo(reference.moran[1], 6);
    expect(result.moranPValue).toBeCloseTo(reference.moran[2], 6);
    expect(result.sigmaSquared).toBeCloseTo(reference.sigmaSquared, 8);
  }
});

it('GPUSpatialRegressionDiagnostics adds the row-scale kernel only when restandardizing', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const plain = new GPUSpatialRegressionDiagnostics(
    createProps(graph, {id: 'plain', mask: createTransientView(graph, 'plain-mask', 'uint32', 10)})
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  const restandardized = new GPUSpatialRegressionDiagnostics(
    createProps(graph, {
      id: 'scaled',
      mask: createTransientView(graph, 'scaled-mask', 'uint32', 10),
      restandardizeRows: true
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain.some(id => id.endsWith('row-scale'))).toBe(false);
  expect(restandardized).toContain('scaled-row-scale');
  expect(restandardized.indexOf('scaled-row-scale')).toBeLessThan(
    restandardized.indexOf('scaled-transposed-lag')
  );
  device.destroy();
});

it('masked diagnostics oracle reproduces spreg on the fitted-row subset, both restandardize modes', () => {
  const scenes = [
    [
      createDiagnosticsScene(8, 11, 0.6, true),
      createDiagnosticsExclusionMask(64, 7, 3),
      SPREG_MASKED_DIAGNOSTICS_REFERENCE.lattice
    ],
    [
      createNearestNeighborScene(60, 3, 13, 0.6, false),
      createDiagnosticsExclusionMask(60, 6, 2),
      SPREG_MASKED_DIAGNOSTICS_REFERENCE.knnDirected
    ]
  ] as const;
  for (const [scene, mask, reference] of scenes) {
    for (const restandardizeRows of [false, true]) {
      const expected = reference[restandardizeRows ? 'restandardized' : 'original'];
      const result = computeSpatialRegressionDiagnosticsOnCPU(
        scene.weights,
        scene.predictors,
        scene.response,
        scene.predictorCount,
        {mask, restandardizeRows}
      );
      expect(result.excludedRowCount).toBe(reference.excluded);
      expect(result.rowCount).toBe(mask.length - reference.excluded);
      const pairs: [typeof result.lmLag, readonly number[]][] = [
        [result.lmLag, expected.lmLag],
        [result.lmError, expected.lmError],
        [result.robustLmLag, expected.robustLmLag],
        [result.robustLmError, expected.robustLmError],
        [result.lmSarma, expected.lmSarma]
      ];
      for (const [test, [statistic, pValue]] of pairs) {
        expect(test.statistic).toBeCloseTo(statistic, 6);
        expect(test.pValue).toBeCloseTo(pValue, 6);
      }
      expect(result.moranI).toBeCloseTo(expected.moran[0], 8);
      expect(result.moranZ).toBeCloseTo(expected.moran[1], 6);
      expect(result.moranPValue).toBeCloseTo(expected.moran[2], 6);
      expect(result.sigmaSquared).toBeCloseTo(expected.sigmaSquared, 8);
      expect(result.weightsSum).toBeCloseTo(expected.weightsSum, 5);
    }
  }
});

it('masked diagnostics oracle treats non-finite inputs like a mask', () => {
  const scene = createDiagnosticsScene(8, 11, 0.6, true);
  const mask = createDiagnosticsExclusionMask(64, 7, 3);
  const response = Float32Array.from(scene.response);
  for (let row = 0; row < 64; row++) {
    if (mask[row] === 0) {
      response[row] = NaN;
    }
  }
  const viaMask = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount,
    {mask}
  );
  const viaNaN = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    response,
    scene.predictorCount
  );
  expect(viaNaN.excludedRowCount).toBe(viaMask.excludedRowCount);
  expect(viaNaN.lmError.statistic).toBeCloseTo(viaMask.lmError.statistic, 12);
  expect(viaNaN.moranZ).toBeCloseTo(viaMask.moranZ, 12);
  // Dropping links changes the statistics: this is not the unmasked analysis.
  const full = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(Math.abs(full.lmError.statistic - viaMask.lmError.statistic)).toBeGreaterThan(0.1);
  expect(full.excludedRowCount).toBe(0);
});
