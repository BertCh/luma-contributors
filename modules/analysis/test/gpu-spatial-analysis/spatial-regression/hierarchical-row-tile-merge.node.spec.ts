// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getPairwiseSumWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/pairwise-sum-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  getRowTileMergeScratchLength,
  getRowTileSumNodes,
  getRowTileMergeNodes,
  REGRESSION_TILE_MAXIMUM_TILE_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterLength} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const [tileCount, groupCount] of [
  [4096, 0],
  [4097, 2],
  [8192, 2],
  [8193, 3]
] as const) {
  it(`declares exactly ${groupCount} group partials per channel at ${tileCount} tiles`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const nodes = getRowTileSumNodes(graph, {
        id: 'sum',
        operation: 'TileMergeTest',
        variant: 'sum',
        rowCount: tileCount,
        layout: {tileRowCount: 1, tileCount},
        sumCount: 1,
        maximumCount: 1,
        bindings: [],
        declarations:
          'fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) { (*terms)[0] += f32(row); (*terms)[1] = max((*terms)[1], f32(row)); }',
        partial: createTransientView(graph, 'partial', 'float32', tileCount * 2),
        output: createTransientView(graph, 'total', 'float32', 2)
      });
      expect(getRowTileMergeScratchLength(tileCount, 2)).toBe(groupCount * 2);
      expect(nodes.map(node => node.id)).toEqual(
        groupCount === 0
          ? ['sum-tiles', 'sum-merge']
          : ['sum-tiles', 'sum-merge-tilePartial-groups', 'sum-merge']
      );
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const merge = kernels.find(kernel => kernel.id === 'sum-merge')!;
      expect(merge.source).toContain('override TILE_COUNT: u32;');
      if (groupCount > 0) {
        const groups = kernels.find(kernel => kernel.id.endsWith('-groups'))!;
        expect(groups.source).toContain(`min(firstTile + 4096u, ${tileCount}u)`);
        expect(groups.source).toContain('if (term == 1u)');
      }
    } finally {
      device.destroy();
    }
  });
}

// Execute the generated kernels through the existing float32 WGSL CPU interpreter. This is
// implementation execution, compared with literals from the independent NumPy oracle, rather
// than a second implementation of the merge used to manufacture its own expectations.
for (const [tileCount, expected, bits] of [
  [4096, 1049429.125, 1233132201],
  [4097, 853.125, 1146439680],
  [8192, 1706.25, 1154828288],
  [8193, 1707.25, 1154836480]
] as const) {
  it(`executes the ${tileCount}-tile shader merge bit for bit against NumPy float32`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const partial = createTransientView(graph, 'partial', 'float32', tileCount * 3);
      const total = createTransientView(graph, 'total', 'float32', 3);
      const memory = createCPUGraphMemory();
      const values = new Float32Array(tileCount * 3);
      for (let tile = 0; tile < tileCount; tile++) {
        values[tile * 3] =
          tile === 0
            ? 2 ** 20
            : tile === 4096
              ? -(2 ** 20)
              : tile === 8192
                ? 1
                : tile % 3 === 0
                  ? 0.125
                  : 0.25;
        values[tile * 3 + 1] = tile === tileCount - 1 ? 77 : 1;
        values[tile * 3 + 2] = tile - tileCount;
      }
      memory.write(partial, values);
      const nodes = getRowTileMergeNodes(
        graph,
        {
          id: 'numeric',
          operation: 'TileMergeNumericTest',
          invocationCount: 3,
          bindings: [
            {name: 'partial', view: partial, type: 'f32', access: 'read'},
            {name: 'total', view: total, type: 'f32', access: 'read_write'}
          ],
          declarations: `const TILE_COUNT: u32 = ${tileCount}u;`,
          body: `if (index == 0u) {
          ${getPairwiseSumWGSL({count: 'TILE_COUNT', term: 'partial[partialOffset + sumItem * 3u]'})}
          total[totalOffset] = sum;
        } else {
          var largest = select(0.0, -3.0e38, index == 2u);
          for (var tile = 0u; tile < TILE_COUNT; tile++) { largest = max(largest, partial[partialOffset + tile * 3u + index]); }
          total[totalOffset + index] = largest;
        }`
        },
        {tileCount, partials: [{name: 'partial', termCount: 3, maxima: {1: 0, 2: -3e38}}]}
      );
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readFloat32(total))).toEqual([expected, 77, -1]);
      expect(memory.readUint32(total)[0]).toBe(bits);
    } finally {
      device.destroy();
    }
  });
}

for (const blockWidth of [1, 2, 6]) {
  it(`executes every blocked channel at width ${blockWidth} against NumPy`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const tileCount = 8193;
      const termCount = 6;
      const blockStride = blockWidth === 1 ? 8256 : tileCount * blockWidth;
      const partial = createTransientView(
        graph,
        'partial',
        'float32',
        (termCount / blockWidth) * blockStride
      );
      const total = createTransientView(graph, 'total', 'float32', termCount);
      const values = new Float32Array(partial.length).fill(Number.NaN);
      for (let term = 0; term < termCount; term++)
        for (let tile = 0; tile < tileCount; tile++) {
          const value =
            tile === 0
              ? 2 ** 20
              : tile === 4096
                ? -(2 ** 20)
                : tile === 8192
                  ? 1
                  : tile % 3 === 0
                    ? 0.125
                    : 0.25;
          values[
            Math.floor(term / blockWidth) * blockStride + tile * blockWidth + (term % blockWidth)
          ] = value * (term + 1);
        }
      const memory = createCPUGraphMemory();
      memory.write(partial, values);
      const nodes = getRowTileMergeNodes(
        graph,
        {
          id: 'blocked',
          operation: 'TileMergeLayoutTest',
          invocationCount: termCount,
          bindings: [
            {name: 'partial', view: partial, type: 'f32', access: 'read'},
            {name: 'total', view: total, type: 'f32', access: 'read_write'}
          ],
          declarations: `const TILE_COUNT: u32 = ${tileCount}u;\nconst SUMS_STRIDE: u32 = ${blockStride}u;`,
          body: `${getPairwiseSumWGSL({count: 'TILE_COUNT', term: `partial[partialOffset + (index / ${blockWidth}u) * ${blockWidth === 1 ? 'SUMS_STRIDE' : `TILE_COUNT * ${blockWidth}u`} + sumItem * ${blockWidth}u + index % ${blockWidth}u]`})}
          total[totalOffset + index] = sum;`
        },
        {
          tileCount,
          partials: [{name: 'partial', termCount, blockWidth, blockStride}],
          ...(blockWidth === 1 ? {strideConstants: ['SUMS_STRIDE']} : {})
        }
      );
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readFloat32(total))).toEqual([
        1707.25, 3414.5, 5122, 6829, 8536, 10244
      ]);
    } finally {
      device.destroy();
    }
  });
}

it('rejects more than the two-level tile limit before declaring merge scratch', () => {
  expect(REGRESSION_TILE_MAXIMUM_TILE_COUNT).toBe(16777216);
  expect(getRowTileMergeScratchLength(16777216, 3)).toBe(12288);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      getRowTileMergeNodes(
        graph,
        {
          id: 'too-many',
          operation: 'TileMergeLimitTest',
          invocationCount: 1,
          bindings: [],
          body: ''
        },
        {tileCount: 16777217, partials: []}
      )
    ).toThrow(/too-many tileCount must be at most 16777216/);
  } finally {
    device.destroy();
  }
});

for (const kind of [
  'gm',
  'two-stage',
  'traces',
  'error-ml',
  'lag-ml',
  'nonstationarity'
] as const) {
  it(`${kind} accepts 8193 one-row tiles and groups its private merges`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const rows = 8193;
      const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `view-${serial++}`, format, length);
      const weights = {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      };
      const props = {
        id: kind,
        weights,
        predictors: view('float32', rows),
        response: view('float32', rows),
        predictorCount: 1,
        tileRowCount: 1,
        maximumPower: 2,
        traceProbeCount: 1,
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)}
        }
      };
      const model =
        kind === 'gm'
          ? new GPUSpatialErrorGM({...props, estimator: 'kelejian-prucha'})
          : kind === 'two-stage'
            ? new GPUSpatialTwoStageLeastSquares(props)
            : kind === 'error-ml'
              ? new GPUSpatialErrorMaximumLikelihood(props)
              : kind === 'lag-ml'
                ? new GPUSpatialLagMaximumLikelihood(props)
                : kind === 'traces'
                  ? new GPUSpatialWeightsTraces({
                      id: kind,
                      weights,
                      tileRowCount: 1,
                      maximumPower: 2,
                      probeCount: 1,
                      includeGramTraces: true,
                      output: {
                        powerTraces: view('float32', 2),
                        powerSums: view('float32', 2),
                        gramTraces: view('float32', 4),
                        infinityNorm: view('float32', 1)
                      }
                    })
                  : new GPUGeographicallyWeightedRegressionNonstationarityTest({
                      id: kind,
                      positions: view('float32x2', rows),
                      predictors: props.predictors,
                      response: props.response,
                      predictorCount: 1,
                      coefficients: view('float32', rows * 2),
                      bandwidthParameters: view(
                        'float32',
                        getGPUGeographicallyWeightedRegressionParameterLength()
                      ),
                      selectedBandwidth: view('float32', 2),
                      parameters: view('uint32', 4),
                      maximumPermutations: 1,
                      tileRowCount: 1,
                      output: {table: view('float32', 12), summary: view('float32', 4)}
                    });
      const nodes = model.getCommandNodes(graph);
      if (kind === 'gm' || kind === 'lag-ml') {
        const childMerges = getGPUCommandNodeKernelSources(nodes).filter(
          kernel => kernel.id.includes('ols-') && /-(means|moments|residual)-merge$/.test(kernel.id)
        );
        // Each child OLS now merges means/moments twice (fit plus BP), raw residuals,
        // and standardized residuals: six phases. Lag ML has two child OLS fits.
        expect(childMerges).toHaveLength(kind === 'gm' ? 6 : 12);
        expect(
          childMerges.filter(kernel => kernel.id.includes('standardized-residual'))
        ).toHaveLength(kind === 'gm' ? 1 : 2);
        for (const child of childMerges)
          expect(child.source.includes('const TILE_COUNT: u32 = 8193u;'), child.id).toBe(false);
      }
      expect(nodes.some(node => node.id.endsWith('-groups'))).toBe(true);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      const groups = kernels.filter(kernel => kernel.id.endsWith('-groups'));
      expect(groups.length).toBeGreaterThan(0);
      for (const group of groups) expect(group.source).toContain('min(firstTile + 4096u, 8193u)');
      // Every private final merge now reads 3 groups, including norm and block-major Gram paths.
      const finalMerges = kernels.filter(
        kernel =>
          !kernel.id.endsWith('-groups') && kernel.source.includes('const TILE_COUNT: u32 = 3u;')
      );
      expect(finalMerges.length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}
