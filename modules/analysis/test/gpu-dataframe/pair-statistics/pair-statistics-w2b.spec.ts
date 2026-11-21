// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {
  GPUSpaceTimeDiggleTest,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {GPUCoregionalizationFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';
import {GPURipley} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {createPairStatisticsWindowHarness} from './pair-statistics-harness';
import {PAIR_STATISTICS_BOUNDARY_SECOND_POINT} from './pair-statistics-boundary-fixture';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: number[] | Float32Array
  ) => {
    const buffer = createInputBuffer(
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
  const output = <Format extends 'float32' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x4' ? 4 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  return {graph, input, output, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

for (const Contributor of [GPUSpaceTimeKFunction, GPUSpaceTimeDiggleTest]) {
  it(`PTS-133 ${Contributor.name} keeps construction mappings and reads live buffer contents`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createContext(device);
    const {graph, input, output} = context;
    const k = output('k', 'float32', 1);
    const times = input('times', 'float32', [4, 5]);
    const props = {
      positions: input('positions', 'float32x2', [4, 4, 5, 4]).view,
      times: times.view,
      parameters: input(
        'parameters',
        'float32',
        getGPUSpaceTimeKFunctionParameterValues({
          bounds: [0, 0, 10, 10],
          timeWindow: [0, 10],
          maximumDistance: 2,
          maximumTimeLag: 2
        })
      ).view,
      randomParameters: input('random', 'uint32', [1, 0]).view,
      gridSize: [2, 2] as [number, number],
      distanceClassCount: 1,
      timeClassCount: 1,
      permutationCount: 1,
      edgeCorrection: 'none' as const,
      output: {
        k: k.view,
        d: output('d', 'float32', 1).view,
        statistics: output('statistics', 'float32', 2).view,
        summary: output('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH).view
      }
    };
    const constructionProps =
      Contributor === GPUSpaceTimeKFunction ? {...props, output: {k: k.view}} : props;
    const contributor =
      Contributor === GPUSpaceTimeKFunction
        ? new GPUSpaceTimeKFunction(constructionProps)
        : new GPUSpaceTimeDiggleTest(props);
    constructionProps.gridSize[0] = 99;
    constructionProps.distanceClassCount = 64;
    constructionProps.output.k = output('replacement-k', 'float32', 1).view;
    graph.add(contributor);
    const compiled = graph.compile();
    try {
      // Two directed pairs: |W||T|/(2*1) * 2 = 1000.
      for (const [secondTime, expectedK] of [
        [5, 1000],
        [9, 0],
        [5, 1000]
      ]) {
        times.buffer.write(Float32Array.of(4, secondTime));
        submitGraph(device, compiled, undefined);
        const actual = (await readFloat32(k.buffer, 1))[0];
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expectedK, 3);
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}

for (const [name, positions, expectedSpatialK] of [
  ['interior', [4, 5, 5, 5], 100],
  ['signed-zero side', [-0, 5, 1, 5], 150],
  ['signed-zero corner', [-0, -0, 1, -0], 300]
] as const) {
  it(`W2-01 Ripley and both space-time estimators retain ordered ${name} weights`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createContext(device);
    const {graph, input, output} = context;
    const positionView = input('positions', 'float32x2', [...positions]).view;
    const ripleyK = output('ripley-k', 'float32', 1);
    const k = output('space-time-k', 'float32', 1);
    const diggleK = output('diggle-k', 'float32', 1);
    const counts = output('ripley-counts', 'uint32', 1);
    const spaceTimeCounts = output('space-time-counts', 'uint32', 1);
    graph.add(
      new GPURipley({
        positions: positionView,
        gridSize: [2, 2],
        radiusCount: 1,
        parameters: input(
          'ripley-parameters',
          'float32',
          getGPURipleyParameterValues({
            bounds: [0, 0, 10, 10],
            maximumDistance: 2,
            edgeCorrection: 'isotropic'
          })
        ).view,
        output: {k: ripleyK.view, pairCounts: counts.view}
      })
    );
    const props = {
      positions: positionView,
      times: input('times', 'float32', [5, 5]).view,
      gridSize: [2, 2] as const,
      distanceClassCount: 1,
      timeClassCount: 1,
      parameters: input(
        'space-time-parameters',
        'float32',
        getGPUSpaceTimeKFunctionParameterValues({
          bounds: [0, 0, 10, 10],
          maximumDistance: 2,
          maximumTimeLag: 2,
          timeWindow: [0, 10]
        })
      ).view
    };
    graph.add(
      new GPUSpaceTimeKFunction({...props, output: {k: k.view, pairCounts: spaceTimeCounts.view}})
    );
    graph.add(
      new GPUSpaceTimeDiggleTest({
        ...props,
        permutationCount: 1,
        randomParameters: input('random', 'uint32', [1, 0]).view,
        output: {
          k: diggleK.view,
          d: output('d', 'float32', 1).view,
          statistics: output('statistics', 'float32', 2).view,
          summary: output('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH).view
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, 1)).toEqual([2]);
      expect(await readUint32(spaceTimeCounts.buffer, 1)).toEqual([2]);
      // Forward/reverse weights: interior 1+1, side 2+1, corner 4+2.
      // K = area/(n(n-1))*sum(weights); space-time adds |T|=10, temporal weight=1.
      for (const [result, expected] of [
        [ripleyK, expectedSpatialK],
        [k, expectedSpatialK * 10],
        [diggleK, expectedSpatialK * 10]
      ] as const) {
        const actual = (await readFloat32(result.buffer, 1))[0];
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 4);
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}

it('W2-01 spatial sharing preserves each space-time estimator temporal and maximum-lag ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input, output} = context;
  const settings = {
    bounds: [0, 0, 10, 10] as const,
    timeWindow: [0, 10] as const,
    maximumDistance: 2,
    maximumTimeLag: 2
  };
  const parameters = input(
    'parameters',
    'float32',
    getGPUSpaceTimeKFunctionParameterValues(settings)
  );
  const props = {
    positions: input('positions', 'float32x2', [4, 4, 5, 4]).view,
    times: input('times', 'float32', [1, 2]).view,
    parameters: parameters.view,
    gridSize: [2, 2] as const,
    distanceClassCount: 1,
    timeClassCount: 1
  };
  const k = output('k', 'float32', 1);
  const diggleK = output('diggle-k', 'float32', 1);
  const counts = output('counts', 'uint32', 1);
  graph.add(new GPUSpaceTimeKFunction({...props, output: {k: k.view, pairCounts: counts.view}}));
  graph.add(
    new GPUSpaceTimeDiggleTest({
      ...props,
      permutationCount: 1,
      randomParameters: input('random', 'uint32', [1, 0]).view,
      output: {
        k: diggleK.view,
        d: output('d', 'float32', 1).view,
        statistics: output('statistics', 'float32', 2).view,
        summary: output('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH).view
      }
    })
  );
  const compiled = graph.compile();
  try {
    // d=1, lag=1. The first focus exactly reaches timeStart: stpp weights 1+1;
    // splancs weights 2+1. With scale |W||T|/2=500 these give K=1000 and 1500.
    // stpp includes ties at either maximum; splancs drops them on either axis.
    for (const [maximumDistance, maximumTimeLag, expectedDiggleK] of [
      [2, 2, 1500],
      [1, 2, 0],
      [2, 1, 0],
      [2, 2, 1500]
    ]) {
      parameters.buffer.write(
        getGPUSpaceTimeKFunctionParameterValues({...settings, maximumDistance, maximumTimeLag})
      );
      submitGraph(device, compiled, undefined);
      expect(await readUint32(counts.buffer, 1)).toEqual([2]);
      for (const [result, expected] of [
        [k, 1000],
        [diggleK, expectedDiggleK]
      ] as const) {
        const actual = (await readFloat32(result.buffer, 1))[0];
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBeCloseTo(expected, 3);
      }
    }
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

it('CORE-106 public fit reaches the extreme rotation then suppresses its overflowing residual', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input, output} = context;
  const matrices = output('matrices', 'float32', 4);
  const loadings = output('loadings', 'float32', 4);
  const diagnostics = output('diagnostics', 'float32x4', 1);
  const status = {
    validity: output('validity', 'uint32', 1),
    iterationCount: output('iterations', 'uint32', 1),
    converged: output('converged', 'uint32', 1),
    iterationLimitReached: output('limit', 'uint32', 1),
    invalidCount: output('invalid', 'uint32', 1)
  };
  const semivariances = input('semivariances', 'float32', [-3e38, 1e37, 3e38]);
  const fit = new GPUCoregionalizationFit({
    variableCount: 2,
    structureCount: 1,
    maximumIterations: 1,
    distances: input('distances', 'float32', [1]).view,
    pairCounts: input('counts', 'uint32', [1]).view,
    semivariances: semivariances.view,
    parameters: input(
      'parameters',
      'float32',
      getGPUCoregionalizationFitParameterValues([{model: 'spherical', range: 1, sill: 1}])
    ).view,
    output: {
      matrices: matrices.view,
      loadings: loadings.view,
      diagnostics: diagnostics.view,
      status: {
        validity: status.validity.view,
        iterationCount: status.iterationCount.view,
        converged: status.converged.view,
        iterationLimitReached: status.iterationLimitReached.view,
        invalidCount: status.invalidCount.view
      }
    }
  });
  recordWGSLKernelNodes(graph);
  const projectedMatrices = output('projected-matrices', 'float32', 4);
  const projectedLoadings = output('projected-loadings', 'float32', 4);
  graph.add({
    getCommandNodes: () => {
      const nodes = fit.getCommandNodes(graph);
      const sweep = getWGSLKernelNodeRecord(graph, `${fit.id}-sweep-0`)!;
      const state = sweep.bindings.find(binding => binding.name === 'state')!.view;
      const matrixOffset = /const MATRIX: u32 = (\d+)u;/.exec(sweep.source)![1];
      const loadingOffset = /const LOADING: u32 = (\d+)u;/.exec(sweep.source)![1];
      // Observe the real public fit's scratch without replacing any of its arithmetic.
      // publish-fit suppresses public outputs but retains the projected state for this probe.
      return [
        ...nodes,
        createWGSLKernelNode(graph, {
          id: 'observe-projection',
          operation: 'CoregionalizationProjectionTest',
          invocationCount: 4,
          bindings: [
            {name: 'state', view: state, type: 'f32', access: 'read'},
            {name: 'matrices', view: projectedMatrices.view, type: 'f32', access: 'read_write'},
            {name: 'loadings', view: projectedLoadings.view, type: 'f32', access: 'read_write'}
          ],
          body: `matrices[matricesOffset + index] = state[stateOffset + ${matrixOffset}u + index];
loadings[loadingsOffset + index] = state[stateOffset + ${loadingOffset}u + index];`
        })
      ];
    }
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // One legal bin, normalized count=1, shape=1: Gram=1, target=the lag matrix.
    // The projection is finite, but its squared residual >= (3e38)^2 overflows f32
    // in publish-fit. Invalid bins remain zero; one completed sweep proves the gate ran.
    for (const [name, expected] of [
      ['validity', 0],
      ['iterationCount', 1],
      ['converged', 0],
      ['iterationLimitReached', 0],
      ['invalidCount', 0]
    ] as const) {
      expect(await readUint32(status[name].buffer, 1), name).toEqual([expected]);
    }
    expect((await readFloat32(matrices.buffer, 4)).every(Number.isNaN)).toBe(true);
    expect((await readFloat32(loadings.buffer, 4)).every(Number.isNaN)).toBe(true);
    const diagnostic = await readFloat32(diagnostics.buffer, 4);
    expect(diagnostic.slice(0, 2).every(Number.isNaN)).toBe(true);
    expect(diagnostic.slice(2)).toEqual([1, 1]);

    // Binary64 projection: A_+=(A+sqrt(9.01)*I)/2 for normalized A=[[-3,.1],[.1,3]].
    // This observes the extreme rotation itself, before the residual's public suppression.
    const expected = [0.0008331019803633, 0.05, 0.05, 3.0008331019803633];
    const projected = await readFloat32(projectedMatrices.buffer, 4);
    const projectedFactors = await readFloat32(projectedLoadings.buffer, 4);
    for (let row = 0; row < 2; row++)
      for (let column = 0; column < 2; column++) {
        const index = row * 2 + column;
        const reconstructed =
          projectedFactors[row * 2] * projectedFactors[column * 2] +
          projectedFactors[row * 2 + 1] * projectedFactors[column * 2 + 1];
        expect(Number.isFinite(projected[index])).toBe(true);
        expect(Number.isFinite(reconstructed)).toBe(true);
        expect(Math.abs(projected[index] / 1e38 - expected[index])).toBeLessThan(2e-5);
        expect(Math.abs(reconstructed / 1e38 - expected[index])).toBeLessThan(2e-5);
      }

    // Same closed-form PSD projection at a scale with a representable squared residual.
    // A=[[-3,.1],[.1,3]], lambda=sqrt(9.01); A_+=(A+lambda*I)/2.
    // Binary64 entries below are independent of either eigensolver.
    semivariances.buffer.write(Float32Array.of(-3e18, 1e17, 3e18));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.validity.buffer, 1)).toEqual([1]);
    const actual = await readFloat32(matrices.buffer, 4);
    const factors = await readFloat32(loadings.buffer, 4);
    for (let row = 0; row < 2; row++)
      for (let column = 0; column < 2; column++) {
        const index = row * 2 + column;
        expect(Number.isFinite(actual[index])).toBe(true);
        expect(Math.abs(actual[index] / 1e18 - expected[index])).toBeLessThan(2e-5);
        const reconstructed =
          factors[row * 2] * factors[column * 2] + factors[row * 2 + 1] * factors[column * 2 + 1];
        expect(Number.isFinite(reconstructed)).toBe(true);
        expect(Math.abs(reconstructed / 1e18 - expected[index])).toBeLessThan(2e-5);
      }
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

it('CORE-105 Ripley counts both boundary members before applying the vertex and reverse weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPairStatisticsWindowHarness(device, {
    window: {
      positions: Float64Array.of(0, 0, -1, 0, 0, -1),
      polygonOffsets: Uint32Array.of(0, 1),
      ringOffsets: Uint32Array.of(0, 3)
    },
    positions: Float32Array.of(0, 0, ...PAIR_STATISTICS_BOUNDARY_SECOND_POINT),
    parameterLength: 8,
    outputs: {k: {format: 'float32', length: 1}, pairCounts: {format: 'uint32', length: 1}},
    createContributor: (views, studyWindow) =>
      new GPURipley({
        positions: views.positions,
        parameters: views.parameters,
        studyWindow,
        gridSize: [2, 2],
        radiusCount: 1,
        output: {k: views.outputs.k as never, pairCounts: views.outputs.pairCounts as never}
      })
  });
  try {
    // d=sqrt(2)/3. At (0,0) the circle has fraction 1/4 (weight 4).
    // At (-1/3,-1/3) the axis cuts have half-angle pi/4 each; the disjoint
    // hypotenuse cut has half-angle pi/3: fraction 1/6 (reverse weight 6).
    // Area=1/2, n=2: K_none=1/2; K_iso=(1/2)/(2*1)*(4+6)=5/2.
    // A bounding-rectangle substitution would give reverse weight 2 and K=3/2.
    for (const [maximumDistance, edgeCorrection, expectedCount, expectedK] of [
      [0.5, 'none', 2, 0.5],
      [0.5, 'isotropic', 2, 2.5],
      [0.25, 'isotropic', 0, 0]
    ] as const) {
      const actual = await harness.run(
        getGPURipleyParameterValues({bounds: [-1, -1, 0, 0], maximumDistance, edgeCorrection})
      );
      expect(actual.pairCounts).toEqual([expectedCount]);
      expect(Number.isFinite(actual.k[0])).toBe(true);
      expect(actual.k[0]).toBeCloseTo(expectedK, 4);
    }
  } finally {
    harness.destroy();
  }
});

for (const gridSize of [undefined, [1, 1] as const, [3, 2] as const]) {
  it(`DPC-123 cross-variogram uses hand-listed bins on ${JSON.stringify(gridSize)} lattice`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createContext(device);
    const {graph, input, output} = context;
    const semivariances = output('semivariances', 'float32', 2);
    const pairCounts = output('pair-counts', 'uint32', 2);
    const status = output('status', 'uint32', 2);
    graph.add(
      new GPUCrossVariogram({
        positions: input('positions', 'float32x2', [0, 0, 1, 0, 3, 0, 2, 0]).view,
        values: input('values', 'float32', [0, 2, 6, 1000]).view,
        secondValues: input('second-values', 'float32', [0, 1, 3, -1000]).view,
        mask: input('mask', 'uint32', [1, 1, 1, 0]).view,
        parameters: input('parameters', 'float32', [4]).view,
        lagCount: 2,
        gridSize,
        output: {
          semivariances: semivariances.view,
          pairCounts: pairCounts.view,
          status: status.view
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // 01: d=1, product/2=1; 02: d=3, product/2=9; 12: d=2, product/2=4.
      // The distance-2 tie starts bin 1; excluded row 3 contributes nothing.
      expect(await readUint32(pairCounts.buffer, 2)).toEqual([1, 2]);
      expect(await readUint32(status.buffer, 2)).toEqual([0, 0]);
      const actual = await readFloat32(semivariances.buffer, 2);
      for (const [index, expected] of [1, 6.5].entries()) {
        expect(Number.isFinite(actual[index])).toBe(true);
        expect(actual[index]).toBeCloseTo(expected, 5);
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}
