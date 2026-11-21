// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LR-D4 conformance of weights consumers (SSR-101, SSR-102, SSR-105 of the 2026-11-19 review):
// weights whose `candidateOverflow` (or `overflow`) is set must suppress the consumer's result,
// and clearing the flag must preserve existing bytes. The guard unit tests use synthetic values;
// the full category-pair graph test below separately proves restoration by actual ordinary writers.
// The suppression nodes
// run on the CPU WGSL executor; the expectations are the documented sentinels (NaN for float
// results, 0 for counts), not a second run of the same code.

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov';
import {GPUSegregation} from '../../../src/gpu-spatial-analysis/segregation/gpu-segregation';
import {getGPUSegregationLayout} from '../../../src/gpu-spatial-analysis/segregation/segregation-layout';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {
  GPULocalCategoryDependence,
  GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS
} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {GPUCategoryPairCounts} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-category-pair-counts';
import {
  GPUSpatialEntropy,
  GPU_SPATIAL_ENTROPY_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-spatial-entropy';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH,
  getGPULocalOutlierFactorParameterValues
} from '../../../src/gpu-spatial-analysis/outlier-detection/outlier-detection-parameters';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {GPULocalMoran} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/global-spatial-statistics-layout';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const ROWS = 4;
const CATEGORIES = 2;

type Scene = ReturnType<typeof createScene>;

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `completeness-${serial++}`, format, length);
  const overflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const weights = {
    offsets: view('uint32', ROWS + 1),
    neighbors: view('uint32', ROWS * 2),
    weights: view('float32', ROWS * 2),
    distances: view('float32', ROWS * 2),
    status: {overflow, candidateOverflow}
  };
  return {device, graph, view, overflow, candidateOverflow, weights};
}

/** A consumer, the views that must be suppressed, and views that must stay untouched. */
type Case = {
  nodes: readonly {id: string}[];
  /** Whole views, or `[view, firstIndex, length]` ranges, expected NaN / 0 under loss. */
  suppressed: (GraphDataView | readonly [GraphDataView, number, number])[];
  kept: GraphDataView[];
  prepare?: (memory: ReturnType<typeof createCPUGraphMemory>) => void;
  lossSources?: readonly {view: GraphDataView; incomplete: number; complete: number}[];
};

const ORDINARY_FLOAT = 2.5;
const ORDINARY_UINT = 7;

function fillOrdinary(memory: ReturnType<typeof createCPUGraphMemory>, view: GraphDataView) {
  memory.write(
    view,
    view.format === 'float32'
      ? new Float32Array(view.length).fill(ORDINARY_FLOAT)
      : new Uint32Array(view.length).fill(ORDINARY_UINT)
  );
}

function readValues(memory: ReturnType<typeof createCPUGraphMemory>, view: GraphDataView) {
  return Array.from(
    view.format === 'float32' ? memory.readFloat32(view) : memory.readUint32(view)
  ) as number[];
}

function expectSetAndClear(scene: Scene, name: string, consumer: Case): void {
  const {graph, overflow, candidateOverflow} = scene;
  const lossSources =
    consumer.lossSources ??
    [overflow, candidateOverflow].map(view => ({
      view,
      incomplete: 1,
      complete: 0
    }));
  for (const {view: flag} of lossSources) {
    expect(
      consumer.nodes.some(node =>
        getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
          binding => binding.view === flag && binding.access === 'read'
        )
      ),
      `${name} must read ${flag === overflow ? 'overflow' : 'candidateOverflow'}`
    ).toBe(true);
  }
  const guards = consumer.nodes.filter(
    node => node.id.includes('-incomplete') || node.id.endsWith('-check-completeness')
  );
  expect(guards.length, `${name} suppression nodes`).toBeGreaterThan(0);
  const memory = createCPUGraphMemory();
  consumer.prepare?.(memory);
  const whole = (entry: Case['suppressed'][number]) => ('buffer' in entry ? entry : entry[0]);
  const views = [...consumer.suppressed.map(whole), ...consumer.kept];
  for (const active of lossSources) {
    // Each loss cause alone, including the opposite polarity of converged.
    for (const source of lossSources) memory.write(source.view, Uint32Array.of(source.complete));
    memory.write(active.view, Uint32Array.of(active.incomplete));
    views.forEach(view => fillOrdinary(memory, view));
    executeCommandNodesOnCPU(guards, memory);
    for (const view of new Set(consumer.suppressed.map(whole))) {
      const values = readValues(memory, view);
      values.forEach((value, index) => {
        const inside = consumer.suppressed.some(entry => {
          const [first, length] = 'buffer' in entry ? [0, view.length] : [entry[1], entry[2]];
          return whole(entry) === view && index >= first && index < first + length;
        });
        const expected = !inside
          ? view.format === 'float32'
            ? ORDINARY_FLOAT
            : ORDINARY_UINT
          : view.format === 'float32'
            ? Number.NaN
            : 0;
        expect(value, `${name} ${view.buffer.id}[${index}] under loss`).toBe(expected);
      });
    }
    for (const view of consumer.kept) {
      expect(readValues(memory, view), `${name} ${view.buffer.id} kept`).toEqual(
        new Array(view.length).fill(view.format === 'float32' ? ORDINARY_FLOAT : ORDINARY_UINT)
      );
    }
    // Clear: the ordinary writers' values stay in place.
    memory.write(active.view, Uint32Array.of(active.complete));
    views.forEach(view => fillOrdinary(memory, view));
    executeCommandNodesOnCPU(guards, memory);
    for (const view of views) {
      const values = readValues(memory, view);
      expect(values.every(Number.isFinite), `${name} ${view.buffer.id} finite`).toBe(true);
      expect(values, `${name} ${view.buffer.id} cleared guard preserves bytes`).toEqual(
        new Array(view.length).fill(view.format === 'float32' ? ORDINARY_FLOAT : ORDINARY_UINT)
      );
    }
  }
}

it('SSR-101 GPUSpatialMarkov passes the loss flags to its lag and suppresses its outputs', () => {
  const scene = createScene();
  const {view, weights} = scene;
  try {
    const periods = 3;
    const output = {
      counts: view('uint32', 2 * CATEGORIES ** 2),
      probabilities: view('float32', 2 * CATEGORIES ** 2),
      rowTotals: view('uint32', 2 * CATEGORIES),
      ignored: view('uint32', 1),
      lagBreaks: view('float32', 3),
      lagClassCount: view('uint32', 1),
      lagValues: view('float32', ROWS * periods)
    };
    const markov = new GPUSpatialMarkov({
      values: view('float32', ROWS * periods),
      classes: view('uint32', ROWS * periods),
      classCount: CATEGORIES,
      weights,
      periodCount: periods,
      maximumLagClassCount: 2,
      lagParameters: view('float32', 16),
      output
    });
    const nodes = markov.getCommandNodes(scene.graph);
    // Every period's lag carries its own suppression of the lag column.
    for (let period = 0; period < periods; period++) {
      expect(
        nodes.some(node => node.id.startsWith(`spatial-markov-lag-${period}-incomplete`))
      ).toBe(true);
    }
    expectSetAndClear(scene, 'GPUSpatialMarkov', {
      nodes: nodes.filter(node => !node.id.startsWith('spatial-markov-lag-')),
      suppressed: [
        output.counts,
        output.probabilities,
        output.rowTotals,
        output.lagBreaks,
        output.lagClassCount
      ],
      kept: [output.ignored]
    });
  } finally {
    scene.device.destroy();
  }
});

it('AMG-59 complete CSR writes independent category counts and entropy', () => {
  const scene = createScene();
  const {graph, view, weights, overflow, candidateOverflow} = scene;
  try {
    const categories = view('uint32', 4);
    const pairCounts = view('uint32', 4),
      categoryRowCounts = view('uint32', 2);
    const nodes = new GPUCategoryPairCounts({
      weights: {...weights, status: undefined},
      categories,
      categoryCount: 2,
      output: {pairCounts, categoryRowCounts}
    }).getCommandNodes(graph);
    const summary = view('float32', GPU_SPATIAL_ENTROPY_LAYOUT.length);
    const entropyPairs = view('uint32', 4);
    const entropyNodes = new GPUSpatialEntropy({
      weights: {...weights, status: undefined},
      categories,
      categoryCount: 2,
      output: {summary, pairCounts: entropyPairs}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2, 3, 4));
    memory.write(weights.neighbors, Uint32Array.of(1, 0, 3, 2));
    memory.write(weights.weights, Float32Array.of(1, 1, 1, 1));
    memory.write(categories, Uint32Array.of(0, 0, 1, 1));
    memory.write(overflow, Uint32Array.of(0));
    memory.write(candidateOverflow, Uint32Array.of(0));
    executeCommandNodesOnCPU([...nodes, ...entropyNodes], memory);
    // Directed pairs 0<->1 are category 0, and 2<->3 category 1: two equiprobable
    // pairs, so H(origin)=H(destination)=H(pair)=ln(2), MI=ln(2), relative MI=1.
    expect(Array.from(memory.readUint32(pairCounts))).toEqual([2, 0, 0, 2]);
    expect(Array.from(memory.readUint32(entropyPairs))).toEqual([2, 0, 0, 2]);
    expect(Array.from(memory.readUint32(categoryRowCounts))).toEqual([2, 2]);
    const expected = [Math.LN2, Math.LN2, Math.LN2, Math.LN2, Math.LN2, 1, 4, 4];
    const actual = Array.from(memory.readFloat32(summary));
    expect(actual).toHaveLength(expected.length);
    actual.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(expected[index], 6);
    });
  } finally {
    scene.device.destroy();
  }
});

it('SSR-101 GPUSpatialMarkov lag suppression keeps one pipeline per kernel across periods', () => {
  const count = (periods: number) => {
    const scene = createScene();
    const {view, weights} = scene;
    try {
      const nodes = new GPUSpatialMarkov({
        values: view('float32', ROWS * periods),
        classes: view('uint32', ROWS * periods),
        classCount: CATEGORIES,
        weights,
        periodCount: periods,
        maximumLagClassCount: 2,
        lagParameters: view('float32', 16),
        output: {
          counts: view('uint32', 2 * CATEGORIES ** 2),
          lagBreaks: view('float32', 3),
          lagClassCount: view('uint32', 1)
        }
      }).getCommandNodes(scene.graph);
      return getGPUCommandNodePipelineKeys(nodes).size;
    } finally {
      scene.device.destroy();
    }
  };
  expect(count(4)).toBe(count(3));
});

it.each([
  'environment',
  'smoothed-population'
] as const)('SSR-102 GPUSegregation suppresses only dependent columns of the last scale (%s)', spatialForm => {
  const scene = createScene();
  const {view, weights} = scene;
  try {
    const {stride, diversity} = getGPUSegregationLayout(CATEGORIES);
    const output = {
      indices: view('float32', 3 * stride),
      localEnvironment: view('float32', 3 * ROWS * CATEGORIES),
      localEntropy: view('float32', 3 * ROWS),
      localDissimilarity: view('float32', 3 * ROWS * CATEGORIES),
      localTheil: view('float32', 3 * ROWS)
    };
    const groupCounts = view('float32', ROWS * CATEGORIES);
    const segregation = new GPUSegregation({
      groupCount: CATEGORIES,
      groupCounts,
      spatialForm,
      scales: [null, {...weights, status: undefined}, weights],
      output
    });
    const nodes = segregation.getCommandNodes(scene.graph);
    expect(nodes.length).toBeLessThanOrEqual(segregation.estimatedCommandNodeCount);
    expectSetAndClear(scene, 'GPUSegregation', {
      nodes,
      prepare(memory) {
        memory.write(groupCounts, Float32Array.of(8, 2, 2, 8, 8, 2, 2, 8));
        memory.write(weights.offsets, Uint32Array.of(0, 1, 2, 3, 4));
        memory.write(weights.neighbors, Uint32Array.of(1, 0, 3, 2));
        memory.write(weights.weights, Float32Array.of(1, 1, 1, 1));
        // Run the actual environment writers and cursor initialization, then exercise the
        // final guard with synthetic canaries so an incorrect block write is visible.
        executeCommandNodesOnCPU(
          nodes.filter(
            node =>
              node.id.startsWith('segregation-parameters') ||
              node.id.endsWith('-scale-cursor-clear') ||
              node.id.endsWith('-environment')
          ),
          memory
        );
        const guard = nodes.find(node => node.id.endsWith('-scale-2-incomplete'))!;
        const ordinal = getWGSLKernelNodeRecord(scene.graph, guard.id)!.bindings.find(
          binding => binding.name === 'scaleOrdinal'
        )!.view;
        expect(Array.from(memory.readUint32(ordinal))).toEqual([2]);
      },
      // Earlier scale blocks stay intact. Environment-form diversity depends only on the
      // original population, so that cell remains intact even at the affected scale.
      suppressed: [
        ...(spatialForm === 'environment'
          ? ([
              [output.indices, 2 * stride, diversity],
              [output.indices, 2 * stride + diversity + 1, stride - diversity - 1]
            ] as const)
          : ([[output.indices, 2 * stride, stride]] as const)),
        [output.localEnvironment, 2 * ROWS * CATEGORIES, ROWS * CATEGORIES],
        [output.localEntropy, 2 * ROWS, ROWS],
        [output.localDissimilarity, 2 * ROWS * CATEGORIES, ROWS * CATEGORIES],
        [output.localTheil, 2 * ROWS, ROWS]
      ],
      kept: []
    });
  } finally {
    scene.device.destroy();
  }
});

it('SSR-102 GPUColocationQuotient suppresses its quotients and keeps the row counts', () => {
  const scene = createScene();
  const {view, weights} = scene;
  try {
    const output = {
      globalQuotients: view('float32', CATEGORIES ** 2),
      localQuotients: view('float32', ROWS * CATEGORIES),
      categoryRowCounts: view('uint32', CATEGORIES)
    };
    const nodes = new GPUColocationQuotient({
      weights,
      categories: view('uint32', ROWS),
      categoryCount: CATEGORIES,
      output
    }).getCommandNodes(scene.graph);
    expectSetAndClear(scene, 'GPUColocationQuotient', {
      nodes,
      suppressed: [output.globalQuotients, output.localQuotients],
      kept: [output.categoryRowCounts]
    });
  } finally {
    scene.device.destroy();
  }
});

it('SSR-102 GPULocalCategoryDependence suppresses its statistics', () => {
  const scene = createScene();
  const {view, weights} = scene;
  try {
    const output = {
      statistics: view('float32', ROWS * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length),
      categoryRowCounts: view('uint32', CATEGORIES)
    };
    const nodes = new GPULocalCategoryDependence({
      weights,
      categories: view('uint32', ROWS),
      categoryCount: CATEGORIES,
      output
    }).getCommandNodes(scene.graph);
    expectSetAndClear(scene, 'GPULocalCategoryDependence', {
      nodes,
      suppressed: [output.statistics],
      kept: [output.categoryRowCounts]
    });
  } finally {
    scene.device.destroy();
  }
});

for (const producer of ['counts', 'entropy'] as const) {
  it.each([
    'overflow',
    'candidateOverflow',
    'requiredCount',
    'complete',
    'converged',
    'iterationLimitReached'
  ] as const)(`AMG-59 ${producer} rejects a standalone %s descriptor, even zero words`, name => {
    const scene = createScene();
    const {view, weights} = scene;
    try {
      const output = {
        pairCounts: view('uint32', CATEGORIES ** 2),
        categoryRowCounts: view('uint32', CATEGORIES)
      };
      const memory = createCPUGraphMemory();
      for (const word of [0, 1]) {
        const flag = view('uint32', 1);
        memory.write(flag, Uint32Array.of(word));
        const props = {
          weights: {...weights, status: {[name]: flag}},
          categories: view('uint32', ROWS),
          categoryCount: CATEGORIES
        };
        if (producer === 'counts') {
          expect(() => new GPUCategoryPairCounts({...props, output}), name).toThrow(
            /weights.status/
          );
        } else {
          expect(
            () =>
              new GPUSpatialEntropy({
                ...props,
                output: {summary: view('float32', GPU_SPATIAL_ENTROPY_LAYOUT.length)}
              }),
            name
          ).toThrow(/weights.status/);
        }
      }
    } finally {
      scene.device.destroy();
    }
  });
}

it('SSR-102 GPULocalOutlierFactor suppresses scores and flags no outlier', () => {
  const scene = createScene();
  const {view, weights} = scene;
  try {
    const iterationLimitReached = view('uint32', 1);
    const converged = view('uint32', 1);
    const complete = view('uint32', 1);
    const requiredCount = view('uint32', 1);
    const output = {
      kDistance: view('float32', ROWS),
      localReachabilityDensity: view('float32', ROWS),
      lof: view('float32', ROWS),
      outlier: view('uint32', ROWS),
      outlierCount: view('uint32', 1)
    };
    const nodes = new GPULocalOutlierFactor({
      neighbors: {
        ...weights,
        status: {...weights.status, iterationLimitReached, converged, complete, requiredCount}
      },
      parameters: view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH),
      output
    }).getCommandNodes(scene.graph);
    expectSetAndClear(scene, 'GPULocalOutlierFactor', {
      nodes,
      lossSources: [
        ...[scene.overflow, scene.candidateOverflow, iterationLimitReached].map(view => ({
          view,
          incomplete: 1,
          complete: 0
        })),
        {view: converged, incomplete: 0, complete: 1},
        {view: complete, incomplete: 0, complete: 1},
        {
          view: requiredCount,
          incomplete: weights.neighbors.length + 1,
          complete: weights.neighbors.length
        }
      ],
      suppressed: [
        output.kDistance,
        output.localReachabilityDensity,
        output.lof,
        output.outlier,
        output.outlierCount
      ],
      kept: []
    });
  } finally {
    scene.device.destroy();
  }
});

it.each([
  'complete',
  'requiredCount'
] as const)('review B1/S3 LOF ordinary writers restore scores after standalone %s loss', name => {
  const scene = createScene();
  const {graph, view} = scene;
  try {
    const evidence = view('uint32', 1);
    const neighbors = {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 3),
      weights: view('float32', 3),
      distances: view('float32', 3),
      status: {[name]: evidence}
    };
    const parameters = view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH);
    const output = {
      kDistance: view('float32', 3),
      localReachabilityDensity: view('float32', 3),
      lof: view('float32', 3),
      outlier: view('uint32', 3),
      outlierCount: view('uint32', 1)
    };
    const nodes = new GPULocalOutlierFactor({neighbors, parameters, output}).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(neighbors.offsets, Uint32Array.of(0, 1, 2, 3));
    memory.write(neighbors.neighbors, Uint32Array.of(1, 0, 1));
    memory.write(neighbors.weights, Float32Array.of(1, 1, 1));
    memory.write(neighbors.distances, Float32Array.of(1, 1, 2));
    memory.write(
      parameters,
      getGPULocalOutlierFactorParameterValues({threshold: 1.5, densityFloor: 0.5})
    );
    const clean = name === 'complete' ? 1 : 3;
    const loss = name === 'complete' ? 0 : 4;
    for (const word of [clean, loss, clean]) {
      memory.write(evidence, Uint32Array.of(word));
      // Do not refill outputs between submissions: recovery must overwrite the previous NaNs.
      executeCommandNodesOnCPU(nodes, memory);
      if (word === loss) {
        for (const result of [output.kDistance, output.localReachabilityDensity, output.lof]) {
          expect(Array.from(memory.readFloat32(result))).toEqual([NaN, NaN, NaN]);
        }
        expect(Array.from(memory.readUint32(output.outlier))).toEqual([0, 0, 0]);
        expect(Array.from(memory.readUint32(output.outlierCount))).toEqual([0]);
      } else {
        // Reachabilities [1,1,2], densities [2/3,2/3,2/5]; row 2 LOF=(2/3)/(2/5)=5/3.
        for (const [result, expected] of [
          [output.kDistance, [1, 1, 2]],
          [output.localReachabilityDensity, [2 / 3, 2 / 3, 2 / 5]],
          [output.lof, [1, 1, 5 / 3]]
        ] as const) {
          const actual = Array.from(memory.readFloat32(result));
          expect(actual).toHaveLength(expected.length);
          actual.forEach((value, index) => {
            expect(Number.isFinite(value)).toBe(true);
            expect(value).toBeCloseTo(expected[index], 6);
          });
        }
        expect(Array.from(memory.readUint32(output.outlier))).toEqual([0, 0, 1]);
        expect(Array.from(memory.readUint32(output.outlierCount))).toEqual([1]);
      }
    }
  } finally {
    scene.device.destroy();
  }
});

it('SSR-105 chunked transpose declares statistics-compatible ports and aggregates their loss', () => {
  const scene = createScene();
  const {graph, view} = scene;
  try {
    const inputOverflow = view('uint32', 1);
    const input: GPUChunkedSpatialWeights = {
      rowCount: ROWS,
      status: {overflow: inputOverflow},
      chunks: [0, 2].map(rowStart => ({
        rowStart,
        rowCount: 2,
        weights: {
          offsets: view('uint32', 3),
          neighbors: view('uint32', 4),
          weights: view('float32', 4)
        }
      }))
    };
    const chunkStatus = [0, 1].map(() => ({
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1)
    }));
    const transpose = new GPUChunkedSpatialWeightsTranspose({
      weights: input,
      output: {
        chunks: chunkStatus.map((status, index) => ({
          rowStart: index * 2,
          rowCount: 2,
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 4),
            weights: view('float32', 4),
            status
          }
        }))
      }
    });
    const transposed = transpose.outputWeights;
    // Before the fix each construction threw "loss flags must be aggregated on the outer
    // weights.status port".
    const moranOutput = {
      zScores: view('float32', ROWS),
      quadrants: view('uint32', ROWS),
      neighborCounts: view('uint32', ROWS)
    };
    const moran = new GPULocalMoran({
      weights: transposed,
      values: view('float32', ROWS),
      parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
      output: moranOutput
    });
    expect(
      () =>
        new GPUGlobalSpatialStatistics({
          mode: 'reference',
          weights: transposed,
          values: view('float32', ROWS),
          statistics: ['moran'],
          output: {summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)}
        })
    ).not.toThrow();
    const transposeNodes = transpose.getCommandNodes(graph);
    expect(transposeNodes.length).toBeLessThanOrEqual(transpose.estimatedCommandNodeCount);
    const moranNodes = moran.getCommandNodes(graph);
    const outerOverflow = transposed.status!.overflow!;
    const outerCandidate = transposed.status!.candidateOverflow!;
    for (const flag of [outerOverflow, outerCandidate]) {
      expect(
        moranNodes.some(node =>
          getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(binding => binding.view === flag)
        )
      ).toBe(true);
    }
    // Set and clear on the aggregation: a chunk overflow is overflow, inherited input loss is
    // candidateOverflow (a larger transpose output cannot recover it).
    const aggregate = transposeNodes.filter(node => node.id.includes('-aggregate-'));
    const memory = createCPUGraphMemory();
    const run = (chunkFlags: [number, number], inputFlag: number) => {
      chunkStatus.forEach((status, index) =>
        memory.write(status.overflow, Uint32Array.of(chunkFlags[index]))
      );
      memory.write(inputOverflow, Uint32Array.of(inputFlag));
      memory.write(outerOverflow, Uint32Array.of(9));
      memory.write(outerCandidate, Uint32Array.of(9));
      executeCommandNodesOnCPU(aggregate, memory);
      return [memory.readUint32(outerOverflow)[0], memory.readUint32(outerCandidate)[0]];
    };
    expect(run([0, 1], 0)).toEqual([1, 0]);
    expect(run([0, 0], 1)).toEqual([0, 1]);
    expect(run([0, 0], 0)).toEqual([0, 0]);
  } finally {
    scene.device.destroy();
  }
});
