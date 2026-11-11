// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getDenseLUWGSL} from '../../src/analysis-core/kernels/dense-solve-wgsl';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {GPUNaturalNeighbor} from '../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {
  GPURasterPhenologyMetrics,
  GPU_RASTER_PHENOLOGY_METRIC,
  GPU_RASTER_PHENOLOGY_METRIC_COUNT,
  getGPURasterPhenologyMetricsParameterValues
} from '../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// A shader-compiler fault of the macOS test GPU (fix review fr-fields-05, found in
// GPUNetworkLouvain): a local `var table: array<u32, N>` whose only stores are those of a loop
// with a constant trip count (here an identity loop, `table[i] = i`), read at an index the
// compiler cannot resolve, reads back as zeros for some sizes. This file pins the fault in a
// kernel of one binding built with `createWGSLKernelNode`, and then runs three kernels of the
// library that hold or held such a table (found by a search of modules/analysis/src on
// 2026-11-10) at the sizes that fail, each against an expectation that does not come from the
// kernel.
//
// Closing pass (2026-11-11): the cause is the Apple GPU back end, any stored value is lost, and
// `var<private>` is NOT a cure: the browser emits a private variable as the same zero-initialised
// local, and the reduced shader loses it at the same sizes
// (`test/analysis-core/shader-compiler-table-fill.spec.ts`). The private table of the kernel
// below reads right, which is an observation about this kernel and is asserted as such. The
// cures are a literal initializer, a module-scope `const`, or a fill loop bounded by a run-time
// value. `test/analysis-core/shader-compiler-table-shapes.spec.ts` runs the old Louvain kernel
// (lost from 12 entries) and `test/analysis-core/table-audit/` the kernels of the library audit.

const SIZES = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 96, 128];

type TableShape = 'function' | 'private' | 'function-float' | 'function-literal';

/**
 * Reads an identity table by a loop index inside a loop nest. The table is filled by an identity
 * loop, or for `'function-literal'` declared with a literal initializer.
 */
async function readIdentityTable(size: number, shape: TableShape): Promise<number[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const isFloat = shape === 'function-float';
  const element = isFloat ? 'f32' : 'u32';
  const declaration = `var${shape === 'private' ? '<private>' : ''} order: array<${element}, ${size}>;`;
  const identity = Array.from({length: size}, (_, entry) => `${entry}u`).join(', ');
  const fill =
    shape === 'function-literal'
      ? `var order = array<u32, ${size}>(${identity});`
      : `${shape === 'private' ? '' : declaration}
  for (var entry = 0u; entry < ${size}u; entry++) { order[entry] = ${isFloat ? 'f32(entry)' : 'entry'}; }`;
  const words = await runWGSLModuleKernel(device, {
    id: `identity-table-${shape}-${size}`,
    inputs: {},
    outputLength: size,
    invocationCount: 1,
    declarations: shape === 'private' ? declaration : '',
    body: `${fill}
  var groups: array<u32, ${size}>;
  for (var entry = 0u; entry < ${size}u; entry++) { groups[entry] = ${size - 1}u - entry; }
  for (var round = 0u; round < 2u; round++) {
    for (var visit = 0u; visit < ${size}u; visit++) {
      let group = ${isFloat ? 'u32(order[visit])' : 'order[visit]'};
      var present = false;
      for (var member = 0u; member < ${size}u; member++) { if (groups[member] == group) { present = true; } }
      if (!present) { continue; }
      output[outputOffset + visit] = group;
    }
  }`
  });
  return Array.from(words);
}

it('a function-scope identity table reads back as zeros at some sizes; one with a literal initializer never does', async () => {
  const wrongSizes: number[] = [];
  for (const size of SIZES) {
    const expected = Array.from({length: size}, (_, entry) => entry);
    // A literal initializer: right at every size, because no loop stores to the table. This is
    // the accepted form.
    expect(await readIdentityTable(size, 'function-literal'), `literal, ${size} entries`).toEqual(
      expected
    );
    // Module scope with the same identity loop reads right in THIS kernel. It is not a cure: the
    // reduced shader of shader-compiler-table-fill.spec.ts loses a private table at the sizes
    // at which it loses a function-scope one.
    expect(await readIdentityTable(size, 'private'), `private, ${size} entries`).toEqual(expected);
    // A float identity table (the shape of the phenology time axis) is not affected here.
    expect(await readIdentityTable(size, 'function-float'), `float, ${size} entries`).toEqual(
      expected
    );
    const values = await readIdentityTable(size, 'function');
    if (!values.every((value, entry) => value === entry)) {
      // When it is wrong it is all zeros, not a partial table.
      expect(values, `function scope, ${size} entries`).toEqual(new Array(size).fill(0));
      wrongSizes.push(size);
    }
  }
  // Measured on the test GPU: 40, 48, 56, 64, 72 and 96 entries read back as zeros in this
  // kernel (from 12 entries up inside the old Louvain kernel). If a compiler update clears the
  // list, the rule in contributor contract section 11 can be revisited; until then a table
  // whose contents are known when the shader is built gets a literal initializer.
  expect(wrongSizes).toContain(64);
});

const LU_CAPACITIES = [8, 12, 16, 24, 32, 40, 48, 56, 64];

it('the dense LU pivot table (identity, then exchanges) is right from 8 to 64 unknowns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of LU_CAPACITIES) {
    const random = createSeededRandom(1000 + capacity);
    // Case 0: diagonally dominant, so no row is exchanged and the pivot table stays the identity
    // it was initialised to. Case 1: the same rows in reverse order, so every pivot is an
    // exchange. Each with 0 and 2 equilibration passes (the scale tables stay at their constant
    // fill with 0 passes).
    const dominant = Array.from({length: capacity}, (_, row) =>
      Array.from({length: capacity}, (_, column) =>
        row === column ? capacity + 1 + random() : random() - 0.5
      )
    );
    const solution = Array.from({length: capacity}, () => 2 * random() - 1);
    const getRightSide = (matrix: number[][]): number[] =>
      matrix.map(row =>
        row.reduce((sum, value, column) => sum + Math.fround(value) * solution[column], 0)
      );
    const reversed = [...dominant].reverse();
    const cases = [
      {matrix: dominant, passes: 0},
      {matrix: dominant, passes: 2},
      {matrix: reversed, passes: 0},
      {matrix: reversed, passes: 2}
    ];
    const square = capacity * capacity;
    const words = await runWGSLModuleKernel(device, {
      id: `dense-lu-pivots-${capacity}`,
      inputs: {
        matrices: Float32Array.from(cases.flatMap(entry => entry.matrix.flat())),
        vectors: Float32Array.from(cases.flatMap(entry => getRightSide(entry.matrix))),
        passes: Uint32Array.from(cases, entry => entry.passes)
      },
      outputLength: cases.length * (capacity + 1),
      invocationCount: cases.length,
      declarations: `${IS_FINITE_VALUE_WGSL}
${getDenseLUWGSL(capacity)}`,
      body: `var matrix: array<f32, ${square}>;
  for (var entry = 0u; entry < ${square}u; entry++) { matrix[entry] = matrices[matricesOffset + index * ${square}u + entry]; }
  var vector: array<f32, ${capacity}>;
  for (var row = 0u; row < ${capacity}u; row++) { vector[row] = vectors[vectorsOffset + index * ${capacity}u + row]; }
  var factorization = factorDenseLU_${capacity}(&matrix, ${capacity}u, passes[passesOffset + index], 0.0, 0.0);
  let finite = solveDenseLU_${capacity}(&matrix, &factorization, &vector);
  let outputBase = outputOffset + index * ${capacity + 1}u;
  output[outputBase] = factorization.status | select(16u, 0u, finite);
  for (var row = 0u; row < ${capacity}u; row++) { output[outputBase + 1u + row] = bitcast<u32>(vector[row]); }`
    });
    const floats = getFloatWords(words);
    cases.forEach((entry, caseIndex) => {
      const base = caseIndex * (capacity + 1);
      expect(words[base], `capacity ${capacity} case ${caseIndex} status`).toBe(0);
      // The right-hand side was built from this solution in float64; the system is well
      // conditioned, so float32 elimination recovers it to about 1e-5.
      const worst = Math.max(
        ...solution.map((value, row) => Math.abs(floats[base + 1 + row] - value))
      );
      expect(worst, `capacity ${capacity} case ${caseIndex} (${entry.passes} passes)`).toBeLessThan(
        2e-4
      );
    });
  }
});

const NATURAL_NEIGHBOR_SAMPLE_COUNTS = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64];

it('GPUNaturalNeighbor without an index reads its sample table right from 8 to 64 samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Sibson interpolation reproduces a linear field exactly. The direct path walks all samples
  // through `sampleIndices`, a function-scope table filled only by an identity loop: if it read
  // back as zeros every sample would be sample 0 and the query would be degenerate or wrong.
  const getField = (x: number, y: number): number => 2 * x - 3 * y + 5;
  for (const sampleCount of NATURAL_NEIGHBOR_SAMPLE_COUNTS) {
    const random = createSeededRandom(4000 + sampleCount);
    // The four corners of a square, then jittered interior points; queries well inside the hull.
    const positions: number[] = [0, 0, 10, 0, 10, 10, 0, 10];
    while (positions.length < 2 * sampleCount) {
      positions.push(0.5 + 9 * random(), 0.5 + 9 * random());
    }
    const stored = Float32Array.from(positions);
    const values = Float32Array.from({length: sampleCount}, (_, sample) =>
      getField(stored[2 * sample], stored[2 * sample + 1])
    );
    const queries = Float32Array.of(5.03, 4.91, 3.3, 6.7, 7.2, 2.4);
    const queryCount = queries.length / 2;
    // With and without the optional weights output: the two are different compiled kernels.
    for (const withWeights of [false, true]) {
      const graph = new GPUCommandGraph(device, {
        id: `natural-neighbor-${sampleCount}-${withWeights}`
      });
      const buffers = {
        positions: createInputBuffer(device, stored),
        values: createInputBuffer(device, values),
        queries: createInputBuffer(device, queries),
        results: createOutputBuffer(device, queryCount * 4),
        status: createOutputBuffer(device, queryCount),
        weights: createOutputBuffer(device, queryCount * sampleCount)
      };
      const label = `${sampleCount} samples${withWeights ? ', weights' : ''}`;
      try {
        graph.add(
          new GPUNaturalNeighbor({
            positions: importGraphBuffer(
              graph,
              'positions',
              buffers.positions,
              'float32x2',
              sampleCount
            ),
            values: importGraphBuffer(graph, 'values', buffers.values, 'float32', sampleCount),
            queries: importGraphBuffer(graph, 'queries', buffers.queries, 'float32x2', queryCount),
            maximumNeighborCount: 64,
            output: {
              values: importGraphBuffer(graph, 'results', buffers.results, 'float32x4', queryCount),
              status: importGraphBuffer(graph, 'status', buffers.status, 'uint32', queryCount),
              ...(withWeights
                ? {
                    weights: importGraphBuffer(
                      graph,
                      'weights',
                      buffers.weights,
                      'float32',
                      queryCount * sampleCount
                    )
                  }
                : {})
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(buffers.status, queryCount), label).toEqual(
            new Array(queryCount).fill(0)
          );
          const results = await readFloat32(buffers.results, queryCount * 4);
          const weights = await readFloat32(buffers.weights, queryCount * sampleCount);
          for (let query = 0; query < queryCount; query++) {
            const expected = getField(queries[2 * query], queries[2 * query + 1]);
            expect(
              Math.abs(results[4 * query] - expected),
              `${label}, query ${query}`
            ).toBeLessThan(2e-3);
            if (withWeights) {
              // The weights are in sample order: they sum to 1 and reproduce the field.
              let weightSum = 0;
              let weighted = 0;
              for (let sample = 0; sample < sampleCount; sample++) {
                weightSum += weights[query * sampleCount + sample];
                weighted += weights[query * sampleCount + sample] * values[sample];
              }
              expect(Math.abs(weightSum - 1), `${label}, query ${query}`).toBeLessThan(1e-4);
              expect(Math.abs(weighted - expected), `${label}, query ${query}`).toBeLessThan(2e-3);
            }
          }
        } finally {
          compiled.destroy();
        }
      } finally {
        for (const buffer of Object.values(buffers)) {
          buffer.destroy();
        }
      }
    }
  }
});

const PHENOLOGY_SLICE_COUNTS = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 96, 128];

it('GPURasterPhenologyMetrics without a date axis reads its time table right from 8 to 128 slices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Without a date axis the times are the slice numbers, a function-scope float table filled
  // only by `times[slice] = f32(slice)`. A triangular curve with its peak at slice p = n / 2
  // rises by 1 per slice from 0, so the peak time is p and the 25% start level is crossed at
  // exactly p / 4. A table of zeros would give 0 for both.
  for (const sliceCount of PHENOLOGY_SLICE_COUNTS) {
    const peak = sliceCount / 2;
    const curve = Float32Array.from(
      {length: sliceCount},
      (_, slice) => peak - Math.abs(slice - peak)
    );
    const graph = new GPUCommandGraph(device, {id: `phenology-${sliceCount}`});
    const buffers = {
      smoothed: createInputBuffer(device, curve),
      parameters: createInputBuffer(
        device,
        getGPURasterPhenologyMetricsParameterValues({
          startFraction: 0.25,
          endFraction: 0.15,
          minimumAmplitude: 0.1
        })
      ),
      metrics: createOutputBuffer(device, GPU_RASTER_PHENOLOGY_METRIC_COUNT),
      flags: createOutputBuffer(device, 1),
      seasonCount: createOutputBuffer(device, 1),
      status: createOutputBuffer(device, 1)
    };
    try {
      graph.add(
        new GPURasterPhenologyMetrics({
          cellCount: 1,
          sliceCount,
          maximumSeasonCount: 1,
          smoothed: importGraphBuffer(graph, 'smoothed', buffers.smoothed, 'float32', sliceCount),
          parameters: importGraphBuffer(graph, 'parameters', buffers.parameters, 'float32', 4),
          output: {
            metrics: importGraphBuffer(
              graph,
              'metrics',
              buffers.metrics,
              'float32',
              GPU_RASTER_PHENOLOGY_METRIC_COUNT
            ),
            flags: importGraphBuffer(graph, 'flags', buffers.flags, 'uint32', 1),
            seasonCount: importGraphBuffer(graph, 'seasonCount', buffers.seasonCount, 'uint32', 1),
            status: importGraphBuffer(graph, 'status', buffers.status, 'uint32', 1)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(buffers.seasonCount, 1), `${sliceCount} slices`).toEqual([1]);
        const metrics = await readFloat32(buffers.metrics, GPU_RASTER_PHENOLOGY_METRIC_COUNT);
        expect(metrics[GPU_RASTER_PHENOLOGY_METRIC.peakTime], `${sliceCount} slices`).toBe(peak);
        expect(
          Math.abs(metrics[GPU_RASTER_PHENOLOGY_METRIC.startTime] - peak / 4),
          `${sliceCount} slices`
        ).toBeLessThan(1e-4);
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of Object.values(buffers)) {
        buffer.destroy();
      }
    }
  }
});
