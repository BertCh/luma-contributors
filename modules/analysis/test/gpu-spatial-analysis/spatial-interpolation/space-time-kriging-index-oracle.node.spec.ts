// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUSpaceTimeKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging';
import {createSpaceTimeKrigingPredictionNode} from '../../../src/gpu-spatial-analysis/spatial-interpolation/space-time-kriging-kernel';
import {getGPUSpaceTimeKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/prediction-covariance';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {selectReferenceNeighbors, type Point} from './space-time-kriging-reference';

const COVARIANCE = {
  type: 'separable' as const,
  sill: 1,
  spatial: {model: 'exponential' as const, nugget: 0, sill: 1, range: 5},
  temporal: {model: 'exponential' as const, nugget: 0, sill: 1, range: 20}
};

for (const neighborSearch of ['metric', 'octant', 'staged'] as const) {
  it(`executes the generated ${neighborSearch} grid walk against independent CPU neighbor IDs`, () => {
    // A sparse wide domain exercises shell pruning. Near points include all octants and exact
    // distance ties; reverse within-cell IDs deliberately remove the original scan visit order.
    const geometryValues: number[] = [];
    for (const extent of [1, 100]) {
      for (const positionX of [-extent, extent])
        for (const positionY of [-extent, extent])
          for (const sampleTime of [-extent, extent])
            geometryValues.push(positionX, positionY, sampleTime);
    }
    geometryValues.push(0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5);
    const sampleCount = geometryValues.length / 3;
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const positions = createTransientView(graph, 'positions', 'float32x2', sampleCount);
    const times = createTransientView(graph, 'times', 'float32', sampleCount);
    const observations = createTransientView(graph, 'observations', 'float32x4', sampleCount);
    const geometry = createTransientView(graph, 'geometry', 'float32x3', sampleCount);
    const queryGeometry = createTransientView(graph, 'query-geometry', 'float32x3', 1);
    const parameters = createTransientView(graph, 'parameters', 'float32', 40);
    const values = createTransientView(graph, 'values', 'float32x4', 1);
    const status = createTransientView(graph, 'status', 'uint32', 1);
    const contributor = new GPUSpaceTimeKriging({
      positions,
      times,
      observations,
      queries: createTransientView(graph, 'queries', 'float32x2', 1),
      queryTimes: createTransientView(graph, 'query-times', 'float32', 1),
      parameters,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      mode: 'simple',
      sampleIndex: 'grid',
      neighborSearch,
      neighborCount: 4,
      neighborsPerOctant: 1,
      stagedCandidateCount: 8,
      output: {values, status}
    });
    // Test the walk independently of the production grid-size heuristic and index builder.
    const gridSide = 16;
    Object.defineProperty(contributor, 'gridSide', {value: gridSide});
    const grid = {
      cellOffsets: createTransientView(graph, 'cell-offsets', 'uint32', gridSide ** 3 + 1),
      sortedRows: createTransientView(graph, 'sorted-rows', 'uint32', sampleCount)
    };
    const node = createSpaceTimeKrigingPredictionNode(
      graph,
      contributor,
      'predict',
      1,
      geometry,
      queryGeometry,
      parameters,
      values,
      status,
      grid
    );
    const kernel = getGPUCommandNodeKernelSources([node])[0];
    // The interpreter does not apply pipeline overrides or implement phony assignments.
    const source = kernel.source
      .replace(
        /override (\w+): u32;/g,
        (_, name: string) => `const ${name}: u32 = ${kernel.constants[name]}u;`
      )
      .replace(/^\s*_ = [^;]+;/gm, '')
      // Scalar casts in wgsl_reflect mutate their argument's type in place. Copy scalar
      // variables before a cast so signed shell coordinates retain WGSL value semantics.
      .replace(/\b(u32|i32|f32)\((\w+(?:\.[xyzw])?)\)/g, '$1($2 + 0)');
    // Expose the selected IDs instead of running the solve. All selection code is the actual
    // generated production shader; public output layouts and kernels remain unchanged.
    const selectionSource =
      source.slice(0, source.indexOf('if (retained == 0u || retained < DRIFT_COUNT)')) +
      `
var selectedRows = vec4<f32>(-1.0);
for (var slot = 0u; slot < retained; slot++) { selectedRows[slot] = f32(neighbors[slot]); }
write_results(index, selectedRows);
statuses[statusesOffset + index] = retained;
}`;
    const positionsCPU: Point[] = [];
    const timesCPU: number[] = [];
    for (let row = 0; row < sampleCount; row++) {
      positionsCPU.push([geometryValues[row * 3], geometryValues[row * 3 + 1]]);
      timesCPU.push(geometryValues[row * 3 + 2]);
    }
    const buckets = Array.from({length: gridSide ** 3}, () => [] as number[]);
    for (let row = sampleCount - 1; row >= 0; row--) {
      const cell = [0, 1, 2].map(axis =>
        Math.floor(((geometryValues[row * 3 + axis] + 100) / 200) * (gridSide - 1))
      );
      buckets[cell[0] + gridSide * (cell[1] + gridSide * cell[2])].push(row);
    }
    const offsets = new Uint32Array(gridSide ** 3 + 1);
    buckets.forEach((bucket, cell) => {
      offsets[cell + 1] = offsets[cell] + bucket.length;
    });
    const sortedRows = Uint32Array.from(buckets.flat());
    const observationValues = Float32Array.from({length: sampleCount * 4}, (_, component) =>
      component % 4 === 3 ? 1 : 0
    );
    observationValues[16 * 4 + 3] = 0; // The exact-hit row is invalid and must not enter selection.
    for (const [queryX, queryY, queryTime, searchRadius, maximumTimeLag] of [
      [0, 0, 0, Infinity, Infinity],
      [0.25, 0, 0, 2, 2],
      [150, 150, 150, Infinity, Infinity],
      [0, 0, 0, 0.1, 0.1]
    ]) {
      const packedParameters = new Float32Array(40);
      packedParameters.set(
        getGPUSpaceTimeKrigingParameterValues({
          covariance: COVARIANCE,
          searchAnisotropy: 0.5,
          searchRadius,
          maximumTimeLag
        })
      );
      packedParameters.set([-100, 100, -100, 100, -100, 100], 32);
      const result = new Float32Array(4);
      const outputStatus = new Uint32Array(1);
      new WgslExec(new WgslParser().parse(selectionSource)).dispatchWorkgroups('main', [1, 1, 1], {
        0: {
          0: Float32Array.from(geometryValues),
          1: observationValues.slice(),
          2: Float32Array.of(queryX, queryY, queryTime),
          3: packedParameters,
          4: offsets.slice(),
          5: sortedRows.slice(),
          6: result,
          7: outputStatus,
          8: Uint32Array.from({length: 64}, (_, word) => (word === 1 || word === 2 ? 1 : 0))
        }
      });
      const usableRows = Array.from({length: sampleCount}, (_, row) => row).filter(
        row =>
          observationValues[row * 4 + 3] > 0 &&
          Math.hypot(positionsCPU[row][0] - queryX, positionsCPU[row][1] - queryY) <=
            searchRadius &&
          Math.abs(timesCPU[row] - queryTime) <= maximumTimeLag
      );
      const expected = selectReferenceNeighbors(
        {
          positions: usableRows.map(row => positionsCPU[row]),
          times: usableRows.map(row => timesCPU[row]),
          observations: usableRows.map(() => [0, 0] as const)
        },
        [queryX, queryY],
        queryTime,
        COVARIANCE,
        {
          search: neighborSearch,
          neighborCount: 4,
          neighborsPerOctant: 1,
          stagedCandidateCount: 8,
          searchAnisotropy: 0.5
        }
      ).map(row => usableRows[row]);
      expect(
        outputStatus[0],
        JSON.stringify({queryX, queryY, queryTime, result: Array.from(result), expected})
      ).toBe(expected.length);
      expect(Array.from(result).slice(0, expected.length)).toEqual(expected);
    }
  });
}

it('keeps row-zero ties when float32 metric squares underflow, including the unpruned reference', () => {
  const sampleCount = 4097;
  for (const sampleIndex of ['grid', 'brute-force'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const observations = view('observations', 'float32x4', sampleCount);
    const geometry = view('geometry', 'float32x3', sampleCount);
    const queryGeometry = view('query-geometry', 'float32x3', 1);
    const parameters = view('parameters', 'float32', 40);
    const values = view('values', 'float32x4', 1);
    const status = view('status', 'uint32', 1);
    const contributor = new GPUSpaceTimeKriging({
      positions: view('positions', 'float32x2', sampleCount),
      times: view('times', 'float32', sampleCount),
      observations,
      queries: view('queries', 'float32x2', 1),
      queryTimes: view('query-times', 'float32', 1),
      parameters,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      mode: 'ordinary',
      sampleIndex,
      neighborCount: 1,
      output: {values, status}
    });
    const side = contributor.gridSide;
    const grid =
      sampleIndex === 'grid'
        ? {
            cellOffsets: view('cell-offsets', 'uint32', side ** 3 + 1),
            sortedRows: view('sorted-rows', 'uint32', sampleCount)
          }
        : undefined;
    const node = createSpaceTimeKrigingPredictionNode(
      graph,
      contributor,
      'predict',
      1,
      geometry,
      queryGeometry,
      parameters,
      values,
      status,
      grid
    );
    const kernel = getGPUCommandNodeKernelSources([node])[0];
    const source = kernel.source
      .replace(/^\s*_ = [^;]+;/gm, '')
      .replace(/\b(u32|i32|f32)\((\w+(?:\.[xyzw])?)\)/g, '$1($2 + 0)');
    const packedGeometry = new Float32Array(sampleCount * 3);
    packedGeometry[2] = 1e-24;
    const observationValues = Float32Array.from({length: sampleCount * 4}, (_, word) =>
      word % 4 === 3 ? 1 : word % 4 === 0 ? (word === 0 ? 10 : 20) : 0
    );
    const settings = new Float32Array(40);
    settings.set(
      getGPUSpaceTimeKrigingParameterValues({
        covariance: {
          type: 'metric',
          joint: {model: 'exponential', nugget: 0, sill: 1, range: 1},
          anisotropy: 1
        }
      })
    );
    settings.set([0, 0, 0, 0, 0, packedGeometry[2]], 32);
    const offsets = new Uint32Array(side ** 3 + 1);
    offsets.fill(sampleCount - 1, 1);
    offsets.fill(sampleCount, side * side * (side - 1) + 1);
    const sortedRows = Uint32Array.from(
      {length: sampleCount},
      (_, slot) => (slot + 1) % sampleCount
    );
    const run = (shader: string) => {
      const output = new Float32Array(4);
      const outputStatus = new Uint32Array(1);
      const uniform = Uint32Array.from({length: 64}, (_, word) =>
        word === 1 || word === 2 ? 1 : 0
      );
      const bindings = grid
        ? {
            0: packedGeometry,
            1: observationValues,
            2: new Float32Array(3),
            3: settings,
            4: offsets.slice(),
            5: sortedRows.slice(),
            6: output,
            7: outputStatus,
            8: uniform
          }
        : {
            0: packedGeometry,
            1: observationValues,
            2: new Float32Array(3),
            3: settings,
            4: output,
            5: outputStatus,
            6: uniform
          };
      new WgslExec(new WgslParser().parse(shader)).dispatchWorkgroups('main', [1, 1, 1], {
        0: bindings
      });
      return {output, outputStatus};
    };
    const prediction = run(source);
    expect(prediction.outputStatus[0]).toBe(0);
    expect(prediction.output[0]).toBeCloseTo(10, 5);
    const countedSource =
      'var<private> visitedSampleCount: u32;\n' +
      source.replace(
        'let observation = read_observations(sample);',
        'visitedSampleCount += 1u; let observation = read_observations(sample);'
      );
    const selectionSource =
      countedSource.slice(
        0,
        countedSource.indexOf('if (retained == 0u || retained < DRIFT_COUNT)')
      ) +
      'write_results(index, vec4<f32>(f32(neighbors[0]), f32(visitedSampleCount), 0.0, 0.0)); statuses[statusesOffset + index] = retained; }';
    const selection = run(selectionSource);
    expect(selection.outputStatus[0]).toBe(1);
    expect(selection.output[0]).toBe(0);
    expect(selection.output[1]).toBe(sampleCount);
    if (grid) {
      // An ordinary exact hit must still prune well-separated cells. Only underflow-sized
      // gaps need the unpruned walk; scanning all 8M rows for every exact hit would regress it.
      packedGeometry.fill(0);
      for (let row = 1; row < sampleCount; row++) packedGeometry[row * 3 + 2] = 100;
      settings[37] = 100;
      offsets.fill(1, 1);
      offsets.fill(sampleCount, side * side * (side - 1) + 1);
      sortedRows.set(Uint32Array.from({length: sampleCount}, (_, row) => row));
      const exactHit = run(selectionSource);
      expect(exactHit.output[0]).toBe(0);
      expect(exactHit.output[1]).toBe(1);
    }
  }
});
