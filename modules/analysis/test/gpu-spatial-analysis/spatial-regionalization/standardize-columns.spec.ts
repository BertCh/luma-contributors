// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {createStandardizeColumnsNode} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-inputs';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

// Math review point-process-clustering-04: the standardizer of AZP, Ward and max-p
// (`createStandardizeColumnsNode`) and the one inside the minimum spanning tree must be one rule.
// Expected values are float64 z-scores (population divisor) of the float32 inputs, and exactly 0
// for a constant column, which is what both documents promise.

const ROWS = 791;
const COLUMNS = 3;

function createTable(): Float32Array {
  const table = new Float32Array(ROWS * COLUMNS);
  for (let row = 0; row < ROWS; row++) {
    // A constant that float32 cannot hold exactly: its rounded mean differs from every entry.
    table[row * COLUMNS] = 37.3;
    // A large mean with a small spread.
    table[row * COLUMNS + 1] = 1e6 + (row % 7);
    table[row * COLUMNS + 2] = Math.sin(row) * 5 - 2;
  }
  // One missing entry: skipped by the statistics, left as it is in the output.
  table[100 * COLUMNS + 2] = Number.NaN;
  return table;
}

function getExpected(table: Float32Array): number[] {
  const expected = new Array<number>(table.length);
  for (let column = 0; column < COLUMNS; column++) {
    const finite: number[] = [];
    for (let row = 0; row < ROWS; row++) {
      const value = table[row * COLUMNS + column];
      if (Number.isFinite(value)) finite.push(value);
    }
    const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
    const deviation = Math.sqrt(
      finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length
    );
    for (let row = 0; row < ROWS; row++) {
      const value = table[row * COLUMNS + column];
      expected[row * COLUMNS + column] = !Number.isFinite(value)
        ? value
        : deviation > 0
          ? (value - mean) / deviation
          : 0;
    }
  }
  return expected;
}

it('standardizes a constant column to exactly 0 in the shared kernel and in the tree', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const table = createTable();
  const rig = new WeightsRig(device);
  const values = rig.input(table, 'float32', table.length);
  const shared = rig.output('float32', table.length);
  const sharedProducer: GPUCommandNodeProducer = {
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) => [
      createStandardizeColumnsNode(graph, {
        id: 'shared-standardize',
        operation: 'StandardizeColumnsSpec',
        values,
        scaled: shared.view,
        rowCount: ROWS,
        columnCount: COLUMNS
      })
    ]
  };
  // A path over the rows: the tree itself is not under test, only its standardized table.
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < ROWS; row++) {
    if (row > 0) neighbors.push(row - 1);
    if (row + 1 < ROWS) neighbors.push(row + 1);
    offsets.push(neighbors.length);
  }
  const weights = rig.uploadWeights(
    {offsets, neighbors, weights: neighbors.map(() => 1), distances: []},
    0
  );
  const tree = rig.output('float32', table.length);
  rig.graph.add(sharedProducer);
  rig.graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values,
      columnCount: COLUMNS,
      output: {
        treeEdgeFlags: rig.output('uint32', neighbors.length).view,
        componentLabels: rig.output('uint32', ROWS).view,
        standardizedValues: tree.view
      }
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const sharedValues = Array.from(await readFloat32(shared.buffer, table.length));
  const treeValues = Array.from(await readFloat32(tree.buffer, table.length));
  compiled.destroy();
  rig.destroy();

  const expected = getExpected(table);
  for (const [name, actual] of [
    ['shared', sharedValues],
    ['tree', treeValues]
  ] as const) {
    for (let row = 0; row < ROWS; row++) {
      expect(actual[row * COLUMNS], `${name} constant column row ${row}`).toBe(0);
      for (let column = 1; column < COLUMNS; column++) {
        const value = actual[row * COLUMNS + column];
        const reference = expected[row * COLUMNS + column];
        if (Number.isNaN(reference)) {
          expect(value, `${name} missing entry`).toBeNaN();
        } else {
          expect(Math.abs(value - reference), `${name} row ${row} column ${column}`).toBeLessThan(
            1e-5
          );
        }
      }
    }
  }
  // One kernel: the two tables are the same numbers, bit for bit (NaN compares by Object.is).
  expect(sharedValues).toEqual(treeValues);
});
