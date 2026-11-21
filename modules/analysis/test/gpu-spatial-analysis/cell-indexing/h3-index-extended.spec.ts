// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {gridDistance, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPointToCell} from '../../../src/gpu-spatial-analysis/cell-indexing/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createCellEdgePoints,
  createPentagonNeighborhoodPoints,
  createSpecialH3Points,
  createUniformSpherePoints
} from './h3-index-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const ROWS = 20000;

/** Compiles one `GPUPointToCell` H3 graph and returns a runner over up to ROWS points. */
function createIndexer(device: Device, resolution: number, precision: 'float32' | 'double-single') {
  const positions = createInputBuffer(device, new Float32Array(2 * ROWS));
  const cells = createOutputBuffer(device, 2 * ROWS);
  const graph = new GPUCommandGraph(device, {id: `h3-${precision}-${resolution}`});
  graph.add(
    new GPUPointToCell({
      family: 'h3',
      resolution,
      precision,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', ROWS),
      output: {cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', ROWS)}
    })
  );
  const compiled = graph.compile();
  return {
    async run(points: Float32Array): Promise<bigint[]> {
      const padded = new Float32Array(2 * ROWS);
      padded.set(points);
      positions.write(padded);
      submitGraph(device, compiled, undefined);
      const words = await readUint32(cells, 2 * ROWS);
      const keys: bigint[] = [];
      for (let row = 0; row < points.length / 2; row++) {
        keys.push((BigInt(words[2 * row + 1]) << 32n) | BigInt(words[2 * row]));
      }
      return keys;
    },
    destroy() {
      compiled.destroy();
      positions.destroy();
      cells.destroy();
    }
  };
}

function countMismatches(points: Float32Array, keys: bigint[], resolution: number) {
  let mismatches = 0;
  let neighbors = 0;
  for (let row = 0; row < keys.length; row++) {
    const expected = latLngToCell(points[2 * row + 1], points[2 * row], resolution);
    const actual = keys[row].toString(16);
    if (actual === expected) {
      continue;
    }
    mismatches++;
    try {
      if (gridDistance(expected, actual) <= 1) {
        neighbors++;
      }
    } catch {
      // Pentagon distortion: not counted as a verified neighbor.
    }
  }
  return {rows: keys.length, mismatches, neighbors};
}

it('GPUPointToCell H3 precision extended matches h3-js on the same f32 inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const table: string[] = ['res | uniform  edge  pentagon  special | f32 uniform'];
  for (let resolution = 0; resolution <= 15; resolution++) {
    const extended = createIndexer(device, resolution, 'double-single');
    const plain = createIndexer(device, resolution, 'float32');
    const sets = {
      uniform: createUniformSpherePoints(ROWS, 1000 + resolution),
      edge: createCellEdgePoints(resolution, 6000, 2000 + resolution),
      pentagon: createPentagonNeighborhoodPoints(resolution, 100, 3000 + resolution),
      special: createSpecialH3Points()
    };
    const results: Record<string, ReturnType<typeof countMismatches>> = {};
    for (const [name, points] of Object.entries(sets)) {
      results[name] = countMismatches(points, await extended.run(points), resolution);
    }
    const plainUniform = countMismatches(sets.uniform, await plain.run(sets.uniform), resolution);
    table.push(
      `${String(resolution).padStart(3)} | ${results['uniform'].mismatches}/${results['uniform'].rows} ` +
        `${results['edge'].mismatches}/${results['edge'].rows} ${results['pentagon'].mismatches}/${results['pentagon'].rows} ` +
        `${results['special'].mismatches}/${results['special'].rows} | ${plainUniform.mismatches}`
    );
    for (const [name, result] of Object.entries(results)) {
      // Verified exact through resolution 15 (headless WebGPU, Apple Metal): no mismatches.
      expect(result.mismatches, `${name} res ${resolution}`).toBe(0);
    }
    // The double-float arithmetic of the extended path rests on error-free transformations
    // (two-sum, Dekker product), which a fast-math shader compiler folds to zero unless the
    // intermediates are kept opaque (`cellIndexH3ExtKeep`). If it folded them here the extended
    // path would be the float32 path, which misses about 2% of uniform points at resolution 12
    // and over 20% at 15. So zero mismatches above, with the plain path failing on the same
    // points in the same run, shows on the GPU that the low words survive.
    if (resolution >= 12) {
      expect(plainUniform.mismatches, `float32 path res ${resolution}`).toBeGreaterThan(
        0.005 * plainUniform.rows
      );
    }
    extended.destroy();
    plain.destroy();
  }
  console.error(`H3 extended precision GPU vs h3-js (mismatches/rows)\n${table.join('\n')}`);
});
