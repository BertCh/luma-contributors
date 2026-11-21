// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsSummary,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import type {OracleCSR} from './spatial-weights-oracle';

// Independent NumPy dense-matrix oracle: weights-summary-audit.oracle.py.
// Self weights enter the moments, but active cardinality excludes them. A listed
// zero slot contributes to structural asymmetry, and row 7 has no listed slot.
const INPUT: OracleCSR = {
  offsets: [0, 3, 5, 8, 9, 10, 11, 12, 12],
  neighbors: [0, 1, 2, 0, 2, 0, 1, 3, 2, 5, 4, 7],
  weights: [0.5, 2, 1, 1, 3, 1, 3, 4, 0, 2, 2, 0],
  distances: []
};
const EXPECTED_COUNTS = [12, 5, 3, 0, 3, 1];
const EXPECTED_CARDINALITY = [2, 2, 3, 0, 1, 1, 0, 0];

function upload(
  rig: WeightsRig,
  chunked: boolean,
  scale: number
): GPUSpatialWeights | GPUChunkedSpatialWeights {
  const csr = {...INPUT, weights: INPUT.weights.map(value => value * scale)};
  if (!chunked) return rig.uploadWeights(csr, 2);
  return {
    rowCount: 8,
    chunks: [
      [0, 3],
      [3, 4],
      [7, 1]
    ].map(([rowStart, rowCount]) => {
      const begin = csr.offsets[rowStart];
      const end = csr.offsets[rowStart + rowCount];
      return {
        rowStart,
        rowCount,
        weights: rig.uploadWeights(
          {
            offsets: csr.offsets
              .slice(rowStart, rowStart + rowCount + 1)
              .map(value => value - begin),
            neighbors: csr.neighbors.slice(begin, end),
            weights: csr.weights.slice(begin, end),
            distances: []
          },
          2
        )
      };
    })
  } satisfies GPUChunkedSpatialWeights;
}

it('summary matches independent moments and every count for whole, uneven and empty chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    for (const scale of [1, 2]) {
      const rig = new WeightsRig(device);
      try {
        const statistics = rig.output('float32', 5);
        const counts = rig.output('uint32', 8);
        const cardinality = rig.output('uint32', 8);
        statistics.buffer.write(new Float32Array(5).fill(-99));
        counts.buffer.write(new Uint32Array(8).fill(99));
        rig.run(
          new GPUSpatialWeightsSummary({
            weights: upload(rig, chunked, scale),
            output: {
              statistics: statistics.view,
              counts: counts.view,
              cardinality: cardinality.view
            }
          })
        );
        // All arithmetic is exactly representable here (halves and small integers).
        expect(await readFloat32(statistics.buffer, 5)).toEqual(
          scale === 1 ? [19.5, 81.5, 309, -99, -99] : [39, 326, 1236, -99, -99]
        );
        expect(await readUint32(counts.buffer, 8)).toEqual([...EXPECTED_COUNTS, 99, 99]);
        expect(await readUint32(cardinality.buffer, 8)).toEqual(EXPECTED_CARDINALITY);
      } finally {
        rig.destroy();
      }
    }
  }
});

it('summary suppresses incomplete inputs and restores all outputs on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const weights: GPUSpatialWeights | GPUChunkedSpatialWeights = upload(rig, chunked, 1);
      const matrixOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
      const matrixCandidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
      weights.status = {overflow: matrixOverflow, candidateOverflow: matrixCandidateOverflow};
      const lossViews = [matrixOverflow, matrixCandidateOverflow];
      if ('chunks' in weights) {
        const overflow = rig.input(new Uint32Array([0]), 'uint32', 1);
        const candidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
        weights.chunks[1].weights.status = {overflow, candidateOverflow};
        lossViews.push(overflow, candidateOverflow);
      }
      const statistics = rig.output('float32', 5);
      const counts = rig.output('uint32', 8);
      const cardinality = rig.output('uint32', 8);
      statistics.buffer.write(new Float32Array(5).fill(-99));
      counts.buffer.write(new Uint32Array(8).fill(99));
      rig.graph.add(
        new GPUSpatialWeightsSummary({
          weights,
          output: {statistics: statistics.view, counts: counts.view, cardinality: cardinality.view}
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const loss of lossViews) {
          // A non-boolean flag still means incomplete.
          rig.bufferOf(loss).write(new Uint32Array([7]));
          submitGraph(device, compiled, undefined);
          const moments = await readFloat32(statistics.buffer, 5);
          expect(moments.slice(0, 3).every(Number.isNaN)).toBe(true);
          expect(moments.slice(3)).toEqual([-99, -99]);
          expect(await readUint32(counts.buffer, 8)).toEqual([0, 0, 0, 0, 0, 0, 99, 99]);
          expect(await readUint32(cardinality.buffer, 8)).toEqual(Array(8).fill(0));
          rig.bufferOf(loss).write(new Uint32Array([0]));
          submitGraph(device, compiled, undefined);
          expect(await readFloat32(statistics.buffer, 5)).toEqual([19.5, 81.5, 309, -99, -99]);
          expect(await readUint32(counts.buffer, 8)).toEqual([...EXPECTED_COUNTS, 99, 99]);
          expect(await readUint32(cardinality.buffer, 8)).toEqual(EXPECTED_CARDINALITY);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});

it('summary loss suppression preserves packed disjoint counts and cardinality views', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const weights = upload(rig, false, 1);
    weights.status = {overflow: rig.input(Uint32Array.of(1), 'uint32', 1)};
    const statistics = rig.output('float32', 3);
    const packed = rig.output('uint32', 14);
    const counts = rig.graph.createDataView(packed.view.buffer, {format: 'uint32', length: 6});
    const cardinality = rig.graph.createDataView(packed.view.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 24
    });
    rig.run(
      new GPUSpatialWeightsSummary({
        weights,
        output: {statistics: statistics.view, counts, cardinality}
      })
    );
    expect((await readFloat32(statistics.buffer, 3)).every(Number.isNaN)).toBe(true);
    expect(await readUint32(packed.buffer, 14)).toEqual(Array(14).fill(0));
  } finally {
    rig.destroy();
  }
});
