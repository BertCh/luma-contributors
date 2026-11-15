// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {compileProjectionProgram, GPUProjectionProgram} from '@luma.gl/graph/gpu-project';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

const SOURCE_VALUES = [10, 20, -999, 40, 50, 60, 70, 80];
const SOURCE_VALIDITY = [1, 0, 1, 1, 1, 1, 1, 1];

// Independent tent-basis oracle: sum contributions of every source centre, rather than selecting
// the sampler's four taps. Exact dyadic coordinates make support and half-cell ties unambiguous.
function sampleReference(
  column: number,
  row: number,
  method: 'nearest' | 'bilinear',
  noDataPolicy: 'strict' | 'renormalize'
): number {
  let numerator = 0;
  let denominator = 0;
  for (let sourceRow = 0; sourceRow < 2; sourceRow++) {
    for (let sourceColumn = 0; sourceColumn < 4; sourceColumn++) {
      const sourceIndex = sourceRow * 4 + sourceColumn;
      if (!SOURCE_VALIDITY[sourceIndex] || SOURCE_VALUES[sourceIndex] === -999) continue;
      const weight =
        method === 'nearest'
          ? Number(sourceColumn === Math.floor(column + 0.5) && sourceRow === Math.floor(row + 0.5))
          : Math.max(0, 1 - Math.abs(column - sourceColumn)) *
            Math.max(0, 1 - Math.abs(row - sourceRow));
      numerator += weight * (SOURCE_VALUES[sourceIndex] * 2 + 5);
      denominator += weight;
    }
  }
  return denominator > 0 && (noDataPolicy === 'renormalize' || denominator === 1)
    ? numerator / denominator
    : NaN;
}

for (const method of ['nearest', 'bilinear'] as const) {
  for (const noDataPolicy of ['strict', 'renormalize'] as const) {
    it(`inline projection matches the materialized path and tent oracle: ${method}/${noDataPolicy} (LUM-26)`, async () => {
      const device = await getWebGPUTestDevice('core');
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const track = (buffer: Buffer) => {
        buffers.push(buffer);
        return buffer;
      };
      const width = 17;
      const height = 3;
      const count = width * height; // Partial final workgroup; two extra output rows are sentinels.
      const sourceGrid = {
        width: 4,
        height: 2,
        affine: [1, 0, -120, 0, 1, -10] as const,
        pixelInterpretation: 'area' as const
      };
      const destinationGrid = {
        width,
        height,
        affine: [0.25, 0, 0, 0, 0.25, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const projection = compileProjectionProgram({
        precision: 'local-f32',
        destinationOrigin: [-120, -10],
        operations: [{type: 'affine', scale: [1, 1], offset: [-120.375, -9.625]}]
      });
      const valuesBuffer = track(createOutputBuffer(device, count + 2));
      const validityBuffer = track(createOutputBuffer(device, count + 2));
      valuesBuffer.write(new Float32Array(count + 2).fill(-777));
      validityBuffer.write(new Uint32Array(count + 2).fill(777));
      const contributor = new GPURasterReprojection({
        sourceGrid,
        destinationGrid,
        inverseProjection: projection,
        method,
        noDataPolicy,
        input: {
          id: 'source',
          format: 'float32',
          scale: 2,
          offset: 5,
          noDataValue: -999,
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(
              graph,
              'source',
              track(createInputBuffer(device, Float32Array.from(SOURCE_VALUES))),
              'float32',
              8
            )
          },
          validity: importGraphBuffer(
            graph,
            'source-validity',
            track(createInputBuffer(device, Uint32Array.from(SOURCE_VALIDITY))),
            'uint32',
            8
          )
        },
        output: {
          values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', count + 2),
          validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', count + 2)
        }
      });
      graph.add(contributor);
      // Preserve the old f32 projection boundary with the published materializing contributor.
      // The uploaded centres equal the former grid kernel exactly (all operations are dyadic).
      const centres = Float32Array.from(
        Array.from({length: count}, (_, index) => [
          ((index % width) + 0.5) * 0.25,
          (Math.floor(index / width) + 0.5) * 0.25
        ]).flat()
      );
      const projectedBuffer = track(createOutputBuffer(device, count * 2));
      const projectedValidityBuffer = track(createOutputBuffer(device, count));
      const materialized = new GPUProjectionProgram({
        projection,
        positions: importGraphBuffer(
          graph,
          'centres',
          track(createInputBuffer(device, centres)),
          'float32x2',
          count
        ),
        output: importGraphBuffer(graph, 'projected', projectedBuffer, 'float32x2', count),
        validity: importGraphBuffer(
          graph,
          'projected-validity',
          projectedValidityBuffer,
          'uint32',
          count
        )
      });
      materialized.addToGraph(graph);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const values = await readFloat32(valuesBuffer, count + 2);
        const validity = await readUint32(validityBuffer, count + 2);
        const projected = await readFloat32(projectedBuffer, count * 2);
        expect(await readUint32(projectedValidityBuffer, count)).toEqual(Array(count).fill(1));
        for (let index = 0; index < count; index++) {
          const expected = sampleReference(
            (index % width) * 0.25 - 0.75,
            Math.floor(index / width) * 0.25,
            method,
            noDataPolicy
          );
          const previous = sampleReference(
            projected[2 * index] - 0.5,
            projected[2 * index + 1] - 0.5,
            method,
            noDataPolicy
          );
          expect(validity[index]).toBe(Number(Number.isFinite(expected)));
          if (Number.isNaN(expected)) {
            expect(previous).toBeNaN();
            expect(values[index]).toBeNaN();
          } else {
            expect(previous).toBeCloseTo(expected, 5);
            expect(Math.abs(values[index] - previous)).toBeLessThanOrEqual(2e-5);
          }
        }
        expect(values.slice(count)).toEqual([-777, -777]);
        expect(validity.slice(count)).toEqual([777, 777]);
      } finally {
        compiled.destroy();
        contributor.destroy();
        materialized.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    });
  }
}

it('rejects invalid projection domains even when their zero position has valid source support (LUM-26)', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const sourceBuffer = createInputBuffer(device, Float32Array.of(10, 20, 30, 40));
  const valuesBuffer = createOutputBuffer(device, 3);
  const validityBuffer = createOutputBuffer(device, 3);
  const sourceGrid = {
    width: 1,
    height: 4,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'point' as const
  };
  const contributor = new GPURasterReprojection({
    sourceGrid,
    destinationGrid: {...sourceGrid, height: 3},
    inverseProjection: compileProjectionProgram({
      precision: 'local-f32',
      operations: [{type: 'web-mercator', arithmetic: 'float32', radius: 1}]
    }),
    method: 'nearest',
    noDataPolicy: 'renormalize',
    input: {
      id: 'source',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'source', sourceBuffer, 'float32', 4)
      }
    },
    output: {
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', 3),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 3)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // y(0)=0; y(1 rad)=asinh(tan(1))=1.226..., nearest source row 1; 2 rad is outside domain.
    expect(await readUint32(validityBuffer, 3)).toEqual([1, 1, 0]);
    const values = await readFloat32(valuesBuffer, 3);
    expect(values.slice(0, 2)).toEqual([10, 20]);
    expect(values[2]).toBeNaN();
  } finally {
    compiled.destroy();
    contributor.destroy();
    sourceBuffer.destroy();
    valuesBuffer.destroy();
    validityBuffer.destroy();
  }
});
