// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('matches a float64 dense covariance oracle across a partial tile, missing samples and re-encodes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let identifier = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      read: async () =>
        Array.from(
          await (format === 'float32' ? readFloat32(buffer, length) : readUint32(buffer, length))
        )
    };
  };
  const cellCount = 257,
    correlationLength = 32,
    cellArea = 6;
  const residuals = Float32Array.from(
    {length: cellCount},
    (_, cell) => 2 + (((cell * 7) % 17) - 8) * 0.25
  );
  residuals[123] = NaN;
  const variances = Float32Array.from({length: cellCount}, (_, cell) => 1 + (cell % 5) * 0.25);
  variances[cellCount - 1] = -1;
  const stableMask = Uint32Array.from({length: cellCount}, (_, cell) => Number(cell % 13 !== 0));
  const validity = new Uint32Array(cellCount).fill(1);
  validity[0] = 0;
  validity[255] = 0;
  const validityInput = input(validity, 'uint32');
  const fit = output(4, 'float32'),
    differences = output(cellCount * 2, 'float32'),
    outputValidity = output(cellCount, 'uint32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: {width: cellCount, height: 1, affine: [2, 0, 0, 0, -3, 0], pixelInterpretation: 'area'},
      reference: input(new Float32Array(cellCount), 'float32').view,
      moving: input(residuals, 'float32').view,
      stableMask: input(stableMask, 'uint32').view,
      differenceVariance: input(variances, 'float32').view,
      validity: validityInput.view,
      correlationLength,
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: outputValidity.view,
        volume: volume.view
      }
    })
  );
  const stableCells = Array.from({length: cellCount}, (_, cell) => cell).filter(
    cell => validity[cell] && stableMask[cell] && Number.isFinite(residuals[cell])
  );
  const validCells = Array.from({length: cellCount}, (_, cell) => cell).filter(
    cell => validity[cell] && Number.isFinite(residuals[cell]) && variances[cell] >= 0
  );
  const median = (values: number[]) => {
    const sorted = [...values].sort((first, second) => first - second);
    return (
      (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) * 0.5
    );
  };
  const bias = median(stableCells.map(cell => residuals[cell]));
  const deviation = 1.4826022 * median(stableCells.map(cell => Math.abs(residuals[cell] - bias)));
  const covariance = (cells: number[], weighted: boolean) =>
    cells.reduce(
      (total, first) =>
        total +
        cells.reduce(
          (row, second) =>
            row +
            Math.exp((-2 * Math.abs(first - second)) / correlationLength) *
              (weighted ? Math.sqrt(variances[first]) * Math.sqrt(variances[second]) : 1),
          0
        ),
      0
    );
  const biasError =
    (1.2533141 * deviation * Math.sqrt(covariance(stableCells, false))) / stableCells.length;
  const expectedVolume = [
    validCells.reduce((sum, cell) => sum + residuals[cell] - bias, 0) * cellArea,
    Math.sqrt(covariance(validCells, true) + validCells.length ** 2 * biasError ** 2) * cellArea,
    validCells.length * cellArea
  ];
  const close = (actual: number, expected: number) =>
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.max(1, Math.abs(expected)) * 1e-5);
  const compiled = graph.compile();
  try {
    for (const excluded of [false, true, false]) {
      validityInput.buffer.write(excluded ? new Uint32Array(cellCount) : validity);
      submitGraph(device, compiled, undefined);
      const actualFit = await fit.read(),
        actualVolume = await volume.read(),
        actualDifferences = await differences.read();
      if (excluded) {
        expect(actualFit).toEqual([NaN, NaN, 0, NaN]);
        expect(actualVolume).toEqual([NaN, NaN, 0]);
        expect(await outputValidity.read()).toEqual(new Array(cellCount).fill(0));
        expect(actualDifferences.every(Number.isNaN)).toBe(true);
      } else {
        [bias, deviation, stableCells.length, biasError].forEach((expected, index) =>
          close(actualFit[index], expected)
        );
        expectedVolume.forEach((expected, index) => close(actualVolume[index], expected));
        const validSet = new Set(validCells);
        expect(await outputValidity.read()).toEqual(
          Array.from({length: cellCount}, (_, cell) => Number(validSet.has(cell)))
        );
        for (let cell = 0; cell < cellCount; cell++) {
          if (validSet.has(cell)) {
            close(actualDifferences[cell * 2], residuals[cell] - bias);
            close(actualDifferences[cell * 2 + 1], Math.sqrt(variances[cell] + biasError ** 2));
          } else {
            expect(actualDifferences[cell * 2]).toBeNaN();
            expect(actualDifferences[cell * 2 + 1]).toBeNaN();
          }
        }
      }
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
