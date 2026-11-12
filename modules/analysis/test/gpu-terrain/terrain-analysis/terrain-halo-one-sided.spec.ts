// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUTerrainDerivativesParameterValues,
  GPUTerrainDerivatives
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {
  getGPUTerrainCurvatureParameterValues,
  GPUTerrainCurvature
} from '../../../src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runQuadraticWindow(
  device: Device,
  bounds: readonly [number, number, number, number]
) {
  const [firstColumn, firstRow, lastColumn, lastRow] = bounds;
  const width = lastColumn - firstColumn;
  const height = lastRow - firstRow;
  const pixelCount = width * height;
  const grid = makeGPURasterMetadata({
    width,
    height,
    cellSize: 1,
    origin: [firstColumn, -firstRow]
  });
  const elevationBuffer = createInputBuffer(
    device,
    Float32Array.from({length: pixelCount}, (_, index) => (firstColumn + (index % width)) ** 2)
  );
  const slopeBuffer = createOutputBuffer(device, pixelCount);
  const laplacianBuffer = createOutputBuffer(device, pixelCount);
  const slopeValidityBuffer = createOutputBuffer(device, pixelCount);
  const curvatureValidityBuffer = createOutputBuffer(device, pixelCount);
  const graph = new GPUCommandGraph(device, {id: 'quadratic-halo'});
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
    }
  };
  const derivativeSettings = new GPUParameterBuffer(device, {
    id: 'derivative-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainDerivativesParameterValues({}, grid)
  });
  const curvatureSettings = new GPUParameterBuffer(device, {
    id: 'curvature-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainCurvatureParameterValues({}, grid)
  });
  const derivatives = new GPUTerrainDerivatives({
    grid,
    elevation,
    borderMode: 'one-sided',
    parameters: derivativeSettings.importToGraph(graph),
    output: {
      slope: importGraphBuffer(graph, 'slope', slopeBuffer, 'float32', pixelCount),
      outputValidity: importGraphBuffer(
        graph,
        'slope-validity',
        slopeValidityBuffer,
        'uint32',
        pixelCount
      )
    }
  });
  const curvature = new GPUTerrainCurvature({
    grid,
    elevation,
    method: 'zevenbergen-thorne',
    borderMode: 'one-sided',
    parameters: curvatureSettings.importToGraph(graph),
    output: {
      curvatures: {
        laplacian: importGraphBuffer(graph, 'laplacian', laplacianBuffer, 'float32', pixelCount)
      },
      outputValidity: importGraphBuffer(
        graph,
        'curvature-validity',
        curvatureValidityBuffer,
        'uint32',
        pixelCount
      )
    }
  });
  graph.add(derivatives);
  graph.add(curvature);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // The one-column, one-row core is [0, 3, 1, 4], at the real west dataset boundary.
    const coreIndex = (3 - firstRow) * width - firstColumn;
    return {
      derivativeHalo: derivatives.requiredHalo,
      curvatureHalo: curvature.requiredHalo,
      slope: (await readFloat32(slopeBuffer, pixelCount))[coreIndex],
      laplacian: (await readFloat32(laplacianBuffer, pixelCount))[coreIndex],
      slopeValidity: (await readUint32(slopeValidityBuffer, pixelCount))[coreIndex],
      curvatureValidity: (await readUint32(curvatureValidityBuffer, pixelCount))[coreIndex]
    };
  } finally {
    compiled.destroy();
    derivativeSettings.destroy();
    curvatureSettings.destroy();
    for (const buffer of [
      elevationBuffer,
      slopeBuffer,
      laplacianBuffer,
      slopeValidityBuffer,
      curvatureValidityBuffer
    ]) {
      buffer.destroy();
    }
  }
}

it('advertised one-sided halos preserve quadratic border slope and Laplacian for a one-column core', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const full = await runQuadraticWindow(device, [0, 0, 7, 7]);
  expect(full.slope).toBeCloseTo(0, 6);
  expect(full.laplacian).toBeCloseTo(2, 6);
  for (const halo of new Set([full.derivativeHalo, full.curvatureHalo])) {
    const window = await runQuadraticWindow(device, [
      0,
      Math.max(0, 3 - halo),
      1 + halo,
      Math.min(7, 4 + halo)
    ]);
    expect(window.slopeValidity).toBe(1);
    expect(window.curvatureValidity).toBe(1);
    expect(window.slope).toBeCloseTo(full.slope, 6);
    expect(window.laplacian).toBeCloseTo(full.laplacian, 6);
  }
  // With only [z0, z1] = [0, 1], extrapolation is linear: z(-1) = -1, slope atan(1),
  // Laplacian -1 - 2*0 + 1 = 0. The third source sample z2 = 4 restores z(-1) = 1.
  const truncated = await runQuadraticWindow(device, [0, 2, 2, 5]);
  expect(truncated.slope).toBeCloseTo(45, 5);
  expect(truncated.laplacian).toBeCloseTo(0, 6);
});
