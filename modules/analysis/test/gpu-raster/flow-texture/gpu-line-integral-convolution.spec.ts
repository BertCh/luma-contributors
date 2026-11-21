// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues,
  GPULineIntegralConvolution,
  type GPULineIntegralConvolutionSettings
} from '../../../src/raster-analysis/flow-texture/index';
import {
  convolveLineIntegralOnCPU,
  getLineIntegralConvolutionNoise
} from '../../oracles/line-integral-convolution-cpu';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createUniformField, createVortexField, type TestField} from './flow-texture-scenes';
import {requireTestDevice} from '../../utils/require-test-device';

type Result = {
  values: Float32Array;
  speeds: Float32Array;
  texture: Float32Array;
};

type Fixture = {
  run(settings: GPULineIntegralConvolutionSettings): Promise<Result>;
  destroy(): void;
};

function createFixture(
  device: Device,
  field: TestField,
  width: number,
  height: number,
  stepCount: number
): Fixture {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'lic-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'lic-parameters',
    format: 'float32',
    length: 12
  });
  const wordBuffer = new GPUParameterBuffer(device, {
    id: 'lic-words',
    format: 'uint32',
    length: 4
  });
  const valuesBuffer = track(createOutputBuffer(device, pixelCount));
  const speedsBuffer = track(createOutputBuffer(device, pixelCount));
  const readbackBuffer = track(createOutputBuffer(device, pixelCount));
  const validityBuffer = track(createOutputBuffer(device, pixelCount));
  const texture = device.createTexture({
    format: 'r32float',
    width,
    height,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
  });
  const textureView = graph.createTextureView(
    graph.importTexture(
      {
        id: 'lic-texture',
        format: 'r32float',
        width,
        height,
        usage: texture.props.usage
      },
      texture
    ),
    {mipLevelCount: 1}
  );
  graph.add(
    new GPULineIntegralConvolution({
      id: 'lic',
      velocities: importGraphBuffer(
        graph,
        'field',
        track(createInputBuffer(device, field.velocities)),
        'float32x2',
        field.width * field.height
      ),
      fieldWidth: field.width,
      fieldHeight: field.height,
      grid: makeGPURasterMetadata({width, height}),
      stepCount,
      parameters: parameterBuffer.importToGraph(graph),
      uint32Parameters: wordBuffer.importToGraph(graph),
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pixelCount),
        speeds: importGraphBuffer(graph, 'speeds', speedsBuffer, 'float32', pixelCount),
        texture: textureView as never
      }
    })
  );
  new GPURasterTextureToBuffer({
    id: 'texture-readback',
    input: {
      id: 'texture-band',
      format: 'float32',
      storage: {kind: 'texture', view: textureView}
    },
    output: importGraphBuffer(graph, 'texture-readback', readbackBuffer, 'float32', pixelCount),
    outputValidity: importGraphBuffer(
      graph,
      'texture-validity',
      validityBuffer,
      'uint32',
      pixelCount
    )
  }).addToGraph(graph);
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPULineIntegralConvolutionParameterValues(settings));
      wordBuffer.write(getGPULineIntegralConvolutionUint32ParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        values: Float32Array.from(await readFloat32(valuesBuffer, pixelCount)),
        speeds: Float32Array.from(await readFloat32(speedsBuffer, pixelCount)),
        texture: Float32Array.from(await readFloat32(readbackBuffer, pixelCount))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      wordBuffer.destroy();
      texture.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Fraction of pixels whose GPU value differs from the oracle by more than `tolerance`. */
function getMismatchFraction(actual: Float32Array, expected: Float32Array, tolerance: number) {
  let mismatches = 0;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(Number.isNaN(actual[index])).toBe(true);
    } else if (!(Math.abs(actual[index] - expected[index]) <= tolerance)) {
      mismatches++;
    }
  }
  return mismatches / expected.length;
}

it('GPULineIntegralConvolution matches the CPU oracle and writes the texture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 32 x 32 vortex over [0, 32]^2 with a NaN cell, rendered at 64 x 64.
  const field = createVortexField(32, 0.1);
  field.velocities[2 * (5 * 32 + 5)] = Number.NaN;
  const fixture = createFixture(device, field, 64, 64, 12);
  try {
    for (const settings of [
      {
        fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
        outputPlacement: {origin: [0, 0], cellSize: [0.5, 0.5]},
        seed: 3
      },
      {
        fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
        outputPlacement: {origin: [0, 0], cellSize: [0.5, 0.5]},
        seed: 3,
        period: 6,
        phase: 0.25,
        minimumSpeed: 0.2
      }
    ] satisfies GPULineIntegralConvolutionSettings[]) {
      const actual = await fixture.run(settings);
      const expected = convolveLineIntegralOnCPU(
        field,
        64,
        64,
        12,
        getGPULineIntegralConvolutionParameterValues(settings),
        settings.seed
      );
      // cos/normalize rounding may move a rare sample across a pixel edge; bound how often.
      expect(getMismatchFraction(actual.values, expected.values, 1e-4)).toBeLessThan(0.02);
      expect(getMismatchFraction(actual.speeds, expected.speeds, 1e-4)).toBe(0);
      expect(Array.from(actual.texture)).toEqual(Array.from(actual.values));
      // The NaN cell blanks the pixels whose bilinear footprint touches it.
      expect(Number.isNaN(actual.values[10 * 64 + 10])).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPULineIntegralConvolution animates by phase without rebuilding and replays bitwise', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const field = createVortexField(16, 0.2);
  const fixture = createFixture(device, field, 32, 32, 8);
  try {
    const base: GPULineIntegralConvolutionSettings = {
      fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
      outputPlacement: {origin: [0, 0], cellSize: [0.5, 0.5]},
      seed: 1,
      period: 4
    };
    const first = await fixture.run({...base, phase: 0});
    const moved = await fixture.run({...base, phase: 0.5});
    const again = await fixture.run({...base, phase: 0});
    expect(new Uint32Array(again.values.buffer)).toEqual(new Uint32Array(first.values.buffer));
    let changed = 0;
    for (let index = 0; index < first.values.length; index++) {
      changed += Math.abs(first.values[index] - moved.values[index]) > 1e-3 ? 1 : 0;
    }
    expect(changed).toBeGreaterThan(first.values.length / 2);
    // A new seed changes the noise on the same graph.
    const reseeded = await fixture.run({...base, phase: 0, seed: 2});
    expect(reseeded.values).not.toEqual(first.values);
  } finally {
    fixture.destroy();
  }
});

it('GPULineIntegralConvolution traces tangent to the field on non-square output cells (raster-fields-01)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Uniform field (1, 1): every streamline is a straight line at 45 degrees in world space. The
  // output has 2 x 1 world-unit cells (16 x 32 pixels over [0, 32]^2), so the same line runs along
  // (1, 2) / sqrt(5) in pixel space. Scaling each axis by its own cell size (the old rule) ran
  // along (1, 1) / sqrt(2) in pixel space, which is 26.6 degrees in world space.
  const width = 16;
  const height = 32;
  const stepCount = 6;
  const stepLength = 1;
  const field = createUniformField(4, 4, 1, 1);
  const fixture = createFixture(device, field, width, height, stepCount);
  try {
    const settings: GPULineIntegralConvolutionSettings = {
      fieldPlacement: {origin: [-16, -16], cellSize: [16, 16]},
      outputPlacement: {origin: [0, 0], cellSize: [2, 1]},
      stepLength,
      seed: 11
    };
    const actual = await fixture.run(settings);
    // Closed form, independent of the kernel's integrator: sample k sits stepLength * k pixels along
    // the pixel-space image of the world diagonal, the weights are the Hann window, and a trace
    // ends at the first sample outside the output.
    const getExpected = (pixelDirection: [number, number]) => {
      const expected = new Float32Array(width * height);
      let minimumMargin = 1;
      for (let row = 0; row < height; row++) {
        for (let column = 0; column < width; column++) {
          let weightSum = 1;
          let sum = getLineIntegralConvolutionNoise(column, row, settings.seed ?? 0);
          for (const directionSign of [1, -1]) {
            for (let step = 1; step <= stepCount; step++) {
              const x = column + 0.5 + directionSign * step * stepLength * pixelDirection[0];
              const y = row + 0.5 + directionSign * step * stepLength * pixelDirection[1];
              if (x < 0 || y < 0 || x >= width || y >= height) {
                break;
              }
              minimumMargin = Math.min(
                minimumMargin,
                Math.abs(x - Math.round(x)),
                Math.abs(y - Math.round(y))
              );
              const weight = 0.5 * (1 + Math.cos((Math.PI * step) / (stepCount + 1)));
              sum +=
                weight *
                getLineIntegralConvolutionNoise(Math.floor(x), Math.floor(y), settings.seed ?? 0);
              weightSum += weight;
            }
          }
          expected[row * width + column] = sum / weightSum;
        }
      }
      return {expected, minimumMargin};
    };
    const tangent = getExpected([1 / Math.sqrt(5), 2 / Math.sqrt(5)]);
    const perAxis = getExpected([Math.SQRT1_2, Math.SQRT1_2]);
    // No sample is close enough to a pixel edge for float32 rounding to change its pixel.
    expect(tangent.minimumMargin).toBeGreaterThan(1e-3);
    let tangentMismatches = 0;
    let perAxisMismatches = 0;
    for (let index = 0; index < width * height; index++) {
      tangentMismatches += Math.abs(actual.values[index] - tangent.expected[index]) > 1e-5 ? 1 : 0;
      perAxisMismatches += Math.abs(actual.values[index] - perAxis.expected[index]) > 1e-2 ? 1 : 0;
    }
    expect(tangentMismatches).toBe(0);
    // The old per-axis rule is a different picture, so this spec fails on it.
    expect(perAxisMismatches).toBeGreaterThan((width * height) / 2);
  } finally {
    fixture.destroy();
  }
});
