// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPURasterContrastEqualization,
  getGPURasterContrastEqualizationParameterValues,
  GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-contrast-equalization/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from './reference.json';

type Scene = {
  name: string;
  mode: string;
  width: number;
  height: number;
  binCount: number;
  tileWidth?: number;
  tileHeight?: number;
  clipLimit: number;
  values: (number | null)[];
  mask?: number[];
  reference?: (number | null)[];
  referenceMask?: number[];
  expected: (number | null)[];
  equalizeAdapthist?: (number | null)[];
};

function createHarness(device: Device, scene: Scene, calibrated = false) {
  const graph = new GPUCommandGraph(device);
  const count = scene.width * scene.height;
  const resources: {destroy(): void}[] = [];
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH
  });
  resources.push(parameters);
  const grid = makeGPURasterMetadata({width: scene.width, height: scene.height});
  const sourceBuffer = createInputBuffer(
    device,
    Float32Array.from(scene.values, value =>
      value === null ? NaN : calibrated ? (value - 2) / 0.5 : value
    )
  );
  const maskBuffer = createInputBuffer(
    device,
    Uint32Array.from(scene.mask ?? scene.values.map(() => 1))
  );
  resources.push(sourceBuffer, maskBuffer);
  const source = makeGPURasterBand(
    importGraphBuffer(graph, 'input', sourceBuffer, 'float32', count),
    grid,
    {
      validity: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', count),
      noDataValue: -9999,
      ...(calibrated ? {scale: 0.5, offset: 2} : {})
    }
  );
  let referenceValuesBuffer: ReturnType<typeof createInputBuffer> | undefined;
  const referenceInput = scene.reference
    ? (() => {
        const referenceGrid = makeGPURasterMetadata({width: scene.reference!.length, height: 1});
        const buffer = createInputBuffer(
          device,
          Float32Array.from(scene.reference!, value => value ?? NaN)
        );
        referenceValuesBuffer = buffer;
        const validityBuffer = createInputBuffer(
          device,
          Uint32Array.from(scene.referenceMask ?? scene.reference!.map(() => 1))
        );
        resources.push(buffer, validityBuffer);
        return {
          grid: referenceGrid,
          input: makeGPURasterBand(
            importGraphBuffer(graph, 'reference', buffer, 'float32', scene.reference!.length),
            referenceGrid,
            {
              validity: importGraphBuffer(
                graph,
                'reference-mask',
                validityBuffer,
                'uint32',
                scene.reference!.length
              )
            }
          )
        };
      })()
    : undefined;
  // Extra output rows start with a sentinel; a rounded-up workgroup must not write padding.
  const valuesBuffer = createOutputBuffer(device, count + 3);
  const validityBuffer = createOutputBuffer(device, count + 3);
  const approximationBuffer = createOutputBuffer(device, 1);
  valuesBuffer.write(new Float32Array(count + 3).fill(-123));
  validityBuffer.write(new Uint32Array(count + 3).fill(123));
  resources.push(valuesBuffer, validityBuffer, approximationBuffer);
  graph.add(
    new GPURasterContrastEqualization({
      grid,
      input: source,
      mode: scene.mode as 'clahe' | 'match-histograms',
      reference: referenceInput,
      binCount: scene.binCount,
      tileWidth: scene.tileWidth,
      tileHeight: scene.tileHeight,
      parameters: parameters.importToGraph(graph),
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', count + 3),
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', count + 3),
        status: {
          approximation: importGraphBuffer(graph, 'approximation', approximationBuffer, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(
      clipLimit = scene.clipLimit,
      inputDomain: readonly [number, number] = [0, 1],
      referenceDomain: readonly [number, number] = [0, 1]
    ) {
      parameters.write(
        getGPURasterContrastEqualizationParameterValues({clipLimit, inputDomain, referenceDomain})
      );
      submitGraph(device, compiled, undefined);
      const values = await readFloat32(valuesBuffer, count + 3);
      const validity = await readUint32(validityBuffer, count + 3);
      expect(values.slice(count)).toEqual([-123, -123, -123]);
      expect(validity.slice(count)).toEqual([123, 123, 123]);
      expect(await readUint32(approximationBuffer, 1)).toEqual([1]);
      return {values: values.slice(0, count), validity: validity.slice(0, count)};
    },
    replaceReference(values: Float32Array) {
      referenceValuesBuffer!.write(values);
    },
    replaceSource(values: Float32Array, mask?: Uint32Array) {
      sourceBuffer.write(values);
      if (mask) maskBuffer.write(mask);
    },
    destroy() {
      compiled.destroy();
      for (const resource of resources.reverse()) resource.destroy();
    }
  };
}

function expectOracle(result: {values: number[]; validity: number[]}, scene: Scene): void {
  expect(result.validity).toEqual(scene.expected.map(value => Number(value !== null)));
  for (let index = 0; index < scene.expected.length; index++) {
    const expected = scene.expected[index];
    if (expected === null) {
      expect(Number.isNaN(result.values[index])).toBe(true);
    } else {
      // CLAHE omits scikit-image's final 14-bit integer truncation. Matching rounds float32 CDFs.
      const tolerance = scene.mode === 'clahe' ? 1.1 / 16383 : 3e-6;
      expect(Math.abs(result.values[index] - expected), `${scene.name} cell ${index}`).toBeLessThan(
        tolerance
      );
    }
  }
  if (scene.equalizeAdapthist && scene.name !== 'constant' && scene.name !== 'one-cell') {
    const minimum = Math.min(...result.values);
    const range = Math.max(...result.values) - minimum;
    const expectedRange =
      Math.max(...(scene.expected as number[])) - Math.min(...(scene.expected as number[]));
    for (let index = 0; index < result.values.length; index++) {
      expect(
        Math.abs((result.values[index] - minimum) / range - scene.equalizeAdapthist[index]!)
      ).toBeLessThan(3 / (16383 * expectedRange));
    }
  }
}

describe('CART-03 scikit-image raster contrast oracles', () => {
  for (const scene of reference.cases as Scene[]) {
    it(scene.name, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createHarness(device, scene);
      try {
        expectOracle(await harness.run(), scene);
      } finally {
        harness.destroy();
      }
    });
  }

  it('keeps an underflowing positive clip limit distinct from zero on an 8 by 8 constant tile', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // All 64 samples start in bin 126. At threshold 1, the residual stride is
    // floor(255 / 63) = 4: bins 0, 4, ..., 252 gain one. At bin 126 the CDF is
    // 32 redistributed counts plus its original clipped count, hence 33/64.
    const mapped = Math.floor((16383 * 33) / 64) / 16383;
    const scene: Scene = {
      name: 'positive-clip-underflow',
      mode: 'clahe',
      width: 8,
      height: 8,
      tileWidth: 8,
      tileHeight: 8,
      binCount: 256,
      clipLimit: 1e-50,
      values: new Array(64).fill(0.5),
      expected: new Array(64).fill(mapped)
    };
    const harness = createHarness(device, scene);
    try {
      const unclipped = await harness.run(0);
      expect(unclipped.validity).toEqual(new Array(64).fill(1));
      for (const value of unclipped.values) expect(Math.abs(value - 1)).toBeLessThan(2e-6);
      const clipped = await harness.run();
      expectOracle(clipped, scene);
      for (const value of clipped.values) expect(Math.abs(value - mapped)).toBeLessThan(2e-6);
      expectOracle(await harness.run(2 ** -126), scene);
    } finally {
      harness.destroy();
    }
  });

  it('rejects a subnormal normalization width before executing the graph', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const minimum = 2 ** -126;
    const maximum = minimum + 2 ** -149;
    const scene: Scene = {
      name: 'subnormal-domain-width',
      mode: 'match-histograms',
      width: 2,
      height: 1,
      binCount: 2,
      clipLimit: 0,
      values: [minimum, maximum],
      reference: [0, 1],
      expected: [0, 1]
    };
    const harness = createHarness(device, scene);
    try {
      await expect(harness.run(0, [minimum, maximum])).rejects.toThrow(
        /inputDomain.*normal float32/
      );
      // The narrowest supported domain still preserves its two endpoint bins.
      harness.replaceSource(Float32Array.from([0, minimum]));
      expectOracle(await harness.run(0, [0, minimum]), scene);
    } finally {
      harness.destroy();
    }
  });

  it('preserves adjacent CDF knots at maximum raster counts', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = reference.largeHistogramMatching;
    const sourceCount = fixture.sourceHistogram.reduce((sum, count) => sum + count, 0);
    const referenceCount = fixture.referenceHistogram.reduce((sum, count) => sum + count, 0);
    expect(sourceCount).toBe(16777216);
    expect(referenceCount).toBe(12582912);
    const graph = new GPUCommandGraph(device);
    const grid = makeGPURasterMetadata({width: 4096, height: sourceCount / 4096});
    const referenceGrid = makeGPURasterMetadata({width: 4096, height: referenceCount / 4096});
    // WebGPU initializes these buffers to zero. Only the final sample is one: no large JS arrays.
    const sourceBuffer = createOutputBuffer(device, sourceCount);
    const referenceBuffer = createOutputBuffer(device, referenceCount);
    const valuesBuffer = createOutputBuffer(device, sourceCount);
    const validityBuffer = createOutputBuffer(device, sourceCount);
    const approximationBuffer = createOutputBuffer(device, 1);
    const parameters = new GPUParameterBuffer(device, {
      id: 'large-count-parameters',
      format: 'float32',
      length: GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH,
      values: getGPURasterContrastEqualizationParameterValues()
    });
    const resources = [
      sourceBuffer,
      referenceBuffer,
      valuesBuffer,
      validityBuffer,
      approximationBuffer,
      parameters
    ];
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      sourceBuffer.write(new Float32Array([1]), (sourceCount - 1) * 4);
      referenceBuffer.write(new Float32Array([1]), (referenceCount - 1) * 4);
      graph.add(
        new GPURasterContrastEqualization({
          grid,
          input: makeGPURasterBand(
            importGraphBuffer(graph, 'large-source', sourceBuffer, 'float32', sourceCount),
            grid
          ),
          mode: 'match-histograms',
          binCount: 2,
          reference: {
            grid: referenceGrid,
            input: makeGPURasterBand(
              importGraphBuffer(
                graph,
                'large-reference',
                referenceBuffer,
                'float32',
                referenceCount
              ),
              referenceGrid
            )
          },
          parameters: parameters.importToGraph(graph),
          output: {
            values: importGraphBuffer(graph, 'large-values', valuesBuffer, 'float32', sourceCount),
            validity: importGraphBuffer(
              graph,
              'large-validity',
              validityBuffer,
              'uint32',
              sourceCount
            ),
            status: {
              approximation: importGraphBuffer(
                graph,
                'large-approximation',
                approximationBuffer,
                'uint32',
                1
              )
            }
          }
        })
      );
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      // Read only four cells. The fixture is public skimage.match_histograms on the full arrays;
      // exact rational interpolation gives (sourceCount - referenceCount) / sourceCount = 1/4.
      for (const index of [0, sourceCount / 2, sourceCount - 2, sourceCount - 1]) {
        const bytes = await valuesBuffer.readAsync(index * 4, 4);
        const value = new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
        const expected = fixture.expected[Number(index === sourceCount - 1)];
        expect(Math.abs(value - expected), `cell ${index}`).toBeLessThan(3e-6);
        const flags = await validityBuffer.readAsync(index * 4, 4);
        expect(new Uint32Array(flags.buffer, flags.byteOffset, 1)[0]).toBe(1);
      }
      expect(await readUint32(approximationBuffer, 1)).toEqual([1]);
    } finally {
      compiled?.destroy();
      for (const resource of resources.reverse()) resource.destroy();
    }
  }, 60000);

  it('changes clip limit and source validity on the same compiled graph, clearing old histograms', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scenes = reference.cases.filter(scene =>
      scene.name.startsWith('odd-reflected')
    ) as Scene[];
    const harness = createHarness(device, scenes[0]);
    try {
      for (const scene of [...scenes, scenes[0]])
        expectOracle(await harness.run(scene.clipLimit), scene);
      harness.replaceSource(new Float32Array(35).fill(NaN), new Uint32Array(35));
      const empty = await harness.run();
      expect(empty.validity).toEqual(new Array(35).fill(0));
      expect(empty.values.every(Number.isNaN)).toBe(true);
      harness.replaceSource(
        Float32Array.from(scenes[0].values as number[]),
        new Uint32Array(35).fill(1)
      );
      expectOracle(await harness.run(), scenes[0]);
    } finally {
      harness.destroy();
    }
  });

  it('canonicalizes calibrated bands and changes both domains without recompiling', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = reference.cases.find(scene => scene.name === 'matching-ties-gaps') as Scene;
    const harness = createHarness(device, scene, true);
    try {
      expectOracle(await harness.run(), scene);
      // Source domain twice as wide: quantized ranks and ties are unchanged for these values.
      harness.replaceReference(Float32Array.from(scene.reference!, value => value! * 10 - 5));
      const shifted = await harness.run(scene.clipLimit, [0, 2], [-5, 5]);
      for (let index = 0; index < shifted.values.length; index++) {
        expect(Math.abs(shifted.values[index] - (scene.expected[index]! * 10 - 5))).toBeLessThan(
          3e-5
        );
      }
    } finally {
      harness.destroy();
    }
  });

  it('rejects NaN, infinity, nodata and invalid-mask samples before histogramming', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene = {
      name: 'invalids',
      mode: 'match-histograms',
      width: 9,
      height: 1,
      binCount: 16,
      clipLimit: 0,
      values: [0, 1, null, null, null, null, 0.5, 1, 0],
      mask: [1, 1, 1, 1, 1, 1, 0, 1, 1],
      reference: [0, 1],
      expected: [0, 1, null, null, null, null, null, 1, 0]
    };
    const harness = createHarness(device, scene);
    try {
      harness.replaceSource(Float32Array.from([0, 1, Infinity, -Infinity, NaN, -9999, 0.5, 1, 0]));
      // Two source zeros have quantile 1/2, identical to the reference's first occupied CDF knot.
      expectOracle(await harness.run(), scene);
    } finally {
      harness.destroy();
    }
  });
});
