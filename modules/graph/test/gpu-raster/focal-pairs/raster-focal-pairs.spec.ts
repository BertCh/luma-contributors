// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterFocalPairs,
  type GPURasterFocalPairsOutput
} from '../../../src/gpu-raster/focal-pairs/index';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import type {GPURasterBand} from '../../../src/gpu-raster/types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import terraReference from '../../oracles/fixtures/raster-terra.json';
import {
  FIXTURES,
  METRICS,
  TOLERANCE,
  type Metric,
  type ReferenceBand,
  type FocalPairsFixture
} from './focal-pairs-fixtures';

const PREFIX_LENGTH = 4;
type Results = Record<Metric, number[]> & {pairedCount: number[]; status: number[]};

function createFixture(
  device: Device,
  fixture: Omit<FocalPairsFixture, 'expected'>,
  integerInput = false,
  textureInput = false
) {
  const graph = new GPUCommandGraph(device);
  const count = fixture.width * fixture.height;
  const buffers: Buffer[] = [];
  const textures: Texture[] = [];
  const sourceBuffers: Record<string, Buffer> = {};
  let serial = 0;
  const importView = <Format extends 'float32' | 'uint32' | 'sint32'>(
    buffer: Buffer,
    format: Format
  ): GraphDataView<Format> => {
    const handle = graph.importBuffer(
      {id: `buffer-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: count, byteOffset: PREFIX_LENGTH * 4});
  };
  const createBand = (id: string, source: ReferenceBand): GPURasterBand => {
    const values = [
      ...Array(PREFIX_LENGTH).fill(-12345),
      ...source.values.map(value => value ?? NaN)
    ];
    const buffer = createInputBuffer(
      device,
      integerInput ? Int32Array.from(values) : Float32Array.from(values)
    );
    const validityBuffer = createInputBuffer(
      device,
      Uint32Array.from([
        ...Array(PREFIX_LENGTH).fill(0),
        ...(source.validity ?? Array(count).fill(1))
      ])
    );
    buffers.push(buffer, validityBuffer);
    sourceBuffers[id] = buffer;
    sourceBuffers[`${id}-validity`] = validityBuffer;
    const metadata = {
      id,
      scale: source.scale,
      offset: source.offset,
      noDataValue: source.noDataValue,
      validity: importView(validityBuffer, 'uint32')
    };
    if (textureInput) {
      const texture = device.createTexture({
        id,
        format: 'r32float',
        width: fixture.width,
        height: fixture.height,
        usage: Texture.SAMPLE | Texture.COPY_DST
      });
      texture.writeData(Float32Array.from(source.values.map(value => value ?? NaN)));
      textures.push(texture);
      const handle = graph.importTexture(
        {
          id,
          format: 'r32float',
          width: fixture.width,
          height: fixture.height,
          usage: texture.props.usage
        },
        texture
      );
      return {
        ...metadata,
        format: 'float32',
        storage: {kind: 'texture', view: graph.createTextureView(handle, {mipLevelCount: 1})}
      };
    }
    return integerInput
      ? {
          ...metadata,
          format: 'sint32',
          storage: {kind: 'buffer', values: importView(buffer, 'sint32')}
        }
      : {
          ...metadata,
          format: 'float32',
          storage: {kind: 'buffer', values: importView(buffer, 'float32')}
        };
  };
  const outputBuffers: Record<string, Buffer> = {};
  const createOutput = <Format extends 'float32' | 'uint32'>(name: string, format: Format) => {
    const buffer = createOutputBuffer(device, count + PREFIX_LENGTH);
    buffers.push(buffer);
    outputBuffers[name] = buffer;
    return importView(buffer, format);
  };
  const output: GPURasterFocalPairsOutput = {
    pairedCount: createOutput('pairedCount', 'uint32'),
    status: createOutput('status', 'uint32')
  };
  for (const metric of METRICS) output[metric] = createOutput(metric, 'float32');
  graph.add(
    new GPURasterFocalPairs({
      ...fixture,
      grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
      predictor: createBand('predictor', fixture.predictor),
      response: createBand('response', fixture.response),
      output
    })
  );
  const compiled = graph.compile();
  return {
    run: async (): Promise<Results> => {
      submitGraph(device, compiled, undefined);
      const result = {} as Results;
      for (const metric of METRICS)
        result[metric] = (await readFloat32(outputBuffers[metric], count + PREFIX_LENGTH)).slice(
          PREFIX_LENGTH
        );
      result.pairedCount = (
        await readUint32(outputBuffers['pairedCount'], count + PREFIX_LENGTH)
      ).slice(PREFIX_LENGTH);
      result.status = (await readUint32(outputBuffers['status'], count + PREFIX_LENGTH)).slice(
        PREFIX_LENGTH
      );
      return result;
    },
    updateResponse: (values: number[]) =>
      sourceBuffers['response'].write(
        Float32Array.from([...Array(PREFIX_LENGTH).fill(-12345), ...values])
      ),
    updatePredictorValidity: (values: number[]) =>
      sourceBuffers['predictor-validity'].write(
        Uint32Array.from([...Array(PREFIX_LENGTH).fill(0), ...values])
      ),
    destroy: () => {
      compiled.destroy();
      expect(buffers.every(buffer => !buffer.destroyed)).toBe(true);
      expect(textures.every(texture => !texture.destroyed)).toBe(true);
      for (const buffer of buffers) buffer.destroy();
      for (const texture of textures) texture.destroy();
    }
  };
}

function expectClose(actual: number[], expected: (number | null)[], label: string) {
  expect(actual).toHaveLength(expected.length);
  for (const [index, value] of expected.entries()) {
    if (value === null || Number.isNaN(value))
      expect(actual[index], `${label}[${index}]`).toBeNaN();
    else
      expect(
        Math.abs(actual[index] - value),
        `${label}[${index}]: ${actual[index]} versus ${value}`
      ).toBeLessThanOrEqual(TOLERANCE.absolute + TOLERANCE.relative * Math.abs(value));
  }
}
function expectReference(actual: Results, expected: FocalPairsFixture['expected']) {
  expect(actual.pairedCount).toEqual(expected.pairedCount);
  expect(actual.status).toEqual(expected.status);
  for (const metric of METRICS) expectClose(actual[metric], expected[metric], metric);
}

for (const fixture of FIXTURES) {
  it(`RS05 matches independent NumPy paired windows: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createFixture(device, fixture);
    try {
      expectReference(await context.run(), fixture.expected);
    } finally {
      context.destroy();
    }
  });
}

it('RS05 updates values and masks on the same compiled graph without destroying borrowed buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positive = FIXTURES.find(fixture => fixture.name === 'positive')!;
  const negative = FIXTURES.find(fixture => fixture.name === 'negative')!;
  const context = createFixture(device, positive);
  try {
    expectReference(await context.run(), positive.expected);
    context.updatePredictorValidity(Array(20).fill(0));
    const empty = await context.run();
    expect(empty.pairedCount).toEqual(Array(20).fill(0));
    expect(empty.status).toEqual(Array(20).fill(1));
    for (const metric of METRICS) expect(empty[metric].every(Number.isNaN)).toBe(true);
    context.updatePredictorValidity(Array(20).fill(1));
    context.updateResponse(negative.response.values.map(value => value!));
    expectReference(await context.run(), negative.expected);
  } finally {
    context.destroy();
  }
});

it('RS05 resolves native integer nodata before calibration', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = FIXTURES.find(value => value.name === 'calibrated-masks-ignore-center-False')!;
  const integerFixture = {
    ...fixture,
    predictor: {...fixture.predictor, values: fixture.predictor.values.map(value => value ?? -999)}
  };
  const context = createFixture(device, integerFixture, true);
  try {
    expectReference(await context.run(), fixture.expected);
  } finally {
    context.destroy();
  }
});

it('RS05 reports nonfinite moments instead of returning a plausible overflowed fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createFixture(device, {
    name: 'overflow',
    width: 3,
    height: 3,
    radius: 1,
    predictor: {values: Array.from({length: 9}, (_, index) => index * 1e30)},
    response: {values: Array.from({length: 9}, (_, index) => index * 2e30)}
  });
  try {
    const actual = await context.run();
    expect(actual.status.every(code => code === 16)).toBe(true);
    for (const metric of METRICS) expect(actual[metric].every(Number.isNaN)).toBe(true);
  } finally {
    context.destroy();
  }
});

for (const borderMode of ['nodata', 'clamp', 'reflect'] as const) {
  it(`RS05 halo-expanded tiles reproduce the whole domain with ${borderMode} borders`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const width = 11;
    const height = 9;
    const radius = 2;
    const wholeFixture = {
      name: 'halo',
      width,
      height,
      radius,
      borderMode,
      predictor: {
        values: Array.from({length: width * height}, (_, index) => (index % 17) - 8),
        validity: Array.from({length: width * height}, (_, index) => Number(index % 19 !== 0))
      },
      response: {
        values: Array.from({length: width * height}, (_, index) => ((index * 7) % 23) - 11),
        validity: Array.from({length: width * height}, (_, index) => Number(index % 13 !== 0))
      }
    };
    const whole = createFixture(device, wholeFixture);
    try {
      const expected = await whole.run();
      let retained = 0;
      for (const [firstColumn, lastColumn] of [
        [0, 5],
        [5, width]
      ]) {
        for (const [firstRow, lastRow] of [
          [0, 4],
          [4, height]
        ]) {
          const sourceLeft = Math.max(0, firstColumn - radius);
          const sourceTop = Math.max(0, firstRow - radius);
          const sourceRight = Math.min(width, lastColumn + radius);
          const sourceBottom = Math.min(height, lastRow + radius);
          const selected = Array.from(
            {length: (sourceRight - sourceLeft) * (sourceBottom - sourceTop)},
            (_, index) =>
              (sourceTop + Math.floor(index / (sourceRight - sourceLeft))) * width +
              sourceLeft +
              (index % (sourceRight - sourceLeft))
          );
          const crop = (band: typeof wholeFixture.predictor) => ({
            values: selected.map(index => band.values[index]),
            validity: selected.map(index => band.validity[index])
          });
          const tile = createFixture(device, {
            ...wholeFixture,
            width: sourceRight - sourceLeft,
            height: sourceBottom - sourceTop,
            predictor: crop(wholeFixture.predictor),
            response: crop(wholeFixture.response)
          });
          try {
            const actual = await tile.run();
            for (let row = firstRow; row < lastRow; row++) {
              for (let column = firstColumn; column < lastColumn; column++) {
                const tileIndex =
                  (row - sourceTop) * (sourceRight - sourceLeft) + column - sourceLeft;
                const wholeIndex = row * width + column;
                expect(actual.pairedCount[tileIndex]).toBe(expected.pairedCount[wholeIndex]);
                expect(actual.status[tileIndex]).toBe(expected.status[wholeIndex]);
                for (const metric of METRICS)
                  expectClose(
                    [actual[metric][tileIndex]],
                    [expected[metric][wholeIndex]],
                    `${metric} tile core ${column},${row}`
                  );
                retained++;
              }
            }
          } finally {
            tile.destroy();
          }
        }
      }
      expect(retained).toBe(width * height);
    } finally {
      whole.destroy();
    }
  });
}

it('RS05 canonicalizes masked calibrated texture bands with the same paired window contract', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = FIXTURES.find(value => value.name === 'calibrated-masks-ignore-center-False')!;
  const context = createFixture(device, fixture, false, true);
  try {
    expectReference(await context.run(), fixture.expected);
  } finally {
    context.destroy();
  }
});

for (const entry of terraReference.cases) {
  it(`RS05 direct terra1.9.50 focalPairs/focalReg: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createFixture(device, {
      ...entry,
      borderMode: entry.borderMode as 'nodata' | 'clamp' | 'reflect',
      noDataPolicy: entry.noDataPolicy as 'ignore' | 'propagate',
      covarianceNormalization: 'sample'
    });
    try {
      const actual = await context.run();
      expect(actual.pairedCount).toEqual(entry.expected.pairedCount);
      for (const metric of ['covariance', 'correlation', 'intercept', 'slope'] as const)
        expectClose(actual[metric], entry.expected[metric], metric);
    } finally {
      context.destroy();
    }
  });
}
