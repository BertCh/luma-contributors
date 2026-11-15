// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterArithmetic,
  getGPURasterArithmeticParameterValues,
  GPURasterConditional,
  GPURasterReclassify,
  getGPURasterConditionalParameterValues,
  getGPURasterReclassifyParameterValues
} from '../../../src/raster-analysis/raster-algebra/index';
import {
  GPURasterSampling,
  GPURasterProfile,
  getGPURasterSamplingParameterValues,
  getGPURasterProfileParameterValues
} from '../../../src/raster-analysis/raster-sampling/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPURasterStretch,
  getGPURasterStretchParameterValues
} from '../../../src/raster-analysis/raster-stretch/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it.each([
  'buffer',
  'texture'
] as const)('raster algebra honours %s band validity, native nodata and calibration across frames', async storageKind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  const importedBuffers = new Map<string, Buffer>();
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    data: Float32Array | Uint32Array
  ) => {
    const buffer = createOutputBuffer(device, data.length);
    buffer.write(data);
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      format === 'float32x2' ? data.length / 2 : data.length
    );
    importedBuffers.set(view.buffer.id, buffer);
    return view;
  };
  // Nine logical cells, twelve backing rows. Tail values must never enter statistics or counts.
  const raw = new Uint32Array([888, 0, 1, 2, 99, 4, 5, 6, 7, 8, 999, 999, 999]);
  const validityBuffer = createInputBuffer(device, new Uint32Array([1, 7, 0, 1, 1, 1, 1, 1, 1]));
  buffers.push(validityBuffer);
  const validity = importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', 9);
  const texture =
    storageKind === 'texture'
      ? device.createTexture({
          format: 'r32uint',
          width: 3,
          height: 3,
          usage: Texture.SAMPLE | Texture.COPY_DST
        })
      : undefined;
  texture?.writeData(raw.subarray(1, 10));
  const rawValues = texture ? undefined : input('uint32', raw);
  const band = {
    id: 'published-input',
    format: 'uint32',
    validity,
    noDataValue: 99,
    scale: 2,
    offset: 10,
    storage: texture
      ? {
          kind: 'texture',
          view: graph.createTextureView(
            graph.importTexture(
              {
                id: 'texture',
                format: 'r32uint',
                width: 3,
                height: 3,
                usage: texture.props.usage
              },
              texture
            )
          )
        }
      : {
          kind: 'buffer',
          values: graph.createDataView(rawValues!.buffer, {
            format: 'uint32',
            byteOffset: 4,
            length: 12
          })
        }
  } as GPURasterBand;
  const mask = input('uint32', new Uint32Array([1, 1, 1, 1, 0, 1, 1, 1, 1]));
  const classes = input('uint32', new Uint32Array(12).fill(777));
  const reclassified = input('float32', new Float32Array(12).fill(777));
  const classCounts = input('uint32', new Uint32Array(4));
  graph.add(
    new GPURasterReclassify({
      id: 'classify-band',
      grid,
      values: band,
      mask,
      // A legacy sentinel must not reject a calibrated band value that happens to equal it.
      noDataValue: 10,
      breaks: input('float32', new Float32Array([12, 20, 24])),
      classValues: input('float32', new Float32Array([100, 200, 300, 400])),
      parameters: input('float32', getGPURasterReclassifyParameterValues({breakCount: 3})),
      output: {classes, reclassified, classCounts}
    })
  );
  const stretched = input('float32', new Float32Array(12).fill(777));
  const colors = input('uint32', new Uint32Array(12).fill(777));
  const statistics = input('float32', new Float32Array(8));
  const histogram = input('uint32', new Uint32Array(16));
  graph.add(
    new GPURasterStretch({
      id: 'stretch-band',
      grid,
      values: band,
      mask,
      noDataValue: 10,
      binCount: 16,
      lutSize: 2,
      regionMask: input('uint32', new Uint32Array([1, 1, 1, 1, 1, 1, 1, 1, 0])),
      palette: input('uint32', new Uint32Array([0x11223344])),
      parameters: input('float32', getGPURasterStretchParameterValues()),
      output: {stretched, colors, statistics, histogram}
    })
  );
  const selected = input('float32', new Float32Array(12).fill(777));
  const conditionResult = input('uint32', new Uint32Array(12).fill(777));
  graph.add(
    new GPURasterConditional({
      id: 'conditional-bands',
      grid,
      cellCount: 9,
      conditionValues: band,
      noDataValue: 10,
      a: {
        id: 'a',
        format: 'float32',
        scale: 3,
        offset: 1,
        noDataValue: -5,
        storage: {
          kind: 'buffer',
          values: input('float32', new Float32Array([100, 101, 102, 103, 104, -5, 106, 107, 108]))
        },
        validity: input('uint32', new Uint32Array([0, 1, 1, 1, 1, 1, 0, 1, 1]))
      },
      b: {
        id: 'b',
        format: 'float32',
        scale: -2,
        offset: 100,
        noDataValue: -5,
        storage: {
          kind: 'buffer',
          values: input('float32', new Float32Array([30, 31, 32, 33, -5, 35, 36, 37, 38]))
        },
        validity: input('uint32', new Uint32Array([1, 1, 1, 1, 1, 1, 1, 0, 1]))
      },
      parameters: input(
        'float32',
        getGPURasterConditionalParameterValues({comparison: '>=', threshold: 20})
      ),
      output: {values: selected, conditionResult}
    })
  );
  const selectedWithConstant = input('float32', new Float32Array(12).fill(777));
  graph.add(
    new GPURasterConditional({
      id: 'conditional-band-and-constant',
      grid,
      cellCount: 9,
      mask,
      a: band,
      noDataValue: 10,
      parameters: input('float32', getGPURasterConditionalParameterValues({constantB: 10})),
      output: {values: selectedWithConstant}
    })
  );
  const arithmeticConstant = input('float32', new Float32Array(12).fill(777));
  const arithmeticBands = input('float32', new Float32Array(12).fill(777));
  for (const [id, b, output] of [
    ['constant', undefined, arithmeticConstant],
    ['bands', band, arithmeticBands]
  ] as const) {
    graph.add(
      new GPURasterArithmetic({
        id: `arithmetic-${id}`,
        grid,
        cellCount: 9,
        a: band,
        b,
        mask,
        noDataValue: 10,
        parameters: input(
          'float32',
          getGPURasterArithmeticParameterValues({operation: 'add', constantB: 10})
        ),
        output: {values: output}
      })
    );
  }
  const centers = input(
    'float32x2',
    Float32Array.from(
      Array.from({length: 9}, (_, index) => [
        (index % 3) + 0.5,
        -(Math.floor(index / 3) + 0.5)
      ]).flat()
    )
  );
  const sampled = input('float32', new Float32Array(12).fill(777));
  const profiled = input('float32', new Float32Array(12).fill(777));
  graph.add(
    new GPURasterSampling({
      grid,
      values: band,
      mask,
      noDataValue: 10,
      positions: centers,
      parameters: input('float32', getGPURasterSamplingParameterValues({}, grid)),
      output: {values: sampled}
    })
  );
  graph.add(
    new GPURasterProfile({
      grid,
      values: band,
      mask,
      noDataValue: 10,
      positions: centers,
      lineOffsets: input(
        'uint32',
        Uint32Array.from({length: 10}, (_, index) => index)
      ),
      sampleCapacity: 9,
      parameters: input('float32', getGPURasterProfileParameterValues({spacing: 1}, grid)),
      output: {
        sampleValues: profiled,
        status: {
          count: input('uint32', new Uint32Array(1)),
          overflow: input('uint32', new Uint32Array(1))
        }
      }
    })
  );
  const compiled = graph.compile();
  const bufferOf = (view: typeof stretched | typeof classes) => {
    const buffer = importedBuffers.get(view.buffer.id);
    if (!buffer) throw new Error('Missing imported buffer');
    return buffer;
  };
  const expectFloats = async (view: typeof stretched, expected: number[]) => {
    const actual = await readFloat32(bufferOf(view), 12);
    for (let index = 0; index < expected.length; index++) {
      if (Number.isNaN(expected[index])) expect(actual[index], `cell ${index}`).toBeNaN();
      else expect(actual[index], `cell ${index}`).toBeCloseTo(expected[index], 5);
    }
    expect(actual.slice(9)).toEqual([777, 777, 777]);
  };
  try {
    submitGraph(device, compiled, undefined);
    await expectFloats(arithmeticConstant, [20, 22, NaN, NaN, NaN, 30, 32, 34, 36]);
    await expectFloats(arithmeticBands, [20, 24, NaN, NaN, NaN, 40, 44, 48, 52]);
    await expectFloats(sampled, [10, 12, NaN, NaN, NaN, 20, 22, 24, 26]);
    await expectFloats(profiled, [10, 12, NaN, NaN, NaN, 20, 22, 24, 26]);
    // Hand-derived from calibrated values [10,12,missing,missing,18,20,22,24,26].
    expect(await readUint32(bufferOf(classes), 12)).toEqual([
      0, 1, 0xffffffff, 0xffffffff, 0xffffffff, 2, 2, 3, 3, 777, 777, 777
    ]);
    expect(await readUint32(bufferOf(classCounts), 4)).toEqual([1, 1, 2, 2]);
    await expectFloats(reclassified, [100, 200, NaN, NaN, NaN, 300, 300, 400, 400]);
    await expectFloats(stretched, [0, 1 / 7, NaN, NaN, NaN, 5 / 7, 6 / 7, 1, 1]);
    expect(await readFloat32(bufferOf(statistics), 5)).toEqual([10, 24, 10, 24, 5]);
    expect((await readUint32(bufferOf(histogram), 16)).reduce((sum, value) => sum + value, 0)).toBe(
      5
    );
    expect(await readUint32(bufferOf(colors), 12)).toEqual([
      0x11223344, 0x11223344, 0, 0, 0, 0x11223344, 0x11223344, 0x11223344, 0x11223344, 777, 777, 777
    ]);
    // A boolean condition and a constant can mix with a band. The constant equals the legacy
    // sentinel, and the calibrated band also legitimately contains that value.
    await expectFloats(selectedWithConstant, [10, 12, NaN, NaN, 10, 20, 22, 24, 26]);
    // An invalid unchosen operand does not invalidate the selected branch (cells 0 and 7).
    await expectFloats(selected, [40, 38, NaN, NaN, NaN, NaN, NaN, 322, 325]);
    expect(await readUint32(bufferOf(conditionResult), 12)).toEqual([
      0, 0, 0, 0, 0, 1, 1, 1, 1, 777, 777, 777
    ]);
    validityBuffer.write(new Uint32Array(9));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(bufferOf(classCounts), 4)).toEqual([0, 0, 0, 0]);
    await expectFloats(reclassified, new Array(9).fill(NaN));
    await expectFloats(stretched, new Array(9).fill(NaN));
    await expectFloats(selected, new Array(9).fill(NaN));
    await expectFloats(selectedWithConstant, [NaN, NaN, NaN, NaN, 10, NaN, NaN, NaN, NaN]);
    for (const view of [arithmeticConstant, arithmeticBands, sampled, profiled]) {
      await expectFloats(view, new Array(9).fill(NaN));
    }
    const emptyStatistics = await readFloat32(bufferOf(statistics), 5);
    expect(emptyStatistics.slice(0, 4).every(Number.isNaN)).toBe(true);
    expect(emptyStatistics[4]).toBe(0);
    expect(await readUint32(bufferOf(histogram), 16)).toEqual(new Array(16).fill(0));
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    texture?.destroy();
  }
});
