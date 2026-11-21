// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterCellStatistics,
  type GPURasterCellStatisticsOutput,
  GPUWeightedOverlay,
  getGPURasterCellStatisticsParameterValues,
  getGPUWeightedOverlayParameterValues
} from '../../../src/raster-analysis/raster-algebra/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const CELL_COUNT = 9;
const MASK = [1, 1, 1, 1, 1, 0, 1, 1, 1];
// Hand-calibrated observations, including native sentinel and per-band validity exclusions.
const OBSERVATIONS = [
  [10, 12, NaN, NaN, 18, 20, 22, 24, 26],
  [10, 11, NaN, 13, NaN, 15, 16, 17, 18],
  [10, 11, 12, NaN, 14, 15, 16, NaN, 18]
];
const RAW_VALUES = [
  [0, 1, 2, 99, 4, 5, 6, 7, 8],
  [0, 2, -99, 6, 8, 10, 12, 14, 16],
  [2, 3, 4, 5, 6, 7, 8, -9, 10]
];
const VALIDITY = [
  [1, 7, 0, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 0, 1, 1, 1, 1],
  [1, 1, 1, 0, 1, 1, 1, 1, 1]
];
const FLOAT_METRICS = [
  'minimum',
  'maximum',
  'range',
  'sum',
  'mean',
  'standardDeviation',
  'majority',
  'minority'
] as const;
const INTEGER_METRICS = ['counts', 'variety'] as const;

function statisticsReference(
  values: number[],
  kept: boolean,
  propagate: boolean,
  minimumCount: number
) {
  const valid = values.filter(Number.isFinite).sort((first, second) => first - second);
  const defined =
    kept && valid.length >= minimumCount && (!propagate || valid.length === values.length);
  const sum = valid.reduce((total, value) => total + value, 0);
  const mean = sum / valid.length;
  const frequencies = [...new Set(valid)].map(value => ({
    value,
    count: valid.filter(other => other === value).length
  }));
  const majority = frequencies
    .slice()
    .sort((first, second) => second.count - first.count || first.value - second.value)[0]?.value;
  const minority = frequencies
    .slice()
    .sort((first, second) => first.count - second.count || first.value - second.value)[0]?.value;
  return {
    counts: kept ? valid.length : 0,
    variety: defined ? frequencies.length : 0,
    minimum: defined ? valid[0] : NaN,
    maximum: defined ? valid[valid.length - 1] : NaN,
    range: defined ? valid[valid.length - 1] - valid[0] : NaN,
    sum: defined ? sum : NaN,
    mean: defined ? mean : NaN,
    standardDeviation: defined
      ? Math.sqrt(valid.reduce((total, value) => total + (value - mean) ** 2, 0) / valid.length)
      : NaN,
    majority: defined ? majority! : NaN,
    minority: defined ? minority! : NaN
  };
}

function overlayReference(values: number[], kept: boolean, ignore: boolean, normalize: boolean) {
  const weights = [0.5, -0.25, 0.75];
  if (!kept || (!ignore && values.some(value => !Number.isFinite(value)))) return NaN;
  const remapped = values.map((value, layer) => {
    if (!Number.isFinite(value)) return undefined;
    if (layer === 1) return value < 12 ? 1 : value < 16 ? 0.5 : NaN;
    return layer === 0 ? value / 32 : 1 - value / 32;
  });
  if (remapped.some(Number.isNaN)) return NaN; // Restricted table class, even under ignore.
  const active = remapped.flatMap((value, layer) =>
    value === undefined ? [] : [{value, weight: weights[layer]}]
  );
  if (!active.length) return NaN;
  const score = active.reduce((total, item) => total + item.value * item.weight, 0);
  return normalize
    ? score / active.reduce((total, item) => total + Math.abs(item.weight), 0)
    : score;
}

it.each([
  'buffer',
  'texture'
] as const)('stacked statistics and overlay preserve %s band validity and calibration on repeated encodings', async storageKind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: CELL_COUNT,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const buffers: Buffer[] = [];
  let sequence = 0;
  const input = <Format extends 'float32' | 'uint32' | 'sint32'>(
    format: Format,
    data: Float32Array | Uint32Array | Int32Array
  ) => {
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `buffer-${sequence++}`, buffer, format, data.length)
    };
  };
  const outputFloat = () => input('float32', new Float32Array(CELL_COUNT + 2).fill(777));
  const outputInteger = () => input('uint32', new Uint32Array(CELL_COUNT + 2).fill(777));
  const validityInputs = VALIDITY.map(values =>
    input('uint32', Uint32Array.from([777, ...values, 777, 777]))
  );
  const texture =
    storageKind === 'texture'
      ? device.createTexture({
          format: 'r32sint',
          width: CELL_COUNT,
          height: 1,
          usage: Texture.SAMPLE | Texture.COPY_DST
        })
      : undefined;
  texture?.writeData(Int32Array.from(RAW_VALUES[1]));
  const bands: GPURasterBand[] = RAW_VALUES.map((values, layer) => {
    const format = (['uint32', 'sint32', 'float32'] as const)[layer];
    const data = [777, ...values, 888, 999];
    const nativeValues = input(
      format,
      format === 'uint32'
        ? Uint32Array.from(data)
        : format === 'sint32'
          ? Int32Array.from(data)
          : Float32Array.from(data)
    );
    return {
      id: `band-${layer}`,
      format,
      scale: [2, 0.5, 1][layer],
      offset: [10, 10, 8][layer],
      noDataValue: [99, -99, -9][layer],
      validity: graph.createDataView(validityInputs[layer].view.buffer, {
        format: 'uint32',
        byteOffset: 4,
        length: CELL_COUNT + 2
      }),
      storage:
        texture && layer === 1
          ? {
              kind: 'texture',
              view: graph.createTextureView(
                graph.importTexture(
                  {
                    id: 'signed-texture',
                    format: 'r32sint',
                    width: CELL_COUNT,
                    height: 1,
                    usage: texture.props.usage
                  },
                  texture
                )
              )
            }
          : {
              kind: 'buffer',
              values: graph.createDataView(nativeValues.view.buffer, {
                format,
                byteOffset: 4,
                length: CELL_COUNT + 2
              })
            }
    } as GPURasterBand;
  });
  const mask = input('uint32', Uint32Array.from(MASK)).view;
  const statistics = (
    [
      ['ignore', 1],
      ['propagate', 1],
      ['ignore', 2]
    ] as const
  ).map(([noDataPolicy, minimumValidCount], index) => {
    const floats = Object.fromEntries(FLOAT_METRICS.map(name => [name, outputFloat()])) as Record<
      (typeof FLOAT_METRICS)[number],
      ReturnType<typeof outputFloat>
    >;
    const integers = Object.fromEntries(
      INTEGER_METRICS.map(name => [name, outputInteger()])
    ) as Record<(typeof INTEGER_METRICS)[number], ReturnType<typeof outputInteger>>;
    graph.add(
      new GPURasterCellStatistics({
        id: `statistics-${index}`,
        grid,
        stack: bands,
        layerCount: bands.length,
        cellCount: CELL_COUNT,
        noDataValue: 10,
        mask,
        parameters: input(
          'float32',
          getGPURasterCellStatisticsParameterValues({noDataPolicy, minimumValidCount})
        ).view,
        output: {
          ...Object.fromEntries(
            [...Object.entries(floats), ...Object.entries(integers)].map(([name, value]) => [
              name,
              value.view
            ])
          ),
          mean: floats.mean.view
        } satisfies GPURasterCellStatisticsOutput
      })
    );
    return {noDataPolicy, minimumValidCount, floats, integers};
  });
  const remapBreaks = input('float32', Float32Array.from([0, 0, 12, 16, 0, 0])).view;
  const remapValues = input('float32', Float32Array.from([0, 0, 0, 1, 0.5, NaN, 0, 0, 0])).view;
  const overlays = [false, true].flatMap(ignore =>
    [false, true].map(normalize => {
      const score = outputFloat();
      const validity = outputInteger();
      const range = input('float32', new Float32Array(2).fill(777));
      graph.add(
        new GPUWeightedOverlay({
          id: `overlay-${ignore}-${normalize}`,
          grid,
          stack: bands,
          layerCount: bands.length,
          cellCount: CELL_COUNT,
          mask,
          noDataValue: 10,
          maximumBreakCount: 2,
          remapBreaks,
          remapValues,
          parameters: input(
            'float32',
            getGPUWeightedOverlayParameterValues({
              layers: [
                {weight: 0.5, inputMinimum: 0, inputMaximum: 32},
                {weight: -0.25, mode: 'table', breakCount: 2},
                {weight: 0.75, inputMinimum: 0, inputMaximum: 32, invert: true}
              ],
              noDataPolicy: ignore ? 'ignore' : 'propagate',
              normalizeWeights: normalize
            })
          ).view,
          output: {score: score.view, validity: validity.view, scoreRange: range.view}
        })
      );
      return {ignore, normalize, score, validity, range};
    })
  );
  const compiled = graph.compile();
  const expectValues = (actual: number[], expected: number[]) => {
    expected.forEach((value, cell) => {
      if (Number.isNaN(value)) expect(actual[cell], `cell ${cell}`).toBeNaN();
      else expect(Math.abs(actual[cell] - value), `cell ${cell}`).toBeLessThan(2e-5);
    });
    expect(actual.slice(CELL_COUNT)).toEqual([777, 777]);
  };
  try {
    for (const empty of [false, true]) {
      if (empty)
        for (const item of validityInputs)
          item.buffer.write(Uint32Array.from([777, ...new Array(CELL_COUNT).fill(0), 777, 777]));
      submitGraph(device, compiled, undefined);
      const observations = Array.from({length: CELL_COUNT}, (_, cell) =>
        OBSERVATIONS.map(values => (empty ? NaN : values[cell]))
      );
      for (const result of statistics) {
        const expected = observations.map((values, cell) =>
          statisticsReference(
            values,
            Boolean(MASK[cell]),
            result.noDataPolicy === 'propagate',
            result.minimumValidCount
          )
        );
        for (const name of FLOAT_METRICS)
          expectValues(
            await readFloat32(result.floats[name].buffer, CELL_COUNT + 2),
            expected.map(value => value[name])
          );
        for (const name of INTEGER_METRICS)
          expect(await readUint32(result.integers[name].buffer, CELL_COUNT + 2)).toEqual([
            ...expected.map(value => value[name]),
            777,
            777
          ]);
      }
      for (const result of overlays) {
        const expected = observations.map((values, cell) =>
          overlayReference(values, Boolean(MASK[cell]), result.ignore, result.normalize)
        );
        expectValues(await readFloat32(result.score.buffer, CELL_COUNT + 2), expected);
        expect(await readUint32(result.validity.buffer, CELL_COUNT + 2)).toEqual([
          ...expected.map(value => Number(Number.isFinite(value))),
          777,
          777
        ]);
        const valid = expected.filter(Number.isFinite);
        const range = await readFloat32(result.range.buffer, 2);
        if (valid.length) {
          expect(Math.abs(range[0] - Math.min(...valid))).toBeLessThan(2e-5);
          expect(Math.abs(range[1] - Math.max(...valid))).toBeLessThan(2e-5);
        } else expect(range.every(Number.isNaN)).toBe(true);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    texture?.destroy();
  }
});
