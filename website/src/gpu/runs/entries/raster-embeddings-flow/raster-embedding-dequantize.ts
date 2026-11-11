// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingDequantize: AlphaEarth-style int8 rasters decode as `sign(q) * (q / 127.5)^2`,
// and `q = -128` in any band is NoData. The int8 bytes here are made on the host from six
// standardized Sentinel-2 bands (unit vectors), with a share of pixels set to NoData; the run decodes
// them on the GPU. The inset is the transfer curve; the probed pixel's six codes sit on it as dots,
// the shown band as a ring.

import {
  GPURasterEmbeddingDequantize,
  RASTER_EMBEDDING_NODATA
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {
  BAND_NAMES,
  createEmbeddingRaster,
  DIMENSIONS,
  getPixelAt,
  getPixelBox,
  packSignedBytes,
  quantizeComponent
} from './_lib/embedding';
import {createInsetBacking, getInsetBackingLayer} from '../space-time-models/_lib/inset';
import {getPaddedRectangle, type InsetFractions} from './_lib/insets';
import {addComponentExtract} from './_lib/kernels';
import {parameters} from './raster-embedding-dequantize.parameters';

const INSET: InsetFractions = [0.04, 0.06, 0.3, 0.42];

/** Decoded value of an int8 code: `sign(q) * (q / 127.5)^2`. */
function decodeCode(code: number): number {
  return Math.sign(code) * (code / 127.5) ** 2;
}

/** Code and value to world units inside the inset: x is q in [-128, 127], y the value in [-1, 1]. */
function getCurvePoint(code: number, value: number, worldWidth: number): [number, number] {
  const [x0, y0, x1, y1] = INSET;
  return [(x0 + ((code + 128) / 255) * (x1 - x0)) * worldWidth, y0 + ((value + 1) / 2) * (y1 - y0)];
}

function getCurvePolyline(worldWidth: number): Float32Array {
  const points: number[] = [];
  for (let code = -127; code <= 127; code += 4) {
    points.push(...getCurvePoint(code, decodeCode(code), worldWidth));
  }
  return Float32Array.from(points);
}

export default defineOperationRun({
  parameters,
  caption:
    "Signed bytes decode through a signed square (inset: the curve, dots are one pixel's codes); NoData pixels drop out (10% injected by default). Six Sentinel-2 bands stand in for a learned embedding.",
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect);
    const bytes = Int8Array.from(raster.vectors, quantizeComponent);
    const order = Float32Array.from({length: raster.pixelCount}, () => random());
    return {...raster, bytes, order, current: new Int8Array(bytes.length)};
  },
  build(builder, raster) {
    const quantized = builder.allocate('quantized', 'uint32', Math.ceil(raster.bytes.length / 4));
    const vectors = builder.allocate('vectors', 'float32', raster.vectors.length);
    const validity = builder.allocate('validity', 'uint32', raster.pixelCount);
    builder.graph.add(
      new GPURasterEmbeddingDequantize({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        quantized: quantized.view,
        output: {vectors: vectors.view, validity: validity.view}
      })
    );
    const component = builder.allocate('component', 'uint32', 1);
    const band = builder.allocate('band', 'float32', raster.pixelCount);
    addComponentExtract(builder, 'band-extract', vectors, DIMENSIONS, component, band);
    return {
      outputs: {
        vectors,
        band,
        backing: createInsetBacking(
          builder,
          'curve-backing',
          getPaddedRectangle(INSET, raster.columns / raster.rows)
        )
      },
      update: current => {
        raster.current.set(raster.bytes);
        for (let pixel = 0; pixel < raster.pixelCount; pixel++) {
          if (raster.order[pixel] < current.noData) {
            raster.current[pixel * DIMENSIONS + (pixel % DIMENSIONS)] = RASTER_EMBEDDING_NODATA;
          }
        }
        quantized.write(packSignedBytes(raster.current));
        component.write(Uint32Array.of(Math.round(current.band)));
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const decoded = peek(outputs.vectors);
      const components = Float32Array.from({length: DIMENSIONS}, (_, band) =>
        decoded ? decoded[index * DIMENSIONS + band] : 0
      );
      const band = Math.round(values.band);
      const byte = inputs.current[index * DIMENSIONS + band];
      return {
        spotlight: null,
        glyphs: [
          getPixelBox(inputs, world, index),
          {shape: 'polyline', points: getCurvePolyline(world.width)},
          ...Array.from({length: DIMENSIONS}, (_, item) => {
            const code = inputs.current[index * DIMENSIONS + item];
            return code === RASTER_EMBEDDING_NODATA
              ? null
              : {shape: 'dot' as const, at: getCurvePoint(code, decodeCode(code), world.width)};
          }).filter(glyph => glyph !== null),
          ...(byte === RASTER_EMBEDDING_NODATA
            ? []
            : [
                {
                  shape: 'ring' as const,
                  at: getCurvePoint(byte, decodeCode(byte), world.width),
                  radius: 0.025
                }
              ])
        ],
        readout: Number.isFinite(components[band])
          ? `${BAND_NAMES[band]} q ${byte} -> ${formatNumber(components[band], 3)}`
          : 'NoData pixel: vectors NaN'
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'raster',
      values: outputs.band,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'diverging',
      range: [-1, 1]
    },
    getInsetBackingLayer(outputs.backing, theme)
  ]
});
