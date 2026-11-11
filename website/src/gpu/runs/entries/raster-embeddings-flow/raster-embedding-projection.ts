// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingProjection: every vector minus the scene mean, projected on the first three
// principal components and stretched to [0, 1] by `stretchRanges` (here
// `getRasterEmbeddingStretchRanges(eigenvalues, sigmaCount)`). That is the false-color view of an
// embedding; one stretched channel is drawn and the probe shows all three. The components come from
// `getRasterEmbeddingPrincipalComponents` on a host covariance of six standardized Sentinel-2 bands.

import {
  getGPURasterEmbeddingProjectionParameterValues,
  getRasterEmbeddingPrincipalComponents,
  getRasterEmbeddingStretchRanges,
  GPURasterEmbeddingProjection
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {formatNumber} from '../../interaction';
import {createEmbeddingRaster, DIMENSIONS, getPixelAt, getPixelBox} from './_lib/embedding';
import {createColoredBars, getColoredBarLayers, writeColoredBars} from './_lib/insets';
import {addComponentExtract} from './_lib/kernels';
import {parameters} from './raster-embedding-projection.parameters';

const COMPONENT_COUNT = 3;

export default defineOperationRun({
  parameters,
  caption:
    "Vectors projected on their first three principal components and stretched; bars are one pixel's three scores. Six Sentinel-2 bands stand in for a learned embedding.",
  createInputs: ({aspect}) => {
    const raster = createEmbeddingRaster(aspect);
    const mean = new Float32Array(DIMENSIONS);
    for (let pixel = 0; pixel < raster.pixelCount; pixel++) {
      for (let band = 0; band < DIMENSIONS; band++) {
        mean[band] += raster.vectors[pixel * DIMENSIONS + band] / raster.pixelCount;
      }
    }
    const covariance = new Float64Array(DIMENSIONS * DIMENSIONS);
    for (let pixel = 0; pixel < raster.pixelCount; pixel++) {
      for (let row = 0; row < DIMENSIONS; row++) {
        for (let column = 0; column < DIMENSIONS; column++) {
          covariance[row * DIMENSIONS + column] +=
            ((raster.vectors[pixel * DIMENSIONS + row] - mean[row]) *
              (raster.vectors[pixel * DIMENSIONS + column] - mean[column])) /
            (raster.pixelCount - 1);
        }
      }
    }
    const principal = getRasterEmbeddingPrincipalComponents(
      covariance,
      DIMENSIONS,
      COMPONENT_COUNT
    );
    return {...raster, mean, principal};
  },
  build(builder, raster) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const rgb = builder.allocate('rgb', 'float32', raster.pixelCount * 3);
    builder.graph.add(
      new GPURasterEmbeddingProjection({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: builder.upload('vectors', 'float32', raster.vectors).view,
        componentCount: COMPONENT_COUNT,
        mean: builder.upload('mean', 'float32', raster.mean).view,
        components: builder.upload(
          'components',
          'float32',
          Float32Array.from(raster.principal.components)
        ).view,
        parameters: parameterBuffer.view,
        output: {rgb: rgb.view}
      })
    );
    const component = builder.allocate('component', 'uint32', 1);
    const channel = builder.allocate('channel', 'float32', raster.pixelCount);
    addComponentExtract(builder, 'channel-extract', rgb, 3, component, channel);
    return {
      outputs: {
        rgb,
        channel,
        inset: createColoredBars(builder, 'inset', 3, raster.columns / raster.rows)
      },
      update: values => {
        parameterBuffer.write(
          getGPURasterEmbeddingProjectionParameterValues({
            stretchRanges: getRasterEmbeddingStretchRanges(
              raster.principal.eigenvalues,
              values.sigmaCount
            )
          })
        );
        component.write(Uint32Array.of(Math.round(values.channel)));
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'probe a pixel',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const rgb = peek(outputs.rgb);
      const channels = Float32Array.from({length: 3}, (_, channel) =>
        rgb ? rgb[index * 3 + channel] : 0
      );
      writeColoredBars(outputs.inset, channels);
      const explained = inputs.principal.explainedVarianceRatio;
      return {
        spotlight: null,
        glyphs: [getPixelBox(inputs, world, index)],
        readout: `scores ${Array.from(channels, value => formatNumber(value)).join(', ')} · PC1 ${(100 * explained[0]).toFixed(0)}% var`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'raster',
      values: outputs.channel,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 1],
      smoothing: true
    },
    ...getColoredBarLayers(outputs.inset, theme)
  ]
});
