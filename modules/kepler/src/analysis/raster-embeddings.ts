// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  GPURasterEmbeddingKMeans,
  GPURasterEmbeddingNearestPixels,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingChange,
  GPURasterEmbeddingCovariance,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingQuantize,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingPyramidLevel,
  GPURasterEmbeddingSimilarity,
  GPURasterEmbeddingZoneMean,
  getRasterEmbeddingPrincipalComponents,
  getRasterEmbeddingStretchRanges,
  getGPURasterEmbeddingProjectionParameterValues,
  getGPURasterEmbeddingSimilarityParameterValues,
  getGPURasterEmbeddingChangeParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {withCatalogGpuSession, type CatalogGpuOutput} from './catalog-gpu-session';
import type {CatalogExecutionStage} from './catalog-binding-types';

/** Shared declaration for discovery and execution evidence, including the host eigensolver. */
export function describeRasterEmbeddingStages(names: readonly string[]): CatalogExecutionStage[] {
  return names.map(exportName => ({
    identity: {package: '@luma-contributors/analysis/raster-analysis', exportName},
    execution: exportName === 'getRasterEmbeddingPrincipalComponents' ? 'cpu' : 'gpu'
  }));
}

export const RASTER_EMBEDDING_OPERATIONS = [
  'kmeans',
  'nearest-prototype',
  'nearest-pixels',
  'knn',
  'linear-probe',
  'change',
  'covariance',
  'pca',
  'projection',
  'dequantize',
  'quantize',
  'pyramid',
  'similarity',
  'zone-mean',
  'usable-mask'
] as const;
export type RasterEmbeddingOperation = (typeof RASTER_EMBEDDING_OPERATIONS)[number];

/** Bounded host tile. All vectors are pixel-major; mask applies to every band together. */
export type KeplerEmbeddingRaster = {
  width: number;
  height: number;
  dimensions: number;
  vectors: Float32Array;
  mask?: Uint32Array;
  /** AlphaEarth signed bytes, packed little-endian, four components per word. */
  quantized?: Uint32Array;
};

export type RasterEmbeddingOptions = {
  operation: RasterEmbeddingOperation;
  clusterCount?: number;
  resultCount?: number;
  neighborCount?: number;
  threshold?: number;
  scale?: number;
  queries?: Float32Array;
  trainingVectors?: Float32Array;
  trainingLabels?: Uint32Array;
  classCount?: number;
  weights?: Float32Array;
  bias?: Float32Array;
  after?: KeplerEmbeddingRaster;
  zoneRaster?: Uint32Array;
  zoneCount?: number;
};

export type RasterEmbeddingResult = {
  width: number;
  height: number;
  rowCount: number;
  /** Pixel, rank, band, matrix element or zone index, according to layout. */
  layout: 'pixels' | 'ranks' | 'bands' | 'matrix' | 'zones' | 'words';
  columns: Record<string, Float32Array | Uint32Array>;
  contributors: string[];
  stages?: readonly CatalogExecutionStage[];
};

/** Generated unit-vector raster: two separated embedding classes, one missing pixel. No dataset license. */
export function createKeplerEmbeddingFixture(): KeplerEmbeddingRaster {
  const width = 8;
  const height = 8;
  const dimensions = 4;
  const vectors = new Float32Array(width * height * dimensions);
  const mask = new Uint32Array(width * height).fill(1);
  for (let pixel = 0; pixel < mask.length; pixel++) {
    const values = [
      pixel % width < 4 ? 1 : -1,
      0.1 + Math.floor(pixel / width) / 20,
      0.2,
      (pixel % 3) / 10
    ];
    const length = Math.hypot(...values);
    values.forEach((value, band) => {
      vectors[pixel * dimensions + band] = value / length;
    });
  }
  mask[63] = 0;
  return {width, height, dimensions, vectors, mask};
}

export function validateKeplerEmbeddingRaster(raster: KeplerEmbeddingRaster): void {
  const {width, height, dimensions, vectors, mask} = raster;
  if (
    ![width, height, dimensions].every(Number.isInteger) ||
    width < 1 ||
    height < 1 ||
    dimensions < 1 ||
    dimensions > 128 ||
    width * height > 262144 ||
    width * height * dimensions > 33554432 ||
    vectors.length !== width * height * dimensions ||
    (mask && mask.length !== width * height)
  ) {
    throw new RangeError(
      'Embedding raster needs 1–128 bands, at most 262144 pixels and 128 MiB of pixel-major Float32 vectors; mask must match pixels.'
    );
  }
}

/** Runs existing contributors only. Readback is bounded to one admitted tile; all GPU resources are released. */
export async function runKeplerRasterEmbedding(
  device: Device,
  raster: KeplerEmbeddingRaster,
  options: RasterEmbeddingOptions
): Promise<RasterEmbeddingResult> {
  validateKeplerEmbeddingRaster(raster);
  if (!RASTER_EMBEDDING_OPERATIONS.includes(options.operation))
    throw new Error('Unknown embedding operation');
  if (options.after) {
    validateKeplerEmbeddingRaster(options.after);
    if (
      options.after.width !== raster.width ||
      options.after.height !== raster.height ||
      options.after.dimensions !== raster.dimensions
    )
      throw new Error('Change requires aligned rasters with identical dimensions');
  }
  const pixelCount = raster.width * raster.height;
  const dimensions = raster.dimensions;
  return withCatalogGpuSession(device, `kepler-embedding-${options.operation}`, async session => {
    const vectors = session.input('vectors', raster.vectors, 'float32');
    const mask = raster.mask ? session.input('mask', raster.mask, 'uint32') : undefined;
    const common = {pixelCount, dimensions, vectors, mask};
    const outputs: Record<string, CatalogGpuOutput<'float32'> | CatalogGpuOutput<'uint32'>> = {};
    const floatOutput = (name: string, count = pixelCount) => {
      const output = session.output(name, 'float32', count);
      outputs[name] = output;
      return output.view;
    };
    const integerOutput = (name: string, count = pixelCount) => {
      const output = session.output(name, 'uint32', count);
      outputs[name] = output;
      return output.view;
    };
    let width = raster.width;
    let height = raster.height;
    let rowCount = pixelCount;
    let layout: RasterEmbeddingResult['layout'] = 'pixels';
    const contributors: string[] = [];
    const add = (contributor: GPUCommandNodeProducer, name: string) => {
      session.graph.add(contributor);
      contributors.push(name);
    };
    const vectorColumns = (view: Float32Array, count: number, length: number, prefix: string) =>
      Object.fromEntries(
        Array.from({length}, (_, band) => [
          `${prefix}_${band + 1}`,
          Float32Array.from({length: count}, (_, pixel) => view[pixel * length + band])
        ])
      );
    let packedVectors: CatalogGpuOutput<'float32'> | undefined;
    let clusteringStatus: CatalogGpuOutput<'uint32'> | undefined;
    const packedLength = dimensions;
    switch (options.operation) {
      case 'kmeans': {
        const clusterCount = options.clusterCount ?? 2;
        clusteringStatus = session.output('kmeans-status', 'uint32', 3);
        add(
          new GPURasterEmbeddingKMeans({
            ...common,
            clusterCount,
            maximumIterations: 32,
            output: {
              labels: integerOutput('label'),
              centroids: session.output('centroids', 'float32', clusterCount * dimensions).view,
              counts: session.output('counts', 'uint32', clusterCount).view,
              status: clusteringStatus.view
            }
          }),
          'GPURasterEmbeddingKMeans'
        );
        break;
      }
      case 'nearest-prototype':
      case 'similarity':
      case 'nearest-pixels': {
        if (!options.queries?.length || options.queries.length % dimensions)
          throw new Error('Supply query vectors with one component per band');
        const queryCount = options.queries.length / dimensions;
        const queries = session.input('queries', options.queries, 'float32');
        if (options.operation === 'nearest-pixels') {
          const resultCount = options.resultCount ?? 8;
          rowCount = queryCount * resultCount;
          layout = 'ranks';
          add(
            new GPURasterEmbeddingNearestPixels({
              ...common,
              queries,
              queryCount,
              resultCount,
              output: {
                pixelRows: integerOutput('pixel_row', rowCount),
                scores: floatOutput('score', rowCount),
                resultCounts: session.output('result-counts', 'uint32', queryCount).view
              }
            }),
            'GPURasterEmbeddingNearestPixels'
          );
          // Keep explicit query/rank identities in the result, including unfilled sentinel slots.
          break;
        }
        add(
          new GPURasterEmbeddingSimilarity({
            ...common,
            queries,
            queryCount,
            parameters: session.input(
              'parameters',
              getGPURasterEmbeddingSimilarityParameterValues({
                matchThreshold: options.threshold ?? 0.9
              }),
              'float32'
            ),
            output: {
              maximum: floatOutput('similarity'),
              bestQuery: integerOutput('prototype'),
              matches: integerOutput('matches')
            }
          }),
          'GPURasterEmbeddingSimilarity'
        );
        break;
      }
      case 'knn': {
        if (!options.trainingVectors || !options.trainingLabels || !options.classCount)
          throw new Error('Supply labelled training vectors and classCount');
        add(
          new GPURasterEmbeddingKNearestClassifier({
            ...common,
            trainingVectors: session.input('training', options.trainingVectors, 'float32'),
            trainingLabels: session.input('training-labels', options.trainingLabels, 'uint32'),
            trainingCount: options.trainingLabels.length,
            classCount: options.classCount,
            neighborCount: options.neighborCount ?? 1,
            metric: 'cosine',
            output: {labels: integerOutput('label'), confidence: floatOutput('confidence')}
          }),
          'GPURasterEmbeddingKNearestClassifier'
        );
        break;
      }
      case 'linear-probe': {
        if (!options.weights || !options.bias || !options.classCount)
          throw new Error('Supply trained probe weights, bias and classCount');
        add(
          new GPURasterEmbeddingLinearProbe({
            ...common,
            classCount: options.classCount,
            weights: session.input('weights', options.weights, 'float32'),
            bias: session.input('bias', options.bias, 'float32'),
            output: {labels: integerOutput('label'), confidence: floatOutput('confidence')}
          }),
          'GPURasterEmbeddingLinearProbe'
        );
        break;
      }
      case 'change': {
        if (!options.after) throw new Error('Select an aligned comparison raster');
        add(
          new GPURasterEmbeddingChange({
            pixelCount,
            dimensions,
            before: vectors,
            beforeMask: mask,
            after: session.input('after', options.after.vectors, 'float32'),
            afterMask: options.after.mask
              ? session.input('after-mask', options.after.mask, 'uint32')
              : undefined,
            parameters: session.input(
              'parameters',
              getGPURasterEmbeddingChangeParameterValues({
                similarityThreshold: options.threshold ?? 0.9
              }),
              'float32'
            ),
            output: {
              similarity: floatOutput('similarity'),
              angle: floatOutput('angle'),
              magnitude: floatOutput('magnitude'),
              changed: integerOutput('changed')
            }
          }),
          'GPURasterEmbeddingChange'
        );
        break;
      }
      case 'pca':
      case 'projection':
      case 'covariance': {
        const mean = session.output('mean', 'float32', dimensions);
        const covariance = session.output('covariance', 'float32', dimensions * dimensions);
        const usableCount = session.output('usable-count', 'uint32', 1);
        add(
          new GPURasterEmbeddingCovariance({
            ...common,
            output: {mean: mean.view, covariance: covariance.view, usableCount: usableCount.view}
          }),
          'GPURasterEmbeddingCovariance'
        );
        session.run();
        if ((await session.readScalar(usableCount)) < 2)
          throw new Error('Covariance and PCA need at least two usable pixels');
        const covarianceValues = await session.readFloat32(covariance);
        if (options.operation === 'covariance')
          return {
            width,
            height,
            rowCount: dimensions * dimensions,
            layout: 'matrix',
            columns: {covariance: covarianceValues},
            contributors,
            stages: describeRasterEmbeddingStages(contributors)
          };
        const componentCount = Math.min(dimensions, 3);
        const principal = getRasterEmbeddingPrincipalComponents(
          covarianceValues,
          dimensions,
          componentCount
        );
        contributors.push('getRasterEmbeddingPrincipalComponents');
        // A second graph avoids re-dispatching the covariance pass after its readback.
        return withCatalogGpuSession(
          device,
          'kepler-embedding-projection',
          async projectionSession => {
            const projected = projectionSession.output(
              'projected',
              'float32',
              pixelCount * componentCount
            );
            const rgb =
              componentCount === 3
                ? projectionSession.output('rgb', 'float32', pixelCount * 3)
                : undefined;
            projectionSession.graph.add(
              new GPURasterEmbeddingProjection({
                pixelCount,
                dimensions,
                vectors: projectionSession.input('vectors', raster.vectors, 'float32'),
                mask: raster.mask
                  ? projectionSession.input('mask', raster.mask, 'uint32')
                  : undefined,
                mean: projectionSession.input('mean', await session.readFloat32(mean), 'float32'),
                componentCount,
                components: projectionSession.input(
                  'components',
                  Float32Array.from(principal.components),
                  'float32'
                ),
                parameters: rgb
                  ? projectionSession.input(
                      'parameters',
                      getGPURasterEmbeddingProjectionParameterValues({
                        // The public helper already supplies nondegenerate ranges for zero
                        // eigenvalues, borrowing the widest spread or using [-1, 1] if all zero.
                        stretchRanges: getRasterEmbeddingStretchRanges(principal.eigenvalues)
                      }),
                      'float32'
                    )
                  : undefined,
                output: {projected: projected.view, rgb: rgb?.view}
              })
            );
            projectionSession.run();
            return {
              width,
              height,
              rowCount,
              layout,
              columns: {
                ...vectorColumns(
                  await projectionSession.readFloat32(projected),
                  pixelCount,
                  componentCount,
                  'component'
                ),
                ...(rgb
                  ? vectorColumns(await projectionSession.readFloat32(rgb), pixelCount, 3, 'rgb')
                  : {})
              },
              contributors: [...contributors, 'GPURasterEmbeddingProjection'],
              stages: describeRasterEmbeddingStages([
                ...contributors,
                'GPURasterEmbeddingProjection'
              ])
            };
          }
        );
      }
      case 'quantize':
      case 'usable-mask': {
        const quantized = session.output(
          'quantized',
          'uint32',
          Math.ceil((pixelCount * dimensions) / 4)
        );
        const validity = integerOutput('usable');
        add(
          new GPURasterEmbeddingQuantize({
            ...common,
            output: {quantized: quantized.view, validity}
          }),
          'GPURasterEmbeddingQuantize'
        );
        if (options.operation === 'quantize') {
          delete outputs['usable'];
          outputs['packed_word'] = quantized;
          rowCount = quantized.length;
          layout = 'words';
        }
        break;
      }
      case 'dequantize': {
        if (
          !raster.quantized ||
          raster.quantized.length !== Math.ceil((pixelCount * dimensions) / 4)
        )
          throw new Error('Dequantize needs packed AlphaEarth int8 input');
        packedVectors = session.output('decoded', 'float32', pixelCount * dimensions);
        add(
          new GPURasterEmbeddingDequantize({
            pixelCount,
            dimensions,
            mask,
            quantized: session.input('quantized', raster.quantized, 'uint32'),
            normalize: true,
            output: {vectors: packedVectors.view, validity: integerOutput('usable')}
          }),
          'GPURasterEmbeddingDequantize'
        );
        break;
      }
      case 'pyramid': {
        const scale = options.scale ?? 2;
        width = Math.ceil(width / scale);
        height = Math.ceil(height / scale);
        rowCount = width * height;
        packedVectors = session.output('overview', 'float32', rowCount * dimensions);
        add(
          new GPURasterEmbeddingPyramidLevel({
            sourceWidth: raster.width,
            sourceHeight: raster.height,
            dimensions,
            vectors,
            mask,
            scale,
            output: {
              vectors: packedVectors.view,
              validity: integerOutput('usable', rowCount),
              validCount: integerOutput('valid_count', rowCount),
              meanLength: floatOutput('mean_length', rowCount)
            }
          }),
          'GPURasterEmbeddingPyramidLevel'
        );
        break;
      }
      case 'zone-mean': {
        if (!options.zoneRaster || options.zoneRaster.length !== pixelCount || !options.zoneCount)
          throw new Error('Supply a zone ID for every pixel and zoneCount');
        rowCount = options.zoneCount;
        layout = 'zones';
        packedVectors = session.output('means', 'float32', rowCount * dimensions);
        let maximumComponentMagnitude = 1;
        for (const value of raster.vectors)
          if (Number.isFinite(value))
            maximumComponentMagnitude = Math.max(maximumComponentMagnitude, Math.abs(value));
        add(
          new GPURasterEmbeddingZoneMean({
            ...common,
            zoneCount: options.zoneCount,
            maximumComponentMagnitude,
            zoneRaster: session.input('zones', options.zoneRaster, 'uint32'),
            output: {
              means: packedVectors.view,
              weights: floatOutput('weight', rowCount),
              meanLengths: floatOutput('mean_length', rowCount)
            }
          }),
          'GPURasterEmbeddingZoneMean'
        );
        break;
      }
    }
    session.run();
    if (clusteringStatus) {
      const [iterations, changedLabels, converged] = await session.readUint32(clusteringStatus);
      if (converged !== 1)
        throw new Error(
          `K-means did not converge after ${iterations} iterations (${changedLabels} changed labels); labels were not published.`
        );
    }
    const columns: RasterEmbeddingResult['columns'] = {};
    for (const [name, output] of Object.entries(outputs))
      columns[name] =
        output.view.format === 'uint32'
          ? await session.readUint32(output)
          : await session.readFloat32(output);
    if (packedVectors)
      Object.assign(
        columns,
        vectorColumns(await session.readFloat32(packedVectors), rowCount, packedLength, 'band')
      );
    if (layout === 'ranks') {
      const resultCount = options.resultCount ?? 8;
      columns['query'] = Uint32Array.from({length: rowCount}, (_, index) =>
        Math.floor(index / resultCount)
      );
      columns['rank'] = Uint32Array.from({length: rowCount}, (_, index) => index % resultCount);
    }
    return {
      width,
      height,
      rowCount,
      layout,
      columns,
      contributors,
      stages: describeRasterEmbeddingStages(contributors)
    };
  });
}
