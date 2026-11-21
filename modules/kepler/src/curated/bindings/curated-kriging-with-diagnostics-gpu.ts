// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createWGSLKernelNode, validateCapacity} from '@luma-contributors/analysis/analysis-core';
import {
  addInterpolationDiagnosticsRecipe,
  getGPUKrigingParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {CatalogGpuContext} from '../../analysis/catalog-binding-types';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession
} from '../../analysis/catalog-gpu-session';
import type {GridSpec, Sample} from '../../analysis/catalog-platform3-b-common';
import type {KrigingSettings} from '../../analysis/catalog-platform3-b-grids';

const BINDING_ID = 'curated-kriging-with-diagnostics';
/** GPUKriging's per-invocation dense system holds at most 16 neighbours and one constraint. */
const MAXIMUM_GPU_NEIGHBORS = 16;
/** Match the existing catalog index policy: roughly four samples per index cell. */
const TARGET_SAMPLES_PER_INDEX_CELL = 4;

/** Upload error allowance in measurement units, tighter than the 3e-4 solve parity tolerance. */
const MAXIMUM_MEASUREMENT_ROUND_TRIP_ERROR = 1e-6;

/**
 * Pack the bounded host samples once, before GPU allocation. The binding has already merged
 * exact coincidences; float32 conversion must not introduce any new coincident locations.
 * Compare zero and the measurement nearest zero as origins: zero preserves mixed magnitudes,
 * while a common measurement baseline preserves small changes in a large-offset field.
 */
export function prepareCuratedKrigingGpuSamples(sample: Sample, valueScale: number) {
  const positions = new Float32Array(sample.count * 2);
  const locationRows = new Map<string, number>();
  let nearestZeroValue = sample.value[0];
  for (let index = 0; index < sample.count; index++) {
    positions[index * 2] = sample.x[index];
    positions[index * 2 + 1] = sample.y[index];
    const packedX = positions[index * 2];
    const packedY = positions[index * 2 + 1];
    if (!Number.isFinite(packedX) || !Number.isFinite(packedY))
      throw new CatalogGpuUnsupportedError(
        `${BINDING_ID}: points coordinates must remain finite in float32; use the CPU engine or rescale the coordinates`
      );
    const key = `${packedX}|${packedY}`;
    const previous = locationRows.get(key);
    if (
      previous !== undefined &&
      (sample.x[previous] !== sample.x[index] || sample.y[previous] !== sample.y[index])
    )
      throw new CatalogGpuUnsupportedError(
        `${BINDING_ID}: points at distinct sample rows ${previous} and ${index} collide in float32 coordinates; use the CPU engine or rescale the coordinates`
      );
    locationRows.set(key, index);
    if (Math.abs(sample.value[index]) < Math.abs(nearestZeroValue))
      nearestZeroValue = sample.value[index];
  }
  let valueOrigin = 0;
  let sampleValues: Float32Array | undefined;
  let smallestRoundTripError = Infinity;
  for (const origin of new Set([0, nearestZeroValue])) {
    const packedValues = Float32Array.from(sample.value, value => (value - origin) / valueScale);
    let maximumRoundTripError = 0;
    for (let index = 0; index < sample.count; index++) {
      const reconstructed = packedValues[index] * valueScale + origin;
      const error = Number.isFinite(reconstructed)
        ? Math.abs(reconstructed - sample.value[index]) / Math.max(1, Math.abs(sample.value[index]))
        : Infinity;
      maximumRoundTripError = Math.max(maximumRoundTripError, error);
    }
    if (maximumRoundTripError < smallestRoundTripError) {
      smallestRoundTripError = maximumRoundTripError;
      sampleValues = packedValues;
      valueOrigin = origin;
    }
  }
  if (!sampleValues || smallestRoundTripError > MAXIMUM_MEASUREMENT_ROUND_TRIP_ERROR)
    throw new CatalogGpuUnsupportedError(
      `${BINDING_ID}: valueProperty measurements must round-trip through float32 within ${MAXIMUM_MEASUREMENT_ROUND_TRIP_ERROR} * max(1, abs(value)); use the CPU engine or rescale the measurements`
    );
  return {positions, sampleValues, valueOrigin};
}

/**
 * Execute the existing diagnostics recipe, including its leave-one-out solves and reductions.
 * Input preparation, fitting and publication belong to the binding, not to another CPU solve.
 * The recipe's residual sign is observed minus predicted; the catalog reports the opposite.
 */
export async function computeCuratedKrigingGpu(
  context: CatalogGpuContext,
  sample: Sample,
  grid: GridSpec,
  settings: KrigingSettings
) {
  validateCapacity(
    BINDING_ID,
    'neighborCount (GPU)',
    settings.neighborCount,
    4,
    MAXIMUM_GPU_NEIGHBORS
  );
  const raster: GPURasterMetadata = {
    width: grid.columns,
    height: grid.rows,
    affine: [grid.cellSize, 0, grid.minX, 0, grid.cellSize, grid.minY],
    pixelInterpretation: 'area'
  };
  // Normalize the stated/fitted total sill, not the empirical sample variance. This preserves
  // squared measurement units even when those two quantities differ. Remove a common value
  // offset before float32 upload so a large measurement baseline does not erase small changes.
  const varianceScale = settings.nugget + settings.sill;
  const valueScale = Math.sqrt(varianceScale);
  const {positions, sampleValues, valueOrigin} = prepareCuratedKrigingGpuSamples(
    sample,
    valueScale
  );
  const side = Math.max(1, Math.ceil(Math.sqrt(sample.count / TARGET_SAMPLES_PER_INDEX_CELL)));
  return withCatalogGpuSession(context.device, BINDING_ID, async session => {
    const cellCount = grid.columns * grid.rows;
    const values = session.output('values', 'float32', cellCount);
    const variance = session.output('variance', 'float32', cellCount);
    const predictions = session.output('predictions', 'float32', sample.count);
    const predictionVariance = session.output('prediction-variance', 'float32', sample.count);
    const validCount = session.output('valid-count', 'uint32', 1);
    const standardizedCount = session.output('standardized-count', 'uint32', 1);
    const bias = session.output('bias', 'float32', 1);
    const meanAbsoluteError = session.output('mean-absolute-error', 'float32', 1);
    const rootMeanSquaredError = session.output('root-mean-squared-error', 'float32', 1);
    const recipe = addInterpolationDiagnosticsRecipe(session.graph, {
      id: BINDING_ID,
      kriging: {
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        positions: session.input('positions', positions, 'float32x2'),
        values: session.input('observed', sampleValues, 'float32'),
        grid: raster,
        indexGridSize: [side, side],
        indexBounds: [
          grid.minX,
          grid.minY,
          grid.minX + grid.columns * grid.cellSize,
          grid.minY + grid.rows * grid.cellSize
        ],
        maximumNeighborCount: settings.neighborCount,
        parameters: session.input(
          'parameters',
          getGPUKrigingParameterValues(
            {
              searchRadius: settings.searchRadius > 0 ? settings.searchRadius : Infinity,
              neighborCount: settings.neighborCount,
              minimumNeighborCount: 1,
              variogram: {
                model: settings.model,
                nugget: settings.nugget / varianceScale,
                sill: settings.sill / varianceScale,
                range: settings.range
              }
            },
            raster
          ),
          'float32'
        )
      },
      output: {
        values: values.view,
        variance: variance.view,
        predictions: predictions.view,
        predictionVariance: predictionVariance.view,
        validCount: validCount.view,
        standardizedCount: standardizedCount.view,
        bias: bias.view,
        meanAbsoluteError: meanAbsoluteError.view,
        rootMeanSquaredError: rootMeanSquaredError.view
      }
    });
    // Recipe statuses are graph-owned transients. Copy their one-row ports before compiling,
    // then read them with the other publication outputs; no status is read inside a contributor.
    const statuses = recipe.status.stages.flatMap(({stage, status}) =>
      Object.entries(status).map(([name, source]) => {
        const output = session.output(`${stage}-${name}`, 'uint32', 1);
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `${BINDING_ID}-publish-${stage}-${name}`,
            operation: BINDING_ID,
            invocationCount: 1,
            bindings: [
              {name: 'source', view: source, type: 'u32', access: 'read'},
              {name: 'destination', view: output.view, type: 'u32', access: 'read_write'}
            ],
            body: 'destination[destinationOffset] = source[sourceOffset];'
          })
        );
        return {key: `${stage}.${name}`, output};
      })
    );
    session.run();
    const statusReadbacks = statuses.map(
      async ({key, output}) => [key, await session.readScalar(output)] as const
    );
    const readbacks = [
      session.readFloat32(values),
      session.readFloat32(variance),
      session.readFloat32(predictions),
      session.readFloat32(predictionVariance),
      session.readScalar(validCount),
      session.readScalar(standardizedCount),
      session.readFloat32(bias),
      session.readFloat32(meanAbsoluteError),
      session.readFloat32(rootMeanSquaredError),
      Promise.all(statusReadbacks)
    ] as const;
    // Let every readback finish before session cleanup releases the buffers, including when
    // one read fails. Promise.all below propagates that failure after the reads have settled.
    await Promise.allSettled([...readbacks, ...statusReadbacks]);
    const [
      cellValues,
      cellVariances,
      predicted,
      validationVariances,
      validationCount,
      standardizedValidationCount,
      meanResidual,
      absoluteError,
      rootSquaredError,
      stages
    ] = await Promise.all(readbacks);
    const status = Object.fromEntries(stages);
    return {
      field: {
        values: Float64Array.from(cellValues, value => value * valueScale + valueOrigin),
        variances: Float64Array.from(cellVariances, value => value * varianceScale),
        // Keep the legacy field for callers; publication uses the producer-named counts.
        singularCount: status['kriging.uncertainCount'],
        uncertainCount: status['kriging.uncertainCount'],
        invalidCount: status['kriging.invalidCount']
      },
      leaveOneOut: {
        predicted: Float64Array.from(predicted, value => value * valueScale + valueOrigin),
        variance: Float64Array.from(validationVariances, value => value * varianceScale),
        failedCount: sample.count - validationCount
      },
      summary: {
        crossValidationCount: validationCount,
        meanError: -meanResidual[0] * valueScale,
        meanAbsoluteError: absoluteError[0] * valueScale,
        rootMeanSquareError: rootSquaredError[0] * valueScale,
        gpuStandardizedCount: standardizedValidationCount,
        gpuInvalidCellCount: status['kriging.invalidCount'],
        gpuUncertainCellCount: status['kriging.uncertainCount'],
        gpuInvalidValidationCount: status['validation-summary.invalidCount']
      }
    };
  });
}
