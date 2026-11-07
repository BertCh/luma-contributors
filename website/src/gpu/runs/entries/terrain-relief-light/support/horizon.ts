// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The terrain horizon (and, for the light operations, slope and aspect) that the sun and sky runs
// build on. Each helper adds the real operation to the run's graph and returns the buffers it wrote.

import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  GPUTerrainDerivatives,
  GPUTerrainHorizon
} from '@luma-contributors/analysis/gpu-terrain';
import type {RunBuilder} from '../../../define-run';
import type {UploadedTerrain} from './terrain';

/** GPUTerrainHorizon: per-direction horizon angles, sky view factor and openness for every cell. */
export function addTerrainHorizon(
  builder: RunBuilder,
  terrain: UploadedTerrain,
  options: {directionCount: number; maximumRadius: number; zFactor?: number}
) {
  const {grid, band, cellCount} = terrain;
  const parameterBuffer = builder.allocate('horizon-parameters', 'float32', 8);
  parameterBuffer.write(getGPUTerrainHorizonParameterValues({zFactor: options.zFactor ?? 1}, grid));
  const horizon = builder.allocate('horizon', 'float32', cellCount * options.directionCount);
  const skyViewFactor = builder.allocate('sky-view-factor', 'float32', cellCount);
  const positiveOpenness = builder.allocate('positive-openness', 'float32', cellCount);
  const negativeOpenness = builder.allocate('negative-openness', 'float32', cellCount);
  builder.graph.add(
    new GPUTerrainHorizon({
      grid,
      elevation: band,
      parameters: parameterBuffer.view,
      directionCount: options.directionCount,
      maximumRadius: options.maximumRadius,
      output: {
        horizon: horizon.view,
        skyViewFactor: skyViewFactor.view,
        positiveOpenness: positiveOpenness.view,
        negativeOpenness: negativeOpenness.view
      }
    })
  );
  return {horizon, skyViewFactor, positiveOpenness, negativeOpenness};
}

/** GPUTerrainDerivatives: slope and aspect, which the sun operations use for surface orientation. */
export function addTerrainSlopeAspect(builder: RunBuilder, terrain: UploadedTerrain) {
  const {grid, band, cellCount} = terrain;
  const parameterBuffer = builder.allocate('derivative-parameters', 'float32', 8);
  parameterBuffer.write(getGPUTerrainDerivativesParameterValues({}, grid));
  const slope = builder.allocate('slope', 'float32', cellCount);
  const aspect = builder.allocate('aspect', 'float32', cellCount);
  builder.graph.add(
    new GPUTerrainDerivatives({
      grid,
      elevation: band,
      parameters: parameterBuffer.view,
      output: {slope: slope.view, aspect: aspect.view}
    })
  );
  return {slope, aspect};
}
