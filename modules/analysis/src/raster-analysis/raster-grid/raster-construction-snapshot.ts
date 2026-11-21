// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPURasterWalkingCostModel} from '../gpu-raster-walking-cost';

/** Capture raster geometry without retaining caller-owned coordinate arrays. @internal */
export function snapshotRasterGrid(grid: GPURasterMetadata): GPURasterMetadata {
  const captured = snapshotConstructionFields(grid, [
    'width',
    'height',
    'affine',
    'pixelInterpretation',
    'coordinateReferenceSystem',
    'level',
    'levelZeroOrigin'
  ]);
  return Object.freeze({
    ...captured,
    affine: Object.freeze([...captured.affine]) as GPURasterMetadata['affine'],
    ...(captured.coordinateReferenceSystem
      ? {
          coordinateReferenceSystem: snapshotConstructionFields(
            captured.coordinateReferenceSystem,
            ['authority', 'wellKnownText', 'projectionJson']
          )
        }
      : {}),
    ...(captured.levelZeroOrigin
      ? {
          levelZeroOrigin: Object.freeze([
            ...captured.levelZeroOrigin
          ]) as GPURasterMetadata['levelZeroOrigin']
        }
      : {})
  });
}

/** Capture the validated model selection and options, without capturing parameter buffer contents. @internal */
export function snapshotRasterWalkingModel(
  model: GPURasterWalkingCostModel | undefined,
  id = 'snapshotRasterWalkingModel'
): GPURasterWalkingCostModel | undefined {
  if (model === undefined) return undefined;
  if (model === null || typeof model !== 'object') {
    validateRequirement(
      id,
      false,
      `model.kind must be tobler or grass (received ${String(model)})`
    );
  }
  const kind = model.kind;
  validateRequirement(
    id,
    kind === 'tobler' || kind === 'grass',
    `model.kind must be tobler or grass (received ${String(kind)})`
  );
  if (kind === 'tobler') return Object.freeze({kind});
  // The discriminator was captured and checked before reading GRASS-only options.
  const captured = snapshotConstructionFields(
    model as Extract<GPURasterWalkingCostModel, {kind: 'grass'}>,
    ['coefficients', 'downhillSlopeThreshold']
  );
  validateRequirement(
    id,
    Array.isArray(captured.coefficients) &&
      captured.coefficients.length === 4 &&
      Array.from(captured.coefficients).every(Number.isFinite),
    `model.coefficients must be four finite numbers (received ${captured.coefficients?.length ?? 'no'} entries)`
  );
  validateRequirement(
    id,
    Number.isFinite(captured.downhillSlopeThreshold) && captured.downhillSlopeThreshold < 0,
    `model.downhillSlopeThreshold must be negative and finite (received ${captured.downhillSlopeThreshold})`
  );
  return Object.freeze({
    ...captured,
    kind,
    coefficients: Object.freeze([...captured.coefficients]) as typeof captured.coefficients
  });
}
