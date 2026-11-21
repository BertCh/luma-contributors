// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Capture declared raster provenance without retaining lazy recipe configuration facades. @internal */
export function snapshotRecipeRasterMetadata(grid: GPURasterMetadata): GPURasterMetadata {
  const snapshot = snapshotConstructionFields(grid, [
    'width',
    'height',
    'affine',
    'pixelInterpretation',
    'coordinateReferenceSystem',
    'level',
    'levelZeroOrigin'
  ]);
  return detachRasterMetadata({
    ...snapshot,
    ...(snapshot.coordinateReferenceSystem
      ? {
          coordinateReferenceSystem: snapshotConstructionFields(
            snapshot.coordinateReferenceSystem,
            ['authority', 'wellKnownText', 'projectionJson']
          )
        }
      : {})
  });
}

/** Raster CRS projectionJson is metadata: copy nested values as well as the declared grid fields. */
function detachRasterMetadata<Value>(value: Value): Value {
  if (!value || typeof value !== 'object') return value;
  return Object.freeze(
    Array.isArray(value)
      ? Array.from(value, entry => detachRasterMetadata(entry))
      : Object.fromEntries(
          Object.entries(value).map(([key, entry]) => [key, detachRasterMetadata(entry)])
        )
  ) as Value;
}
