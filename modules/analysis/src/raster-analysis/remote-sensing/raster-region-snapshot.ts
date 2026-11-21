// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Copies projection JSON through ordinary field reads, including transparent proxy objects. */
function snapshotRegionProjectionValue(
  value: unknown,
  copies = new WeakMap<object, unknown>()
): unknown {
  if (value === null || typeof value !== 'object') return value;
  if (copies.has(value)) return copies.get(value);
  if (Array.isArray(value)) {
    const result: unknown[] = [];
    copies.set(value, result);
    for (const item of value) result.push(snapshotRegionProjectionValue(item, copies));
    return Object.freeze(result);
  }
  const result: Record<string, unknown> = {};
  copies.set(value, result);
  for (const key of Object.keys(value))
    Object.defineProperty(result, key, {
      value: snapshotRegionProjectionValue((value as Record<string, unknown>)[key], copies),
      enumerable: true,
      writable: true,
      configurable: true
    });
  return Object.freeze(result);
}

/** Captures region-grid metadata without cloning any graph view or GPU resource. */
export function snapshotRasterRegionGrid(input: GPURasterMetadata): GPURasterMetadata {
  const grid = snapshotConstructionFields(input, [
    'width',
    'height',
    'affine',
    'pixelInterpretation',
    'coordinateReferenceSystem',
    'level',
    'levelZeroOrigin'
  ]);
  const coordinateReferenceSystem =
    grid.coordinateReferenceSystem &&
    snapshotConstructionFields(grid.coordinateReferenceSystem, [
      'authority',
      'wellKnownText',
      'projectionJson'
    ]);
  return {
    ...grid,
    affine: [...grid.affine],
    ...(grid.levelZeroOrigin
      ? {levelZeroOrigin: [...grid.levelZeroOrigin] as [number, number]}
      : {}),
    ...(coordinateReferenceSystem
      ? {
          coordinateReferenceSystem: {
            ...coordinateReferenceSystem,
            ...(coordinateReferenceSystem.projectionJson
              ? {
                  projectionJson: snapshotRegionProjectionValue(
                    coordinateReferenceSystem.projectionJson
                  ) as Readonly<Record<string, unknown>>
                }
              : {})
          }
        }
      : {})
  };
}
