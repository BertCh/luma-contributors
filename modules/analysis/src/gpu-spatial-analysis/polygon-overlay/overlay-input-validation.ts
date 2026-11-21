// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPolygonOverlayProps, GPUPolygonOverlayOutput} from './gpu-polygon-overlay';
import {validatePackedUint32View} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
type PackedPolygons = GPUPolygonOverlayProps['left'];

export function validatePolygons(id: string, name: string, polygons: PackedPolygons): void {
  if (polygons.kind !== 'polygons') {
    throw new Error(`${id} ${name}.kind must be polygons`);
  }
  validateSinglePackedView(polygons.positions, ['float32x2'], `${id} ${name}.positions`);
  for (const [field, view] of [
    ['featureOffsets', polygons.featureOffsets],
    ['polygonOffsets', polygons.polygonOffsets],
    ['ringOffsets', polygons.ringOffsets]
  ] as const) {
    validatePackedUint32View(view, `${id} ${name}.${field}`);
    if (view.length < 2) {
      throw new Error(`${id} ${name}.${field} must contain at least two entries`);
    }
  }
  if (polygons.sourceIds) {
    validatePackedUint32View(polygons.sourceIds, `${id} ${name}.sourceIds`);
    validateRequirement(
      id,
      polygons.sourceIds.length >= polygons.featureOffsets.length - 1,
      `${name}.sourceIds must hold at least ${polygons.featureOffsets.length - 1} feature identities`
    );
  }
  if (polygons.featureOffsets.length - 1 >= 0x80000000) {
    throw new Error(`${id} ${name} has too many features for packed overlay provenance`);
  }
}

export function validateStatus(
  id: string,
  name: string,
  status: GPUPolygonOverlayOutput['status']
): void {
  for (const [field, view] of Object.entries(status)) {
    if (view) {
      validatePackedUint32View(view, `${id} ${name}.${field}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name}.${field} must contain one row`);
      }
    }
  }
}

/** Shared enum checks for construction and device-only planning. @internal */
export function validateOverlayOptions(
  id: string,
  operation: string,
  fillRule: string | undefined
): void {
  validateRequirement(
    id,
    ['intersection', 'union', 'difference', 'symmetric-difference', 'dissolve'].includes(operation),
    `operation must be intersection, union, difference, symmetric-difference or dissolve; received ${operation}`
  );
  validateRequirement(
    id,
    fillRule === undefined || ['even-odd', 'positive', 'negative', 'nonzero'].includes(fillRule),
    `fillRule must be even-odd, positive, negative or nonzero; received ${fillRule}`
  );
}
