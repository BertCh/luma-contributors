// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.

import type {GPUSpatialContext} from '../analysis-core/contracts/spatial-context';
import type {GPUSpatialFrame} from '../analysis-core/contracts/spatial-context';
import {
  createGPUSpatialFrameFromIdentifier,
  getGPUSpatialFrameContextUnits
} from '../analysis-core/contracts/spatial-frame';

/**
 * Planar context of rows projected into the projected CRS `targetCrs`, or `undefined` when no
 * target is given. The `'crs'` frame makes rows projected into different CRSs incompatible; its
 * id is `targetCrs` as written. This module stays free of math.gl runtime code, so spellings are
 * not normalized here: pass a `frame` built by `createGPUSpatialFrameFromCRS` for normalized ids,
 * signatures and PROJ-string units. `units` is declared (`'meters'`, `'kilometers'`) when the
 * frame's linear unit is known, which for the default frame covers EPSG:3857 and WGS84 UTM.
 * Otherwise it stays undeclared (native).
 *
 * @internal
 */
export function getGPUProjectionSpatialContext(
  targetCrs?: string,
  frame?: GPUSpatialFrame
): GPUSpatialContext | undefined {
  if (!targetCrs && !frame) {
    return undefined;
  }
  // A frame, when given, wins; otherwise targetCrs is defined here.
  const targetFrame = frame ?? createGPUSpatialFrameFromIdentifier(targetCrs!);
  const units = getGPUSpatialFrameContextUnits(targetFrame);
  return Object.freeze({
    coordinateSpace: 'planar',
    metric: 'native',
    ...(units ? {units} : {}),
    frame: targetFrame
  });
}

/**
 * Destination origin carried by a target frame, or `undefined` when the plan should compute its
 * own. Only frames whose origin is a coordinate of the destination CRS qualify: anchor
 * `'relative-origin'`, or precision `'float32-origin-relative'` without a `'projection-center'`
 * anchor (whose origin is a longitude/latitude, not a destination coordinate).
 *
 * @internal
 */
export function getGPUProjectionFrameOrigin(frame?: GPUSpatialFrame): [number, number] | undefined {
  if (!frame?.origin) {
    return undefined;
  }
  const isRelativeOrigin =
    frame.anchor === 'relative-origin' ||
    (frame.anchor !== 'projection-center' && frame.precision === 'float32-origin-relative');
  return isRelativeOrigin ? [frame.origin[0], frame.origin[1]] : undefined;
}
