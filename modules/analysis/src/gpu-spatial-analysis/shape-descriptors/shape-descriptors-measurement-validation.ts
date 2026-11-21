// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {getKnownGPUSpatialFrameUnit} from '../../analysis-core/contracts/spatial-frame';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** @internal Apply the nested measures' dimensional contract at every construction boundary. */
export function validateShapeDescriptorMeasurements(
  id: string,
  spatialContext: GPUSpatialContext,
  output: {
    areas?: unknown;
    perimeters?: unknown;
    polsbyPopper?: unknown;
    schwartzberg?: unknown;
    sliver?: unknown;
    convexity?: unknown;
  }
): void {
  const needsCompactness = Boolean(output.polsbyPopper || output.schwartzberg || output.sliver);
  const needsAreas = needsCompactness || Boolean(output.convexity) || Boolean(output.areas);
  const needsPerimeters = needsCompactness || Boolean(output.perimeters);
  const frame = spatialContext.frame;
  const registryUnit = frame?.crs
    ? getKnownGPUSpatialFrameUnit(frame.crs.trim().toUpperCase())
    : undefined;
  const angularUnit = [frame?.unit, registryUnit].find(unit => unit?.degreesPerUnit !== undefined);
  validateRequirement(
    id,
    (!needsAreas && !needsPerimeters) ||
      spatialContext.coordinateSpace !== 'planar' ||
      angularUnit === undefined,
    `frame units ${angularUnit?.name} are angular; project positions to a planar linear frame before measuring length or area`
  );
}
