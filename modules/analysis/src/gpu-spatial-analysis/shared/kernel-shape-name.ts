// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Resolves the quartic/bisquare shape alias, then validates the family's canonical accepted set.
 * This resolves names only: normalization, bandwidth units and additional kernels belong to callers.
 * Pass the contributor ID for a construction/packing error that identifies the caller.
 */
export function resolveKernelShapeName<Shape extends string>(
  id: string,
  kernel: string,
  acceptedShapes: readonly Shape[]
): Shape {
  const canonical = kernel === 'bisquare' ? 'quartic' : kernel;
  const accepted = acceptedShapes.includes(canonical as Shape);
  const names = [
    ...acceptedShapes,
    ...(acceptedShapes.includes('quartic' as Shape) ? ['bisquare'] : [])
  ];
  validateRequirement(
    id,
    accepted,
    `kernel received ${String(kernel)}; expected one of ${names.join(', ')}`
  );
  return canonical as Shape;
}
