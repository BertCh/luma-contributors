// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Raster outer-corner origin and signed cell sizes, in the field's planar units. */
export type FlowRasterPlacement = {
  origin: readonly [number, number];
  cellSize: readonly [number, number];
};

type PlacementTuple = readonly [number, number, number, number];

/** @internal Validate and flatten a placement without losing binary64 origins. */
export function resolveFlowRasterPlacement<Name extends string>(
  id: string,
  settings: Record<NoInfer<Name>, FlowRasterPlacement>,
  name: Name,
  removedName: string
): PlacementTuple {
  validateRequirement(id, !(removedName in settings), `${removedName} was removed; use ${name}`);
  const placement = settings[name];
  validateRequirement(
    id,
    placement?.origin?.length === 2 && placement?.cellSize?.length === 2,
    `${name} must contain origin [x, y] and cellSize [dx, dy]; received ${JSON.stringify(placement)}`
  );
  const values: PlacementTuple = [
    placement.origin[0],
    placement.origin[1],
    placement.cellSize[0],
    placement.cellSize[1]
  ];
  for (let axis = 0; axis < 2; axis++) {
    validateRequirement(
      id,
      Number.isFinite(values[axis]),
      `${name} origin[${axis}] must be finite; received ${values[axis]}`
    );
  }
  return values;
}

/** @internal A local origin is binary64 host metadata, never a narrowed GPU state word. */
export function resolveFlowLocalOrigin(
  id: string,
  origin?: readonly [number, number]
): readonly [number, number] {
  const values = origin ?? [0, 0];
  validateRequirement(
    id,
    values.length === 2 && values.every(Number.isFinite),
    `localOrigin must contain two finite numbers; received ${JSON.stringify(values)}`
  );
  return [values[0], values[1]];
}
