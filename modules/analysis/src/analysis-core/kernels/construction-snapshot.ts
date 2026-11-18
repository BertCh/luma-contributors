// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext, GPUSpatialFrame} from '../contracts/spatial-context';

/** @internal Capture declared fields, including inherited/non-enumerable getters, once. */
export function snapshotConstructionFields<Props extends object>(
  props: Props,
  fields: readonly (keyof Props)[]
): Props {
  return Object.freeze(
    Object.fromEntries(fields.filter(field => field in props).map(field => [field, props[field]]))
  ) as Props;
}

/** @internal Copy and freeze context metadata without cloning graph views or buffers. */
export function snapshotConstructionSpatialContext(context: GPUSpatialContext): GPUSpatialContext {
  const snapshot = snapshotConstructionFields(context, [
    'coordinateSpace',
    'metric',
    'units',
    'sphereRadius',
    'ellipsoid',
    'edgeModel',
    'datum',
    'gridFamily',
    'gridResolution',
    'frame'
  ]);
  const frame = snapshot.frame;
  const origin = frame?.origin;
  const areaOfUse = frame?.areaOfUse;
  const unit = frame?.unit;
  return Object.freeze({
    ...snapshot,
    ...(snapshot.ellipsoid
      ? {
          ellipsoid: snapshotConstructionFields(snapshot.ellipsoid, ['semiMajorAxis', 'flattening'])
        }
      : {}),
    ...(frame
      ? {
          frame: Object.freeze({
            ...snapshotConstructionFields(frame, [
              'id',
              'kind',
              'origin',
              'crs',
              'signature',
              'unit',
              'axisOrder',
              'areaOfUse',
              'anchor',
              'precision'
            ]),
            ...(origin
              ? {origin: Object.freeze([...origin]) as unknown as GPUSpatialFrame['origin']}
              : {}),
            ...(areaOfUse
              ? {areaOfUse: Object.freeze([...areaOfUse]) as GPUSpatialFrame['areaOfUse']}
              : {}),
            ...(unit
              ? {
                  unit: snapshotConstructionFields(unit, [
                    'name',
                    'metersPerUnit',
                    'degreesPerUnit'
                  ])
                }
              : {})
          })
        }
      : {})
  });
}
