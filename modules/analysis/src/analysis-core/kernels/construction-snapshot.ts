// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext, GPUSpatialFrame} from '../contracts/spatial-context';

const constructionSpatialContextSnapshots = new WeakSet<GPUSpatialContext>();

/** @internal Capture declared fields, including inherited/non-enumerable getters, once. */
export function snapshotConstructionFields<Props extends object>(
  props: Props,
  fields: readonly (keyof Props)[]
): Props {
  const entries: [keyof Props, unknown][] = [];
  for (const field of fields) {
    const value = props[field];
    // A proxy may supply a value without reporting membership. Only an undefined value needs
    // the presence check, so absent fields remain distinct from explicitly undefined fields.
    if (value !== undefined || field in props) entries.push([field, value]);
  }
  return Object.freeze(Object.fromEntries(entries)) as Props;
}

/** @internal Copy and freeze context metadata without cloning graph views or buffers. */
export function snapshotConstructionSpatialContext(context: GPUSpatialContext): GPUSpatialContext {
  // Only reuse snapshots produced here. A caller-frozen descriptor may still have changing
  // getters or mutable nested metadata, so Object.isFrozen alone is insufficient.
  if (constructionSpatialContextSnapshots.has(context)) return context;
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
  const frame =
    snapshot.frame &&
    snapshotConstructionFields(snapshot.frame, [
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
    ]);
  const origin = frame?.origin;
  const areaOfUse = frame?.areaOfUse;
  const unit = frame?.unit;
  const resolved = Object.freeze({
    ...snapshot,
    ...(snapshot.ellipsoid
      ? {
          ellipsoid: snapshotConstructionFields(snapshot.ellipsoid, ['semiMajorAxis', 'flattening'])
        }
      : {}),
    ...(frame
      ? {
          frame: Object.freeze({
            ...frame,
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
  constructionSpatialContextSnapshots.add(resolved);
  return resolved;
}
