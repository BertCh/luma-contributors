// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Runtime-backed CRS resolution for {@link GPUSpatialFrame}. Synchronous and provider-free:
 * identifiers are normalized by math.gl, PROJ strings are parsed, and callers holding resolved
 * math.gl metadata (for example from `projectionEngine.getCRS`) pass it for exact signatures and
 * units. Not imported by the `gpu-project` execution entry, which must stay math.gl-free.
 */

import {
  getCRSSignature,
  linearUnits,
  normalizeCRSIdentifier,
  parsePROJString
} from '@math.gl/geospatial/crs';
import {validateRequirement} from '../kernels/gpu-contributor-utils';
import type {GPUSpatialFrame} from './spatial-context';
import {
  GPU_METERS_FRAME_UNIT,
  getGPUSpatialFrameMetadataUnit,
  getKnownGPUSpatialFrameUnit,
  type GPUSpatialFrameAnchor,
  type GPUSpatialFrameMetadata,
  type GPUSpatialFramePrecision,
  type GPUSpatialFrameUnit
} from './spatial-frame';

/** Options of {@link createGPUSpatialFrameFromCRS}. */
export type GPUSpatialFrameFromCRSOptions = {
  /** Resolved math.gl metadata; gives exact signature, units, axis order and area of use. */
  metadata?: GPUSpatialFrameMetadata;
  /** Overrides the derived frame id. */
  id?: string;
  precision?: GPUSpatialFramePrecision;
  /** Storage origin in target CRS units; projection centers use longitude-latitude degrees. */
  origin?: readonly [number, number];
  anchor?: GPUSpatialFrameAnchor;
};

const IDENTIFIER_PATTERN = /^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.-]+$/;

/**
 * Creates a `'crs'` frame. The id is the normalized authority identifier (`epsg:3857` and
 * `urn:ogc:def:crs:EPSG::3857` both give `EPSG:3857`), else the metadata signature, else the input
 * string. `crs` keeps the input as written. Unit comes from `metadata`, then the PROJ string
 * `+units`, then a known registry family (`EPSG:4326`, `EPSG:3857`, WGS84 UTM `EPSG:326xx`/`327xx`);
 * it is omitted when unknown.
 */
export function createGPUSpatialFrameFromCRS(
  crs: string,
  options: GPUSpatialFrameFromCRSOptions = {}
): GPUSpatialFrame {
  const {metadata} = options;
  validateRequirement(
    'createGPUSpatialFrameFromCRS',
    options.anchor === undefined ||
      ['projection-center', 'relative-origin'].includes(options.anchor),
    `anchor must be projection-center or relative-origin; received ${options.anchor}`
  );
  validateRequirement(
    'createGPUSpatialFrameFromCRS',
    options.precision === undefined ||
      ['float32', 'float32-origin-relative', 'float64-split'].includes(options.precision),
    `precision must be float32, float32-origin-relative or float64-split; received ${options.precision}`
  );
  validateRequirement(
    'createGPUSpatialFrameFromCRS',
    options.origin === undefined ||
      (options.origin.length === 2 && options.origin.every(Number.isFinite)),
    `origin must contain two finite coordinates; received ${options.origin}`
  );
  const relative =
    options.anchor === 'relative-origin' ||
    (options.precision === 'float32-origin-relative' && options.anchor !== 'projection-center');
  const effectiveOrigin = options.origin ?? (relative ? ([0, 0] as const) : undefined);
  const origin = effectiveOrigin
    ? (Object.freeze([...effectiveOrigin]) as unknown as [number, number])
    : undefined;
  const identifier = normalizeCRSIdentifier(crs);
  const signature = metadata ? (metadata.signature ?? getCRSSignature(metadata)) : undefined;
  const unit = metadata
    ? getGPUSpatialFrameMetadataUnit(metadata)
    : identifier.startsWith('+')
      ? getProjStringUnit(identifier)
      : getKnownGPUSpatialFrameUnit(identifier);
  return Object.freeze({
    id:
      options.id ??
      `${IDENTIFIER_PATTERN.test(identifier) ? identifier : (signature ?? crs)}${relative ? `;origin=${origin ?? [0, 0]}` : ''}`,
    kind: 'crs',
    ...(origin ? {origin} : {}),
    ...(options.anchor ? {anchor: options.anchor} : {}),
    crs,
    ...(signature !== undefined ? {signature} : {}),
    ...(unit ? {unit: Object.freeze({...unit})} : {}),
    ...(metadata ? {axisOrder: metadata.axis} : {}),
    ...(metadata?.areaOfUse
      ? {
          areaOfUse: Object.freeze([...metadata.areaOfUse]) as readonly [
            number,
            number,
            number,
            number
          ]
        }
      : {}),
    ...(options.precision ? {precision: options.precision} : {})
  });
}

function getProjStringUnit(definition: string): GPUSpatialFrameUnit | undefined {
  try {
    const parameters = parsePROJString(definition).parameters;
    const projection = parameters.find(parameter => parameter.name === 'proj')?.value;
    if (projection && ['longlat', 'latlong', 'lonlat', 'latlon'].includes(projection)) {
      return {name: 'degrees', degreesPerUnit: 1};
    }
    const units = parameters.find(parameter => parameter.name === 'units')?.value;
    const toMeter = parameters.find(parameter => parameter.name === 'to_meter')?.value;
    if (units !== undefined) {
      return linearUnits[units];
    }
    if (toMeter !== undefined) {
      return Number(toMeter) > 0 ? {name: 'custom', metersPerUnit: Number(toMeter)} : undefined;
    }
    // PROJ projected systems default to meters.
    return parameters.some(parameter => parameter.name === 'proj' && parameter.value !== 'longlat')
      ? GPU_METERS_FRAME_UNIT
      : undefined;
  } catch {
    return undefined;
  }
}
