// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Facade over math.gl CRS metadata for {@link GPUSpatialFrame}. This module and
 * `spatial-frame-crs.ts` are the only analysis contract modules that touch math.gl CRS types.
 * This one imports types only, so the `gpu-project` execution bundle stays free of math.gl
 * runtime code; `spatial-frame-crs.ts` adds the runtime-backed CRS resolution.
 */

import type {CRSMetadata, CRSUnit} from '@math.gl/geospatial/crs';
import {GPU_EARTH_MEAN_RADIUS_METERS} from './earth';
import type {GPUSpatialFrame, GPUSpatialFrameKind, GPUSpatialUnit} from './spatial-context';

/** Unit of frame coordinates; math.gl's `CRSUnit` (`name`, `metersPerUnit`, `degreesPerUnit`). */
export type GPUSpatialFrameUnit = CRSUnit;

/** Resolved math.gl CRS metadata (a `CRSDescriptor` also qualifies and carries its `signature`). */
export type GPUSpatialFrameMetadata = CRSMetadata & {readonly signature?: string};

/**
 * What the frame `origin` means.
 * - `'projection-center'`: the projection is centered on the origin (local frames); the origin is
 *   the coordinate zero.
 * - `'relative-origin'`: coordinates are the CRS's own, stored relative to the origin to keep
 *   float32 precision.
 */
export type GPUSpatialFrameAnchor = 'projection-center' | 'relative-origin';

/** How frame coordinates are held for GPU execution. */
export type GPUSpatialFramePrecision = 'float32' | 'float32-origin-relative' | 'float64-split';

/**
 * Local frame kinds and the {@link GPUSpatialFrameKind} each maps to. The single source for
 * `GPULocalFrameKind`.
 */
export const GPU_LOCAL_FRAME_KINDS = {
  'azimuthal-equidistant': 'local-azimuthal-equidistant',
  'lambert-equal-area': 'local-lambert-equal-area'
} as const satisfies Record<string, GPUSpatialFrameKind>;

/** Frame kind of a local frame kind. */
export function getGPUSpatialFrameKind(
  localKind: keyof typeof GPU_LOCAL_FRAME_KINDS
): (typeof GPU_LOCAL_FRAME_KINDS)[keyof typeof GPU_LOCAL_FRAME_KINDS] {
  return GPU_LOCAL_FRAME_KINDS[localKind];
}

/** Meters, the unit of local frames and of most projected CRSs. */
export const GPU_METERS_FRAME_UNIT: GPUSpatialFrameUnit = Object.freeze({
  name: 'meters',
  metersPerUnit: 1
});
const DEGREES: GPUSpatialFrameUnit = Object.freeze({name: 'degrees', degreesPerUnit: 1});

/** Props of {@link createGPULocalSpatialFrame}. */
export type GPULocalSpatialFrameProps = {
  kind: keyof typeof GPU_LOCAL_FRAME_KINDS;
  origin: readonly [number, number];
  /** Sphere radius in meters; suffixed to the default id when it differs from the mean radius. */
  sphereRadius?: number;
  id?: string;
};

/** Creates a local frame in meters, centered on its origin and held origin-relative. */
export function createGPULocalSpatialFrame(props: GPULocalSpatialFrameProps): GPUSpatialFrame {
  const [originLongitude, originLatitude] = props.origin;
  const kind = getGPUSpatialFrameKind(props.kind);
  const radiusSuffix =
    props.kind === 'azimuthal-equidistant' &&
    props.sphereRadius !== undefined &&
    props.sphereRadius !== GPU_EARTH_MEAN_RADIUS_METERS
      ? `,R=${props.sphereRadius}`
      : '';
  return Object.freeze({
    id: props.id ?? `${kind}(${originLongitude},${originLatitude}${radiusSuffix})`,
    kind,
    origin: [originLongitude, originLatitude] as [number, number],
    unit: GPU_METERS_FRAME_UNIT,
    axisOrder: 'enu',
    anchor: 'projection-center',
    precision: 'float32-origin-relative'
  });
}

/**
 * Declared units accepted by a native planar context: `'meters'` for meter frames, otherwise
 * `undefined` (native). Other units, including kilometers, remain in `frame.unit`; the library
 * never converts coordinates or thresholds.
 */
export function getGPUSpatialFrameContextUnits(frame: GPUSpatialFrame): GPUSpatialUnit | undefined {
  const metersPerUnit = frame.unit?.metersPerUnit;
  if (metersPerUnit === 1) {
    return 'meters';
  }
  return undefined;
}

/**
 * Creates a `'crs'` frame from an authority identifier without math.gl runtime code: the id and
 * `crs` are the input as written, and the unit comes from {@link getKnownGPUSpatialFrameUnit}.
 * Use `createGPUSpatialFrameFromCRS` for normalized ids, signatures and PROJ-string units.
 */
export function createGPUSpatialFrameFromIdentifier(crs: string): GPUSpatialFrame {
  const unit = getKnownGPUSpatialFrameUnit(crs);
  return Object.freeze({id: crs, kind: 'crs', crs, ...(unit ? {unit} : {})});
}

/**
 * Unit of a recognized registry family: `EPSG:4326` and `OGC:CRS84` (degrees), `EPSG:3857` and
 * WGS84 UTM `EPSG:326xx`/`EPSG:327xx` (meters). Undefined for everything else, since reading other
 * codes needs a CRS provider.
 */
export function getKnownGPUSpatialFrameUnit(identifier: string): GPUSpatialFrameUnit | undefined {
  if (identifier === 'EPSG:4326' || identifier === 'OGC:CRS84') {
    return DEGREES;
  }
  return identifier === 'EPSG:3857' || /^EPSG:32[67](0[1-9]|[1-5]\d|60)$/.test(identifier)
    ? GPU_METERS_FRAME_UNIT
    : undefined;
}

/** Common unit of the horizontal axes of resolved metadata, or undefined when they differ. */
export function getGPUSpatialFrameMetadataUnit(
  metadata: CRSMetadata
): GPUSpatialFrameUnit | undefined {
  const {xUnit, yUnit} = metadata;
  return xUnit.name === yUnit.name &&
    xUnit.metersPerUnit === yUnit.metersPerUnit &&
    xUnit.degreesPerUnit === yUnit.degreesPerUnit
    ? xUnit
    : undefined;
}
