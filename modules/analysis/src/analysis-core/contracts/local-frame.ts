// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {projectionEngine} from '@math.gl/geospatial/projection';
import {GPU_EARTH_MEAN_RADIUS_METERS} from './earth';
import {GPU_PLANAR_SPATIAL_CONTEXT, type GPUSpatialContext} from './spatial-context';
import {GPU_LOCAL_FRAME_KINDS, createGPULocalSpatialFrame} from './spatial-frame';

/**
 * Projection of a local frame.
 * - `'azimuthal-equidistant'`: spherical azimuthal equidistant on the mean Earth radius. Distances
 *   and bearings from the origin are exact great-circle values; other distances are within about
 *   0.1% up to 200 km from the origin. The frame for distance bands, neighbor radii and buffers.
 * - `'lambert-equal-area'`: ellipsoidal (WGS84) Lambert azimuthal equal-area, as
 *   `+proj=laea +ellps=WGS84`. Areas are exact ellipsoidal areas at any scale. The frame for
 *   densities, areal weights and dot sampling.
 */
export type GPULocalFrameKind = keyof typeof GPU_LOCAL_FRAME_KINDS;

/** Options of {@link createGPULocalFrame}. */
export type GPULocalFrameProps = {
  /** Frame origin `[longitude, latitude]` in degrees, kept in binary64 on the CPU. */
  origin: readonly [number, number];
  /** Projection of the frame. Defaults to `'azimuthal-equidistant'`. */
  kind?: GPULocalFrameKind;
  /** Sphere radius in meters for `'azimuthal-equidistant'`. Defaults to the mean Earth radius. */
  sphereRadius?: number;
  /** Frame identity carried by `spatialContext.frame.id`. Defaults to one derived from kind and origin. */
  id?: string;
  /**
   * Horizontal datum of the origin and projected positions, copied to spatialContext.datum.
   * The frame id is unchanged by it.
   */
  datum?: string;
};

/**
 * A binary64 CPU-side local planar frame around an origin, in meters.
 *
 * Projecting longitude/latitude into a local frame before upload is the "project first" step that
 * planar spatial-analysis operators require: the origin is subtracted in binary64, so float32
 * frame coordinates keep about 1e-7 of the frame extent (millimeters at 10 km, centimeters at
 * 100 km) instead of the 0.4 to 1 m quantization of float32 longitude/latitude. Pass
 * {@link GPULocalFrame.spatialContext} to the planar operators; it carries the frame identity so
 * data projected around different origins is rejected as incompatible. Use
 * `GPULocalFrameProjection` to project GPU-resident positions without a readback.
 */
export type GPULocalFrame = {
  readonly kind: GPULocalFrameKind;
  /** Frame origin `[longitude, latitude]` in degrees. */
  readonly origin: readonly [number, number];
  /**
   * Planar native context of frame coordinates, `units: 'meters'`, with the frame identity. Its
   * frame kind lets distance operators reject Lambert frames and area operators reject AEQD frames.
   */
  readonly spatialContext: GPUSpatialContext;
  /** Projects a longitude/latitude in degrees to frame meters. Antipodes give NaN. */
  project(longitude: number, latitude: number): [number, number];
  /** Inverse of {@link project}; longitudes are returned in [-180, 180]. */
  unproject(x: number, y: number): [number, number];
  /**
   * Projects rows of `[longitude, latitude, ...]` with `stride` numbers per row (default 2) to
   * interleaved float32 `[x, y]` frame meters, two per row.
   */
  projectArray(lngLats: ArrayLike<number>, stride?: number, target?: Float32Array): Float32Array;
  /** Unprojects interleaved `[x, y]` frame meters to interleaved binary64 longitude/latitude. */
  unprojectArray(xys: ArrayLike<number>, target?: Float64Array): Float64Array;
};

/**
 * Creates a binary64 local frame for projecting longitude/latitude positions before planar
 * analysis. See {@link GPULocalFrame}.
 */
export function createGPULocalFrame(props: GPULocalFrameProps): GPULocalFrame {
  const {
    origin,
    kind = 'azimuthal-equidistant',
    sphereRadius = GPU_EARTH_MEAN_RADIUS_METERS
  } = props;
  const [originLongitude, originLatitude] = origin;
  if (
    !Number.isFinite(originLongitude) ||
    !Number.isFinite(originLatitude) ||
    Math.abs(originLatitude) > 90
  ) {
    throw new Error('createGPULocalFrame origin must be a finite longitude/latitude');
  }
  if (!(sphereRadius > 0) || !Number.isFinite(sphereRadius)) {
    throw new Error('createGPULocalFrame sphereRadius must be positive');
  }
  if (props.datum !== undefined && (typeof props.datum !== 'string' || props.datum.trim() === '')) {
    throw new Error('createGPULocalFrame datum must be a non-empty string');
  }
  const equidistant = kind === 'azimuthal-equidistant';
  const transform = equidistant
    ? createLocalFrameTransform(originLongitude, originLatitude, `+proj=aeqd +R=${sphereRadius}`)
    : createLocalFrameTransform(originLongitude, originLatitude, '+proj=laea +ellps=WGS84');
  const spatialContext: GPUSpatialContext = Object.freeze({
    ...GPU_PLANAR_SPATIAL_CONTEXT,
    units: 'meters',
    ...(props.datum !== undefined ? {datum: props.datum} : {}),
    frame: createGPULocalSpatialFrame({
      kind,
      origin: [originLongitude, originLatitude],
      sphereRadius,
      id: props.id
    })
  });

  return Object.freeze({
    kind,
    origin: Object.freeze([originLongitude, originLatitude] as const),
    spatialContext,
    project: (longitude: number, latitude: number): [number, number] => {
      const scratch = new Float64Array(2);
      transform.projectTo(longitude, latitude, scratch);
      return [scratch[0], scratch[1]];
    },
    unproject: (x: number, y: number): [number, number] => {
      const scratch = new Float64Array(2);
      transform.unprojectTo(x, y, scratch);
      return [scratch[0], scratch[1]];
    },
    projectArray(lngLats: ArrayLike<number>, stride = 2, target?: Float32Array): Float32Array {
      if (!Number.isInteger(stride) || stride < 2) {
        throw new Error('createGPULocalFrame projectArray stride must be an integer >= 2');
      }
      const rowCount = Math.floor(lngLats.length / stride);
      const output = target ?? new Float32Array(rowCount * 2);
      const scratch = new Float64Array(2);
      for (let row = 0; row < rowCount; row++) {
        transform.projectTo(lngLats[row * stride], lngLats[row * stride + 1], scratch);
        output[row * 2] = scratch[0];
        output[row * 2 + 1] = scratch[1];
      }
      return output;
    },
    unprojectArray(xys: ArrayLike<number>, target?: Float64Array): Float64Array {
      const output = target ?? new Float64Array(xys.length);
      const scratch = new Float64Array(2);
      for (let index = 0; index + 1 < xys.length; index += 2) {
        transform.unprojectTo(xys[index], xys[index + 1], scratch);
        output[index] = scratch[0];
        output[index + 1] = scratch[1];
      }
      return output;
    }
  });
}

type LocalFrameTransform = {
  /** Writes `[x, y]` meters to `target`; antipodes and non-invertible points give NaN. */
  projectTo(longitude: number, latitude: number, target: Float64Array): void;
  /** Writes `[longitude, latitude]` degrees to `target`, longitude in [-180, 180]. */
  unprojectTo(x: number, y: number, target: Float64Array): void;
};

/** Longitude wrapped to [-180, 180], keeping +180 for positive inputs. */
function wrapLongitudeDegrees(longitude: number): number {
  const wrapped = ((((longitude + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 && longitude > 0 ? 180 : wrapped;
}

/**
 * Local frame transform delegated to math.gl's PROJ-compatible azimuthal projections, centered on
 * the origin. `projectionString` is `+proj=aeqd +R=<radius>` (spherical) or
 * `+proj=laea +ellps=WGS84` (ellipsoidal).
 */
function createLocalFrameTransform(
  originLongitude: number,
  originLatitude: number,
  projectionString: string
): LocalFrameTransform {
  const projection = projectionEngine.createProjection({
    from: '+proj=longlat +ellps=WGS84',
    to: `${projectionString} +lon_0=${originLongitude} +lat_0=${originLatitude}`
  });
  return {
    projectTo(longitude, latitude, target) {
      target[0] = longitude;
      target[1] = latitude;
      try {
        projection.projectFlatSync(target, 2);
      } catch {
        // math.gl throws at the antipode, which has no single image.
        target[0] = NaN;
        target[1] = NaN;
      }
    },
    unprojectTo(x, y, target) {
      target[0] = x;
      target[1] = y;
      try {
        projection.unprojectFlatSync(target, 2);
        target[0] = wrapLongitudeDegrees(target[0]);
      } catch {
        // Points beyond the projection extent have no geographic position.
        target[0] = NaN;
        target[1] = NaN;
      }
    }
  };
}
