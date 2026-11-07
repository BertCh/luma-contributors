// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPULocalFrame,
  type GPULocalFrame,
  type GPULocalFrameKind
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {CatalogCoordinateSystem, CatalogDiagnostic, CatalogPosition} from './catalog-geojson';
import {addCatalogDiagnostic} from './catalog-geojson';

/**
 * Projection of a geographic local frame.
 * - `'azimuthal-equidistant'`: distances and bearings from the frame origin are exact. Use it for
 *   distance, buffer, cluster, neighbor, network and grid-spacing operations.
 * - `'lambert-equal-area'`: areas are exact (WGS84 ellipsoid). Use it for operations that publish
 *   areas or densities.
 */
export type LocalFrameKind = GPULocalFrameKind;

/**
 * Where frame coordinates start, in projected meters (or coordinate units for planar data).
 * - `'none'`: the projection origin (geographic) or the coordinate origin (planar).
 * - `'minimum'`: the lower left corner of the projected data extent. Frame coordinates are never
 *   negative, which lattice and raster code relies on for grid alignment.
 * - `'center'`: the center of the projected data extent, so float32 keeps offsets from the middle.
 */
export type LocalFrameShift = 'none' | 'minimum' | 'center';

export type LocalFrameOptions = {
  /** Geographic projection. Defaults to `'azimuthal-equidistant'`; ignored for planar data. */
  kind?: LocalFrameKind;
  /** Frame coordinate origin relative to the projection. Defaults to `'none'`. */
  shift?: LocalFrameShift;
  /**
   * Explicit geographic projection origin `[longitude, latitude]`. Defaults to the center of the
   * data extent (antimeridian aware). Ignored for planar data.
   */
  origin?: readonly [number, number];
};

/**
 * The one local planar frame of the kepler catalog operations, in meters for EPSG:4326 and in
 * coordinate units for planar data. `forward` and `inverse` work in binary64.
 */
export type LocalFrame = {
  /** The binary64 graph frame, null for planar data. Carries the frame identity in its spatial context. */
  readonly gpuFrame: GPULocalFrame | null;
  /** Projected coordinates of the frame coordinate origin. */
  readonly shiftX: number;
  readonly shiftY: number;
  /** Projects a position; a position that cannot be projected (see below) gives `[NaN, NaN]`. */
  forward(position: CatalogPosition): [number, number];
  /** Number of `forward` calls so far that returned a non-finite coordinate. See {@link addLocalFrameDiagnostic}. */
  getNonFiniteCount(): number;
  inverse(x: number, y: number): [number, number];
};

/**
 * Local frame around the data.
 *
 * EPSG:4326 positions are projected with `createGPULocalFrame` around the center of the
 * longitude/latitude bounding box of `positions` (the bounding box center, not the mean, so the
 * origin does not drift with vertex density). The longitude extent is antimeridian aware: the
 * smaller of the plain span and the span with negative longitudes shifted by 360 is used, so data
 * from 179 E to 179 W is centered on 180, not on 0. Inverse longitudes are wrapped to [-180, 180].
 * Positions on the far side of the Earth from the origin (antipodal) project to NaN; `forward`
 * counts them and {@link addLocalFrameDiagnostic} reports the count. Callers decide whether to
 * drop those positions; the frame never silently substitutes a value.
 *
 * Planar positions use an identity frame, optionally shifted by `options.shift`.
 */
export function makeLocalFrame(
  coordinateSystem: CatalogCoordinateSystem,
  positions: readonly CatalogPosition[],
  options: LocalFrameOptions = {}
): LocalFrame {
  const {kind = 'azimuthal-equidistant', shift = 'none'} = options;
  let gpuFrame: GPULocalFrame | null = null;
  if (coordinateSystem !== 'planar') {
    let minimumLongitude = Infinity;
    let maximumLongitude = -Infinity;
    let minimumShiftedLongitude = Infinity;
    let maximumShiftedLongitude = -Infinity;
    let minimumLatitude = Infinity;
    let maximumLatitude = -Infinity;
    for (const position of positions) {
      if (!Number.isFinite(position[0]) || !Number.isFinite(position[1])) continue;
      const shiftedLongitude = position[0] < 0 ? position[0] + 360 : position[0];
      minimumLongitude = Math.min(minimumLongitude, position[0]);
      maximumLongitude = Math.max(maximumLongitude, position[0]);
      minimumShiftedLongitude = Math.min(minimumShiftedLongitude, shiftedLongitude);
      maximumShiftedLongitude = Math.max(maximumShiftedLongitude, shiftedLongitude);
      minimumLatitude = Math.min(minimumLatitude, position[1]);
      maximumLatitude = Math.max(maximumLatitude, position[1]);
    }
    const hasPositions = Number.isFinite(minimumLongitude);
    let originLongitude = 0;
    if (hasPositions) {
      const plainSpan = maximumLongitude - minimumLongitude;
      const shiftedSpan = maximumShiftedLongitude - minimumShiftedLongitude;
      originLongitude =
        shiftedSpan < plainSpan
          ? wrapLongitude((minimumShiftedLongitude + maximumShiftedLongitude) / 2)
          : (minimumLongitude + maximumLongitude) / 2;
    }
    const originLatitude = hasPositions ? (minimumLatitude + maximumLatitude) / 2 : 0;
    gpuFrame = createGPULocalFrame({
      origin: options.origin ?? [originLongitude, Math.max(-90, Math.min(90, originLatitude))],
      kind
    });
  }
  const project = (position: CatalogPosition): [number, number] =>
    gpuFrame ? gpuFrame.project(position[0], position[1]) : [position[0], position[1]];
  let nonFiniteCount = 0;

  let shiftX = 0;
  let shiftY = 0;
  if (shift !== 'none') {
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumY = Infinity;
    let maximumY = -Infinity;
    for (const position of positions) {
      const [x, y] = project(position);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }
    if (Number.isFinite(minimumX)) {
      shiftX = shift === 'minimum' ? minimumX : (minimumX + maximumX) / 2;
      shiftY = shift === 'minimum' ? minimumY : (minimumY + maximumY) / 2;
    }
  }
  return {
    gpuFrame,
    shiftX,
    shiftY,
    forward: position => {
      const [x, y] = project(position);
      if (!Number.isFinite(x) || !Number.isFinite(y)) nonFiniteCount++;
      return [x - shiftX, y - shiftY];
    },
    getNonFiniteCount: () => nonFiniteCount,
    inverse: (x, y) =>
      gpuFrame ? gpuFrame.unproject(x + shiftX, y + shiftY) : [x + shiftX, y + shiftY]
  };
}

/** Longitude wrapped to [-180, 180). */
function wrapLongitude(longitude: number): number {
  return ((((longitude + 180) % 360) + 360) % 360) - 180;
}

/**
 * Adds a `projection-nan` warning when `frame.forward` met positions it could not project, which
 * happens for positions antipodal to the frame origin. Call it after the operation has projected
 * its inputs.
 */
export function addLocalFrameDiagnostic(
  diagnostics: CatalogDiagnostic[],
  frame: Pick<LocalFrame, 'getNonFiniteCount'>
): void {
  const count = frame.getNonFiniteCount();
  if (count) {
    addCatalogDiagnostic(
      diagnostics,
      'projection-nan',
      'warning',
      'Positions antipodal to the local frame origin cannot be projected and produce non-finite coordinates.',
      count
    );
  }
}
