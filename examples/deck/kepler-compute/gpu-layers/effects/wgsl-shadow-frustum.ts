// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Matrix4} from '@math.gl/core';

/**
 * Light frustums for views where Deck's shadow module cannot size the shadow map.
 *
 * Deck fits the light's orthographic frustum to the intersection of the view frustum with the
 * ground plane (geospatial map views) or to the view far plane (everything else). The far plane
 * makes orbit and other non-geospatial views unusable and the ground plane does not exist on a
 * globe. These helpers fit the frustum to the scene bounds instead:
 *
 * - non-geospatial views: explicit or layer-derived scene bounds, else the view frustum with its
 *   far plane clamped to twice the camera distance;
 * - globe views: the visible part of the sphere (receivers, for the light-space extent) and the
 *   full sphere plus an elevation margin (casters, for the light-space depth range).
 *
 * All points are in Deck common space, the same space the shadow module projects.
 */

/** Deck's `GLOBE_RADIUS` (common space units of the globe projection). */
export const WGSL_SHADOW_GLOBE_RADIUS = 256;
/** Fraction of the globe radius that casters may rise above the surface (about 64 km). */
const GLOBE_ELEVATION_MARGIN = 0.01;
const GLOBE_SAMPLE_COUNT = 9;
const FRUSTUM_PADDING = 0.02;

/** Axis-aligned scene bounds [[minX, minY, minZ], [maxX, maxY, maxZ]] in the layers' world coordinates. */
export type WGSLShadowSceneBounds = readonly [ArrayLike<number>, ArrayLike<number>];

/** Minimal viewport surface used for frustum fitting. */
export type WGSLShadowFrustumViewport = {
  width: number;
  height: number;
  isGeospatial: boolean;
  projectionMode?: number;
  pixelUnprojectionMatrix: ArrayLike<number>;
  cameraPosition?: ArrayLike<number>;
  center?: ArrayLike<number>;
  projectPosition(position: number[]): number[];
};

/** Orthographic light view-projection, ready for the WGSL shadow uniforms (center is zero). */
export type WGSLShadowProjection = {
  viewProjectionMatrix: number[];
  projectCenter: [number, number, number, number];
  /** Padded light-space extent in common space units (before the optional aspect fit). */
  extentSize: [number, number];
  /** Rotation of the light-space axes about the light direction, in radians. */
  rotation: number;
  /**
   * Shader bias parameters [common units per depth unit, bias start in common units]. The bias is
   * a world-space distance, so it stays valid whatever the depth range of the frustum is.
   */
  depthParams: [number, number];
};

/** Options of `makeWGSLShadowProjection`. */
export type WGSLShadowProjectionOptions = {
  /**
   * Target texture aspect ratio (width / height). When set, the light-space axes are rotated about
   * the light direction to the angle that needs the smallest map of that aspect, which raises the
   * texel density on a visible cap that is a tilted ellipse in light space.
   */
  aspectRatio?: number;
  /** Fixed part of the depth bias in common space units; default 0.1% of the depth range (legacy). */
  depthBias?: number;
  /** Shadow map size in texels; adds one texel of footprint to the bias to suppress acne. */
  mapSize?: readonly [number, number];
};

/** Earth radius in meters divided by `WGSL_SHADOW_GLOBE_RADIUS`: meters per globe common unit. */
export const WGSL_SHADOW_GLOBE_METERS_PER_UNIT = 6371008.8 / WGSL_SHADOW_GLOBE_RADIUS;

/** Light-space extent sources: xy from receivers, depth from receivers and casters. */
export type WGSLShadowFramePoints = {
  /** Points whose light-space x/y define the map extent (everything that can receive a shadow). */
  extentPoints: number[][];
  /** Points whose light-space depth defines near and far (everything that can cast or receive). */
  depthPoints: number[][];
  /** Which source produced the points, for diagnostics. */
  source: 'scene-bounds' | 'view-frustum' | 'globe';
};

function unprojectPixel(matrix: ArrayLike<number>, x: number, y: number, z: number): number[] {
  const result = [0, 0, 0, 0];
  for (let row = 0; row < 4; row++) {
    result[row] = matrix[row] * x + matrix[4 + row] * y + matrix[8 + row] * z + matrix[12 + row];
  }
  const w = result[3] || 1;
  return [result[0] / w, result[1] / w, result[2] / w];
}

function getBoundsCorners(bounds: WGSLShadowSceneBounds): number[][] {
  const [minimum, maximum] = bounds;
  const corners: number[][] = [];
  for (const x of [minimum[0], maximum[0]]) {
    for (const y of [minimum[1], maximum[1]]) {
      for (const z of [minimum[2] ?? 0, maximum[2] ?? 0]) corners.push([x, y, z]);
    }
  }
  return corners;
}

/** Union of the layers' data bounds, or null when no layer reports finite bounds. */
export function mergeWGSLShadowLayerBounds(
  layers: readonly {getBounds?: () => unknown}[]
): WGSLShadowSceneBounds | null {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const layer of layers) {
    const bounds = layer.getBounds?.() as [number[], number[]] | null | undefined;
    if (!bounds) continue;
    for (let axis = 0; axis < 3; axis++) {
      const low = bounds[0][axis] ?? 0;
      const high = bounds[1][axis] ?? 0;
      if (!Number.isFinite(low) || !Number.isFinite(high)) continue;
      minimum[axis] = Math.min(minimum[axis], low);
      maximum[axis] = Math.max(maximum[axis], high);
    }
  }
  return Number.isFinite(minimum[0]) && Number.isFinite(minimum[1]) ? [minimum, maximum] : null;
}

/** View frustum corners in common space with the far plane clamped to twice the camera distance. */
function getClampedFrustumPoints(viewport: WGSLShadowFrustumViewport): number[][] {
  const matrix = viewport.pixelUnprojectionMatrix;
  const camera = viewport.cameraPosition;
  const center = viewport.center;
  let maximumDistance = Infinity;
  if (camera && center) {
    const distance = Math.hypot(
      camera[0] - center[0],
      camera[1] - center[1],
      camera[2] - center[2]
    );
    if (distance > 0 && Number.isFinite(distance)) maximumDistance = distance * 2;
  }
  const points: number[][] = [];
  for (const [x, y] of [
    [0, 0],
    [viewport.width, 0],
    [0, viewport.height],
    [viewport.width, viewport.height]
  ]) {
    const near = unprojectPixel(matrix, x, y, -1);
    const far = unprojectPixel(matrix, x, y, 1);
    const length = Math.hypot(far[0] - near[0], far[1] - near[1], far[2] - near[2]);
    const scale = length > maximumDistance ? maximumDistance / length : 1;
    points.push(near, [
      near[0] + (far[0] - near[0]) * scale,
      near[1] + (far[1] - near[1]) * scale,
      near[2] + (far[2] - near[2]) * scale
    ]);
  }
  return points;
}

/**
 * Nearest intersection of a ray with the origin-centered sphere, or null.
 * Kept hand-written: math.gl's Ray/BoundingSphere intersection does not match these semantics (the
 * segment is bounded by `length`, a ray starting inside the sphere returns null rather than the far
 * root, and tangent hits count).
 */
function intersectSphere(origin: number[], target: number[], radius: number): number[] | null {
  const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (!length) return null;
  const unit = direction.map(value => value / length);
  const b = origin[0] * unit[0] + origin[1] * unit[1] + origin[2] * unit[2];
  const c = origin[0] ** 2 + origin[1] ** 2 + origin[2] ** 2 - radius * radius;
  const discriminant = b * b - c;
  if (discriminant < 0) return null;
  const distance = -b - Math.sqrt(discriminant);
  if (distance < 0 || distance > length) return null;
  return [
    origin[0] + unit[0] * distance,
    origin[1] + unit[1] * distance,
    origin[2] + unit[2] * distance
  ];
}

/** Visible globe surface samples (a screen grid intersected with the sphere), common space. */
function getVisibleGlobePoints(viewport: WGSLShadowFrustumViewport): number[][] {
  const points: number[][] = [];
  for (let row = 0; row < GLOBE_SAMPLE_COUNT; row++) {
    for (let column = 0; column < GLOBE_SAMPLE_COUNT; column++) {
      const x = (column / (GLOBE_SAMPLE_COUNT - 1)) * viewport.width;
      const y = (row / (GLOBE_SAMPLE_COUNT - 1)) * viewport.height;
      const near = unprojectPixel(viewport.pixelUnprojectionMatrix, x, y, -1);
      const far = unprojectPixel(viewport.pixelUnprojectionMatrix, x, y, 1);
      const hit = intersectSphere(near, far, WGSL_SHADOW_GLOBE_RADIUS);
      if (hit) points.push(hit);
    }
  }
  return points;
}

/**
 * Chooses the points that bound the light frustum for a viewport Deck cannot fit itself, or null
 * when the view shows nothing the light could shade (for example a globe looking at empty space).
 */
export function getWGSLShadowFramePoints(
  viewport: WGSLShadowFrustumViewport,
  options: {globe: boolean; sceneBounds?: WGSLShadowSceneBounds | null}
): WGSLShadowFramePoints | null {
  if (options.globe) {
    const visible = getVisibleGlobePoints(viewport);
    if (!visible.length) return null;
    const radius = WGSL_SHADOW_GLOBE_RADIUS * (1 + GLOBE_ELEVATION_MARGIN);
    // Casters rise above the surface, so their shadows land inside the receiver extent along the
    // light direction; only the depth range has to cover the whole shell.
    const shell: number[][] = [];
    for (const x of [-radius, radius])
      for (const y of [-radius, radius]) for (const z of [-radius, radius]) shell.push([x, y, z]);
    return {extentPoints: visible, depthPoints: [...visible, ...shell], source: 'globe'};
  }
  if (options.sceneBounds) {
    const common = getBoundsCorners(options.sceneBounds).map(corner =>
      viewport.projectPosition(corner)
    );
    return {extentPoints: common, depthPoints: common, source: 'scene-bounds'};
  }
  const frustum = getClampedFrustumPoints(viewport);
  return {extentPoints: frustum, depthPoints: frustum, source: 'view-frustum'};
}

function getPaddedRange(values: number[]): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max((high - low) * FRUSTUM_PADDING, 1e-4);
  return [low - padding, high + padding];
}

const ROTATION_SAMPLE_COUNT = 36;

/** Size of the smallest `aspectRatio` rectangle that covers the rotated points, per rotation. */
function getCoveringSize(
  points: ArrayLike<number>[],
  rotation: number,
  aspectRatio: number
): {width: number; height: number} {
  const cosine = Math.cos(rotation);
  const sine = Math.sin(rotation);
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (const point of points) {
    const x = cosine * point[0] - sine * point[1];
    const y = sine * point[0] + cosine * point[1];
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }
  const width = maximumX - minimumX;
  const height = maximumY - minimumY;
  return {
    width: Math.max(width, height * aspectRatio),
    height: Math.max(height, width / aspectRatio)
  };
}

/**
 * Orthographic light view-projection fitted to the frame points. `lightViewMatrix` is the light
 * view matrix built from the light direction, exactly as Deck builds it.
 */
export function makeWGSLShadowProjection(
  lightViewMatrix: Matrix4,
  points: WGSLShadowFramePoints,
  options: WGSLShadowProjectionOptions = {}
): WGSLShadowProjection {
  let extent = points.extentPoints.map(point => lightViewMatrix.transform(point));
  const depth = points.depthPoints.map(point => lightViewMatrix.transform(point));
  let rotation = 0;
  if (options.aspectRatio) {
    // Rotation about the light axis keeps depth unchanged; pick the angle with the tightest map.
    let bestArea = Infinity;
    for (let sample = 0; sample < ROTATION_SAMPLE_COUNT; sample++) {
      const candidate = (sample / ROTATION_SAMPLE_COUNT) * Math.PI;
      const size = getCoveringSize(extent, candidate, options.aspectRatio);
      const area = size.width * size.height;
      if (area < bestArea - 1e-9) {
        bestArea = area;
        rotation = candidate;
      }
    }
  }
  const rotationMatrix = new Matrix4().rotateZ(rotation);
  const fittedView = new Matrix4(lightViewMatrix).multiplyLeft(rotationMatrix);
  if (rotation) extent = points.extentPoints.map(point => fittedView.transform(point));
  const [left, right] = getPaddedRange(extent.map(point => point[0]));
  const [bottom, top] = getPaddedRange(extent.map(point => point[1]));
  const [near, far] = getPaddedRange(depth.map(point => -point[2]));
  const projection = new Matrix4().ortho({left, right, bottom, top, near, far});
  const depthRange = far - near;
  const texelSize = options.mapSize
    ? Math.max((right - left) / options.mapSize[0], (top - bottom) / options.mapSize[1])
    : 0;
  return {
    viewProjectionMatrix: Array.from(projection.multiplyRight(fittedView)),
    projectCenter: [0, 0, 0, 0],
    extentSize: [right - left, top - bottom],
    rotation,
    depthParams: [depthRange, (options.depthBias ?? depthRange * 0.001) + texelSize]
  };
}
