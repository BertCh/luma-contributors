// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Feature, Geometry} from 'geojson';

import {clamp, toRadians} from '@math.gl/core';

/** Kepler 3.3 has no native replay: this walks drawn shapes and annotations in creation order. */
export type DrawingReplayStep = {
  kind: 'feature' | 'annotation';
  id: string;
  label: string;
  bounds: readonly [west: number, south: number, east: number, north: number];
  longitude: number;
  latitude: number;
  zoom: number;
};

export type DrawingReplayAnnotation = {
  id: string;
  kind: string;
  isVisible?: boolean;
  anchorPoint: readonly number[];
  label?: string;
  radiusInMeters?: number;
};

export type DrawingReplayViewport = {width: number; height: number};

const METERS_PER_DEGREE_LATITUDE = 111_320;
const SINGLE_POINT_ZOOM = 15;
const MAXIMUM_REPLAY_ZOOM = 18;
const PADDING_RATIO = 0.8;

type Bounds = [number, number, number, number];

function extendBounds(bounds: Bounds, coordinates: unknown): void {
  if (!Array.isArray(coordinates)) return;
  if (typeof coordinates[0] === 'number') {
    const [longitude, latitude] = coordinates as number[];
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return;
    bounds[0] = Math.min(bounds[0], longitude);
    bounds[1] = Math.min(bounds[1], latitude);
    bounds[2] = Math.max(bounds[2], longitude);
    bounds[3] = Math.max(bounds[3], latitude);
    return;
  }
  for (const child of coordinates) extendBounds(bounds, child);
}

export function getGeometryBounds(geometry: Geometry | null | undefined): Bounds | null {
  if (!geometry) return null;
  const bounds: Bounds = [Infinity, Infinity, -Infinity, -Infinity];
  if (geometry.type === 'GeometryCollection')
    for (const child of geometry.geometries) {
      const childBounds = getGeometryBounds(child);
      if (childBounds) extendBounds(bounds, [childBounds.slice(0, 2), childBounds.slice(2, 4)]);
    }
  else extendBounds(bounds, geometry.coordinates);
  return Number.isFinite(bounds[0]) ? bounds : null;
}

function getMercatorY(latitude: number): number {
  const clamped = clamp(latitude, -85.05, 85.05);
  const radians = toRadians(clamped);
  return Math.log(Math.tan(Math.PI / 4 + radians / 2)) / (2 * Math.PI);
}

/** Web Mercator zoom (512 px tiles) that fits the bounds with padding; points use a fixed zoom. */
export function fitReplayBounds(
  bounds: Bounds,
  viewport: DrawingReplayViewport
): {longitude: number; latitude: number; zoom: number} {
  const longitude = (bounds[0] + bounds[2]) / 2;
  const latitude = (bounds[1] + bounds[3]) / 2;
  const widthFraction = (bounds[2] - bounds[0]) / 360;
  const heightFraction = Math.abs(getMercatorY(bounds[3]) - getMercatorY(bounds[1]));
  if (widthFraction <= 0 && heightFraction <= 0)
    return {longitude, latitude, zoom: SINGLE_POINT_ZOOM};
  const zoomForWidth =
    widthFraction > 0
      ? Math.log2((viewport.width * PADDING_RATIO) / (512 * widthFraction))
      : Infinity;
  const zoomForHeight =
    heightFraction > 0
      ? Math.log2((viewport.height * PADDING_RATIO) / (512 * heightFraction))
      : Infinity;
  const zoom = Math.max(0, Math.min(MAXIMUM_REPLAY_ZOOM, zoomForWidth, zoomForHeight));
  return {longitude, latitude, zoom};
}

/** Drawn features first (their array order is creation order), then annotations. */
export function planDrawingReplay(
  features: readonly Feature<Geometry>[],
  annotations: readonly DrawingReplayAnnotation[],
  viewport: DrawingReplayViewport
): DrawingReplayStep[] {
  const steps: DrawingReplayStep[] = [];
  for (const [index, feature] of features.entries()) {
    const bounds = getGeometryBounds(feature.geometry);
    if (!bounds) continue;
    steps.push({
      kind: 'feature',
      id: String(feature.id ?? index),
      label: `${feature.geometry.type} ${index + 1}`,
      bounds,
      ...fitReplayBounds(bounds, viewport)
    });
  }
  // Native annotation actions prepend new entries, so reverse them into creation order.
  for (const annotation of [...annotations].reverse()) {
    if (annotation.isVisible === false) continue;
    const [longitude, latitude] = annotation.anchorPoint;
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) continue;
    const radiusDegrees = (annotation.radiusInMeters ?? 0) / METERS_PER_DEGREE_LATITUDE;
    const longitudeRadius = radiusDegrees / Math.max(0.01, Math.cos(toRadians(latitude)));
    const bounds: Bounds = [
      longitude - longitudeRadius,
      latitude - radiusDegrees,
      longitude + longitudeRadius,
      latitude + radiusDegrees
    ];
    steps.push({
      kind: 'annotation',
      id: annotation.id,
      label: annotation.label || `${annotation.kind} annotation`,
      bounds,
      ...fitReplayBounds(bounds, viewport)
    });
  }
  return steps;
}

export type DrawingReplayState = {index: number; playing: boolean};

/** Pure transitions: next wraps to the first step, previous wraps to the last, stop keeps the index. */
export function stepDrawingReplay(
  state: DrawingReplayState,
  stepCount: number,
  action: 'play' | 'pause' | 'next' | 'previous' | 'restart' | 'tick'
): DrawingReplayState {
  if (!stepCount) return {index: 0, playing: false};
  switch (action) {
    case 'play':
      return {index: state.index, playing: true};
    case 'pause':
      return {index: state.index, playing: false};
    case 'restart':
      return {index: 0, playing: true};
    case 'previous':
      return {index: (state.index - 1 + stepCount) % stepCount, playing: false};
    case 'next':
      return {index: (state.index + 1) % stepCount, playing: false};
    case 'tick':
      // Playback stops after the last step instead of looping forever.
      return state.index + 1 >= stepCount
        ? {index: state.index, playing: false}
        : {index: state.index + 1, playing: true};
  }
}
