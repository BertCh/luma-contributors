// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_WGS84_SEMI_MAJOR_AXIS_METERS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {LocalMetricProjection} from '../../engine/projection';
import {getLatitudeFromMercatorY, getMercatorY} from '../../engine/web-mercator';
import type {UtmFrame} from './b16-common';
import {toRadians} from '@math.gl/core';

/** Exact local Mercator analysis coordinates, separately mapped into deck's display frame. */
export function createMercatorGridFrame(
  origin: readonly [number, number],
  center: readonly [number, number],
  size: number,
  cellSize: number
): UtmFrame {
  const radiansPerDegree = toRadians(1);
  const radius = GPU_WGS84_SEMI_MAJOR_AXIS_METERS;
  const cosine = Math.cos(center[1] * radiansPerDegree);
  const centerNorthing = getMercatorY(center[1], radius);
  const halfWidth = (size * cellSize) / 2;
  const display = new LocalMetricProjection(origin);
  const toLngLat = (eastingOffset: number, northingOffset: number): [number, number] => [
    center[0] + eastingOffset / (radius * cosine * radiansPerDegree),
    getLatitudeFromMercatorY(centerNorthing + northingOffset / cosine, radius)
  ];
  const toMeters = (eastingOffset: number, northingOffset: number): [number, number] =>
    display.project(...toLngLat(eastingOffset, northingOffset));
  const translation = toMeters(0, 0);
  const east = toMeters(1000, 0);
  const north = toMeters(0, 1000);
  return {
    columns: size,
    rows: size,
    cellSize,
    halfWidth,
    halfHeight: halfWidth,
    origin: [origin[0], origin[1]],
    translation,
    rotation: [
      (east[0] - translation[0]) / 1000,
      (east[1] - translation[1]) / 1000,
      (north[0] - translation[0]) / 1000,
      (north[1] - translation[1]) / 1000
    ],
    lngLatToCell(longitude, latitude) {
      const eastingOffset = (longitude - center[0]) * radiansPerDegree * radius * cosine;
      const northingOffset = (getMercatorY(latitude, radius) - centerNorthing) * cosine;
      return [(eastingOffset + halfWidth) / cellSize, (halfWidth - northingOffset) / cellSize];
    },
    localToMeters: toMeters,
    cellToMeters(column, row) {
      return toMeters((column + 0.5) * cellSize - halfWidth, halfWidth - (row + 0.5) * cellSize);
    },
    cellToLngLat(column, row) {
      return toLngLat((column + 0.5) * cellSize - halfWidth, halfWidth - (row + 0.5) * cellSize);
    }
  };
}
