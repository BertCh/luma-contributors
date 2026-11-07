// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU rendering of a field layer for the static fallback: a tiny BMP data URI plus contour paths.
// Mirrors the field shader closely enough that the live canvas replaces it without a visual jump.

import {clamp, sampleField} from './data.ts';
import {getRampColor, type Rgb} from './palette.ts';
import type {FieldLayer} from './types.ts';

const IMAGE_COLUMNS = 48;
const IMAGE_ROWS = 30;
/** Expansion cutoff shown by the still frame of isochrones. */
const STILL_CUTOFF = 0.55;
const CORNER_OFFSETS = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1]
];

function encodeBase64(bytes: Uint8Array): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let text = '';
  for (let index = 0; index < bytes.length; index += 3) {
    const chunk = (bytes[index] << 16) | ((bytes[index + 1] ?? 0) << 8) | (bytes[index + 2] ?? 0);
    text += alphabet[(chunk >> 18) & 63] + alphabet[(chunk >> 12) & 63];
    text += index + 1 < bytes.length ? alphabet[(chunk >> 6) & 63] : '=';
    text += index + 2 < bytes.length ? alphabet[chunk & 63] : '=';
  }
  return text;
}

function getFieldColor(layer: FieldLayer, u: number, v: number): Rgb {
  const {values, columns, rows} = layer;
  const value = sampleField(values, columns, rows, u, v);
  if (layer.style === 'hillshade') {
    const step = 1 / layer.columns;
    const strength = layer.strength ?? 5;
    const gradientX =
      clamp(
        (sampleField(values, columns, rows, u + step, v) -
          sampleField(values, columns, rows, u - step, v)) *
          strength,
        -0.5,
        0.5
      ) * 8;
    const gradientY =
      clamp(
        (sampleField(values, columns, rows, u, v + 1 / layer.rows) -
          sampleField(values, columns, rows, u, v - 1 / layer.rows)) *
          strength,
        -0.5,
        0.5
      ) * 8;
    const length = Math.hypot(gradientX, gradientY, 1);
    const angle = 2.2;
    const light: Rgb = [Math.cos(angle) * 0.6, Math.sin(angle) * 0.6, 0.55];
    const lightLength = Math.hypot(...light);
    const shade = clamp(
      (-gradientX * light[0] - gradientY * light[1] + light[2]) / (length * lightLength),
      0,
      1
    );
    return getRampColor(layer.ramp, value).map(channel => channel * (0.35 + 0.9 * shade)) as Rgb;
  }
  if (layer.style === 'contours') {
    return getRampColor(layer.ramp, value).map(channel => channel * 0.16) as Rgb;
  }
  if (layer.style === 'expand') {
    const inside = value < STILL_CUTOFF;
    const color = getRampColor(layer.ramp, (value / STILL_CUTOFF) * 0.9);
    return inside ? color : (color.map(channel => channel * 0.12) as Rgb);
  }
  return getRampColor(layer.ramp, value);
}

/** 24-bit BMP of the field as a data URI. */
export function createFieldImageUri(layer: FieldLayer): string {
  const columns = layer.filter === 'nearest' ? layer.columns : IMAGE_COLUMNS;
  const rows = layer.filter === 'nearest' ? layer.rows : IMAGE_ROWS;
  const rowBytes = Math.ceil((columns * 3) / 4) * 4;
  const bytes = new Uint8Array(54 + rowBytes * rows);
  const view = new DataView(bytes.buffer);
  bytes[0] = 0x42;
  bytes[1] = 0x4d;
  view.setUint32(2, bytes.length, true);
  view.setUint32(10, 54, true);
  view.setUint32(14, 40, true);
  view.setInt32(18, columns, true);
  view.setInt32(22, rows, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 24, true);
  view.setUint32(34, rowBytes * rows, true);
  for (let row = 0; row < rows; row++) {
    const offset = 54 + (rows - 1 - row) * rowBytes;
    for (let column = 0; column < columns; column++) {
      const color = getFieldColor(layer, (column + 0.5) / columns, (row + 0.5) / rows);
      bytes[offset + column * 3] = Math.round(clamp(color[2], 0, 1) * 255);
      bytes[offset + column * 3 + 1] = Math.round(clamp(color[1], 0, 1) * 255);
      bytes[offset + column * 3 + 2] = Math.round(clamp(color[0], 0, 1) * 255);
    }
  }
  return `data:image/bmp;base64,${encodeBase64(bytes)}`;
}

export type FieldContour = {path: string; color: Rgb; width: number};

/** Marching squares contour paths in scene units. */
export function createFieldContours(
  layer: FieldLayer,
  sceneWidth: number,
  sceneHeight: number
): FieldContour[] {
  const levels = layer.levels ?? 10;
  const columns = 64;
  const rows = 40;
  const grid = new Float32Array((columns + 1) * (rows + 1));
  for (let row = 0; row <= rows; row++) {
    for (let column = 0; column <= columns; column++) {
      grid[row * (columns + 1) + column] = sampleField(
        layer.values,
        layer.columns,
        layer.rows,
        column / columns,
        row / rows
      );
    }
  }
  const cellWidth = sceneWidth / columns;
  const cellHeight = sceneHeight / rows;
  const contours: FieldContour[] = [];
  const maximumLevel = layer.style === 'expand' ? Math.floor(STILL_CUTOFF * levels) : levels;
  for (let level = 1; level < maximumLevel + (layer.style === 'expand' ? 1 : 0); level++) {
    const threshold = level / levels;
    let path = '';
    const at = (column: number, row: number) => grid[row * (columns + 1) + column];
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const corners = [
          at(column, row),
          at(column + 1, row),
          at(column + 1, row + 1),
          at(column, row + 1)
        ];
        const crossings: [number, number][] = [];
        for (let edge = 0; edge < 4; edge++) {
          const first = corners[edge];
          const second = corners[(edge + 1) % 4];
          if (first < threshold !== second < threshold) {
            const amount = (threshold - first) / (second - first);
            const [firstX, firstY] = CORNER_OFFSETS[edge];
            const [secondX, secondY] = CORNER_OFFSETS[(edge + 1) % 4];
            crossings.push([
              (column + firstX + (secondX - firstX) * amount) * cellWidth,
              (row + firstY + (secondY - firstY) * amount) * cellHeight
            ]);
          }
        }
        for (let index = 0; index + 1 < crossings.length; index += 2) {
          path += `M${crossings[index][0].toFixed(1)} ${crossings[index][1].toFixed(1)}L${crossings[index + 1][0].toFixed(1)} ${crossings[index + 1][1].toFixed(1)}`;
        }
      }
    }
    if (path) {
      contours.push({
        path,
        color: getRampColor(layer.ramp, threshold),
        width: level % 4 === 0 ? 0.55 : 0.3
      });
    }
  }
  return contours;
}
