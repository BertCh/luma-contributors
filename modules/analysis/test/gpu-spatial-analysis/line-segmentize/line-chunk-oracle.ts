// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCentralAngle, interpolateGreatCircle} from '../geometry-measures/geodesic-oracle';
import {getFlatVertex, type FlatLineResult, type FlatLines} from './line-segmentize-oracle';

type Piece = {line: number; start: number; end: number; copy: boolean};

function getRequiredChunks(length: number, chunkLength: number): number {
  if (!(chunkLength > 0)) {
    return 1;
  }
  // Search rounded boundaries independently of the device's division estimate.
  let high = 2 ** 24;
  if (!Number.isFinite(length) || Math.fround(high * chunkLength) < length) {
    return Math.min(0xffffffff, Math.max(1, Math.ceil(Math.fround(length / chunkLength))));
  }
  let low = 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (Math.fround(middle * chunkLength) < length) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

/**
 * Reference of `GPULineChunk` with independently rounded float32 measures for cut selection
 * and f64 measures for interpolation. Interior cuts at a vertex use the first row with
 * its published measure; a terminal cut uses the last row. Input rows between the cuts
 * are retained, including coincident rows.
 */
export function chunkLines(
  lines: FlatLines,
  options: {
    mode: 'chunk' | 'substring';
    chunkLength?: number;
    startMeasure?: number;
    endMeasure?: number;
    lineCapacity: number;
    spherical?: boolean;
    radius?: number;
  }
): FlatLineResult & {sourceLineIndices: number[]; pieceTotal: number} {
  const spherical = Boolean(options.spherical);
  const radius = options.radius ?? 6371008.8;
  const lineCount = lines.lineOffsets.length - 1;
  const measures: number[] = [];
  const publishedMeasures = new Float32Array(lines.positions.length / 2);
  const shifted: number[][] = [];
  for (let line = 0; line < lineCount; line++) {
    let measure = 0;
    let publishedSum = 0;
    let shift = 0;
    for (let row = lines.lineOffsets[line]; row < lines.lineOffsets[line + 1]; row++) {
      const vertex = getFlatVertex(lines, row);
      if (row > lines.lineOffsets[line]) {
        const previous = getFlatVertex(lines, row - 1);
        const segmentLength = spherical
          ? getCentralAngle(previous, vertex) * radius
          : Math.hypot(vertex[0] - previous[0], vertex[1] - previous[1]);
        measure += segmentLength;
        // Sum independently measured float32 segment lengths in f64, rounding at publication.
        publishedSum += Math.fround(segmentLength);
        shift -= 360 * Math.round((vertex[0] - previous[0]) / 360);
      }
      measures[row] = measure;
      publishedMeasures[row] = publishedSum;
      shifted[row] = spherical ? [vertex[0] + shift, vertex[1]] : vertex;
    }
  }
  const pieces: Piece[] = [];
  let pieceTotal = 0;
  for (let line = 0; line < lineCount; line++) {
    const start = lines.lineOffsets[line];
    const end = lines.lineOffsets[line + 1];
    if (end <= start) {
      continue;
    }
    const length = publishedMeasures[end - 1];
    if (!(length > 0)) {
      pieceTotal = Math.min(0xffffffff, pieceTotal + 1);
      pieces.push({line, start: 0, end: 0, copy: true});
      continue;
    }
    if (options.mode === 'chunk') {
      const chunkLength = Math.fround(options.chunkLength ?? 0);
      const required = getRequiredChunks(length, chunkLength);
      pieceTotal = Math.min(0xffffffff, pieceTotal + required);
      const count = Math.min(required, options.lineCapacity);
      const getBoundary = (index: number) => Math.fround(Math.fround(index) * chunkLength);
      for (let piece = 0; piece < count; piece++) {
        pieces.push({
          line,
          start: chunkLength > 0 ? getBoundary(piece) : 0,
          end: piece + 1 < required ? getBoundary(piece + 1) : length,
          copy: false
        });
      }
    } else {
      pieceTotal = Math.min(0xffffffff, pieceTotal + 1);
      const clamp = (value: number) => Math.min(Math.max(Math.fround(value), 0), length);
      pieces.push({
        line,
        start: clamp(options.startMeasure ?? 0),
        end: clamp(options.endMeasure ?? length),
        copy: false
      });
    }
  }
  const interpolate = (row: number, measure: number): number[] => {
    const segmentMeasure = measures[row + 1] - measures[row];
    const fraction =
      segmentMeasure > 0 ? Math.min(Math.max((measure - measures[row]) / segmentMeasure, 0), 1) : 0;
    const a = shifted[row];
    const b = shifted[row + 1];
    return spherical
      ? interpolateGreatCircle(a, b, fraction)
      : [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction];
  };
  const result = {
    positions: [] as number[][],
    lineOffsets: [0],
    sourceRows: [] as number[],
    measures: [] as number[],
    sourceLineIndices: [] as number[],
    pieceTotal
  };
  for (const piece of pieces.slice(0, options.lineCapacity)) {
    const start = lines.lineOffsets[piece.line];
    const end = lines.lineOffsets[piece.line + 1];
    const push = (vertex: number[], measure: number, row: number) => {
      result.positions.push(vertex);
      result.measures.push(measure);
      result.sourceRows.push(row);
    };
    if (piece.copy) {
      for (let row = start; row < end; row++) {
        push(shifted[row], measures[row], row);
      }
    } else if (piece.start <= piece.end) {
      const findCutRow = (measure: number) => {
        if (measure >= publishedMeasures[end - 1]) {
          return end - 1;
        }
        let row = start;
        while (publishedMeasures[row] < measure) {
          row++;
        }
        return row;
      };
      const startCut = findCutRow(piece.start);
      const endCut = findCutRow(piece.end);
      const firstInterior = startCut + (publishedMeasures[startCut] === piece.start ? 1 : 0);
      const clampRow = (row: number) => Math.min(Math.max(row, start + 1), end - 1) - 1;
      const cutPosition = (cut: number, segment: number, measure: number) =>
        publishedMeasures[cut] === measure ? shifted[cut] : interpolate(segment, measure);
      const startSegment = clampRow(firstInterior);
      push(cutPosition(startCut, startSegment, piece.start), piece.start, startSegment);
      for (let row = firstInterior; row < endCut; row++) {
        push(shifted[row], measures[row], row);
      }
      const endSegment = clampRow(Math.max(endCut, firstInterior));
      push(cutPosition(endCut, endSegment, piece.end), piece.end, endSegment);
    }
    result.lineOffsets.push(result.positions.length);
    result.sourceLineIndices.push(piece.line);
  }
  while (result.lineOffsets.length < options.lineCapacity + 1) {
    result.lineOffsets.push(result.positions.length);
    result.sourceLineIndices.push(0xffffffff);
  }
  return result;
}
