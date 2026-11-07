// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getDirectedEdgeDestination, getDirectedEdgeOrigin, vertexToLatLng} from 'h3-js';

/** Decodes the GPU's low-word/high-word H3 identity storage to canonical bare hex. */
export function keyToHex(keyWords: Uint32Array, row: number): string {
  const lowWord = keyWords[row * 2];
  const highWord = keyWords[row * 2 + 1];
  if (lowWord === 0 && highWord === 0) return '';
  return ((BigInt(highWord) << 32n) | BigInt(lowWord)).toString(16).padStart(15, '0');
}

export function toCellIndex(key: string): string {
  return key ? `0x${key}` : '—';
}

export function sharedVertexKeys(
  vertices: Uint32Array,
  firstRow: number,
  secondRow: number
): string[] {
  const getRowKeys = (row: number) => {
    const keys = new Set<string>();
    for (let slot = 0; slot < 6; slot++) {
      const key = keyToHex(vertices, row * 6 + slot);
      if (key) keys.add(key);
    }
    return keys;
  };
  const first = getRowKeys(firstRow);
  const second = getRowKeys(secondRow);
  return [...first].filter(key => second.has(key)).sort();
}

/** Converts GPU-produced vertex identities to display coordinates using h3-js bare-hex keys. */
export function makeVertexPoints(keys: readonly string[]): [number, number][] {
  const points: [number, number][] = [];
  for (const key of keys) {
    try {
      const [latitude, longitude] = vertexToLatLng(key);
      points.push([longitude, latitude]);
    } catch {
      // A malformed GPU key is omitted from the display overlay.
    }
  }
  return points;
}

function getDirectedEdgesForCell(edges: Uint32Array, cellRow: number): string[] {
  const keys: string[] = [];
  for (let slot = 0; slot < 6; slot++) {
    const key = keyToHex(edges, cellRow * 6 + slot);
    if (key) keys.push(key);
  }
  return keys;
}

/** Decodes the opposed directed edges for two adjacent GPU-produced cell identities. */
export function findOpposedEdgePair(
  edges: Uint32Array,
  firstCell: string,
  secondCell: string
): string {
  const firstEdges = getDirectedEdgesForCell(edges, 0);
  const secondEdges = getDirectedEdgesForCell(edges, 1);
  for (const firstEdge of firstEdges) {
    try {
      if (
        getDirectedEdgeOrigin(firstEdge).toLowerCase() === firstCell.toLowerCase() &&
        getDirectedEdgeDestination(firstEdge).toLowerCase() === secondCell.toLowerCase()
      ) {
        const reverseEdge = secondEdges.find(edge => {
          try {
            return (
              getDirectedEdgeOrigin(edge).toLowerCase() === secondCell.toLowerCase() &&
              getDirectedEdgeDestination(edge).toLowerCase() === firstCell.toLowerCase()
            );
          } catch {
            return false;
          }
        });
        if (reverseEdge) return `0x${firstEdge} ↔ 0x${reverseEdge}`;
      }
    } catch {
      // A malformed GPU identity is omitted from the display summary.
    }
  }
  return 'No opposed edge pair decoded';
}
