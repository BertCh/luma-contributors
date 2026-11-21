// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {cellToVertexes, originToDirectedEdges} from 'h3-js';
import {expect, it} from 'vitest';
import fixture from './face-edge-identities-fixture.json';

const {S2} = createRequire(import.meta.url)('s2-geometry');

it('pins independent S2 face transitions and H3 identities including deleted directions', () => {
  expect(fixture.s2.length).toBe(108);
  for (const row of fixture.s2) {
    const cell = S2.S2Cell.FromFaceIJ(row.face, row.ij, row.level);
    expect(S2.toId(cell.toHilbertQuadkey())).toBe(row.key);
    const neighbors = cell
      .getNeighbors()
      .map((neighbor: {toHilbertQuadkey(): string}) => S2.toId(neighbor.toHilbertQuadkey()));
    expect(new Set(neighbors)).toEqual(new Set(row.neighbors));
  }
  expect(fixture.h3.filter(row => row.pentagon)).toHaveLength(60);
  for (const row of fixture.h3) {
    expect(row.edges.filter(key => key !== '0')).toEqual(originToDirectedEdges(row.cell));
    expect(row.vertices.filter(key => key !== '0')).toEqual(cellToVertexes(row.cell));
    if (row.pentagon) {
      expect(row.edges[0]).toBe('0');
      expect(row.vertices[5]).toBe('0');
    }
  }
});
