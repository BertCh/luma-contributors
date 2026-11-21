// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/index';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';

export function createSplitOracleFixture(
  view: OracleView,
  selfContact: boolean,
  optional: 'edges' | 'uncertain' | 'absent'
) {
  const positions = view('float32x2', 8);
  const lineOffsets = view('uint32', 5);
  const sourceIds = view('uint32', 4);
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const vertexSourceEdges = optional === 'edges' ? view('uint32', 8) : undefined;
  const uncertainCount = optional === 'uncertain' ? view('uint32', 1) : undefined;
  const contributor = new GPULineSplit({
    // Both fixtures intersect exactly at (1,1). Separate lines yield four pieces;
    // a single bow tie yields three pieces, with edges 0 and 2 marked self-contact.
    lines: {
      kind: 'lines',
      positions: view('float32x2', 4, [0, 0, 2, 2, 0, 2, 2, 0]),
      lineOffsets: selfContact ? view('uint32', 2, [0, 4]) : view('uint32', 3, [0, 2, 4])
    },
    intersectionCapacity: 8,
    output: {
      pieces: {
        geometry: {kind: 'lines', positions, lineOffsets},
        sourceIds,
        vertexSourceEdges,
        status: {count, overflow}
      },
      status: {uncertainCount}
    }
  });
  return {
    contributor,
    positions,
    lineOffsets,
    sourceIds,
    count,
    overflow,
    vertexSourceEdges,
    uncertainCount
  };
}
