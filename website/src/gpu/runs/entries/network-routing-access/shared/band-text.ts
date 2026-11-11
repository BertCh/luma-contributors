// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Text for the adjacency matrix pictures: the diagonal blocks that grouped order gives each
// north-to-south band of tracts (outlined, named), and a note saying what rows and columns are.
// The block extents are the band sizes of the input groups, which is what the order sorts by.

import type {RunLabel, RunRule, RunText} from '../../../text';
import type {CommunityGraph} from './graphs';

/** Outlined, numbered diagonal blocks for the bands of `graph` in grouped order. */
export function createBandText(
  graph: CommunityGraph,
  world: {width: number; height: number},
  bandCount: number
): RunText {
  const sizes = new Array<number>(bandCount).fill(0);
  for (const group of graph.groups) sizes[group]++;
  const labels: RunLabel[] = [];
  const rules: RunRule[] = [];
  let start = 0;
  sizes.forEach((size, band) => {
    const left = (start / graph.nodeCount) * world.width;
    const right = ((start + size) / graph.nodeCount) * world.width;
    const top = world.height * (1 - start / graph.nodeCount);
    const bottom = world.height * (1 - (start + size) / graph.nodeCount);
    const corners: [number, number][] = [
      [left, top],
      [right, top],
      [right, bottom],
      [left, bottom],
      [left, top]
    ];
    for (let side = 0; side < 4; side++) {
      rules.push({
        from: corners[side],
        to: corners[side + 1],
        tone: 'ink',
        width: 1.2,
        opacity: 0.6,
        dash: 5
      });
    }
    labels.push({
      text:
        band === 0
          ? 'band 1 (north)'
          : band === bandCount - 1
            ? `band ${band + 1} (south)`
            : `band ${band + 1}`,
      at: [left, top],
      anchor: 'top-left',
      offset: [4, -3],
      role: 'label',
      group: 'bands'
    });
    start += size;
  });
  labels.push({
    text: 'tract links, either direction (1 in 6 cross-band kept)',
    at: [0, 1],
    space: 'canvas',
    anchor: 'bottom-left',
    offset: [4, 4],
    role: 'tick',
    priority: 3
  });
  return {labels, rules};
}

/** The note alone, for a matrix in input order (no blocks to name). */
export function createMatrixNote(): RunText {
  return {
    labels: [
      {
        text: 'tract links, either direction (1 in 6 cross-band kept)',
        at: [0, 1],
        space: 'canvas',
        anchor: 'bottom-left',
        offset: [4, 4],
        role: 'tick',
        priority: 3
      }
    ]
  };
}
