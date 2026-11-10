// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import blocks from './chicago-workplace-blocks.json';
import {createTracts, type Random, type World} from '../runs/inputs';

/** LODES workplace blocks and their Census tract keys, in the same local coordinate system. */
export function getWorkplaceRows(random: Random, world: World) {
  const tracts = createTracts(random, world, 8);
  const positions = new Float32Array(blocks.length * 2);
  const keys = new Uint32Array(blocks.length);
  const jobs = new Float32Array(blocks.length);
  blocks.forEach(([east, north, count, tract], row) => {
    positions.set(tracts.projectMeters(east, north), row * 2);
    keys[row] = tract;
    jobs[row] = count;
  });
  return {tracts, positions, keys, jobs};
}
