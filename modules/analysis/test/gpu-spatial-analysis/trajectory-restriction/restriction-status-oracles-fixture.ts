// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Hand oracle: a segment (-2,0)->(2,0), t=0..4 crosses the box [-1,1]^2
// during [1,3]. First-slot flags: temporal support 32 plus interval 1 = 33.
// Open geometry excludes both crossing instants, adding 64+128 => 225.
// Triangle [(0,0),(1,0),(0,1)] touches y=-x only at (0,0), hence
// 32+singleton 2=34 when closed and just 32 when open. A wholly contained
// segment adds full-support 16: 49, and its final point gives 32+16+2=50.
// Invalid box bounds mark both observations 4; a duration-4 gap at limit 1
// gives 32+8=40 at its initial outside point. Unused second slots are zero.
export const RESTRICTION_STATUS_CASES = [
  {
    name: 'crossing closed',
    positions: [-2, 0, 2, 0],
    parameters: [0, 4, -1, -1, 1, 1, 0],
    boundary: 'closed',
    rings: undefined,
    status: [33, 0, 32, 0]
  },
  {
    name: 'crossing open',
    positions: [-2, 0, 2, 0],
    parameters: [0, 4, -1, -1, 1, 1, 0],
    boundary: 'open',
    rings: undefined,
    status: [225, 0, 32, 0]
  },
  {
    name: 'touch closed',
    positions: [-1, 1, 1, -1],
    parameters: [0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    boundary: 'closed',
    rings: [3],
    status: [34, 0, 32, 0]
  },
  {
    name: 'touch open',
    positions: [-1, 1, 1, -1],
    parameters: [0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    boundary: 'open',
    rings: [3],
    status: [32, 0, 32, 0]
  },
  {
    name: 'fully inside',
    positions: [0, 0, 0.5, 0],
    parameters: [0, 4, -1, -1, 1, 1, 0],
    boundary: 'closed',
    rings: undefined,
    status: [49, 0, 50, 0]
  },
  {
    name: 'invalid box',
    positions: [-2, 0, 2, 0],
    parameters: [0, 4, 1, -1, -1, 1, 0],
    boundary: 'closed',
    rings: undefined,
    status: [4, 0, 4, 0]
  },
  {
    name: 'gap',
    positions: [-2, 0, 2, 0],
    parameters: [0, 4, -1, -1, 1, 1, 1],
    boundary: 'closed',
    rings: undefined,
    status: [40, 0, 32, 0]
  }
] as const;
