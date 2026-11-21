// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createPlaneTransferFixture,
  PLANE_TRANSFER_BLOCK_ROWS,
  PLANE_TRANSFER_PAIRS
} from './pointcloud-icp-transfer-fixture';

for (const center of [
  [0, 0, 0],
  [500, 300, 0]
]) {
  it(`plane transfer fixture has zero within-block rotation information at ${center}`, () => {
    const fixture = createPlaneTransferFixture(center);
    expect(fixture.source).toHaveLength(6 * 256 * 3);
    expect(fixture.target).toHaveLength(6 * 256 * 3);
    expect(fixture.normals).toHaveLength(6 * 256 * 3);
    // A constant source position equals its block centre; cross(position-centre,normal)=0.
    // Thus each untransferred block contributes only translation information, of combined rank 3.
    for (let block = 0; block < PLANE_TRANSFER_PAIRS.length; block++) {
      const begin = block * PLANE_TRANSFER_BLOCK_ROWS * 3;
      const first = fixture.source.slice(begin, begin + 3);
      for (let row = 0; row < PLANE_TRANSFER_BLOCK_ROWS; row++) {
        expect(fixture.source.slice(begin + row * 3, begin + row * 3 + 3)).toEqual(first);
        expect(fixture.normals.slice(begin + row * 3, begin + row * 3 + 3)).toEqual(
          PLANE_TRANSFER_PAIRS[block].normal
        );
      }
    }
  });
}
