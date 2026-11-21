// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeHealpixPointToCellOnCPU} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';

it('CPU HEALPix pins base faces, valid zero key, masked and invalid rows', () => {
  // At order 0 the equatorial face centers at longitude 0 and 90 are faces 4 and 5.
  // (45,60) is strictly inside northern face 0. UNIQ adds 4^(0+1)=4.
  // The last two rows are nonfinite and masked respectively, and always get key zero.
  for (const scheme of ['nested', 'uniq'] as const) {
    const result = computeHealpixPointToCellOnCPU(
      Float32Array.of(0, 0, 90, 0, 45, 60, NaN, 0, 180, 0),
      {order: 0, scheme, mask: Uint32Array.of(1, 1, 1, 1, 0)}
    );
    expect(Array.from(result.cells)).toEqual(
      scheme === 'nested' ? [4, 0, 5, 0, 0, 0, 0, 0, 0, 0] : [8, 0, 9, 0, 4, 0, 0, 0, 0, 0]
    );
    expect(Array.from(result.validity)).toEqual([1, 1, 1, 0, 0]);
  }
});
