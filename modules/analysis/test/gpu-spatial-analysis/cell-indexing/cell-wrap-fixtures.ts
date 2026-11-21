// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent resolution-5 keys at (-150, 0) and (-150, 10).
// Quad tiles: x=floor(30*32/360)=2, y=floor((1-asinh(tan(lat))/pi)*16)=16 or 15;
// interleave (x,y) to paths 516 and 174. Geohash interleaves floor(30*8192/360)
// and floor((lat+90)*4096/180). S2: face 3, quadratic ST(z/x,y/x), level-5 Hilbert
// paths 748 and 801. Python derivation retained in this stream's scratch/expectations.py.
// H3: h3-js latLngToCell(latitude,-150,5), an independent compiled reference.
export const WRAPPED_CELL_KEYS = {
  quadbin: [0x485813ffffffffffn, 0x4852bbffffffffffn],
  quadkey: [0x1400000000000204n, 0x14000000000000aen],
  geohash: [0x5000000000844444n, 0x500000000084ec4en],
  h3: [0x85888bdbfffffffn, 0x855c19bbfffffffn],
  s2: [0x7764000000000000n, 0x790c000000000000n]
} as const;
