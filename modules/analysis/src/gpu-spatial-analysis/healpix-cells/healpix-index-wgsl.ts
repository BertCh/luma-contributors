// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL} from '../cell-aggregation/cell-keys';
import {getExactDegreeReductionWGSL} from '../cell-indexing/cell-index-wgsl';

/** Largest HEALPix order (depth) of the nested scheme in a 64-bit index, as in healpix-geo. */
export const HEALPIX_MAXIMUM_ORDER = 29;

/**
 * HEALPix nested forward indexing (Gorski et al. 2005, `ang2pix_nest`) in f32. Concatenate it
 * after nothing else: it includes `CELL_KEY_WGSL` and the Morton helper of `QUADBIN_TILE_WGSL`.
 *
 * Defines `cellIndexHealpixNestedFromLngLat(lng, lat, order) -> vec2u(high, low)`, the nested
 * index `face * 4^order + interleave(ix, iy)` with `ix` on the even bits. Longitude and latitude
 * are degrees on the sphere (geodetic equals geocentric latitude; there is no ellipsoid).
 *
 * The projection is closed form: with `z = sin(lat)`, the equatorial belt (`|z| <= 2/3`) takes the
 * ascending and descending diagonal coordinates `jp`, `jm` of `nside (0.5 + tt) -+ 0.75 nside z`
 * with `tt = lng / 90` in [0, 4), and the polar caps use `nside sqrt(3 (1 - |z|))`, where
 * `1 - |z|` is evaluated as `2 sin^2` of half the colatitude so it keeps its digits near a pole.
 *
 * Longitude may be any finite value: from 2^24 up it is first reduced modulo 360 exactly. The belt
 * diagonals are clamped to `5 nside - 1`: just west of a 90 degree meridian at the belt edge, f32
 * rounds `0.5 + tt` up to 4.5 and `jp` or `jm` can reach `5 nside`, which without the clamp gave a
 * cell on the far side of face 4 (north) or the invalid face 12 (south). The clamp is a no-op in
 * exact arithmetic and leaves the key in one of the cells at that face corner.
 *
 * Accuracy: every product above is an f32, so `nside * tt` carries a relative error of about
 * `6e-8`; the cell index is exact except for points within about `nside * 6e-8 * 4` cell widths
 * of a cell edge, which is a fraction `4 nside 6e-8` of the points (see the contributor).
 *
 * @internal
 */
export const HEALPIX_INDEX_WGSL = /* wgsl */ `
${CELL_KEY_WGSL}
${QUADBIN_TILE_WGSL}
// asin(2/3) in degrees: the latitude where the equatorial belt ends.
const CELL_INDEX_HEALPIX_BELT_LATITUDE: f32 = 41.810314895778596;
${getExactDegreeReductionWGSL('cellIndexHealpixReduceDegrees')}

fn cellIndexHealpixNestedFromLngLat(longitude: f32, latitude: f32, order: u32) -> vec2u {
  let nside = f32(1u << order);
  let sideMask = (1u << order) - 1u;
  let reduced = cellIndexHealpixReduceDegrees(longitude);
  var wrapped = reduced - 360.0 * floor(reduced / 360.0);
  // Reciprocal division may round a quotient just below an integer upward. Repair the
  // negative residual using its sign bits; do not clamp a point west of the meridian to zero.
  if ((bitcast<u32>(wrapped) & 0x80000000u) != 0u) {
    wrapped = wrapped + 360.0;
  }
  if ((bitcast<u32>(wrapped) & 0x7fffffffu) >= 0x43b40000u) {
    wrapped = wrapped - 360.0;
  }
  // A rounded floor can leave the turn a hair outside [0, 4): clamp, then wrap 4 to 0.
  var tt = clamp(wrapped / 90.0, 0.0, 4.0);
  if (tt >= 4.0) {
    tt = 0.0;
  }
  let clamped = clamp(latitude, -90.0, 90.0);
  var face = 0u;
  var ix = 0u;
  var iy = 0u;
  if (abs(clamped) <= CELL_INDEX_HEALPIX_BELT_LATITUDE) {
    let z = sin(radians(clamped));
    let temp1 = nside * (0.5 + tt);
    let temp2 = nside * z * 0.75;
    // In exact arithmetic both diagonals stay below 5 nside; f32 rounding can reach it.
    let diagonalLimit = (5u << order) - 1u;
    let jp = min(u32(max(temp1 - temp2, 0.0)), diagonalLimit);
    let jm = min(u32(max(temp1 + temp2, 0.0)), diagonalLimit);
    let ifp = jp >> order;
    let ifm = jm >> order;
    if (ifp == ifm) {
      face = ifp | 4u;
    } else if (ifp < ifm) {
      face = ifp;
    } else {
      face = ifm + 8u;
    }
    ix = jm & sideMask;
    iy = sideMask - (jp & sideMask);
  } else {
    var ntt = u32(tt);
    ntt = min(ntt, 3u);
    let tp = tt - f32(ntt);
    let halfColatitude = sin(radians(90.0 - abs(clamped)) * 0.5);
    let tmp = nside * sqrt(6.0 * halfColatitude * halfColatitude);
    let jp = min(u32(tp * tmp), sideMask);
    let jm = min(u32((1.0 - tp) * tmp), sideMask);
    if (clamped >= 0.0) {
      face = ntt;
      ix = sideMask - jm;
      iy = sideMask - jp;
    } else {
      face = ntt + 8u;
      ix = jp;
      iy = jm;
    }
  }
  let path = quadbinGetCompactKey(ix, iy);
  return path | cellShiftLeft(vec2u(0u, face), 2u * order);
}

/** The nested index plus \`4^(order + 1)\`: the UNIQ scheme, never zero and self-describing. */
fn cellIndexHealpixUniq(nested: vec2u, order: u32) -> vec2u {
  let marker = cellShiftLeft(vec2u(0u, 1u), 2u * order + 2u);
  let low = nested.y + marker.y;
  let carry = select(0u, 1u, low < nested.y);
  return vec2u(nested.x + marker.x + carry, low);
}
`;
