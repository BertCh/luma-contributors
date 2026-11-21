// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../kernels/wgsl-common';

const CELL_BIAS = 0x40000000;
const SIGNIFICAND_BITS = 23;
const EXPONENT_BIAS = 127;
const MINIMUM_NORMAL = 2 ** -126;
const GPU_RING_DEFAULT_TOLERANCE_SCALE = 2 ** -21;

/** CPU mirror of the WGSL vertex quantizer used by `GPUSegmentRingAssembly`. */
export type GPURingVertexCell = {
  /** Ordered `u32` cell index. */
  cell: number;
  /** True for a non-finite coordinate. */
  overflow: boolean;
};

/**
 * Cells near zero have a power-of-two width strictly greater than the float32 tolerance, with
 * a minimum width of 2^-126 to accommodate flushed subnormal differences. Where the spacing
 * of float32 coordinates reaches the cell width, put at most two consecutive coordinates in
 * each cell. This keeps keys in the positive i32 range, including at float32 extremes.
 * Matching coordinates are still at most one key apart on each axis.
 */
export function quantizeRingVertexCoordinate(value: number, tolerance: number): GPURingVertexCell {
  const floats = Float32Array.of(value, tolerance);
  const bits = new Uint32Array(floats.buffer);
  const coordinate = floats[0];
  if (!Number.isFinite(coordinate)) return {cell: 0xffffffff, overflow: true};
  const magnitude = bits[0] & 0x7fffffff;
  const cellPower = ((bits[1] & 0x7fffffff) >>> SIGNIFICAND_BITS) + 1 - EXPONENT_BIAS;
  const cellSize = Math.max(MINIMUM_NORMAL, 2 ** cellPower);
  const sparseStart = cellSize * 2 ** SIGNIFICAND_BITS;
  if (Math.abs(coordinate) >= sparseStart) {
    const sparseBits = new Uint32Array(Float32Array.of(sparseStart).buffer)[0];
    const sparseRank = magnitude - sparseBits + (coordinate < 0 ? 1 : 0);
    const magnitudeCell = 2 ** SIGNIFICAND_BITS + (sparseRank >>> 1);
    return {cell: CELL_BIAS + Math.sign(coordinate) * magnitudeCell, overflow: false};
  }
  return {cell: CELL_BIAS + Math.floor(coordinate / cellSize), overflow: false};
}

/**
 * WGSL helpers shared by the hash, cancel and match kernels. The frame extent in `counters`
 * is needed only when the ring assembler derives the matching tolerance.
 */
export function getRingQuantizerWGSL(
  toleranceLiteral: string,
  atomicCounters: boolean,
  extentIndex: number
): string {
  const tolerance = Float32Array.of(Number(toleranceLiteral));
  const toleranceBits = new Uint32Array(tolerance.buffer)[0] & 0x7fffffff;
  const load = atomicCounters
    ? `atomicLoad(&counters[countersOffset + ${extentIndex}u])`
    : `counters[countersOffset + ${extentIndex}u]`;
  return `${IS_FINITE_VALUE_WGSL}
const TOLERANCE_BITS: u32 = ${toleranceBits}u;
const CELL_BIAS: u32 = ${CELL_BIAS}u;
// Kept for the ring assembly and contact-peeling callers' shared signature.
fn ringOrigin() -> vec2f { return vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]); }
fn ringFrameExtent() -> f32 { return bitcast<f32>(${load}); }
fn ringCellToleranceBits() -> u32 {
  if (TOLERANCE_BITS != 0u) { return TOLERANCE_BITS; }
  return bitcast<u32>(ringFrameExtent() * ${GPU_RING_DEFAULT_TOLERANCE_SCALE});
}
fn ringQuantize(value: f32, _origin: f32) -> u32 {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  let exponent = magnitude >> 23u;
  if (exponent == 255u) { return 0xffffffffu; }
  let negative = (bits >> 31u) != 0u;
  let cellExponent = (ringCellToleranceBits() >> 23u) + 1u;
  let sparseExponent = cellExponent + 23u;
  var magnitudeCell = 0u;
  if (exponent >= sparseExponent) {
    // Pack two consecutive coordinates per cell where the float32 step is wide.
    // The negative boundary cell also contains the preceding dense coordinate.
    let sparseRank = magnitude - (sparseExponent << 23u) + select(0u, 1u, negative);
    magnitudeCell = 0x00800000u + (sparseRank >> 1u);
  } else {
    let significand = (magnitude & 0x007fffffu) | select(0u, 0x00800000u, exponent != 0u);
    let shift = sparseExponent - max(exponent, 1u);
    if (shift < 24u) {
      magnitudeCell = significand >> shift;
      if (negative && (magnitudeCell << shift) != significand) { magnitudeCell++; }
    } else if (negative && significand != 0u) {
      magnitudeCell = 1u;
    }
  }
  return select(CELL_BIAS + magnitudeCell, CELL_BIAS - magnitudeCell, negative);
}
fn ringHash(qx: u32, qy: u32, group: u32) -> u32 {
  var h = qx * 0x9E3779B1u;
  h = h ^ (h >> 15u);
  h = (h + qy * 0x85EBCA77u) * 0xC2B2AE3Du;
  h = h ^ (h >> 13u);
  h = (h + group * 0x27D4EB2Fu) * 0x165667B1u;
  h = h ^ (h >> 16u);
  return select(h, 0xfffffffeu, h == 0xffffffffu);
}`;
}
