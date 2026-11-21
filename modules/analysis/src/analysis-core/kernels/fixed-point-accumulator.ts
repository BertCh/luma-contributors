// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from './gpu-contributor-utils';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';

/**
 * Shared deterministic fixed-point accumulator (algorithms roadmap decision 4).
 *
 * A sum is held as one 96-bit two's-complement integer per cell (three `u32` words, low word
 * first) in units of `2^unitExponent`. Integer addition is associative, so the sum is the same
 * for every add order, workgroup split, chunking and atomic interleaving: results are bitwise
 * reproducible across runs and dispatch orders.
 *
 * The unit is data-derived but quantized to a power of two, in the style of ReproBLAS binned
 * summation: `unitExponent = E + 1 + h - 95`, where `E` is the largest binary exponent of any term
 * (an order-independent maximum, reduced on the GPU into one scale word) and `h` is the bit length
 * of the maximum number of terms per cell. Then every term is below `2^(95 - h)` units and no cell
 * can overflow. A term keeps all 24 significand bits while its magnitude is at least
 * `max * 2^(h - 71)`, a dynamic range of `2^(71 - h)` (about `1e15` at a million terms), which is
 * why one outlier no longer zeroes typical terms (library rule 7, review B-01, which pegged a
 * 24-bit unit to the maximum). Smaller terms round to the nearest unit (ties to even); the
 * absolute error of a sum of `n` terms is at most `n / 2` units. The legacy shared scale
 * spans unrelated cells too: with two terms [1, 2^100], h=2 and unit=256, so the singleton
 * 1 rounds to zero. Opt into createFixedPointGroupedScaleNodes to isolate cells. Accumulation
 * must use scale[cell] with getFixedPointUnitExponent; decode with scaleMode: 'per-cell'.
 *
 * Quantizing and decoding use integer bit manipulation only (no f32 arithmetic before the final
 * `ldexp`), so the result does not depend on fused multiply-add or reassociation. Decoding rounds
 * the exact integer to the nearest f32 (ties to even); sums below `2^-126` may flush to zero and
 * sums above the f32 range decode to `±Infinity`.
 *
 * @internal
 */
export const FIXED_POINT_WORD_COUNT = 3;

/** Largest number of terms per cell a fixed-point sum supports (`h <= 32`). @internal */
export const FIXED_POINT_MAXIMUM_TERM_COUNT = 0xffffffff;

/**
 * Returns `h`, the headroom bits of a fixed-point sum of at most `maximumTermCount` terms per
 * cell: the bit length of `maximumTermCount` (at least 1).
 *
 * @internal
 */
export function getFixedPointHeadroomBits(maximumTermCount: number): number {
  validateFixedPointMaximumTermCount('fixed-point', 'maximumTermCount', maximumTermCount);
  return Math.max(1, maximumTermCount.toString(2).length);
}

/**
 * Select units from a finite, nonnegative host magnitude bound WITHOUT narrowing it to f32.
 * For a positive bound with binary exponent E, returns max(E, -126) + 1 + headroomBits - 95.
 * Squared f32 observation bounds can exceed f32's range while their three-word moments still
 * fit. Read the binary64 exponent directly: log2 can round values just below a power of two
 * up to that power. Zero and subnormal-f32 bounds retain the legacy minimum-normal-f32 exponent
 * (-126), required by quantizeFixedPoint's bounded shifts of unnormalized subnormal significands.
 * @internal
 */
export function getFixedPointMagnitudeUnitExponent(
  maximumMagnitude: number,
  maximumTermCount: number
): number {
  validateRequirement(
    'fixed-point',
    Number.isFinite(maximumMagnitude) && maximumMagnitude >= 0,
    `maximumMagnitude must be finite and nonnegative; received ${maximumMagnitude}`
  );
  const magnitudeView = new DataView(new ArrayBuffer(8));
  magnitudeView.setFloat64(0, maximumMagnitude);
  const highWord = magnitudeView.getUint32(0);
  const biasedExponent = (highWord >>> 20) & 0x7ff;
  const exponent = Math.max(biasedExponent - 1023, -126);
  return exponent + 1 + getFixedPointHeadroomBits(maximumTermCount) - 95;
}

/** Validate a caller's public bound before a packer writes any destination words. */
export function validateFixedPointMaximumTermCount(
  id: string,
  propPath: string,
  value: number
): void {
  validateCapacity(id, propPath, value, 0, FIXED_POINT_MAXIMUM_TERM_COUNT);
}

/**
 * WGSL helpers of the fixed-point accumulator. A value is a `vec3<u32>` (`x` low, `z` high word,
 * two's complement over 96 bits). Requires no bindings.
 *
 * - `getFixedPointExponentBits(value: f32) -> u32`: the biased exponent of a finite nonzero
 *   value (1 for subnormals), 0 for zero and non-finite values. Reduce it with `atomicMax` into
 *   the scale word.
 * - `getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32`.
 * - `quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32>`: round to the nearest unit,
 *   ties to even; non-finite values give 0.
 * - `addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32>` and `negateFixedPoint`.
 * - `getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32`: nearest f32, ties to even.
 *
 * @internal
 */
export const FIXED_POINT_WGSL = /* wgsl */ `
fn getFixedPointExponentBits(value: f32) -> u32 {
  let magnitudeBits = bitcast<u32>(value) & 0x7fffffffu;
  if (magnitudeBits == 0u || magnitudeBits >= 0x7f800000u) { return 0u; }
  return max(magnitudeBits >> 23u, 1u);
}
fn getFixedPointUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  // A term with biased exponent E is below 2^(E - 126) = 2^(E - 127 + 1); 95 - h bits remain.
  return i32(max(maximumExponentBits, 1u)) - 127 + 1 + i32(headroomBits) - 95;
}
fn addFixedPoint(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
  let low = a.x + b.x;
  let lowCarry = select(0u, 1u, low < a.x);
  let middlePartial = a.y + b.y;
  let middleCarry = select(0u, 1u, middlePartial < a.y);
  let middle = middlePartial + lowCarry;
  let middleRoundCarry = select(0u, 1u, middle < middlePartial);
  return vec3<u32>(low, middle, a.z + b.z + middleCarry + middleRoundCarry);
}
fn negateFixedPoint(value: vec3<u32>) -> vec3<u32> {
  return addFixedPoint(~value, vec3<u32>(1u, 0u, 0u));
}
fn shiftFixedPointLeft(value: u32, shift: u32) -> vec3<u32> {
  var words = array<u32, 4>(0u, 0u, 0u, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  words[word] = value << bit;
  // (value >> 1) >> (31 - bit) is value >> (32 - bit) without a shift by 32 when bit is 0.
  words[word + 1u] = (value >> 1u) >> (31u - bit);
  return vec3<u32>(words[0], words[1], words[2]);
}
fn shiftFixedPointRight(value: vec3<u32>, shift: u32) -> u32 {
  var words = array<u32, 4>(value.x, value.y, value.z, 0u);
  let word = shift / 32u;
  let bit = shift % 32u;
  return (words[word] >> bit) | ((words[word + 1u] << 1u) << (31u - bit));
}
fn getFixedPointBit(value: vec3<u32>, position: u32) -> u32 {
  var words = array<u32, 3>(value.x, value.y, value.z);
  return (words[position / 32u] >> (position % 32u)) & 1u;
}
fn hasFixedPointBitsBelow(value: vec3<u32>, count: u32) -> bool {
  var words = array<u32, 3>(value.x, value.y, value.z);
  let word = count / 32u;
  var found = (words[word] & ((1u << (count % 32u)) - 1u)) != 0u;
  if (word > 0u) { found = found || words[0] != 0u; }
  if (word > 1u) { found = found || words[1] != 0u; }
  return found;
}
fn quantizeFixedPoint(value: f32, unitExponent: i32) -> vec3<u32> {
  let bits = bitcast<u32>(value);
  let exponentBits = (bits >> 23u) & 0xffu;
  if (exponentBits == 0xffu) { return vec3<u32>(0u); }
  var significand = bits & 0x7fffffu;
  var exponent = -149;
  if (exponentBits != 0u) {
    significand = significand | 0x800000u;
    exponent = i32(exponentBits) - 150;
  }
  // value = significand * 2^exponent; the term in units is significand * 2^shift.
  let shift = exponent - unitExponent;
  var magnitude = vec3<u32>(0u);
  if (shift >= 0) {
    // The scale bounds shift by 71 - h; the clamp only keeps a misused scale memory-safe.
    magnitude = shiftFixedPointLeft(significand, u32(min(shift, 71)));
  } else if (shift >= -24) {
    let right = u32(-shift);
    var quotient = significand >> right;
    let remainder = significand & ((1u << right) - 1u);
    let half = 1u << (right - 1u);
    if (remainder > half || (remainder == half && (quotient & 1u) != 0u)) {
      quotient = quotient + 1u;
    }
    magnitude = vec3<u32>(quotient, 0u, 0u);
  }
  return select(magnitude, negateFixedPoint(magnitude), (bits >> 31u) != 0u);
}
fn getFixedPointFloat(value: vec3<u32>, unitExponent: i32) -> f32 {
  let negative = (value.z >> 31u) != 0u;
  let magnitude = select(value, negateFixedPoint(value), negative);
  var leading = 0u;
  if (magnitude.z != 0u) {
    leading = 64u + firstLeadingBit(magnitude.z);
  } else if (magnitude.y != 0u) {
    leading = 32u + firstLeadingBit(magnitude.y);
  } else if (magnitude.x != 0u) {
    leading = firstLeadingBit(magnitude.x);
  } else {
    return 0.0;
  }
  var significand = magnitude.x;
  var exponent = unitExponent;
  if (leading >= 24u) {
    let shift = leading - 23u;
    significand = shiftFixedPointRight(magnitude, shift);
    let roundBit = getFixedPointBit(magnitude, shift - 1u);
    if (roundBit != 0u && (hasFixedPointBitsBelow(magnitude, shift - 1u) || (significand & 1u) != 0u)) {
      significand = significand + 1u;
    }
    exponent = unitExponent + i32(shift);
  }
  // significand <= 2^24, so f32(significand) is exact and the value is significand * 2^exponent.
  var infinityBits = 0x7f800000u;
  var result = bitcast<f32>(infinityBits);
  if (exponent < 104 || (exponent == 104 && significand < 0x1000000u)) {
    // Two steps keep the ldexp exponent in range for normal results below 2^-103.
    let leadingExponent = i32(firstLeadingBit(significand));
    result = ldexp(ldexp(f32(significand), -leadingExponent), exponent + leadingExponent);
  }
  return select(result, -result, negative);
}`;

/**
 * Centered streaming second-moment update; include after {@link FIXED_POINT_WGSL}.
 *
 * `getFixedPointMomentIncrement(priorSum: vec3<u32>, priorCount: u32, value: f32,
 * sumUnitExponent: i32, momentUnitExponent: i32) -> vec3<u32>` returns the increment to
 * sum-of-squared-deviations, in units of `2^momentUnitExponent` (squared observation units).
 * `priorSum` is the sum BEFORE this observation in units of `2^sumUnitExponent`; `priorCount`
 * is its observation count, strictly below FIXED_POINT_MAXIMUM_TERM_COUNT. The first observation
 * (`priorCount == 0`) contributes exactly zero, without reading/decoding the prior sum.
 *
 * Both the prior sum and `priorCount * quantizeFixedPoint(value, sumUnitExponent)` must fit
 * signed 96 bits, as guaranteed by the ordinary fixed-point term/count bounds. Their difference
 * may need all 96 UNSIGNED bits. Subtraction happens before float conversion, preserving small
 * residuals on a large baseline. For n prior samples with sum S, the increment is
 * `(n * value - S)^2 / (n * (n + 1))`. Its integer residual is normalized before multiplication;
 * the removed power of two is applied to the quantizer's unit, so neither the absolute sum nor
 * the unscaled square needs to fit f32. Uses the existing nearest-unit, ties-to-even quantizer
 * and unchanged three-word state. This corrects cancellation; it does not promise order-independent
 * f32 rounding of general moment updates.
 *
 * @internal
 */
export const FIXED_POINT_MOMENTS_WGSL = /* wgsl */ `
fn getFixedPointMomentIncrement(
  priorSum: vec3<u32>, priorCount: u32, value: f32,
  sumUnitExponent: i32, momentUnitExponent: i32
) -> vec3<u32> {
  if (priorCount == 0u) { return vec3<u32>(0u); }
  let observation = quantizeFixedPoint(value, sumUnitExponent);
  let observationNegative = (observation.z >> 31u) != 0u;
  let sumNegative = (priorSum.z >> 31u) != 0u;
  var addend = select(observation, negateFixedPoint(observation), observationNegative);
  var multipliedObservation = vec3<u32>(0u);
  var remainingCount = priorCount;
  // Exactly multiply by the integer count, including counts not representable as f32.
  // A uint32 count has at most 32 bits; all partial sums stay within its final magnitude.
  for (var bit = 0u; bit < 32u; bit++) {
    if ((remainingCount & 1u) != 0u) {
      multipliedObservation = addFixedPoint(multipliedObservation, addend);
    }
    remainingCount = remainingCount >> 1u;
    if (remainingCount == 0u) { break; }
    addend = vec3<u32>(addend.x << 1u, (addend.y << 1u) | (addend.x >> 31u),
      (addend.z << 1u) | (addend.y >> 31u));
  }
  let sumMagnitude = select(priorSum, negateFixedPoint(priorSum), sumNegative);
  var residual = addFixedPoint(multipliedObservation, sumMagnitude);
  if (observationNegative == sumNegative) {
    residual = addFixedPoint(multipliedObservation, negateFixedPoint(sumMagnitude));
    residual = select(residual, negateFixedPoint(residual), (residual.z >> 31u) != 0u);
  }
  var leading = 0u;
  if (residual.z != 0u) { leading = 64u + firstLeadingBit(residual.z); }
  else if (residual.y != 0u) { leading = 32u + firstLeadingBit(residual.y); }
  else if (residual.x != 0u) { leading = firstLeadingBit(residual.x); }
  else { return vec3<u32>(0u); }

  let residualExponent = sumUnitExponent + i32(leading);
  var normalized = 0.0;
  if (leading == 95u) {
    // The unsigned residual can exceed signed-96 range. Shift it one bit with a sticky low
    // bit, preserving round-to-nearest-even when getFixedPointFloat retains its top 24 bits.
    let halfResidual = vec3<u32>((residual.x >> 1u) | (residual.y << 31u) | (residual.x & 1u),
      (residual.y >> 1u) | (residual.z << 31u), residual.z >> 1u);
    normalized = getFixedPointFloat(halfResidual, 1 - i32(leading));
  } else {
    normalized = getFixedPointFloat(residual, -i32(leading));
  }
  let scaledIncrement = (normalized / f32(priorCount)) * (normalized / f32(priorCount + 1u));
  return quantizeFixedPoint(scaledIncrement, momentUnitExponent - 2 * residualExponent);
}`;

/**
 * Returns WGSL `fn addFixedPointTo<Name>(cell: u32, value: vec3<u32>)` that adds a fixed-point
 * value to cell `cell` of the `atomic<u32>` storage binding `bindingName` (three words per cell,
 * the binding's `${bindingName}Offset` applied). Each word takes one `atomicAdd`, skipped when
 * its addend is 0; the carry out of a word is detected from the returned old value. The final
 * words are the exact 96-bit sum whatever the interleaving (intermediate states are not).
 *
 * @internal
 */
export function getFixedPointAtomicAddWGSL(bindingName: string): string {
  const name = `addFixedPointTo${bindingName[0].toUpperCase()}${bindingName.slice(1)}`;
  return /* wgsl */ `
fn ${name}(cell: u32, value: vec3<u32>) {
  let base = ${bindingName}Offset + cell * ${FIXED_POINT_WORD_COUNT}u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&${bindingName}[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&${bindingName}[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&${bindingName}[base + 2u], highAddend);
  }
}`;
}

/**
 * Builds the scale word of a fixed-point sum: one clear node, then per chunk of `values` one
 * workgroup max of {@link FIXED_POINT_WGSL}'s `getFixedPointExponentBits` and one `atomicMax`
 * per workgroup (library rule 14). Non-finite values do not raise the scale. Consumers whose
 * terms are computed in-kernel write the same word with their own `atomicMax` of the exponent
 * bits of their terms.
 *
 * @internal
 */
export function createFixedPointScaleNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    /** Terms whose largest magnitude sets the unit. */
    values: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    /** One `uint32` word: the largest biased exponent, 0 when every term is zero or non-finite. */
    scale: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  validateSinglePackedView(props.scale, ['uint32'], `${props.id} scale`);
  for (const chunk of getGraphViewChunks(props.values)) {
    validateSinglePackedView(chunk, ['float32'], `${props.id} values chunk`);
  }
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-clear`,
      operation: props.operation,
      variant: 'fixed-point-scale-clear',
      bindings: [{name: 'fixedPointScale', view: props.scale, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      body: 'fixedPointScale[fixedPointScaleOffset] = 0u;'
    })
  ];
  const chunks = getGraphViewChunks(props.values);
  for (const [chunkIndex, chunk] of chunks.entries()) {
    if (chunk.length === 0) {
      continue;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-${chunkIndex}`,
        operation: props.operation,
        variant: 'fixed-point-scale',
        bindings: [
          {name: 'fixedPointValues', view: chunk, type: 'f32', access: 'read'},
          {
            name: 'fixedPointScale',
            view: props.scale,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: chunk.length,
        guardIndex: false,
        declarations: `${FIXED_POINT_WGSL}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&fixedPointScale[fixedPointScaleOffset], workgroupMaximum);
    }
  }`
      })
    );
  }
  return nodes;
}

/** Sets every word of a fixed-point sum array to zero. @internal */
export function createFixedPointClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; words: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  validateSinglePackedView(props.words, ['uint32'], `${props.id} words`);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'fixed-point-clear',
    bindings: [{name: 'fixedPointWords', view: props.words, type: 'u32', access: 'read_write'}],
    invocationCount: props.words.length,
    // A fixed size keeps one module for every cell count (rule 12).
    workgroupSize: 64,
    body: 'fixedPointWords[fixedPointWordsOffset + index] = 0u;'
  });
}

/**
 * Decodes `output.length` fixed-point cells (three words each) into the nearest f32 values.
 * `maximumTermCount` must be the one the accumulating kernels used for their unit.
 *
 * @internal
 */
export function createFixedPointDecodeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    words: GraphDataView<'uint32'>;
    scale: GraphDataView<'uint32'>;
    maximumTermCount: number;
    output: GraphDataView<'float32'>;
    /** One scale per output cell instead of the legacy shared scale. */
    scaleMode?: 'shared' | 'per-cell';
  }
): GPUCommandNode<Parameters> {
  validateSinglePackedView(props.scale, ['uint32'], `${props.id} scale`);
  validateSinglePackedView(props.words, ['uint32'], `${props.id} words`);
  validateSinglePackedView(props.output, ['float32'], `${props.id} output`);
  validateRequirement(
    props.id,
    props.scaleMode === undefined || props.scaleMode === 'shared' || props.scaleMode === 'per-cell',
    `scaleMode must be shared or per-cell; received ${props.scaleMode}`
  );
  validateRequirement(
    props.id,
    props.scale.length >= (props.scaleMode === 'per-cell' ? props.output.length : 1),
    'scale must hold one word per selected scale'
  );
  if (props.words.length < props.output.length * FIXED_POINT_WORD_COUNT) {
    throw new Error(`${props.operation}: fixed-point words must hold three words per output`);
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'fixed-point-decode',
    bindings: [
      {name: 'fixedPointWords', view: props.words, type: 'u32', access: 'read'},
      {name: 'fixedPointScale', view: props.scale, type: 'u32', access: 'read'},
      {name: 'fixedPointOutput', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.output.length,
    workgroupSize: 64,
    declarations: `override FIXED_POINT_HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}`,
    constants: {FIXED_POINT_HEADROOM_BITS: getFixedPointHeadroomBits(props.maximumTermCount)},
    body: `let base = fixedPointWordsOffset + index * ${FIXED_POINT_WORD_COUNT}u;
  let unitExponent = getFixedPointUnitExponent(
    fixedPointScale[fixedPointScaleOffset${props.scaleMode === 'per-cell' ? ' + index' : ''}],
    FIXED_POINT_HEADROOM_BITS
  );
  fixedPointOutput[fixedPointOutputOffset + index] = getFixedPointFloat(
    vec3<u32>(fixedPointWords[base], fixedPointWords[base + 1u], fixedPointWords[base + 2u]),
    unitExponent
  );`
  });
}

/**
 * Opt-in independent scale selection for groups/bins. `cellRows` parallels `values`, addressing
 * the scale/output cell (not caller IDs). Each resident row must be less than scale.length;
 * callers that produce indices must establish that invariant. All chunks and scale must be packed.
 * Scale is cleared on every run; zero terms and zero cells return no nodes.
 * Accumulating kernels MUST quantize each term using scale[cellRows[row]], and publication must
 * use createFixedPointDecodeNode({scaleMode: 'per-cell'}). Shared-scale callers are unchanged.
 * @internal
 */
export function createFixedPointGroupedScaleNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    cellRows: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
    scale: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const values = getGraphViewChunks(props.values);
  const cells = getGraphViewChunks(props.cellRows);
  validateSinglePackedView(props.scale, ['uint32'], `${props.id} scale`);
  for (const chunk of values)
    validateSinglePackedView(chunk, ['float32'], `${props.id} values chunk`);
  for (const chunk of cells)
    validateSinglePackedView(chunk, ['uint32'], `${props.id} cellRows chunk`);
  validateRequirement(
    props.id,
    props.scale.length > 0 || props.values.length === 0,
    'scale.length must be positive when values is nonempty'
  );
  validateRequirement(
    props.id,
    values.length === cells.length &&
      values.every((chunk, index) => chunk.length === cells[index].length),
    'cellRows must have the same chunk topology as values'
  );
  if (props.scale.length === 0) return [];
  const nodes = [
    createFixedPointClearNode(graph, {
      id: `${props.id}-clear`,
      operation: props.operation,
      words: props.scale
    })
  ];
  for (const [chunkIndex, chunk] of values.entries()) {
    if (chunk.length === 0) continue;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${props.id}-${chunkIndex}`,
        operation: props.operation,
        variant: 'fixed-point-grouped-scale',
        invocationCount: chunk.length,
        workgroupSize: 64,
        bindings: [
          {name: 'fixedPointValues', view: chunk, type: 'f32', access: 'read'},
          {name: 'fixedPointCells', view: cells[chunkIndex], type: 'u32', access: 'read'},
          {name: 'fixedPointScale', view: props.scale, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: FIXED_POINT_WGSL,
        body: `let cell = fixedPointCells[fixedPointCellsOffset + index];
  let exponentBits = getFixedPointExponentBits(fixedPointValues[fixedPointValuesOffset + index]);
  atomicMax(&fixedPointScale[fixedPointScaleOffset + cell], exponentBits);`
      })
    );
  }
  return nodes;
}
