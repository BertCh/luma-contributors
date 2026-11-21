// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Bins of one 8-bit digit of the neighbor radix select and sort (a 1 KiB private histogram). */
const NEIGHBOR_RADIX_BINS = 256;

/** Population under the boundary circle, accumulated during the existing neighbor search. @internal */
export type ScanBoundaryPopulation = {
  view: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  integer: boolean;
  buckets: number;
  /** Append ten exact mass words after the existing rounded-mass/count tails. */
  exact?: boolean;
};

/** @internal Exact eligibility threshold; the accumulator must also be declared. */
export const SCAN_EXACT_POPULATION_LIMIT_WGSL = `
var<private> populationLimitWords: array<u32, 10>;
fn preparePopulationLimit(fraction: f32) {
  let bits = bitcast<u32>(fraction);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  let shift = select(149u, 150u - exponent, exponent != 0u);
  // Base-256 multiplication keeps every intermediate below 65536, without overflowing
  // scalar arithmetic or depending on signed right-shift behavior in host shader interpreters.
  var product: array<u32, 44>;
  for (var byte = 0u; byte < 40u; byte++) {
    let inputByte = (totals[totalsOffset + 7u + byte / 4u] / (1u << ((byte % 4u) * 8u))) & 255u;
    var carry = 0u;
    for (var digit = 0u; digit < 3u; digit++) {
      let fractionByte = (significand / (1u << (digit * 8u))) & 255u;
      let combined = product[byte + digit] + inputByte * fractionByte + carry;
      product[byte + digit] = combined & 255u;
      carry = combined / 256u;
    }
    product[byte + 3u] = carry;
  }
  let firstByte = shift / 8u;
  let offset = shift % 8u;
  for (var word = 0u; word < 10u; word++) {
    var value = 0u;
    for (var digit = 0u; digit < 4u; digit++) {
      let source = firstByte + word * 4u + digit;
      var byteValue = 0u;
      if (source < 44u) { byteValue = product[source] / (1u << offset); }
      if (offset != 0u && source + 1u < 44u) { byteValue = byteValue | ((product[source + 1u] << (8u - offset)) & 255u); }
      value = value | (byteValue << (digit * 8u));
    }
    populationLimitWords[word] = value;
  }
}
fn isBoundaryPopulationEligible() -> bool {
  for (var reverse = 0u; reverse < 10u; reverse++) {
    let word = 9u - reverse;
    if (boundaryPopulationWords[word] != populationLimitWords[word]) {
      return boundaryPopulationWords[word] < populationLimitWords[word];
    }
  }
  return true;
}`;

/** @internal */
export const SCAN_EXACT_POPULATION_ACCUMULATOR_WGSL = `
// Exact positive-f32 accumulation in units of 2^-149. Ten words cover a sum of
// up to 2^24 finite f32 inputs; round once, so traversal order cannot alter admission.
var<private> boundaryPopulationWords: array<u32, 10>;
fn resetBoundaryPopulation() {
  for (var word = 0u; word < 10u; word++) { boundaryPopulationWords[word] = 0u; }
}
fn addBoundaryPopulationWord(word: u32, value: u32) {
  var destination = word;
  var carry = value;
  while (destination < 10u && carry != 0u) {
    let previous = boundaryPopulationWords[destination];
    let next = previous + carry;
    boundaryPopulationWords[destination] = next;
    carry = select(0u, 1u, next < previous);
    destination++;
  }
}
fn addBoundaryPopulationBits(bits: u32) {
    let exponent = (bits >> 23u) & 255u;
    if ((bits & 0x80000000u) != 0u || exponent == 255u) { return; }
    let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
    let shift = select(0u, exponent - 1u, exponent != 0u);
    let word = shift / 32u;
    let offset = shift % 32u;
    addBoundaryPopulationWord(word, significand << offset);
    if (offset != 0u) { addBoundaryPopulationWord(word + 1u, significand >> (32u - offset)); }
}
fn readBoundaryPopulation() -> f32 {
  var highestWord = 9u;
  loop {
    if (boundaryPopulationWords[highestWord] != 0u) { break; }
    if (highestWord == 0u) { return 0.0; }
    highestWord--;
  }
  var highestBit = 31u;
  while ((boundaryPopulationWords[highestWord] >> highestBit) == 0u) { highestBit--; }
  let highest = highestWord * 32u + highestBit;
  if (highest < 23u) { return bitcast<f32>(boundaryPopulationWords[0]); }
  let shift = highest - 23u;
  let word = shift / 32u;
  let offset = shift % 32u;
  // offset is in [0,31], so this unsigned power-of-two divisor is never zero.
  var significand = boundaryPopulationWords[word] / (1u << offset);
  if (offset != 0u && word < 9u) { significand = significand | (boundaryPopulationWords[word + 1u] << (32u - offset)); }
  if (shift != 0u) {
    let roundWord = (shift - 1u) / 32u;
    let roundOffset = (shift - 1u) % 32u;
    let roundBit = (boundaryPopulationWords[roundWord] >> roundOffset) & 1u;
    var sticky = false;
    for (var lower = 0u; lower < roundWord; lower++) { sticky = sticky || boundaryPopulationWords[lower] != 0u; }
    if (roundOffset != 0u) { sticky = sticky || (boundaryPopulationWords[roundWord] & ((1u << roundOffset) - 1u)) != 0u; }
    if (roundBit != 0u && (sticky || (significand & 1u) != 0u)) { significand++; }
  }
  var exponent = highest - 22u;
  if (significand == 0x1000000u) { significand = significand >> 1u; exponent++; }
  if (exponent >= 255u) { var infinityBits = 0x7f800000u; return bitcast<f32>(infinityBits); }
  return bitcast<f32>((exponent << 23u) | (significand & 0x7fffffu));
}`;

/** @internal */
export function getScanBoundaryPopulationWGSL(
  population: ScanBoundaryPopulation | undefined
): string {
  if (!population) return '';
  const type = population.integer ? 'u32' : 'f32';
  return `fn getZonePopulation(zone: u32) -> ${type} {
  var total = ${population.integer ? '0u' : '0.0'};
  for (var bucket = 0u; bucket < ${population.buckets}u; bucket++) {
    total += population[populationOffset + zone * ${population.buckets}u + bucket];
  }
  return total;
}${
    population.integer
      ? ''
      : `
${SCAN_EXACT_POPULATION_ACCUMULATOR_WGSL}
fn addZoneBoundaryPopulation(zone: u32) {
  for (var bucket = 0u; bucket < ${population.buckets}u; bucket++) {
    addBoundaryPopulationBits(bitcast<u32>(population[populationOffset + zone * ${population.buckets}u + bucket]));
  }
}`
  }`;
}

/**
 * The `nearest` kernel shared by the scan statistics: per center, the `listLength` nearest zones
 * in (separation key, zone index) order into `neighbors` and the separation keys into `distances`.
 * With boundaryPopulation, distances also holds a population tail followed by a complete-circle
 * row-count tail, each zones words. Both include every tie at the final stored distance.
 * The exact flag additionally appends ten unrounded floating-mass words per center.
 *
 * @internal
 */
export function createScanNearestZonesKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    neighbors: GraphDataView<'uint32'>;
    distances: GraphDataView<'float32'>;
    sortKeys: GraphDataView<'uint32'>;
    sortZones: GraphDataView<'uint32'>;
    zones: number;
    listLength: number;
    separationWGSL: string;
    boundaryPopulation?: ScanBoundaryPopulation;
  }
): GPUCommandNode<Parameters> {
  const {positions, neighbors, distances, sortKeys, sortZones, zones, listLength, separationWGSL} =
    props;
  const population = props.boundaryPopulation;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'nearest',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
      {name: 'distances', view: distances, type: 'u32', access: 'read_write'},
      {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
      {name: 'sortZones', view: sortZones, type: 'u32', access: 'read_write'},
      ...(population
        ? [
            {
              name: 'population',
              view: population.view,
              type: population.integer ? ('u32' as const) : ('f32' as const),
              access: 'read' as const
            }
          ]
        : [])
    ],
    invocationCount: zones,
    declarations: `${separationWGSL}
${getScanBoundaryPopulationWGSL(population)}
const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;
const RADIX_BINS: u32 = ${NEIGHBOR_RADIX_BINS}u;
// Separation keys are non-negative f32, so their bits order as u32 keys.
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}`,
    // The row of a center lists its LIST nearest zones in (separation key, zone index) order.
    // When LIST < ZONES a radix select over the key bits (8-bit digits, most significant
    // first, one pass over the zones per digit) finds the key of the LIST-th nearest zone and
    // how many zones at that key still fit. One more pass gathers the selected zones in zone
    // order, and an LSD radix sort (four stable 8-bit passes, ping-ponging with the sort
    // scratch rows) orders them by key, so equal keys keep ascending zone order. Work per
    // center is O(ZONES + LIST), not the O(ZONES * LIST) of a selection by repeated scans.
    body: `var histogram: array<u32, RADIX_BINS>;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let row = index * LIST;
  var threshold = 0xffffffffu;
  var equalCount = LIST;
  if (LIST < ZONES) {
    var prefix = 0u;
    var prefixMask = 0u;
    var rank = LIST - 1u;
    for (var level = 0u; level < 4u; level++) {
      let shift = 24u - 8u * level;
      for (var bin = 0u; bin < RADIX_BINS; bin++) {
        histogram[bin] = 0u;
      }
      for (var zone = 0u; zone < ZONES; zone++) {
        let key = getZoneSeparationKey(zone, centerX, centerY);
        if ((key & prefixMask) == prefix) {
          let bin = (key >> shift) & 255u;
          histogram[bin] = histogram[bin] + 1u;
        }
      }
      var digit = 0u;
      loop {
        if (digit == RADIX_BINS - 1u || rank < histogram[digit]) {
          break;
        }
        rank -= histogram[digit];
        digit++;
      }
      prefix = prefix | (digit << shift);
      prefixMask = prefixMask | (255u << shift);
    }
    threshold = prefix;
    equalCount = rank + 1u;
  }
  var filled = 0u;
  var equalTaken = 0u;
  ${
    population
      ? `${population.integer ? 'var boundaryPopulation = 0u;' : 'resetBoundaryPopulation();'}
  var boundaryCount = 0u;`
      : ''
  }
  for (var zone = 0u; zone < ZONES; zone++) {
    ${
      population
        ? ''
        : `if (filled >= LIST) {
      break;
    }`
    }
    let key = getZoneSeparationKey(zone, centerX, centerY);
    ${
      population
        ? `if (key <= threshold) { ${population.integer ? 'boundaryPopulation += getZonePopulation(zone);' : 'addZoneBoundaryPopulation(zone);'} boundaryCount++; }
    if (filled >= LIST) { continue; }`
        : ''
    }
    var take = key < threshold;
    if (!take && key == threshold && equalTaken < equalCount) {
      take = true;
      equalTaken++;
    }
    if (take) {
      distances[distancesOffset + row + filled] = key;
      neighbors[neighborsOffset + row + filled] = zone;
      filled++;
    }
  }
  ${
    population
      ? `distances[distancesOffset + ZONES * LIST + index] = ${population.integer ? 'boundaryPopulation' : 'bitcast<u32>(readBoundaryPopulation())'};
  distances[distancesOffset + ZONES * (LIST + 1u) + index] = boundaryCount;
  ${
    population.exact && !population.integer
      ? `for (var word = 0u; word < 10u; word++) {
    distances[distancesOffset + ZONES * (LIST + 2u) + index * 10u + word] = boundaryPopulationWords[word];
  }`
      : ''
  }`
      : ''
  }
  for (var shift = 0u; shift < 32u; shift += 16u) {
    // Digit shift: the table rows to the sort scratch.
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (distances[distancesOffset + row + slot] >> shift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    var running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = distances[distancesOffset + row + slot];
      let bin = (key >> shift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      sortKeys[sortKeysOffset + row + destination] = key;
      sortZones[sortZonesOffset + row + destination] = neighbors[neighborsOffset + row + slot];
    }
    // Digit shift + 8: the sort scratch back to the table rows.
    let nextShift = shift + 8u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (sortKeys[sortKeysOffset + row + slot] >> nextShift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = sortKeys[sortKeysOffset + row + slot];
      let bin = (key >> nextShift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      distances[distancesOffset + row + destination] = key;
      neighbors[neighborsOffset + row + destination] = sortZones[sortZonesOffset + row + slot];
    }
  }`
  });
}
