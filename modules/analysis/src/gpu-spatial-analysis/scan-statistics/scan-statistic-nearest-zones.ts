// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Bins of one 8-bit digit of the neighbor radix select and sort (a 1 KiB private histogram). */
const NEIGHBOR_RADIX_BINS = 256;

/**
 * The `nearest` kernel shared by the scan statistics: per center, the `listLength` nearest zones
 * in (separation key, zone index) order into `neighbors` and the separation keys into `distances`.
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
  }
): GPUCommandNode<Parameters> {
  const {positions, neighbors, distances, sortKeys, sortZones, zones, listLength, separationWGSL} =
    props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'nearest',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
      {name: 'distances', view: distances, type: 'u32', access: 'read_write'},
      {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
      {name: 'sortZones', view: sortZones, type: 'u32', access: 'read_write'}
    ],
    invocationCount: zones,
    declarations: `${separationWGSL}
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
  for (var zone = 0u; zone < ZONES; zone++) {
    if (filled >= LIST) {
      break;
    }
    let key = getZoneSeparationKey(zone, centerX, centerY);
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
