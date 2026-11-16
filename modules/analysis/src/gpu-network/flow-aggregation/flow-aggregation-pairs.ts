// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Zone ID written for rejected or unused rows. */
export const GPU_FLOW_AGGREGATION_NO_ZONE = 0xffffffff;

/** Largest legacy packed-key zone count; larger counts use two-column lexicographic keys. */
export const GPU_FLOW_AGGREGATION_MAXIMUM_PACKED_ZONE_COUNT = 65535;

/** Largest zone count with uint32 IDs and room for the zone-sum offsets sentinel. */
export const GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT = 0xfffffffe;

/**
 * Legacy packed pair key (zoneCount <= 65535): `originZone * zoneCount + destinationZone`.
 *
 * Keys are always below `0xffffffff`, which the hash table reserves as its empty marker.
 */
export function getGPUFlowPairKey(
  originZone: number,
  destinationZone: number,
  zoneCount: number
): number {
  return originZone * zoneCount + destinationZone;
}

/** Inverse of {@link getGPUFlowPairKey}: `[originZone, destinationZone]`. */
export function getGPUFlowPairZones(pairKey: number, zoneCount: number): [number, number] {
  return [Math.floor(pairKey / zoneCount), pairKey % zoneCount];
}
