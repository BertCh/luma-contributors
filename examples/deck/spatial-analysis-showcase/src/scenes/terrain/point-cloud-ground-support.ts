// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';

export const LIDAR_GROUND_GRID = {
  origin: [-110, -110] as const,
  cellSize: 1,
  width: 220,
  height: 220
};
// Larger windows have no fully supported acceptances on this sparse class-2 seed raster.
export const LIDAR_GROUND_RADII = [1] as const;

export const LIDAR_GROUND_DISPLAY_CATEGORIES = [
  {label: 'Rejected with complete support', color: [204, 109, 97]},
  {label: 'Accepted ground with complete support', color: [150, 211, 166]},
  {label: 'Provisional: incomplete morphology window', color: [233, 181, 101]},
  {label: 'Unsupported / invalid', color: [150, 155, 160]}
] as const;

export function getLidarGroundDisplayClass(mask: number, status: number): number {
  return status === 2 ? 2 : status === 0 && (mask === 0 || mask === 1) ? mask : 3;
}

/** The ground layer consumes these categories, never the unqualified acceptance mask. */
export function createGroundDisplayClassesNode(
  graph: GPUCommandGraph,
  mask: GraphDataView<'uint32'>,
  status: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'lidar-ground-display-classes',
    operation: 'GroundDisplayClasses',
    invocationCount: output.length,
    bindings: [
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'status', view: status, type: 'u32', access: 'read'},
      {name: 'display', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `let code = status[statusOffset + index];
let accepted = mask[maskOffset + index];
var category = select(3u, 2u, code == 2u);
if (code == 0u && accepted <= 1u) { category = accepted; }
display[displayOffset + index] = category;`
  });
}

/** Occupancy alone cannot seed terrain: Surfaces bit 2 identifies a ground return. */
export function createGroundSeedCoverageNode(
  graph: GPUCommandGraph,
  coverage: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'lidar-ground-seed-support',
    operation: 'GroundSeedSupport',
    invocationCount: output.length,
    bindings: [
      {name: 'coverage', view: coverage, type: 'u32', access: 'read'},
      {name: 'seed', view: output, type: 'u32', access: 'read_write'}
    ],
    body: 'seed[seedOffset + index] = select(0u, 1u, (coverage[coverageOffset + index] & 2u) != 0u);'
  });
}

/** Suppress terrain classification whenever morphology reports incomplete or absent support. */
export function createSupportedGroundClassesNode(
  graph: GPUCommandGraph,
  classifications: GraphDataView<'uint32'>,
  status: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
) {
  return createWGSLKernelNode(graph, {
    id: 'lidar-ground-supported-classes',
    operation: 'SupportedGroundClasses',
    invocationCount: output.length,
    bindings: [
      {name: 'classes', view: classifications, type: 'u32', access: 'read'},
      {name: 'status', view: status, type: 'u32', access: 'read'},
      {name: 'supported', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `let classification = classes[classesOffset + index];
supported[supportedOffset + index] = select(classification, 1u, classification == 2u && status[statusOffset + index] != 0u);`
  });
}
