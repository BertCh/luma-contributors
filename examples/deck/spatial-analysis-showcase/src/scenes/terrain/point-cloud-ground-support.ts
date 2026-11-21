// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {
  GPUPointCloudCoverageValidity,
  decodeGPUPointCloudFeaturesStatus
} from '@luma-contributors/analysis/pointcloud-analysis';

export const LIDAR_GROUND_GRID = {
  origin: [-110, -110] as const,
  cellSize: 1,
  width: 220,
  height: 220
};
// Larger windows have no fully supported acceptances on this sparse class-2 seed raster.
export const LIDAR_GROUND_RADII = [1] as const;

export const LIDAR_FEATURE_DISPLAY_CATEGORIES = [
  {label: 'Usable fit; halo uncertified', color: [150, 211, 166]},
  {label: 'Usable fit; incomplete support', color: [233, 181, 101]},
  {label: 'Truncated neighbourhood', color: [229, 145, 83]},
  {label: 'Degenerate / insufficient fit', color: [150, 155, 160]},
  {label: 'Eigensolve did not converge', color: [204, 109, 97]}
] as const;

export function getLidarFilterSupportText(thresholdComplete: boolean): string {
  return thresholdComplete
    ? 'Threshold from retained nearest neighbours; capped support and the clipped sample halo remain uncertified'
    : 'Incomplete threshold; classifications suppressed (hidden)';
}

/** Five GPU-reduced counts: incomplete, usable incomplete, truncated, failed solve, degenerate. */
export function addLidarFeatureSummary(
  graph: GPUCommandGraph,
  status: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): void {
  const names = ['incomplete', 'usableIncomplete', 'truncated', 'failed', 'degenerate'] as const;
  const masks = names.map(name =>
    createTransientView(graph, `lidar-feature-${name}`, 'uint32', status.length)
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'lidar-feature-summary-flags',
      operation: 'LidarFeatureSummaryFlags',
      invocationCount: status.length,
      bindings: [
        {name: 'status', view: status, type: 'u32', access: 'read'},
        ...names.map((name, index) => ({
          name,
          view: masks[index],
          type: 'u32' as const,
          access: 'read_write' as const
        }))
      ],
      body: `let flags = status[statusOffset + index];
incomplete[incompleteOffset + index] = u32((flags & 8u) != 0u);
usableIncomplete[usableIncompleteOffset + index] = u32((flags & 13u) == 8u);
truncated[truncatedOffset + index] = u32((flags & 2u) != 0u);
failed[failedOffset + index] = u32((flags & 4u) != 0u);
degenerate[degenerateOffset + index] = u32((flags & 1u) != 0u);`
    })
  );
  for (const [index, name] of names.entries()) {
    graph.add(
      new GPUReduction({
        id: `lidar-feature-summary-${name}`,
        input: masks[index],
        output: graph.createDataView(output.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: output.byteOffset + index * 4
        }),
        operation: 'sum'
      })
    );
  }
}

/** Fit failure, truncation and incomplete halo support are separate causes. */
export function getLidarFeatureDisplayClass(flags: number): number {
  const status = decodeGPUPointCloudFeaturesStatus(Uint32Array.of(flags));
  return status.notConverged
    ? 4
    : status.degenerate
      ? 3
      : status.truncated
        ? 2
        : status.incompleteSupport
          ? 1
          : 0;
}

export function createFeatureDisplayClassesNode(
  graph: GPUCommandGraph,
  status: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>,
  id = 'lidar-feature-display-classes'
) {
  return createWGSLKernelNode(graph, {
    id,
    operation: 'LidarFeatureDisplayClasses',
    invocationCount: output.length,
    bindings: [
      {name: 'status', view: status, type: 'u32', access: 'read'},
      {name: 'display', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `let flags = status[statusOffset + index];
var category = select(0u, 1u, (flags & 8u) != 0u);
if ((flags & 2u) != 0u) { category = 2u; }
if ((flags & 1u) != 0u) { category = 3u; }
if ((flags & 4u) != 0u) { category = 4u; }
display[displayOffset + index] = category;`
  });
}

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
  return new GPUPointCloudCoverageValidity({
    id: 'lidar-ground-seed-support',
    coverage,
    output: {
      surfaceValidity: createTransientView(
        graph,
        'lidar-seed-surface-validity',
        'uint32',
        coverage.length
      ),
      terrainValidity: output,
      canopyValidity: createTransientView(
        graph,
        'lidar-seed-canopy-validity',
        'uint32',
        coverage.length
      )
    }
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
