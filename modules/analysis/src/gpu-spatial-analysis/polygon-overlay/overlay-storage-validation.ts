// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPolygonOverlayOutput} from './gpu-polygon-overlay';
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getNextPowerOfTwo,
  getGraphOutputViews,
  validateCapacity,
  validateRequirement,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Captures known status views once, including inherited and hidden refused fields. */
export function snapshotOverlayStatusFields(
  id: string,
  output: GPUPolygonOverlayOutput
): {
  status: GPUPolygonOverlayOutput['status'];
  boundaryStatus: GPUPolygonOverlayOutput['boundary']['status'];
} {
  const refused = [
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation',
    'uncertainCount',
    'requiredCount',
    'ringCount',
    'requiredRingCount',
    'nodedSegmentRequiredCount'
  ];
  const snapshots: Record<string, unknown>[] = [];
  for (const [path, status, supported] of [
    [
      'output.status',
      output.status,
      [
        'count',
        'overflow',
        'candidateOverflow',
        'ringCount',
        'requiredRingCount',
        'uncertainCount',
        'nodedSegmentRequiredCount'
      ]
    ],
    [
      'output.boundary.status',
      output.boundary.status,
      ['count', 'requiredCount', 'overflow', 'candidateOverflow']
    ]
  ] as const) {
    const fields = new Set([...supported, ...refused, ...Object.getOwnPropertyNames(status)]);
    const captured: Record<string, unknown> = {};
    for (const name of fields) {
      if (!(name in status)) continue;
      const view = (status as Record<string, unknown>)[name];
      validateRequirement(
        id,
        view === undefined || (supported as readonly string[]).includes(name),
        `${path}.${name} is unsupported because this pipeline does not publish it; omit the view and use the documented count and overflow diagnostics`
      );
      if ((supported as readonly string[]).includes(name)) captured[name] = view;
    }
    snapshots.push(Object.freeze(captured));
  }
  return {
    status: snapshots[0] as GPUPolygonOverlayOutput['status'],
    boundaryStatus: snapshots[1] as GPUPolygonOverlayOutput['boundary']['status']
  };
}

/** Preflights the overlay and its noder, grid, scans, two BVHs and ring assembly without mutation. */
export function validateOverlayScratchByteLengths<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: OverlayScratchDimensions
): void {
  validateOverlayScratchByteLengthsForDevice(graph.device, id, props);
}

type OverlayScratchDimensions = {
  expandedVertexCount: number;
  inputRingCount: number;
  intersections: number;
  nodedSegments: number;
  boundaryCapacity: number;
  outputRingCapacity: number;
  outputVertexCapacity: number;
  leafCapacity?: number;
};

/** Checks each derived scratch binding without requiring a graph. @internal */
export function validateOverlayScratchByteLengthsForDevice(
  device: Device,
  id: string,
  props: OverlayScratchDimensions
): void {
  const hotCapacity = props.expandedVertexCount + props.intersections;
  const incidenceCapacity = Math.max(
    props.nodedSegments + props.expandedVertexCount,
    2 * hotCapacity
  );
  const ringLeafCapacity = getNextPowerOfTwo(props.outputRingCapacity);
  const groupLeafCapacity = Math.max(2, getNextPowerOfTwo(props.nodedSegments));
  for (const [name, bytes] of Object.entries({
    combinedPositions: props.expandedVertexCount * 8,
    combinedOffsets: (props.inputRingCount + 1) * 4,
    hotPixelSegmentTables: hotCapacity * 128,
    crossingTerms: props.intersections * 8 * 4,
    candidateListsAndScan: incidenceCapacity * 2 * 4,
    nodedLatticeAndGroups: props.nodedSegments * 16,
    groupHierarchy: (2 * groupLeafCapacity - 1) * 8,
    latticeBoundary: props.boundaryCapacity * 16,
    ringPositions: props.outputVertexCapacity * 8,
    ringBounds: props.outputRingCapacity * 16,
    ringShellRecords: props.outputRingCapacity * 3 * 4,
    ringOffsets: (props.outputRingCapacity + 1) * 4,
    ringShellHierarchy: (2 * ringLeafCapacity - 1) * 8,
    boundsPartials:
      Math.min(
        256,
        Math.max(
          1,
          Math.ceil(Math.max(4 * hotCapacity, props.nodedSegments, props.outputRingCapacity) / 4096)
        )
      ) * 16,
    ...(props.leafCapacity ? {requestedHierarchy: (2 * props.leafCapacity - 1) * 8} : {})
  })) {
    const byteLength = Math.max(4, bytes);
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      byteLength,
      'reduce capacities or split input features'
    );
    validateCapacity(id, `${name} byteLength`, byteLength, 4, device.limits.maxBufferSize);
  }
}

/** Every bound view of the nested polygon/boundary/provenance output, without flattening data. */
export function getOverlayOutputViews(output: GPUPolygonOverlayOutput) {
  return [
    ...getGraphOutputViews(output.geometry),
    ...getGraphOutputViews(output.boundary),
    ...getGraphOutputViews(output.status),
    ...getGraphOutputViews(output.featureSources ?? {})
  ];
}
