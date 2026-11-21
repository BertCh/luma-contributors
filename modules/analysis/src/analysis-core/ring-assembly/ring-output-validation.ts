// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUSegmentRingAssemblyOutput} from './gpu-segment-ring-assembly';
import {validateDisjointWritableViews} from '../kernels/graph-chunk-dispatch';
import {validateGraphOutputsDisjointFromInputs} from '../kernels/gpu-contributor-utils';

/** Construction-time caller-output access plan; private scratch never aliases these views. */
export function validateRingOutputDispatches(
  id: string,
  output: GPUSegmentRingAssemblyOutput,
  pairedOpposites: boolean
): void {
  validateDisjointWritableViews(id, Object.fromEntries(getOutputViewEntries(output)), {
    separateDispatches: true
  });
  const validateDispatch = (
    inputs: readonly (GraphDataView | undefined)[],
    writable: Readonly<Record<string, GraphDataView | undefined>>
  ) => {
    validateDisjointWritableViews(id, writable);
    // A disjoint byte range is still a conflicting read-only/write usage of one buffer.
    validateGraphOutputsDisjointFromInputs(id, Object.values(writable), inputs);
  };
  validateDispatch([], {
    segmentRings: output.segmentRings,
    segmentVertices: output.segmentVertices,
    segmentFlags: pairedOpposites ? output.segmentFlags : undefined
  });
  validateDispatch([output.segmentVertices], {positions: output.positions});
  validateDispatch([output.positions, output.ringOffsets], {
    ringAreas: output.ringAreas,
    ringIsHole: output.ringIsHole
  });
  validateDispatch([output.positions, output.ringOffsets], {ringShells: output.ringShells});
  if (output.polygons) {
    validateDispatch([output.ringGroups], {sourceIds: output.polygons.sourceIds});
    validateDispatch([], {
      'polygons.ringOffsets': output.polygons.ringOffsets,
      'polygons.polygonOffsets': output.polygons.polygonOffsets,
      'polygons.featureOffsets': output.polygons.featureOffsets
    });
    validateDispatch([output.positions, output.ringOffsets, output.polygons.ringOffsets], {
      'polygons.positions': output.polygons.positions
    });
  }
  validateDispatch([], {
    'status.count': output.status.count,
    'status.overflow': output.status.overflow,
    'status.requiredCount': output.status.requiredCount
  });
  validateDispatch([output.segmentFlags], {
    'status.candidateOverflow': output.status.candidateOverflow
  });
  validateDispatch([output.status.candidateOverflow, output.status.overflow], {
    'status.complete': output.status.complete
  });
  validateDispatch([output.polygons?.featureOffsets], {
    'status.polygonCount': output.status.polygonCount
  });
  validateDispatch([], {
    'status.openSegmentCount': output.status.openSegmentCount,
    'status.touchingSegmentCount': output.status.touchingSegmentCount
  });
}

function getOutputViewEntries(output: GPUSegmentRingAssemblyOutput) {
  const {kind: _kind, ...polygonViews} = output.polygons ?? {};
  const {status, polygons: _polygons, ...views} = output;
  return [
    ...Object.entries(views).map(([name, view]) => [`output.${name}`, view] as const),
    ...Object.entries(polygonViews).map(
      ([name, view]) => [`output.polygons.${name}`, view] as const
    ),
    ...Object.entries(status).map(([name, view]) => [`output.status.${name}`, view] as const)
  ];
}

export function getOutputViews(output: GPUSegmentRingAssemblyOutput) {
  return getOutputViewEntries(output).map(([, view]) => view);
}
