// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';

export function getPointCloudFeatureClass(
  linearity: number,
  planarity: number,
  scattering: number,
  status: number | undefined
): number {
  if (
    status === undefined ||
    (status & 5) !== 0 ||
    ![linearity, planarity, scattering].every(Number.isFinite)
  )
    return NaN;
  return planarity >= linearity && planarity >= scattering ? 1 : scattering > linearity ? 2 : 0;
}

export function getPointCloudFeatureSupportText(status: number | undefined): string {
  return status === undefined
    ? 'support pending'
    : (status & 5) !== 0
      ? 'unusable shape'
      : (status & 2) !== 0
        ? 'truncated support'
        : 'complete support';
}

export function getPointCloudChangeSupportText(status: number | undefined): string {
  if (status === undefined) return 'support pending';
  if ((status & 0xff) === 2) return 'insufficiently supported';
  if ((status & 0xff) === 1) return 'invalid core';
  if ((status & 0xff) === 3) return 'nonfinite uncertainty';
  return status === 0 ? 'supported' : 'incomplete grid support';
}

export function summarizePointCloudChanges(
  changes: ArrayLike<number> | null | undefined,
  status: ArrayLike<number> | null | undefined
) {
  let strongest: number | undefined;
  let changed = 0;
  let supported = 0;
  for (let row = 0; changes && status && row < changes.length; row++) {
    if (status[row] !== 0 || !Number.isFinite(changes[row])) continue;
    supported++;
    if (Math.abs(changes[row]) > 0.01) changed++;
    if (strongest === undefined || Math.abs(changes[row]) > Math.abs(strongest))
      strongest = changes[row];
  }
  return {strongest, changed, supported};
}

export function getPointCloudHeightSupportText(status: number | undefined): string {
  return status === undefined
    ? 'support pending'
    : (['valid height', 'invalid point', 'unsupported terrain', 'nonfinite height'][status] ??
        'unknown height status');
}

export function getPointCloudICPSupportText(status: number | undefined): string {
  return status === undefined
    ? 'fit pending'
    : ([
        'converged',
        'iteration budget exhausted',
        'insufficient matches',
        'degenerate fit',
        'eigensolve exhausted'
      ][status] ?? 'unknown fit status');
}

/** Only complete classifications enter the ground count; provisional rows stay separate. */
export function summarizePointCloudGround(
  mask: ArrayLike<number> | null | undefined,
  status: ArrayLike<number> | null | undefined
) {
  let ground = 0;
  let supported = 0;
  let incomplete = 0;
  for (let row = 0; mask && status && row < mask.length; row++) {
    if (status[row] !== 0) {
      incomplete++;
      continue;
    }
    supported++;
    if (mask[row] !== 0) ground++;
  }
  return {ground, supported, incomplete};
}

/**
 * Ground filter classes for display: 0 removed, 1 ground (complete support), 2 provisionally
 * removed, 3 provisionally ground (a status bit outside `unclassifiedBits` is set, so the
 * classification is less certain), NaN when a bit of `unclassifiedBits` says the row was not
 * classified. Provisional rows never enter the ground count, as in the showcase lidar scene.
 */
export const POINT_CLOUD_GROUND_CLASS_LABELS = [
  'removed',
  'ground',
  'provisionally removed',
  'provisionally ground'
] as const;

export function summarizePointCloudGroundClasses(
  mask: ArrayLike<number> | null | undefined,
  status: ArrayLike<number> | null | undefined,
  unclassifiedBits: number
) {
  const counts = {
    ground: 0,
    removed: 0,
    provisionalGround: 0,
    provisionalRemoved: 0,
    unclassified: 0
  };
  for (let row = 0; mask && status && row < mask.length; row++) {
    if ((status[row] & unclassifiedBits) !== 0) counts.unclassified++;
    else if (status[row] !== 0) {
      if (mask[row] !== 0) counts.provisionalGround++;
      else counts.provisionalRemoved++;
    } else if (mask[row] !== 0) counts.ground++;
    else counts.removed++;
  }
  return counts;
}

export function createPointCloudGroundClassNode(
  graph: GPUCommandGraph,
  props: {
    id: string;
    mask: GraphDataView;
    status: GraphDataView;
    output: GraphDataView;
    unclassifiedBits: number;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'PointCloudGroundClass',
    invocationCount: props.output.length,
    bindings: [
      {name: 'mask', view: props.mask, type: 'u32', access: 'read'},
      {name: 'status', view: props.status, type: 'u32', access: 'read'},
      {name: 'classes', view: props.output, type: 'f32', access: 'read_write'}
    ],
    body: `let code = status[statusOffset + index];
var value = bitcast<f32>(0x7fc00000u | (index & 0u));
if ((code & ${props.unclassifiedBits}u) == 0u) {
  value = f32(select(0u, 1u, mask[maskOffset + index] != 0u) + select(0u, 2u, code != 0u));
}
classes[classesOffset + index] = value;`
  });
}

/** Suppress unusable measurements and coordinates before rendering, without CPU readback. */
export function createPointCloudDisplayCopyNode(
  graph: GPUCommandGraph,
  props: {
    id: string;
    source: GraphDataView;
    status: GraphDataView<'uint32'>;
    output: GraphDataView;
    componentStart?: number;
    invalidBits?: number;
    statusIndex?: number;
  }
) {
  const components = props.output.rowByteLength / 4;
  const stride = props.source.rowByteLength / 4;
  const sourceType = props.source.format.startsWith('uint') ? 'u32' : 'f32';
  const statusIndex = props.statusIndex === undefined ? 'index' : `${props.statusIndex}u`;
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'PointCloudDisplay',
    invocationCount: props.output.length,
    bindings: [
      {name: 'values', view: props.source, type: sourceType, access: 'read'},
      {name: 'status', view: props.status, type: 'u32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    body: `let usable = ${props.invalidBits === undefined ? `status[statusOffset + ${statusIndex}] == 0u` : `(status[statusOffset + ${statusIndex}] & ${props.invalidBits}u) == 0u`};
for (var component = 0u; component < ${components}u; component++) {
  var value = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (usable) { value = f32(values[valuesOffset + index * ${stride}u + ${props.componentStart ?? 0}u + component]); }
  display[displayOffset + index * ${components}u + component] = value;
}`
  });
}
