// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedWorkgroupDispatchLayout
} from '../../../../gpu-core/gpu-dispatch-utils';

export type WebGPUDispatchLayout = GPUBoundedDispatchLayout;

const WEBGPU_MINIMUM_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65_535;

/** Plans a bounded dispatch; delegates to the shared gpu-core implementation. */
export function getWebGPUDispatchLayout(
  workgroupCount: number,
  maxWorkgroupsPerDimension?: number
): WebGPUDispatchLayout {
  return getBoundedWorkgroupDispatchLayout(
    'WebGPU dispatch',
    workgroupCount,
    getMaximumWorkgroupsPerDimension(maxWorkgroupsPerDimension)
  );
}

export function getWebGPUDispatchWorkgroupIndex(
  layout: WebGPUDispatchLayout,
  workgroupId: string = 'workgroupId'
): string {
  return (
    `((${workgroupId}.z * ${layout.y}u + ${workgroupId}.y) * ` + `${layout.x}u + ${workgroupId}.x)`
  );
}

export function getWebGPUDispatchRowIndex(
  layout: WebGPUDispatchLayout,
  workgroupSize: number,
  workgroupId: string = 'workgroupId',
  localId: string = 'localId'
): string {
  return (
    `(${getWebGPUDispatchWorkgroupIndex(layout, workgroupId)} * ` +
    `${workgroupSize}u + ${localId}.x)`
  );
}

/** Pipeline constants matching {@link getWebGPUDispatchRowIndexOverride}. */
export function getWebGPUDispatchOverrides(layout: WebGPUDispatchLayout): {
  DISPATCH_X: number;
  DISPATCH_Y: number;
} {
  return getBoundedDispatchOverrides(layout);
}

/**
 * Like {@link getWebGPUDispatchWorkgroupIndex}, but the dispatch shape comes from the
 * `DISPATCH_X`/`DISPATCH_Y` pipeline overrides, so one shader serves every row count.
 */
export function getWebGPUDispatchWorkgroupIndexOverride(
  workgroupId: string = 'workgroupId'
): string {
  return `((${workgroupId}.z * DISPATCH_Y + ${workgroupId}.y) * DISPATCH_X + ${workgroupId}.x)`;
}

/** Like {@link getWebGPUDispatchRowIndex}, with the dispatch shape taken from pipeline overrides. */
export function getWebGPUDispatchRowIndexOverride(
  workgroupSize: number,
  workgroupId: string = 'workgroupId',
  localId: string = 'localId'
): string {
  return `(${getWebGPUDispatchWorkgroupIndexOverride(workgroupId)} * ${workgroupSize}u + ${localId}.x)`;
}

function getMaximumWorkgroupsPerDimension(maxWorkgroupsPerDimension?: number): number {
  if (Number.isFinite(maxWorkgroupsPerDimension) && maxWorkgroupsPerDimension! > 0) {
    return Math.floor(maxWorkgroupsPerDimension!);
  }
  return WEBGPU_MINIMUM_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
}
