// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const MAXIMUM_UINT32 = 0xffffffff;
const UINT32_VALUE_COUNT = 0x100000000;

/** Bounded three-dimensional WebGPU workgroup layout. @internal */
export type GPUBoundedDispatchLayout = {
  x: number;
  y: number;
  z: number;
};

/** Plans a bounded three-dimensional dispatch for a linear element range. @internal */
export function getBoundedDispatchLayout(
  operationName: string,
  elementCount: number,
  workgroupSize: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  if (!Number.isSafeInteger(elementCount) || elementCount < 0 || elementCount > MAXIMUM_UINT32) {
    throw new Error(`${operationName} element count must be a non-negative uint32`);
  }
  validateWorkgroupSize(operationName, workgroupSize);
  const maximum = Math.floor(maxComputeWorkgroupsPerDimension);
  if (!Number.isSafeInteger(maximum) || maximum < 1) {
    throw new Error('maxComputeWorkgroupsPerDimension must be a positive integer');
  }

  return getBoundedWorkgroupDispatchLayout(
    operationName,
    Math.ceil(elementCount / workgroupSize),
    maximum
  );
}

/**
 * Spreads a workgroup count over a bounded three-dimensional dispatch. This is the single
 * implementation of the x/y/z expansion; prefer `getBoundedDispatchLayout` when starting from an
 * element count. @internal
 */
export function getBoundedWorkgroupDispatchLayout(
  operationName: string,
  workgroupCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  const maximum = Math.floor(maxComputeWorkgroupsPerDimension);
  const totalWorkgroupCount = Math.max(1, Math.ceil(workgroupCount));
  const x = Math.min(totalWorkgroupCount, maximum);
  const y = Math.min(Math.ceil(totalWorkgroupCount / x), maximum);
  const z = Math.ceil(totalWorkgroupCount / x / y);
  if (z > maximum) {
    throw new Error(
      `${operationName} requires ${totalWorkgroupCount} workgroups, exceeding the 3D dispatch limit of ${maximum} per dimension`
    );
  }
  return {x, y, z};
}

/** Pipeline constants matching `getBoundedInvocationIndexOverrideSource`. @internal */
export function getBoundedDispatchOverrides(layout: GPUBoundedDispatchLayout): {
  DISPATCH_X: number;
  DISPATCH_Y: number;
} {
  return {DISPATCH_X: layout.x, DISPATCH_Y: layout.y};
}

/**
 * Like `getBoundedInvocationIndexSource`, but the dispatch shape comes from pipeline overrides so
 * one shader module serves every problem size. Declares `DISPATCH_X`/`DISPATCH_Y`; pass
 * `getBoundedDispatchOverrides(layout)` as pipeline constants. @internal
 */
export function getBoundedInvocationIndexOverrideSource(workgroupSize: number): {
  declarations: string;
  body: string;
} {
  validateWorkgroupSize('GPU dispatch', workgroupSize);
  const maximumLinearWorkgroupCount = Math.floor(MAXIMUM_UINT32 / workgroupSize) + 1;
  return {
    declarations: 'override DISPATCH_X: u32;\noverride DISPATCH_Y: u32;',
    body: `let workgroupIndex = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (workgroupIndex >= ${maximumLinearWorkgroupCount}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`
  };
}

/** Returns WGSL that maps a bounded 3D dispatch to one linear element index. @internal */
export function getBoundedInvocationIndexSource(
  layout: GPUBoundedDispatchLayout,
  workgroupSize: number
): string {
  validateWorkgroupSize('GPU dispatch', workgroupSize);
  const maximumLinearWorkgroupCount = Math.floor(MAXIMUM_UINT32 / workgroupSize) + 1;
  return `let workgroupIndex = (workgroupId.z * ${layout.y}u + workgroupId.y) * ${layout.x}u + workgroupId.x;
  if (workgroupIndex >= ${maximumLinearWorkgroupCount}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;`;
}

/** Ensures the pre-multiplication guard exactly partitions the uint32 invocation range. */
function validateWorkgroupSize(operationName: string, workgroupSize: number): void {
  if (
    !Number.isSafeInteger(workgroupSize) ||
    workgroupSize < 2 ||
    workgroupSize > MAXIMUM_UINT32 ||
    UINT32_VALUE_COUNT % workgroupSize !== 0
  ) {
    throw new Error(`${operationName} workgroup size must be a power of two greater than one`);
  }
}
