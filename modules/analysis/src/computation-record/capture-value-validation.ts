// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCaptureRasterGeometryIssues} from './capture-raster-validation';

const getTypedArrayName = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Uint8Array.prototype),
  Symbol.toStringTag
)!.get!;

/** Uses intrinsic typed-array branding, including views created in another JavaScript realm. */
export function isMatchingCaptureArray(
  values: unknown,
  elementType: unknown
): values is Float32Array | Uint32Array | Int32Array {
  if (!ArrayBuffer.isView(values)) return false;
  const name = getTypedArrayName.call(values);
  return (
    (elementType === 'f32' && name === 'Float32Array') ||
    (elementType === 'u32' && name === 'Uint32Array') ||
    (elementType === 'i32' && name === 'Int32Array')
  );
}

/** Metadata must be valid before JSON can replace nonfinite numbers with null. */
export function getCaptureMetadataIssues(
  capture: Record<string, unknown>
): {field: string; message: string}[] {
  const issues: {field: string; message: string}[] = [];
  for (const field of ['id', 'name']) {
    if (typeof capture[field] !== 'string' || !capture[field])
      issues.push({field, message: 'Expected a nonempty string'});
  }
  if (capture['provenance'] !== 'gpu-readback' && capture['provenance'] !== 'gpu-replay')
    issues.push({field: 'provenance', message: 'Expected gpu-readback or gpu-replay'});
  if (
    capture['components'] !== undefined &&
    (typeof capture['components'] !== 'number' ||
      !Number.isSafeInteger(capture['components']) ||
      capture['components'] <= 0)
  )
    issues.push({field: 'components', message: 'Expected a positive safe integer'});
  if (
    typeof capture['elementType'] !== 'string' ||
    !['f32', 'u32', 'i32'].includes(capture['elementType'])
  )
    issues.push({field: 'elementType', message: 'Expected f32, u32 or i32'});
  for (const issue of getCaptureRasterGeometryIssues(capture['geometry']))
    issues.push({...issue, field: `geometry.${issue.field}`});
  return issues;
}
