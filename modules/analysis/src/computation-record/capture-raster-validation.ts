// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCaptureGeometry} from './computation-record-types';

/** Checks additive raster placement fields on both in-memory and saved capture paths. */
export function getCaptureRasterGeometryIssues(
  geometry: unknown
): {field: string; message: string}[] {
  if (
    geometry === null ||
    typeof geometry !== 'object' ||
    !('kind' in geometry) ||
    geometry.kind !== 'raster'
  )
    return [];
  const raster = geometry as Record<string, unknown>;
  const issues: {field: string; message: string}[] = [];
  if (
    raster['affine'] !== undefined &&
    (!Array.isArray(raster['affine']) ||
      raster['affine'].length !== 6 ||
      Array.from({length: 6}, (_, index) => (raster['affine'] as unknown[])[index]).some(
        value => typeof value !== 'number' || !Number.isFinite(value)
      ))
  )
    issues.push({
      field: 'affine',
      message: 'Expected six finite affine coefficients [a,b,c,d,e,f]'
    });
  if (
    raster['pixelInterpretation'] !== undefined &&
    (typeof raster['pixelInterpretation'] !== 'string' ||
      !['area', 'point'].includes(raster['pixelInterpretation']))
  )
    issues.push({field: 'pixelInterpretation', message: 'Expected area or point'});
  for (const field of ['frameId', 'crs']) {
    if (raster[field] !== undefined && (typeof raster[field] !== 'string' || !raster[field]))
      issues.push({field, message: 'Expected a nonempty identity string'});
  }
  return issues;
}

/** Effective sample placement, preserving the historical top-edge default for southward rows. */
export function getGPUComputationRasterSamplePosition(
  geometry: Extract<GPUComputationCaptureGeometry, {kind: 'raster'}>,
  column: number,
  row: number
): readonly [number, number] {
  const issue = getCaptureRasterGeometryIssues(geometry)[0];
  if (issue) throw new Error(`computation-captures geometry.${issue.field}: ${issue.message}`);
  const cellSize = geometry.cellSize ?? 1;
  const north = geometry.rowDirection === 'north';
  const origin = geometry.origin ?? [0, north ? 0 : geometry.shape[1] * cellSize];
  const [a, b, c, d, e, f] = geometry.affine ?? [
    cellSize,
    0,
    origin[0],
    0,
    north ? cellSize : -cellSize,
    origin[1]
  ];
  const shift = geometry.pixelInterpretation === 'point' ? 0 : 0.5;
  return [
    a * (column + shift) + b * (row + shift) + c,
    d * (column + shift) + e * (row + shift) + f
  ];
}
