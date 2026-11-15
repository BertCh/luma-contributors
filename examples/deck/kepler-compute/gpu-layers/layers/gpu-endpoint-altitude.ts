// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';

/** Endpoint position vectors: longitude/latitude, or longitude/latitude/altitude in meters. */
export type GPUEndpointPositionFormat = 'float32x2' | 'float32x3';

/** Endpoint vector accepted by line, arc and arrow-head layers. */
export type GPUEndpointPositionVector = GPUVector<'float32x2'> | GPUVector<'float32x3'>;

/** Formats accepted for endpoint positions; the 2D default stays valid. */
export const GPU_ENDPOINT_POSITION_FORMATS: GPUEndpointPositionFormat[] = [
  'float32x2',
  'float32x3'
];

/** Longitude/latitude variable names a transformed shader body and brushing expression use. */
export const GPU_ENDPOINT_SOURCE_LONGITUDE_LATITUDE = 'sourceLongitudeLatitude';
export const GPU_ENDPOINT_TARGET_LONGITUDE_LATITUDE = 'targetLongitudeLatitude';

const VERTEX_BODY_MARKER = '-> VertexOutput {';

/**
 * Types the `sourcePositions` / `targetPositions` vertex inputs from their vector format (the
 * attribute names stay the layer contract) and exposes `sourceLongitudeLatitude` /
 * `targetLongitudeLatitude` (vec2) plus `sourceAltitude` / `targetAltitude` (f32, 0 for 2D vectors).
 * Every other body use of the original names is renamed. A float32x2 vector never feeds a vec3 input.
 */
export function applyEndpointAltitudeToShader(
  source: string,
  sourceVectorFormat: GPUEndpointPositionFormat | undefined,
  targetVectorFormat: GPUEndpointPositionFormat | undefined,
  altitudeScaleExpression = '1.0'
): string {
  // Vectors without a recorded format are the 2D default.
  const sourceFormat = sourceVectorFormat ?? 'float32x2';
  const targetFormat = targetVectorFormat ?? 'float32x2';
  const sourceDeclaration = '@location(0) sourcePositions: vec2<f32>';
  const targetDeclaration = '@location(1) targetPositions: vec2<f32>';
  // The layer shader no longer declares endpoint inputs or a vertex entry point.
  if (
    !source.includes(sourceDeclaration) ||
    !source.includes(targetDeclaration) ||
    !source.includes(VERTEX_BODY_MARKER)
  ) {
    throw new Error('Endpoint altitude requires endpoint inputs');
  }
  const renamed = source
    .replace(sourceDeclaration, '@@SOURCE@@')
    .replace(targetDeclaration, '@@TARGET@@')
    .replace(/\bsourcePositions\b/g, GPU_ENDPOINT_SOURCE_LONGITUDE_LATITUDE)
    .replace(/\btargetPositions\b/g, GPU_ENDPOINT_TARGET_LONGITUDE_LATITUDE)
    .replace('@@SOURCE@@', `@location(0) sourcePositions: ${getType(sourceFormat)}`)
    .replace('@@TARGET@@', `@location(1) targetPositions: ${getType(targetFormat)}`);
  const prelude = [
    getPrelude('source', sourceFormat, altitudeScaleExpression),
    getPrelude('target', targetFormat, altitudeScaleExpression)
  ].join('\n');
  return renamed.replace(VERTEX_BODY_MARKER, `${VERTEX_BODY_MARKER}\n${prelude}`);
}

function getType(format: GPUEndpointPositionFormat): string {
  return format === 'float32x3' ? 'vec3<f32>' : 'vec2<f32>';
}

function getPrelude(
  side: 'source' | 'target',
  format: GPUEndpointPositionFormat,
  altitudeScaleExpression: string
): string {
  const input = `${side}Positions`;
  return format === 'float32x3'
    ? `  let ${side}LongitudeLatitude = ${input}.xy;\n  let ${side}Altitude = ${input}.z * ${altitudeScaleExpression};`
    : `  let ${side}LongitudeLatitude = ${input};\n  let ${side}Altitude = 0.0;`;
}

/** Reports whether any endpoint carries altitude. */
export function hasEndpointAltitude(sourcePosition: GPUVector, targetPosition: GPUVector): boolean {
  return sourcePosition.format === 'float32x3' || targetPosition.format === 'float32x3';
}
