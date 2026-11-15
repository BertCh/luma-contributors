// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type GeometryPropertyName =
  | 'fillColor'
  | 'lineColor'
  | 'lineWidth'
  | 'radius'
  | 'elevation'
  | 'elevationOffset';
export type GeometryStyleDiagnostic = {
  sourceRowIndex: number;
  property: GeometryPropertyName;
  reason: string;
};
export type PreparedGeometryProperties = {
  fillColor?: {values: Uint8Array; validity: Uint32Array};
  lineColor?: {values: Uint8Array; validity: Uint32Array};
  lineWidth?: {values: Float32Array; validity: Uint32Array};
  radius?: {values: Float32Array; validity: Uint32Array};
  elevation?: {values: Float32Array; validity: Uint32Array};
  elevationOffset?: {values: Float32Array; validity: Uint32Array};
};
export const GEOMETRY_PROPERTY_NAMES: readonly GeometryPropertyName[] = [
  'fillColor',
  'lineColor',
  'lineWidth',
  'radius',
  'elevation',
  'elevationOffset'
];

/** Root normalized Feature properties only; native FeatureCollections use their first Feature. */
export function getGeometryFeatureProperties(raw: unknown): Record<string, unknown> | undefined {
  try {
    let value = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (!value || typeof value !== 'object') return undefined;
    if (value.type === 'FeatureCollection') value = value.features?.[0];
    if (!value || value.type !== 'Feature' || !value.geometry || typeof value.geometry !== 'object')
      return undefined;
    if (
      ![
        'Point',
        'MultiPoint',
        'LineString',
        'MultiLineString',
        'Polygon',
        'MultiPolygon',
        'GeometryCollection'
      ].includes(value.geometry.type)
    )
      return undefined;
    return value.properties && typeof value.properties === 'object' ? value.properties : undefined;
  } catch {
    return undefined;
  }
}

export function ensureGeometryProperty(
  properties: PreparedGeometryProperties,
  property: GeometryPropertyName,
  rowCount: number
): void {
  if (property === 'fillColor' || property === 'lineColor') {
    properties[property] ??= {
      values: new Uint8Array(rowCount * 4),
      validity: new Uint32Array(rowCount)
    };
  } else {
    properties[property] ??= {
      values: new Float32Array(rowCount),
      validity: new Uint32Array(rowCount)
    };
  }
}

/** Native truthy fallback, except finite-presence elevationOffset; mutable defaults belong to runtime. */
export function writeGeometryProperty(
  properties: PreparedGeometryProperties,
  property: GeometryPropertyName,
  value: unknown,
  rowIndex: number,
  rowCount: number
): string | undefined {
  ensureGeometryProperty(properties, property, rowCount);
  // Pinned GeoJSON accepts finite offsets including zero; other types/nonfinite values fall back.
  // This intentionally differs from elevation || 500 and the other truthy scalar properties.
  if (property === 'elevationOffset') {
    if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
    if (!Number.isFinite(Math.fround(value)))
      return 'Embedded elevation offset exceeds finite float32 storage';
    const channel = properties.elevationOffset!;
    channel.values[rowIndex] = value;
    channel.validity[rowIndex] = 1;
    return undefined;
  }
  if (!value) return undefined;
  if (property === 'fillColor' || property === 'lineColor') {
    if (!Array.isArray(value) && !(ArrayBuffer.isView(value) && !(value instanceof DataView)))
      return 'Truthy embedded color requires an array';
    const color = value as ArrayLike<unknown>;
    const channel = properties[property]!;
    // Match Deck's component defaults and Uint8 attribute conversion, including RGB alpha255.
    for (let component = 0; component < 4; component++) {
      const componentValue = color[component];
      channel.values[rowIndex * 4 + component] =
        typeof componentValue === 'number' && Number.isFinite(componentValue)
          ? componentValue
          : component === 3
            ? 255
            : 0;
    }
    channel.validity[rowIndex] = 1;
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value)))
    return property === 'elevation'
      ? 'Truthy embedded elevation requires a finite float32 number'
      : 'Truthy embedded size requires a finite float32 number';
  const channel = properties[property]!;
  channel.values[rowIndex] = value;
  channel.validity[rowIndex] = 1;
  return undefined;
}
