// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, ComputePass, Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {
  makeProjectionWgsl,
  normalizeProjectionLabel,
  parseProjectionDescriptor,
  projectDescriptor,
  unprojectDescriptor
} from './resident-spatial-join-crs';
import {transformLngLatToMercator, transformMercatorToLngLat} from './raster-reprojection';
import type {PreparedArrowAnalyticsTable} from '@luma-contributors/graph/arrow';
import type {KeplerDatasetRegistry} from '@luma-contributors/kepler';
import {
  estimateTabularPointPositionBytes,
  makeGPUVectorsFromTabularPointPositions,
  type CoordinateRoles,
  type TabularPointGPUVectors
} from './point-position-source';

/**
 * Canonical CRS label: 'EPSG:4326', 'EPSG:3857' or a canonical transverse Mercator definition
 * (see resident-spatial-join-crs.ts). No other CRS is guessed or inferred.
 */
export type ResidentSpatialJoinCrs = string;
export type ResidentSpatialJoinPointTransform =
  | 'none'
  | 'longitude-latitude-to-mercator'
  | 'mercator-to-longitude-latitude'
  /** Any other supported pair: points leave their CRS through longitude/latitude into the polygon CRS. */
  | 'projected';

/** Normalizes a declared CRS label, or returns undefined when the join cannot run in that CRS. */
export function normalizeResidentSpatialJoinCrs(
  declared: string
): ResidentSpatialJoinCrs | undefined {
  return normalizeProjectionLabel(declared);
}

/**
 * Points are moved into the polygon CRS, so polygon vertices are never rewritten. Same CRS needs no
 * transform: point-in-polygon is a planar test in either system.
 */
export function getResidentSpatialJoinPointTransform(
  pointCrs: ResidentSpatialJoinCrs,
  polygonCrs: ResidentSpatialJoinCrs
): ResidentSpatialJoinPointTransform {
  if (pointCrs === polygonCrs) return 'none';
  if (pointCrs === 'EPSG:4326' && polygonCrs === 'EPSG:3857')
    return 'longitude-latitude-to-mercator';
  if (pointCrs === 'EPSG:3857' && polygonCrs === 'EPSG:4326')
    return 'mercator-to-longitude-latitude';
  return 'projected';
}

/** CPU reference for one point, used by specs and the Float32 error accounting. */
export function transformResidentSpatialJoinPoint(
  transform: ResidentSpatialJoinPointTransform,
  x: number,
  y: number,
  crs?: {points: ResidentSpatialJoinCrs; polygons: ResidentSpatialJoinCrs}
): [number, number] {
  if (transform === 'none') return [x, y];
  if (transform === 'projected') {
    const from = parseProjectionDescriptor(crs!.points)!;
    const to = parseProjectionDescriptor(crs!.polygons)!;
    const [longitude, latitude] = unprojectDescriptor(from, x, y);
    return projectDescriptor(to, longitude, latitude);
  }
  return transform === 'longitude-latitude-to-mercator'
    ? transformLngLatToMercator(x, y)
    : transformMercatorToLngLat(x, y);
}

const WORKGROUP_SIZE = 256;
const EARTH_RADIUS = 6378137;
const MAXIMUM_LATITUDE = 85.0511287798066;

/** In-place GPU reprojection of one chunk of float32x2 points; Float32 limits apply to the result. */
export class ResidentSpatialJoinPointReprojector {
  private readonly kernel: Kernel;
  constructor(
    device: Device,
    transform: Exclude<ResidentSpatialJoinPointTransform, 'none'>,
    private readonly rowCount: number,
    crs?: {points: ResidentSpatialJoinCrs; polygons: ResidentSpatialJoinCrs}
  ) {
    const projectedPrelude =
      transform === 'projected'
        ? `${makeProjectionWgsl('source', parseProjectionDescriptor(crs!.points)!)}
${makeProjectionWgsl('target', parseProjectionDescriptor(crs!.polygons)!)}`
        : '';
    const body =
      transform === 'projected'
        ? 'points[index] = targetProject(sourceUnproject(point));'
        : transform === 'longitude-latitude-to-mercator'
          ? `let latitude = clamp(point.y, -${MAXIMUM_LATITUDE}, ${MAXIMUM_LATITUDE});
  points[index] = vec2<f32>(${EARTH_RADIUS}.0 * point.x * 0.017453292519943295, ${EARTH_RADIUS}.0 * log(tan(0.7853981633974483 + latitude * 0.008726646259971648)));`
          : `points[index] = vec2<f32>(point.x / ${EARTH_RADIUS}.0 * 57.29577951308232, (2.0 * atan(exp(point.y / ${EARTH_RADIUS}.0)) - 1.5707963267948966) * 57.29577951308232);`;
    this.kernel = new Kernel(device, {
      id: 'resident-spatial-join-reproject',
      source: `
${projectedPrelude}
@group(0) @binding(0) var<storage,read_write> points:array<vec2<f32>>;
@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  let index = id.x;
  if (index >= ${rowCount}u) {return;}
  let point = points[index];
  ${body}
}`,
      shaderLayout: {bindings: [{name: 'points', location: 0, group: 0, type: 'storage' as const}]}
    });
  }
  /** Records the transform pass; the chunk must already hold the copied (and padded) points. */
  encode(computePass: ComputePass, points: Buffer): void {
    this.kernel.dispatch(computePass, {
      bindings: {points},
      x: Math.ceil(this.rowCount / WORKGROUP_SIZE)
    });
  }
  destroy(): void {
    this.kernel.destroy();
  }
}

/**
 * Resident point positions are validated as longitude/latitude at ingestion, so a point table in a
 * projected CRS (UTM or Web Mercator meters) holds NaN positions. A Spatial Join whose left CRS is
 * not EPSG:4326 uploads the raw coordinate columns instead; the caller destroys the result.
 */
export function makeResidentSpatialJoinSourcePositions(
  device: Device,
  prepared: PreparedArrowAnalyticsTable,
  roles: CoordinateRoles,
  memoryRegistry?: KeplerDatasetRegistry
): TabularPointGPUVectors {
  const longitudeColumn = prepared.table.getChild(roles.longitude)!;
  const latitudeColumn = prepared.table.getChild(roles.latitude)!;
  const readNumber = (value: unknown): number =>
    typeof value === 'number' && Number.isFinite(value) ? value : NaN;
  const batches = prepared.sourceBatches.map(batch => {
    const positions = new Float32Array(batch.sourceRowCount * 2);
    for (let row = 0; row < batch.sourceRowCount; row++) {
      const sourceRow = batch.sourceRowIndexOffset + row;
      positions[row * 2] = readNumber(longitudeColumn.get(sourceRow));
      positions[row * 2 + 1] = readNumber(latitudeColumn.get(sourceRow));
    }
    return {sourceRowOffset: batch.sourceRowIndexOffset, rowCount: batch.sourceRowCount, positions};
  });
  const estimate = estimateTabularPointPositionBytes(
    batches.map(batch => batch.rowCount),
    false
  );
  return makeGPUVectorsFromTabularPointPositions(
    device,
    {
      roles: {longitude: roles.longitude, latitude: roles.latitude},
      batches,
      invalidRowCount: 0,
      typedStorageByteLength: estimate.cpuByteLength,
      gpuByteLength: estimate.gpuByteLength
    },
    {memoryRegistry}
  );
}
