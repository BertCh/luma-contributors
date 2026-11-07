// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {makeGPURasterMetadata} from '@luma-contributors/graph/gpu-raster';
import {
  GPUTerrainCurvature,
  GPUTerrainDerivatives,
  getGPUTerrainCurvatureParameterValues,
  getGPUTerrainDerivativesParameterValues
} from '@luma-contributors/graph/gpu-terrain';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuOutput,
  type CatalogGpuSession
} from './catalog-gpu-session';

export type TerrainGpuOperation = 'slope' | 'aspect' | 'hillshade' | 'curvature';

export type TerrainGpuInput = Readonly<{
  /** Row-major, north-up elevation samples. */
  elevation: ArrayLike<number>;
  /** 1 where the elevation sample is valid. */
  validity: Uint8Array;
  width: number;
  height: number;
  /** [west, south, east, north] in the declared CRS. */
  extent: readonly [number, number, number, number];
  /** EPSG:4326 extents are degrees, EPSG:3857 extents are meters. */
  crs: 'EPSG:4326' | 'EPSG:3857';
  operation: TerrainGpuOperation;
  zFactor?: number;
  azimuthDegrees?: number;
  altitudeDegrees?: number;
}>;

export type TerrainGpuGrid = Readonly<{
  width: number;
  height: number;
  values: Float32Array;
  validity: Uint8Array;
  /** Contributors that were dispatched. */
  contributors: readonly string[];
}>;

/** Kepler hillshade is reported on 0..255, the contributor returns 0..1. */
const HILLSHADE_SCALE = 255;
/** ESRI curvature is -100 times the Laplacian of the Zevenbergen-Thorne surface. */
const CURVATURE_SCALE = -100;

/** Output-unit scale applied to the contributor value: hillshade 0..255, curvature ESRI sign. */
export function getTerrainOutputScale(operation: TerrainGpuOperation): number {
  return operation === 'hillshade'
    ? HILLSHADE_SCALE
    : operation === 'curvature'
      ? CURVATURE_SCALE
      : 1;
}

/** Dispatches the terrain contributor into session outputs without running or reading anything. */
export function addTerrainGraph(
  session: CatalogGpuSession,
  input: TerrainGpuInput
): {
  values: CatalogGpuOutput<'float32'>;
  outputValidity: CatalogGpuOutput<'uint32'>;
  contributor: string;
} {
  const {width, height} = input;
  const pixelCount = width * height;
  const [west, south, east, north] = input.extent;
  const grid = makeGPURasterMetadata({
    width,
    height,
    cellSize: [(east - west) / width, (north - south) / height],
    origin: [west, north]
  });
  const cellSizeMode = input.crs === 'EPSG:3857' ? 'web-mercator' : 'geographic';
  const elevationValues = new Float32Array(pixelCount);
  const validityMask = new Uint32Array(pixelCount);
  for (let sampleIndex = 0; sampleIndex < pixelCount; sampleIndex++) {
    if (input.validity[sampleIndex]) {
      elevationValues[sampleIndex] = input.elevation[sampleIndex];
      validityMask[sampleIndex] = 1;
    }
  }
  const zFactor = input.zFactor ?? 1;
  const elevationView = session.input('elevation', elevationValues, 'float32');
  const maskView = session.input('mask', validityMask, 'uint32');
  const elevation = {
    id: 'elevation',
    format: 'float32',
    storage: {kind: 'buffer', values: elevationView},
    validity: maskView
  } as const;
  const outputValidity = session.output('validity', 'uint32', pixelCount);
  const values = session.output('values', 'float32', pixelCount);
  let contributor: string;
  if (input.operation === 'curvature') {
    const parameters = session.input(
      'parameters',
      getGPUTerrainCurvatureParameterValues({zFactor}, grid),
      'float32'
    );
    session.graph.add(
      new GPUTerrainCurvature({
        grid,
        elevation,
        parameters,
        method: 'zevenbergen-thorne',
        cellSizeMode,
        borderMode: 'nodata',
        output: {curvatures: {laplacian: values.view}, outputValidity: outputValidity.view}
      })
    );
    contributor = 'GPUTerrainCurvature';
  } else {
    const parameters = session.input(
      'parameters',
      getGPUTerrainDerivativesParameterValues(
        {
          zFactor,
          azimuthDegrees: input.azimuthDegrees,
          altitudeDegrees: input.altitudeDegrees
        },
        grid
      ),
      'float32'
    );
    session.graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation,
        parameters,
        cellSizeMode,
        borderMode: 'nodata',
        output: {[input.operation]: values.view, outputValidity: outputValidity.view}
      })
    );
    contributor = 'GPUTerrainDerivatives';
  }
  return {values, outputValidity, contributor};
}

/**
 * Computes one Horn slope, aspect or hillshade, or an ESRI curvature, for a whole elevation tile
 * on the GPU with the shared terrain contributors. Ground cell size is derived per row by the
 * contributor from the declared CRS (WGS84 ellipsoid, geographic degrees or Web Mercator meters).
 * Samples on the tile border and samples with an invalid neighbor are invalid (borderMode
 * 'nodata'); pad the tile with a neighbor halo to avoid seams.
 */
export async function computeTerrainGridOnGpu(
  device: Device,
  input: TerrainGpuInput
): Promise<TerrainGpuGrid> {
  const {width, height} = input;
  const pixelCount = width * height;
  // Unsupported tile size: a storage binding cannot hold the elevation band.
  if (pixelCount * 4 > device.limits.maxStorageBufferBindingSize) {
    throw new CatalogGpuUnsupportedError('Terrain tile exceeds the storage binding limit');
  }
  return withCatalogGpuSession(device, `terrain-${input.operation}`, async session => {
    const {values, outputValidity, contributor} = addTerrainGraph(session, input);
    session.run();
    const [rawValues, rawValidity] = await Promise.all([
      session.readFloat32(values),
      session.readUint32(outputValidity)
    ]);
    const scale = getTerrainOutputScale(input.operation);
    const resultValues = new Float32Array(pixelCount);
    const resultValidity = new Uint8Array(pixelCount);
    for (let sampleIndex = 0; sampleIndex < pixelCount; sampleIndex++) {
      const value = rawValues[sampleIndex];
      if (rawValidity[sampleIndex] && Number.isFinite(value)) {
        resultValues[sampleIndex] = value * scale;
        resultValidity[sampleIndex] = 1;
      }
    }
    return {
      width,
      height,
      values: resultValues,
      validity: resultValidity,
      contributors: [contributor]
    };
  });
}
