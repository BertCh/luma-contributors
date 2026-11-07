// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphTextureView} from '@luma.gl/graph/gpu-core';
import {
  getGPURasterProfileParameterValues,
  makeGPURasterMetadata,
  GPURasterProfile
} from '@luma-contributors/analysis/raster-analysis';
import {
  GPURasterBufferToTexture,
  GPURasterContrast,
  GPURasterHistogram,
  GPURasterTextureToBuffer
} from '@luma.gl/graph/gpu-raster';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {createGraphViewer} from './b16-common';

export type SpectralDisplaySettings = {
  displayMode: 'linear' | 'gamma' | 'equalize';
  displayDomain: 'calibrated' | 'gpu';
  displayContrast: number;
  displayGamma: number;
  displayBins: 64 | 256;
  profileSpacing: 60 | 120 | 240;
  profileMethod: 'nearest' | 'bilinear';
  profileNoData: 'strict' | 'renormalize';
  transportChannel: 'R' | 'G' | 'B' | 'A';
};

export type SpectralDisplayOutputs = {
  transformed: Buffer;
  transformedValidity: Buffer;
  displayHistogram: Buffer;
  calibratedHistogram: Buffer;
  gpuHistogram: Buffer;
  gpuDomain: Buffer;
  profileValues: Buffer;
  profileDistances: Buffer;
  profileStatus: Buffer;
  profileRequired: Buffer;
  profileOverflow: Buffer;
  profileLength: Buffer;
  profileMinimum: Buffer;
  profileMaximum: Buffer;
  returnedValues: Buffer;
  returnedValidity: Buffer;
  transportTexture: ReturnType<Device['createTexture']>;
  profileLine: Buffer;
};

export const PROFILE_CAPACITY = 512;
export const PROFILE_CELL_SIZE_METERS = 60;
export const PROFILE_VERTICES = [
  [450, 390],
  [3930, 3690],
  [7290, 7290]
] as const;

export function addSpectralDisplayGraph(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  view: ReturnType<typeof createGraphViewer>,
  device: Device,
  sourceNir: Buffer,
  sourceValidity: Buffer,
  settings: SpectralDisplaySettings
): SpectralDisplayOutputs {
  const pixelCount = 128 * 128;
  const transformed = resources.createBuffer('display-transformed', pixelCount * 4);
  const transformedValidity = resources.createBuffer(
    'display-transformed-validity',
    pixelCount * 4
  );
  const sourceHistogram = resources.createBuffer(
    'equalize-source-histogram',
    settings.displayBins * 4
  );
  const selectedHistogram = resources.createBuffer('display-histogram', 256 * 4);
  const calibratedHistogram = resources.createBuffer('calibrated-histogram', 256 * 4);
  const gpuHistogram = resources.createBuffer('gpu-domain-histogram', 256 * 4);
  const gpuDomain = resources.createBuffer('gpu-domain', 2 * 4);
  const source = {
    id: 'display-nir-source',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: view('display-nir', sourceNir, 'float32', pixelCount)
    },
    validity: view('display-source-validity', sourceValidity, 'uint32', pixelCount)
  };
  const histogramDomain =
    settings.displayDomain === 'gpu'
      ? view('display-gpu-domain', gpuDomain, 'float32', 2)
      : ([0, 0.6] as const);
  new GPURasterHistogram({
    id: 'spectral-display-literal-histogram',
    input: source,
    output: view('display-literal-histogram', calibratedHistogram, 'uint32', 256),
    domain: [0, 0.6]
  }).addToGraph(graph);
  new GPURasterHistogram({
    id: 'spectral-display-gpu-domain-histogram',
    input: source,
    output: view('display-gpu-domain-histogram', gpuHistogram, 'uint32', 256),
    domain: 'valid-auto',
    domainOutput: view('display-gpu-domain', gpuDomain, 'float32', 2)
  }).addToGraph(graph);
  new GPURasterHistogram({
    id: 'spectral-display-selected-histogram',
    input: source,
    output: view('equalize-source-histogram', sourceHistogram, 'uint32', settings.displayBins),
    domain: histogramDomain
  }).addToGraph(graph);
  new GPURasterContrast({
    id: 'spectral-display-contrast',
    width: 128,
    height: 128,
    input: source,
    output: view('display-transformed', transformed, 'float32', pixelCount),
    outputValidity: view('display-transformed-validity', transformedValidity, 'uint32', pixelCount),
    domain: histogramDomain,
    contrast: settings.displayContrast,
    gamma: settings.displayGamma,
    mode: settings.displayMode,
    ...(settings.displayMode === 'equalize'
      ? {
          histogram: view(
            'equalize-source-histogram',
            sourceHistogram,
            'uint32',
            settings.displayBins
          )
        }
      : {})
  }).addToGraph(graph);
  const transformedBand = {
    id: 'transformed-display-band',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: view('display-transformed', transformed, 'float32', pixelCount)
    },
    validity: view('display-transformed-validity', transformedValidity, 'uint32', pixelCount)
  };
  new GPURasterHistogram({
    id: 'spectral-transformed-display-histogram',
    input: transformedBand,
    output: view('display-selected-histogram', selectedHistogram, 'uint32', 256),
    domain: histogramDomain
  }).addToGraph(graph);

  const grid = makeGPURasterMetadata({
    width: 128,
    height: 128,
    cellSize: 1,
    origin: [0, 0]
  });
  // The display vertices use east/south distances from the northwest corner.
  // Sampling uses source-cell coordinates so authored cell centers stay exact.
  // GPU profile distances are in cells and are converted to metres at readout.
  const profilePositionData = new Float32Array(
    PROFILE_VERTICES.flatMap(([east, south]) => [east, -south])
  );
  const profileLine = resources.createBuffer('profile-line', profilePositionData, Buffer.VERTEX);
  const profileCellPositions = resources.createBuffer(
    'profile-cell-positions',
    profilePositionData.map(coordinate => coordinate / PROFILE_CELL_SIZE_METERS)
  );
  const lineOffsets = resources.createBuffer('profile-line-offsets', new Uint32Array([0, 3]));
  const profileValues = resources.createBuffer('profile-values', PROFILE_CAPACITY * 4);
  const profileDistances = resources.createBuffer('profile-distances', PROFILE_CAPACITY * 4);
  const profileCount = resources.createBuffer('profile-count', 4);
  const profileRequired = resources.createBuffer('profile-required', 4);
  const profileOverflow = resources.createBuffer('profile-overflow', 4);
  const profileLength = resources.createBuffer('profile-length', 4);
  const profileMinimum = resources.createBuffer('profile-minimum', 4);
  const profileMaximum = resources.createBuffer('profile-maximum', 4);
  const samplingSettings = getGPURasterProfileParameterValues(
    {
      spacing: settings.profileSpacing / PROFILE_CELL_SIZE_METERS,
      method: settings.profileMethod,
      noDataPolicy: settings.profileNoData,
      localOrigin: [0, 0]
    },
    grid
  );
  const profileParameters = resources.createParameterBuffer(
    'profile-parameters',
    'float32',
    samplingSettings.length,
    samplingSettings
  );
  const parameterView = profileParameters.importToGraph(graph);
  graph.add(
    new GPURasterProfile({
      id: 'spectral-profile',
      grid,
      values: view('profile-nir', sourceNir, 'float32', pixelCount),
      validity: view('profile-validity', sourceValidity, 'uint32', pixelCount),
      positions: view(
        'profile-vertices',
        profileCellPositions,
        'float32x2',
        PROFILE_VERTICES.length
      ),
      lineOffsets: view('profile-line-offsets', lineOffsets, 'uint32', 2),
      sampleCapacity: PROFILE_CAPACITY,
      parameters: parameterView,
      output: {
        status: {
          count: view('profile-count', profileCount, 'uint32', 1),
          requiredCount: view('profile-required', profileRequired, 'uint32', 1),
          overflow: view('profile-overflow', profileOverflow, 'uint32', 1)
        },
        sampleValues: view('profile-values', profileValues, 'float32', PROFILE_CAPACITY),
        sampleDistances: view('profile-distances', profileDistances, 'float32', PROFILE_CAPACITY),
        lineLengths: view('profile-length', profileLength, 'float32', 1),
        lineMinimums: view('profile-minimum', profileMinimum, 'float32', 1),
        lineMaximums: view('profile-maximum', profileMaximum, 'float32', 1)
      }
    })
  );

  if (!device.getTextureFormatCapabilities('rgba32float').store) {
    throw new Error('Spectral texture roundtrip requires rgba32float storage writes.');
  }
  const transportTexture = resources.track(
    device.createTexture({
      id: 'spectral-transport-rgba32float',
      format: 'rgba32float',
      width: 128,
      height: 128,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    })
  );
  const textureHandle = graph.importTexture(
    {
      id: 'spectral-transport-rgba32float',
      format: 'rgba32float',
      width: 128,
      height: 128,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    },
    transportTexture
  );
  const textureView: GraphTextureView<'rgba32float'> = graph.createTextureView(textureHandle);
  const textureValidity = resources.createBuffer('transport-texture-validity', pixelCount * 4);
  const returnedValues = resources.createBuffer('transport-returned-values', pixelCount * 4);
  const returnedValidity = resources.createBuffer('transport-returned-validity', pixelCount * 4);
  const channel = ({R: 0, G: 1, B: 2, A: 3} as const)[settings.transportChannel];
  new GPURasterBufferToTexture({
    id: 'spectral-buffer-to-texture',
    input: source,
    output: textureView,
    outputValidity: view('transport-texture-validity', textureValidity, 'uint32', pixelCount),
    channel,
    applyCalibration: false
  }).addToGraph(graph);
  new GPURasterTextureToBuffer({
    id: 'spectral-texture-to-buffer',
    input: {
      id: 'transport-texture-band',
      format: 'float32',
      storage: {kind: 'texture', view: textureView, channel},
      validity: view('transport-texture-validity', textureValidity, 'uint32', pixelCount)
    },
    output: view('transport-returned-values', returnedValues, 'float32', pixelCount),
    outputValidity: view('transport-returned-validity', returnedValidity, 'uint32', pixelCount),
    applyCalibration: false
  }).addToGraph(graph);

  return {
    transformed,
    transformedValidity,
    displayHistogram: selectedHistogram,
    calibratedHistogram,
    gpuHistogram,
    gpuDomain,
    profileValues,
    profileDistances,
    profileStatus: profileCount,
    profileRequired,
    profileOverflow,
    profileLength,
    profileMinimum,
    profileMaximum,
    returnedValues,
    returnedValidity,
    transportTexture,
    profileLine
  };
}
