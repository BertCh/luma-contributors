// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterBoxBlur,
  GPURasterConvolution,
  GPURasterGaussianBlur,
  GPURasterGradient,
  GPURasterGradientMagnitude,
  GPURasterLaplacian,
  GPURasterNeighborhood,
  GPURasterScharr,
  GPURasterSobel
} from '@luma.gl/graph/gpu-raster';
import {
  GPURasterFocalPairs,
  GPURasterTexture,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import type {SpatialAnalysisColormap} from '../../engine/layers';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {fetchBytes, getDataFileUrl} from '../../data/loaders';
import {
  addSpectralDisplayGraph,
  PROFILE_CAPACITY,
  PROFILE_CELL_SIZE_METERS,
  PROFILE_VERTICES,
  type SpectralDisplayOutputs
} from './spectral-texture-display';
import {
  getSpectralLayerStyle,
  getSpectralTextureRange,
  SPECTRAL_MAP_LAYERS,
  type FocalLayer,
  type SpectralLayer
} from './spectral-texture-style';

export type SpectralTextureOptions = {
  date: 'before' | 'after';
  layer: SpectralLayer | 'focal' | 'display' | 'transport';
  metric:
    | 'contrast'
    | 'dissimilarity'
    | 'homogeneity'
    | 'ASM'
    | 'energy'
    | 'correlation'
    | 'entropy';
  radius: number;
  sigma: number;
  borderMode: 'clamp' | 'reflect' | 'nodata';
  noDataPolicy: 'propagate' | 'ignore-renormalize';
  centerPropagation: boolean;
  gradientOperator: 'sobel' | 'scharr';
  laplacianConnectivity: '4' | '8';
  textureRadius: number;
  textureDirection: 'E' | 'N' | 'NE' | 'NW';
  textureLevels: '8' | '16';
  textureSymmetry: 'on' | 'off';
  textureDisplayDomain: 'detail' | 'theoretical';
  convolutionKernel: 'sobel-x' | 'east-derivative' | 'south-derivative';
  convolutionBorder: 'inherit' | 'constant';
  convolutionBorderValue: number;
  focalLayer: FocalLayer;
  covariance: 'population' | 'sample';
  minimumCount: number;
  focalNoData: 'ignore' | 'propagate';
  displayMode: 'linear' | 'gamma' | 'equalize';
  displayDomain: 'calibrated' | 'gpu';
  displayContrast: '0.5' | '1' | '2';
  displayGamma: '0.5' | '1' | '2';
  displayBins: '64' | '256';
  profileSpacing: '60' | '120' | '240';
  profileMethod: 'nearest' | 'bilinear';
  profileNoData: 'strict' | 'renormalize';
  transportChannel: 'R' | 'G' | 'B' | 'A';
};

const CROP_SIZE = 128;
const CROP_COLUMN = 60;
const CROP_ROW = 50;
const FULL_SIZE = 250;
const PROJECTED_BOUNDS = [662800, 4441040, 677800, 4456040] as const;
const TEXTURE_OFFSETS = {E: [1, 0], N: [0, -1], NE: [1, -1], NW: [-1, -1]} as const;
const TEXTURE_METRICS = [
  'contrast',
  'dissimilarity',
  'homogeneity',
  'ASM',
  'energy',
  'correlation',
  'entropy'
] as const;
const FOCAL_METRICS = [
  'covariance',
  'correlation',
  'slope',
  'intercept',
  'rSquared',
  'residualSumSquares',
  'pairedCount',
  'status'
] as const;
const COMPILE_OPTIONS = new Set<keyof SpectralTextureOptions>([
  'date',
  'radius',
  'sigma',
  'borderMode',
  'noDataPolicy',
  'centerPropagation',
  'gradientOperator',
  'laplacianConnectivity',
  'textureRadius',
  'textureDirection',
  'textureLevels',
  'textureSymmetry',
  'convolutionKernel',
  'convolutionBorder',
  'convolutionBorderValue',
  'covariance',
  'minimumCount',
  'focalNoData',
  'displayMode',
  'displayDomain',
  'displayContrast',
  'displayGamma',
  'displayBins',
  'profileSpacing',
  'profileMethod',
  'profileNoData',
  'transportChannel'
]);
type Source = {nir: Float32Array; swir: Float32Array; validity: Uint32Array};
type FocalBuffers = {
  pairedCount: Buffer;
  status: Buffer;
  covariance: Buffer;
  correlation: Buffer;
  slope: Buffer;
  intercept: Buffer;
  rSquared: Buffer;
  residualSumSquares: Buffer;
};
type Outputs = {
  values: Record<SpectralLayer, Buffer>;
  focal: Record<FocalLayer, Buffer>;
  focalCountDisplay: Buffer;
  focalStatusDisplay: Buffer;
  textureFeatures: Buffer;
  textureDisplay: Buffer;
  texturePairs: Buffer;
};
type Summary = {
  values: Partial<Record<SpectralLayer, Float32Array>>;
  texture: Float32Array;
  texturePairs: Uint32Array;
  focal: Partial<Record<FocalLayer, Float32Array | Uint32Array>>;
  diagnostics: {
    transformed: Float32Array;
    transformedValidity: Uint32Array;
    displayHistogram: Uint32Array;
    calibratedHistogram: Uint32Array;
    gpuHistogram: Uint32Array;
    gpuDomain: Float32Array;
    profileValues: Float32Array;
    profileDistances: Float32Array;
    profileCount: number;
    profileRequired: number;
    profileOverflow: number;
    profileLength: number;
    profileMinimum: number;
    profileMaximum: number;
    returnedValues: Float32Array;
    returnedValidity: Uint32Array;
  };
};
type Variant = {
  optionsKey: string;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  outputs: Outputs;
  diagnostics: SpectralDisplayOutputs;
  summary: Summary | null;
};

async function loadSource(
  ctx: SceneContext<SpectralTextureOptions>,
  date: 'before' | 'after'
): Promise<Source> {
  const dataset = ctx.datasets.get('dixie-spectral');
  const [nirRaster, swirRaster, validityBytes] = await Promise.all([
    dataset.loadRaster(`nir_${date}`, ctx.signal),
    dataset.loadRaster(`swir22_${date}`, ctx.signal),
    fetchBytes(getDataFileUrl('dixie-spectral', `validity_${date}.bin`), ctx.signal)
  ]);
  const nir = new Float32Array(CROP_SIZE * CROP_SIZE);
  const swir = new Float32Array(CROP_SIZE * CROP_SIZE);
  const validity = new Uint32Array(CROP_SIZE * CROP_SIZE);
  const nativeValidity = new Uint8Array(validityBytes);
  for (let row = 0; row < CROP_SIZE; row++) {
    for (let column = 0; column < CROP_SIZE; column++) {
      const targetIndex = row * CROP_SIZE + column;
      const sourceIndex = (row + CROP_ROW) * FULL_SIZE + column + CROP_COLUMN;
      const valid =
        nativeValidity[sourceIndex] === 1 &&
        Number.isFinite(nirRaster.values[sourceIndex]) &&
        Number.isFinite(swirRaster.values[sourceIndex]);
      nir[targetIndex] = valid ? nirRaster.values[sourceIndex] : Number.NaN;
      swir[targetIndex] = valid ? swirRaster.values[sourceIndex] : Number.NaN;
      validity[targetIndex] = valid ? 1 : 0;
    }
  }
  return {nir, swir, validity};
}

function makeKernel(radius: number): number[] {
  const width = radius * 2 + 1;
  const values: number[] = [];
  for (let row = -radius; row <= radius; row++) {
    for (let column = -radius; column <= radius; column++) {
      values.push((radius + 1 - Math.abs(column)) * (radius + 1 - Math.abs(row)));
    }
  }
  return values.map(value => value / (width * width));
}

function createFocalBuffers(resources: SpatialAnalysisResources, prefix: string): FocalBuffers {
  const floatBuffer = (name: string) =>
    resources.createBuffer(`${prefix}-${name}`, CROP_SIZE * CROP_SIZE * 4);
  return {
    pairedCount: resources.createBuffer(`${prefix}-paired-count`, CROP_SIZE * CROP_SIZE * 4),
    status: resources.createBuffer(`${prefix}-status`, CROP_SIZE * CROP_SIZE * 4),
    covariance: floatBuffer('covariance'),
    correlation: floatBuffer('correlation'),
    slope: floatBuffer('slope'),
    intercept: floatBuffer('intercept'),
    rSquared: floatBuffer('r-squared'),
    residualSumSquares: floatBuffer('residual-sum-squares')
  };
}

function getCompileKey(options: SpectralTextureOptions): string {
  return [
    options.date,
    options.radius,
    options.sigma,
    options.borderMode,
    options.noDataPolicy,
    options.centerPropagation,
    options.gradientOperator,
    options.laplacianConnectivity,
    options.textureRadius,
    options.textureDirection,
    options.textureLevels,
    options.textureSymmetry,
    options.convolutionKernel,
    options.convolutionBorder,
    options.convolutionBorderValue,
    options.covariance,
    options.minimumCount,
    options.focalNoData,
    options.displayMode,
    options.displayDomain,
    options.displayContrast,
    options.displayGamma,
    options.displayBins,
    options.profileSpacing,
    options.profileMethod,
    options.profileNoData,
    options.transportChannel
  ].join('|');
}

export async function createSpectralTexture(
  ctx: SceneContext<SpectralTextureOptions>
): Promise<SceneInstance<SpectralTextureOptions>> {
  ctx.setStatus('Loading calibrated Sentinel-2 reflectance');
  const sources = {before: await loadSource(ctx, 'before'), after: await loadSource(ctx, 'after')};
  ctx.signal.throwIfAborted();
  const frame = createUtmFrame(ctx.datasets.get('dixie-spectral').defaultOrigin, {
    boundsProjected: [
      PROJECTED_BOUNDS[0] + CROP_COLUMN * 60,
      PROJECTED_BOUNDS[1] + (FULL_SIZE - CROP_ROW - CROP_SIZE) * 60,
      PROJECTED_BOUNDS[0] + (CROP_COLUMN + CROP_SIZE) * 60,
      PROJECTED_BOUNDS[3] - CROP_ROW * 60
    ],
    cellSizeM: 60,
    width: CROP_SIZE,
    height: CROP_SIZE
  });
  const rootResources = new SpatialAnalysisResources(ctx.device, 'spectral-texture-owner');
  const graphSwap = rootResources.track(new GraphSwap());
  const cropCorners = [
    frame.cellToMeters(-0.5, -0.5),
    frame.cellToMeters(CROP_SIZE - 0.5, -0.5),
    frame.cellToMeters(CROP_SIZE - 0.5, CROP_SIZE - 0.5),
    frame.cellToMeters(-0.5, CROP_SIZE - 0.5)
  ];
  const cropOutline = new Float32Array(16);
  for (let index = 0; index < 4; index++) {
    const start = cropCorners[index];
    const end = cropCorners[(index + 1) % 4];
    cropOutline.set([start[0], start[1], end[0], end[1]], index * 4);
  }
  const cropOutlineBuffer = rootResources.createBuffer('crop-outline', cropOutline, Buffer.VERTEX);
  const profileDisplayVertices = new Float32Array(8);
  for (let segment = 0; segment < 2; segment++) {
    const first = PROFILE_VERTICES[segment];
    const second = PROFILE_VERTICES[segment + 1];
    const start = frame.cellToMeters(first[0] / 60 - 0.5, first[1] / 60 - 0.5);
    const end = frame.cellToMeters(second[0] / 60 - 0.5, second[1] / 60 - 0.5);
    profileDisplayVertices.set([start[0], start[1], end[0], end[1]], segment * 4);
  }
  const profileDisplayLine = rootResources.createBuffer(
    'profile-display-line',
    profileDisplayVertices,
    Buffer.VERTEX
  );
  let variantIndex = 0;
  let active: Variant;
  let dirty = true;

  const compileVariant = async (options: SpectralTextureOptions): Promise<Variant> => {
    const variantResources = new SpatialAnalysisResources(
      ctx.device,
      `spectral-texture-${variantIndex++}`
    );
    const source = sources[options.date];
    const pixelCount = CROP_SIZE * CROP_SIZE;
    const sourceNir = variantResources.createBuffer('source-nir', source.nir);
    const sourceSwir = variantResources.createBuffer('source-swir', source.swir);
    const sourceValidity = variantResources.createBuffer('source-validity', source.validity);
    const outputs: Outputs = {
      values: {} as Record<SpectralLayer, Buffer>,
      focal: {} as Record<FocalLayer, Buffer>,
      focalCountDisplay: variantResources.createBuffer(
        'focal-count-display',
        new Float32Array(pixelCount).fill(Number.NaN)
      ),
      focalStatusDisplay: variantResources.createBuffer(
        'focal-status-display',
        new Float32Array(pixelCount).fill(Number.NaN)
      ),
      textureFeatures: variantResources.createBuffer('texture-features', pixelCount * 7 * 4),
      textureDisplay: variantResources.createBuffer(
        'texture-display',
        new Float32Array(pixelCount).fill(Number.NaN)
      ),
      texturePairs: variantResources.createBuffer('texture-pairs', pixelCount * 4)
    };
    for (const layerName of SPECTRAL_MAP_LAYERS) {
      if (layerName === 'source') outputs.values[layerName] = sourceNir;
      else if (layerName === 'texture') outputs.values[layerName] = outputs.textureDisplay;
      else
        outputs.values[layerName] = variantResources.createBuffer(
          `layer-${layerName}`,
          pixelCount * 4
        );
    }
    const focalBuffers = createFocalBuffers(variantResources, 'focal');
    outputs.focal = {
      covariance: focalBuffers.covariance,
      correlation: focalBuffers.correlation,
      slope: focalBuffers.slope,
      intercept: focalBuffers.intercept,
      rSquared: focalBuffers.rSquared,
      residualSumSquares: focalBuffers.residualSumSquares,
      pairedCount: outputs.focalCountDisplay,
      status: outputs.focalStatusDisplay
    };
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `spectral-texture-${variantIndex}`});
    const view = createGraphViewer(graph);
    const floatView = (name: string, buffer: Buffer, length = pixelCount) =>
      view(name, buffer, 'float32', length);
    const wordView = (name: string, buffer: Buffer, length = pixelCount) =>
      view(name, buffer, 'uint32', length);
    const input = {
      id: `nir-${options.date}`,
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: floatView('source-nir', sourceNir)},
      validity: wordView('source-validity', sourceValidity)
    };
    const response = {
      id: `swir-${options.date}`,
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: floatView('source-swir', sourceSwir)},
      validity: wordView('response-validity', sourceValidity)
    };
    const output = (name: SpectralLayer) => floatView(`layer-${name}`, outputs.values[name]);
    const outputValidity = (name: string) =>
      wordView(
        `${name}-validity`,
        variantResources.createBuffer(`${name}-validity`, pixelCount * 4)
      );
    const diagnostics = addSpectralDisplayGraph(
      graph,
      variantResources,
      view,
      ctx.device,
      sourceNir,
      sourceValidity,
      {
        displayMode: options.displayMode,
        displayDomain: options.displayDomain,
        displayContrast: Number(options.displayContrast),
        displayGamma: Number(options.displayGamma),
        displayBins: Number(options.displayBins) as 64 | 256,
        profileSpacing: Number(options.profileSpacing) as 60 | 120 | 240,
        profileMethod: options.profileMethod,
        profileNoData: options.profileNoData,
        transportChannel: options.transportChannel
      }
    );
    const borderMode = options.borderMode;
    const noDataPolicy = options.noDataPolicy;
    const smoothRadius = options.radius;
    const nodataPolicy = noDataPolicy;
    new GPURasterGaussianBlur({
      id: 'spectral-gaussian',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('smooth'),
      outputValidity: outputValidity('smooth'),
      radius: smoothRadius,
      sigma: options.sigma,
      borderMode,
      noDataPolicy: nodataPolicy
    }).addToGraph(graph);
    new GPURasterBoxBlur({
      id: 'spectral-box',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('box'),
      outputValidity: outputValidity('box'),
      radius: smoothRadius,
      borderMode,
      noDataPolicy: nodataPolicy
    }).addToGraph(graph);
    new GPURasterNeighborhood({
      id: 'spectral-weighted',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('weighted'),
      outputValidity: outputValidity('weighted'),
      radius: smoothRadius,
      kernel: makeKernel(smoothRadius),
      normalize: true,
      borderMode,
      noDataPolicy: nodataPolicy
    }).addToGraph(graph);
    const convolutionKernels = {
      'sobel-x': {
        kernel: [-1, 0, 1, -2, 0, 2, -1, 0, 1].map(value => value / 8),
        width: 3,
        height: 3
      },
      'east-derivative': {kernel: [-1, -2, 0, 2, 1].map(value => value / 8), width: 5, height: 1},
      'south-derivative': {kernel: [-1, -2, 0, 2, 1].map(value => value / 8), width: 1, height: 5}
    } as const;
    const convolution = convolutionKernels[options.convolutionKernel];
    const convolutionBorderMode =
      options.convolutionBorder === 'constant' ? 'constant' : borderMode;
    new GPURasterConvolution({
      id: 'spectral-convolution',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('convolution'),
      outputValidity: outputValidity('convolution'),
      kernel: convolution.kernel,
      kernelWidth: convolution.width,
      kernelHeight: convolution.height,
      borderMode: convolutionBorderMode,
      ...(convolutionBorderMode === 'constant'
        ? {borderValue: options.convolutionBorderValue}
        : {}),
      noDataPolicy: 'propagate',
      normalize: false
    }).addToGraph(graph);
    const gradientScale = options.gradientOperator === 'scharr' ? 1 / 32 : 1 / 8;
    for (const direction of ['x', 'y'] as const) {
      const validity = outputValidity(`gradient-${direction}`);
      new GPURasterGradient({
        id: `spectral-gradient-${direction}`,
        width: CROP_SIZE,
        height: CROP_SIZE,
        input,
        output: output(direction === 'x' ? 'gradientX' : 'gradientY'),
        outputValidity: validity,
        operator: options.gradientOperator,
        direction,
        scale: gradientScale,
        borderMode
      }).addToGraph(graph);
      new GPURasterSobel({
        id: `spectral-sobel-${direction}`,
        width: CROP_SIZE,
        height: CROP_SIZE,
        input,
        output: output(direction === 'x' ? 'sobelX' : 'sobelY'),
        outputValidity: outputValidity(`sobel-${direction}`),
        direction,
        scale: 1 / 8,
        borderMode
      }).addToGraph(graph);
      new GPURasterScharr({
        id: `spectral-scharr-${direction}`,
        width: CROP_SIZE,
        height: CROP_SIZE,
        input,
        output: output(direction === 'x' ? 'scharrX' : 'scharrY'),
        outputValidity: outputValidity(`scharr-${direction}`),
        direction,
        scale: 1 / 32,
        borderMode
      }).addToGraph(graph);
    }
    new GPURasterGradientMagnitude({
      id: 'spectral-gradient-magnitude',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('gradient'),
      outputValidity: outputValidity('gradient'),
      operator: options.gradientOperator,
      scale: gradientScale,
      borderMode
    }).addToGraph(graph);
    new GPURasterLaplacian({
      id: 'spectral-laplacian',
      width: CROP_SIZE,
      height: CROP_SIZE,
      input,
      output: output('laplacian'),
      outputValidity: outputValidity('laplacian'),
      connectivity: options.laplacianConnectivity === '4' ? 4 : 8,
      borderMode
    }).addToGraph(graph);
    const grid = makeGPURasterMetadata({
      width: CROP_SIZE,
      height: CROP_SIZE,
      cellSize: 60,
      origin: [PROJECTED_BOUNDS[0] + CROP_COLUMN * 60, PROJECTED_BOUNDS[3] - CROP_ROW * 60]
    });
    graph.add(
      new GPURasterTexture({
        id: 'spectral-glcm',
        grid,
        values: floatView('texture-input', sourceNir),
        validity: wordView('texture-validity', sourceValidity),
        levels: Number(options.textureLevels),
        range: [0, 0.6],
        radius: options.textureRadius,
        offset: TEXTURE_OFFSETS[options.textureDirection],
        symmetric: options.textureSymmetry === 'on',
        features: floatView('texture-features', outputs.textureFeatures, pixelCount * 7),
        pairCount: wordView('texture-pairs', outputs.texturePairs)
      })
    );
    graph.add(
      new GPURasterFocalPairs({
        id: 'spectral-focal-pairs',
        grid,
        predictor: input,
        response,
        radius: options.radius,
        borderMode: borderMode === 'nodata' ? 'nodata' : borderMode,
        noDataPolicy: options.focalNoData,
        propagateCenterNoData: options.centerPropagation,
        minimumCount: options.minimumCount,
        covarianceNormalization: options.covariance,
        output: {
          pairedCount: wordView('focal-paired-count', focalBuffers.pairedCount),
          status: wordView('focal-status', focalBuffers.status),
          covariance: floatView('focal-covariance', focalBuffers.covariance),
          correlation: floatView('focal-correlation', focalBuffers.correlation),
          slope: floatView('focal-slope', focalBuffers.slope),
          intercept: floatView('focal-intercept', focalBuffers.intercept),
          rSquared: floatView('focal-r-squared', focalBuffers.rSquared),
          residualSumSquares: floatView(
            'focal-residual-sum-squares',
            focalBuffers.residualSumSquares
          )
        }
      })
    );
    const compiled = variantResources.track(await graph.compileAsync());
    let summary: Summary | null = null;
    const summaryBuffers = [
      ...SPECTRAL_MAP_LAYERS.filter(name => name !== 'source' && name !== 'texture').map(
        name => outputs.values[name]
      ),
      outputs.textureFeatures,
      outputs.texturePairs,
      diagnostics.transformed,
      diagnostics.transformedValidity,
      diagnostics.displayHistogram,
      diagnostics.calibratedHistogram,
      diagnostics.gpuHistogram,
      diagnostics.gpuDomain,
      diagnostics.profileValues,
      diagnostics.profileDistances,
      diagnostics.profileStatus,
      diagnostics.profileRequired,
      diagnostics.profileOverflow,
      diagnostics.profileLength,
      diagnostics.profileMinimum,
      diagnostics.profileMaximum,
      diagnostics.returnedValues,
      diagnostics.returnedValidity,
      focalBuffers.covariance,
      focalBuffers.correlation,
      focalBuffers.slope,
      focalBuffers.intercept,
      focalBuffers.rSquared,
      focalBuffers.residualSumSquares,
      focalBuffers.pairedCount,
      focalBuffers.status
    ];
    const reader = new SummaryReader(
      variantResources,
      'spectral-summary',
      summaryBuffers.map(buffer => ({buffer, size: buffer.byteLength})),
      bytes => {
        let byteOffset = 0;
        const readFloat = (length = pixelCount) => {
          const values = new Float32Array(bytes, byteOffset, length);
          byteOffset += length * 4;
          return values;
        };
        const readUint = (length = pixelCount) => {
          const values = new Uint32Array(bytes, byteOffset, length);
          byteOffset += length * 4;
          return values;
        };
        const values: Partial<Record<SpectralLayer, Float32Array>> = {};
        for (const name of SPECTRAL_MAP_LAYERS.filter(
          value => value !== 'source' && value !== 'texture'
        ))
          values[name] = readFloat();
        const texture = readFloat(pixelCount * 7);
        const texturePairs = readUint();
        const diagnosticSummary = {
          transformed: readFloat(),
          transformedValidity: readUint(),
          displayHistogram: readUint(256),
          calibratedHistogram: readUint(256),
          gpuHistogram: readUint(256),
          gpuDomain: readFloat(2),
          profileValues: readFloat(PROFILE_CAPACITY),
          profileDistances: readFloat(PROFILE_CAPACITY),
          profileCount: readUint(1)[0],
          profileRequired: readUint(1)[0],
          profileOverflow: readUint(1)[0],
          profileLength: readFloat(1)[0] * PROFILE_CELL_SIZE_METERS,
          profileMinimum: readFloat(1)[0],
          profileMaximum: readFloat(1)[0],
          returnedValues: readFloat(),
          returnedValidity: readUint()
        };
        const focal: Partial<Record<FocalLayer, Float32Array | Uint32Array>> = {};
        for (const metric of FOCAL_METRICS)
          focal[metric] =
            metric === 'pairedCount' || metric === 'status' ? readUint() : readFloat();
        const pairedCountValues = focal.pairedCount;
        const statusValues = focal.status;
        if (pairedCountValues instanceof Uint32Array) {
          outputs.focalCountDisplay.write(Float32Array.from(pairedCountValues));
        }
        if (statusValues instanceof Uint32Array) {
          const statusClasses = Float32Array.from(statusValues, classifyFocalStatus);
          outputs.focalStatusDisplay.write(statusClasses);
        }
        summary = {values, texture, texturePairs, focal, diagnostics: diagnosticSummary};
        variant.summary = summary;
        writeTextureDisplay(variant, ctx.options.metric);
        if (active === variant && active.optionsKey === getCompileKey(ctx.options)) {
          ctx.setLegendData('displayDomain', Array.from(summary.diagnostics.gpuDomain));
          publishReadouts();
          ctx.refreshTooltip();
          ctx.requestLayers();
        }
      }
    );
    const variant: Variant = {
      optionsKey: getCompileKey(options),
      resources: variantResources,
      compiled,
      reader,
      outputs,
      diagnostics,
      summary
    };
    return variant;
  };

  let selectedCell = 56 * CROP_SIZE + 73;
  const marker = rootResources.createBuffer('selected-cell', new Float32Array(2), Buffer.VERTEX);
  const writeMarker = () =>
    marker.write(
      Float32Array.of(
        ...frame.cellToMeters(selectedCell % CROP_SIZE, Math.floor(selectedCell / CROP_SIZE))
      )
    );
  writeMarker();
  const getCell = (coordinate: readonly number[] | null) => {
    if (!coordinate) return -1;
    const [column, row] = frame.lngLatToCell(coordinate[0], coordinate[1]);
    const cellColumn = Math.floor(column);
    const cellRow = Math.floor(row);
    return cellColumn >= 0 && cellRow >= 0 && cellColumn < CROP_SIZE && cellRow < CROP_SIZE
      ? cellRow * CROP_SIZE + cellColumn
      : -1;
  };
  const publishReadouts = () => {
    if (!active || active.optionsKey !== getCompileKey(ctx.options)) {
      ctx.setReadout(
        'selectedEvidence',
        'Raster recipe pending · evidence updates when the graph swap commits.'
      );
      ctx.setReadout(
        'textureEvidence',
        'Texture evidence pending · evidence updates when the graph swap commits.'
      );
      ctx.setReadout(
        'focalEvidence',
        'Focal evidence pending · evidence updates when the graph swap commits.'
      );
      ctx.setReadout('displayEvidence', 'Display transform pending.');
      ctx.setReadout('profileEvidence', 'Transect profile pending.');
      ctx.setReadout('transportEvidence', 'Texture transport pending.');
      ctx.setChart('pairChart', null);
      ctx.setChart('displayChart', null);
      ctx.setChart('profileChart', null);
      return;
    }
    const summary = active?.summary;
    if (!summary) {
      ctx.setReadout(
        'selectedEvidence',
        'GPU readback pending · selected-cell evidence will update when ready.'
      );
      ctx.setReadout(
        'textureEvidence',
        'GPU readback pending · texture evidence will update when ready.'
      );
      ctx.setReadout(
        'focalEvidence',
        'GPU readback pending · focal evidence will update when ready.'
      );
      ctx.setReadout('displayEvidence', 'Display transform readback pending.');
      ctx.setReadout('profileEvidence', 'Transect profile readback pending.');
      ctx.setReadout('transportEvidence', 'Texture transport readback pending.');
      ctx.setChart('pairChart', null);
      ctx.setChart('displayChart', null);
      ctx.setChart('profileChart', null);
      return;
    }
    const focalValue = (name: FocalLayer) => summary.focal[name]?.[selectedCell] ?? Number.NaN;
    const focalStatus = Number(focalValue('status'));
    const focalPairs = Number(focalValue('pairedCount'));
    const values = [
      `cov ${Number(focalValue('covariance')).toFixed(6)}`,
      `r ${Number(focalValue('correlation')).toFixed(6)}`,
      `slope ${Number(focalValue('slope')).toFixed(6)}`,
      `intercept ${Number(focalValue('intercept')).toFixed(6)}`,
      `R² ${Number(focalValue('rSquared')).toFixed(6)}`,
      `RSS ${Number(focalValue('residualSumSquares')).toFixed(6)}`
    ];
    const statusDescription =
      [
        (focalStatus & 1) !== 0 ? 'insufficient pairs' : '',
        (focalStatus & 2) !== 0 ? 'missing pair' : '',
        (focalStatus & 4) !== 0 ? 'constant predictor' : '',
        (focalStatus & 8) !== 0 ? 'constant response' : '',
        (focalStatus & 16) !== 0 ? 'non-finite result' : ''
      ]
        .filter(Boolean)
        .join(' + ') || 'supported';
    ctx.setReadout(
      'focalEvidence',
      `${values.join(' · ')} · n ${focalPairs} · status mask ${focalStatus}: ${statusDescription}`
    );
    const diagnostics = summary.diagnostics;
    const sourceForDiagnostics = sources[ctx.options.date];
    const diagnosticPixelCount = CROP_SIZE * CROP_SIZE;
    const invalidDisplayCount = diagnostics.transformedValidity.reduce(
      (count, valid) => count + (valid === 0 ? 1 : 0),
      0
    );
    const sourceProbe = sourceForDiagnostics.nir[selectedCell];
    const displayProbe = diagnostics.transformed[selectedCell];
    ctx.setReadout(
      'displayEvidence',
      `${ctx.options.displayMode} · NIR ${sourceProbe.toFixed(6)} → display ${displayProbe.toFixed(6)} · ${invalidDisplayCount} invalid / ${diagnosticPixelCount} · gamma exponent pow(normalized, 1/γ)`
    );
    const displayDomain = ctx.options.displayDomain === 'gpu' ? diagnostics.gpuDomain : [0, 0.6];
    const measuredCounts =
      ctx.options.displayDomain === 'gpu'
        ? diagnostics.gpuHistogram
        : diagnostics.calibratedHistogram;
    const distributionPositions = Array.from(
      {length: 256},
      (_, index) => displayDomain[0] + ((index + 0.5) / 256) * (displayDomain[1] - displayDomain[0])
    );
    ctx.setChart('displayChart', {
      kind: 'line',
      xLabel: 'reflectance-domain bin midpoint',
      yLabel: 'GPU pixel count',
      series: [
        {label: 'measured NIR', x: distributionPositions, y: Array.from(measuredCounts)},
        {
          label: 'transformed display values',
          x: distributionPositions,
          y: Array.from(diagnostics.displayHistogram)
        }
      ]
    });
    const profileCount = Math.min(diagnostics.profileCount, PROFILE_CAPACITY);
    const profileDistances = Array.from(
      diagnostics.profileDistances.slice(0, profileCount),
      distanceInCells => distanceInCells * PROFILE_CELL_SIZE_METERS
    );
    const profileValues = Array.from(diagnostics.profileValues.slice(0, profileCount));
    ctx.setReadout(
      'profileEvidence',
      `${profileCount}/${diagnostics.profileRequired} emitted · ${profileValues.filter(Number.isFinite).length} finite · overflow ${diagnostics.profileOverflow} · ${ctx.options.profileMethod}/${ctx.options.profileNoData} · min ${diagnostics.profileMinimum.toFixed(6)} / max ${diagnostics.profileMaximum.toFixed(6)} · ${diagnostics.profileLength.toFixed(1)} m`
    );
    ctx.setChart('profileChart', {
      kind: 'line',
      xLabel: 'distance along transect (m)',
      yLabel: 'NIR reflectance (unitless)',
      series: [{label: 'measured NIR', x: profileDistances, y: profileValues}]
    });
    let transportMaximumError = 0;
    let transportMismatchCount = 0;
    let transportFiniteCount = 0;
    for (let cell = 0; cell < diagnosticPixelCount; cell++) {
      const sourceValid =
        sourceForDiagnostics.validity[cell] !== 0 &&
        Number.isFinite(sourceForDiagnostics.nir[cell]);
      if (sourceValid !== (diagnostics.returnedValidity[cell] !== 0)) transportMismatchCount++;
      if (sourceValid && Number.isFinite(diagnostics.returnedValues[cell])) {
        transportFiniteCount++;
        transportMaximumError = Math.max(
          transportMaximumError,
          Math.abs(sourceForDiagnostics.nir[cell] - diagnostics.returnedValues[cell])
        );
      }
    }
    ctx.setReadout(
      'transportEvidence',
      `${ctx.options.transportChannel} channel · max |source − returned| ${transportMaximumError.toExponential(2)} · validity mismatches ${transportMismatchCount} · ${transportFiniteCount} finite comparisons · calibration off`
    );
    const metricIndex = TEXTURE_METRICS.indexOf(ctx.options.metric);
    const textureValue = summary.texture[selectedCell * 7 + metricIndex];
    ctx.setReadout(
      'textureEvidence',
      `${ctx.options.metric} ${Number.isFinite(textureValue) ? textureValue.toFixed(6) : 'undefined'} · ${summary.texturePairs[selectedCell]} ${ctx.options.textureSymmetry === 'on' ? 'symmetric' : 'directed'} pairs`
    );
    const shownLayer = ctx.options.layer === 'focal' ? ctx.options.focalLayer : ctx.options.layer;
    let outputValue = Number.NaN;
    if (ctx.options.layer === 'focal') {
      outputValue = summary.focal[ctx.options.focalLayer]?.[selectedCell] ?? Number.NaN;
    } else if (ctx.options.layer === 'source') {
      outputValue = sources[ctx.options.date].nir[selectedCell];
    } else if (ctx.options.layer === 'texture') {
      outputValue = textureValue;
    } else if (ctx.options.layer === 'display') {
      outputValue = summary.diagnostics.transformed[selectedCell];
    } else if (ctx.options.layer === 'transport') {
      outputValue = summary.diagnostics.returnedValues[selectedCell];
    } else {
      outputValue = summary.values[ctx.options.layer]?.[selectedCell] ?? Number.NaN;
    }
    const smoothedValue = summary.values.smooth?.[selectedCell] ?? Number.NaN;
    const supportRadius = ctx.options.radius;
    const centerRow = Math.floor(selectedCell / CROP_SIZE);
    const centerColumn = selectedCell % CROP_SIZE;
    let validSourceCells = 0;
    let availableSourceCells = 0;
    for (let rowOffset = -supportRadius; rowOffset <= supportRadius; rowOffset++) {
      for (let columnOffset = -supportRadius; columnOffset <= supportRadius; columnOffset++) {
        let row = centerRow + rowOffset;
        let column = centerColumn + columnOffset;
        if (ctx.options.borderMode === 'clamp') {
          row = Math.max(0, Math.min(CROP_SIZE - 1, row));
          column = Math.max(0, Math.min(CROP_SIZE - 1, column));
        } else if (ctx.options.borderMode === 'reflect') {
          row = reflectCoordinate(row, CROP_SIZE);
          column = reflectCoordinate(column, CROP_SIZE);
        } else if (row < 0 || row >= CROP_SIZE || column < 0 || column >= CROP_SIZE) {
          continue;
        }
        availableSourceCells++;
        if (sources[ctx.options.date].validity[row * CROP_SIZE + column]) validSourceCells++;
      }
    }
    ctx.setReadout(
      'selectedEvidence',
      `cell row ${centerRow}, col ${centerColumn} · ${shownLayer} ${Number.isFinite(outputValue) ? outputValue.toFixed(6) : 'no data'} · NIR support ${validSourceCells}/${availableSourceCells} source cells · source − Gaussian ${Number.isFinite(smoothedValue) ? (sources[ctx.options.date].nir[selectedCell] - smoothedValue).toFixed(6) : 'undefined'}`
    );
    const source = sources[ctx.options.date];
    const predictorSamples: number[] = [];
    const responseSamples: number[] = [];
    const radius = ctx.options.radius;
    for (let rowOffset = -radius; rowOffset <= radius; rowOffset++) {
      for (let columnOffset = -radius; columnOffset <= radius; columnOffset++) {
        let row = centerRow + rowOffset;
        let column = centerColumn + columnOffset;
        if (ctx.options.borderMode === 'clamp') {
          row = Math.max(0, Math.min(CROP_SIZE - 1, row));
          column = Math.max(0, Math.min(CROP_SIZE - 1, column));
        } else if (ctx.options.borderMode === 'reflect') {
          row = reflectCoordinate(row, CROP_SIZE);
          column = reflectCoordinate(column, CROP_SIZE);
        } else if (row < 0 || row >= CROP_SIZE || column < 0 || column >= CROP_SIZE) {
          continue;
        }
        const index = row * CROP_SIZE + column;
        if (
          source.validity[index] &&
          Number.isFinite(source.nir[index]) &&
          Number.isFinite(source.swir[index])
        ) {
          predictorSamples.push(source.nir[index]);
          responseSamples.push(source.swir[index]);
        }
      }
    }
    const slope = Number(focalValue('slope'));
    const intercept = Number(focalValue('intercept'));
    const fit =
      focalStatus === 0 && Number.isFinite(slope) && Number.isFinite(intercept)
        ? {slope, intercept, label: 'GPU local OLS'}
        : undefined;
    ctx.setChart('pairChart', {
      kind: 'scatter',
      title: `Selected ${radius * 2 + 1} × ${radius * 2 + 1} NIR / SWIR₂ pairs`,
      x: predictorSamples,
      y: responseSamples,
      xLabel: 'NIR reflectance (unitless)',
      yLabel: 'SWIR₂ reflectance (unitless)',
      description:
        'Bounded source pairs use the GPU radius, border mapping, and valid support; the fit line is the GPU result.',
      ...(fit ? {fit} : {})
    });
  };
  function writeTextureDisplay(variant: Variant, metric: SpectralTextureOptions['metric']): void {
    const summary = variant.summary;
    if (!summary) return;
    const metricIndex = TEXTURE_METRICS.indexOf(metric);
    const displayValues = new Float32Array(CROP_SIZE * CROP_SIZE);
    for (let cell = 0; cell < displayValues.length; cell++)
      displayValues[cell] = summary.texture[cell * 7 + metricIndex];
    variant.outputs.textureDisplay.write(displayValues);
  }

  active = await compileVariant(ctx.options);
  ctx.setStatus('');
  const validCount = sources[ctx.options.date].validity.reduce((sum, value) => sum + value, 0);
  ctx.setReadout(
    'sourceSupport',
    `${ctx.options.date === 'before' ? 'July 13 ignition day' : 'September 21'} strict QA · ${validCount.toLocaleString()} valid crop cells / 16,384`
  );

  return {
    getCompiledGraphs: () => [active.compiled],
    encode(commandEncoder) {
      if (dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        active.reader.markStale();
      }
      active.reader.flush(commandEncoder);
    },
    getLayers() {
      const isCurrent = active.optionsKey === getCompileKey(ctx.options);
      if (!isCurrent) {
        return [
          new SpatialAnalysisSegmentLayer({
            id: 'spectral-profile-line',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: profileDisplayLine,
            instanceCount: 2,
            widthPixels: 3,
            color: [27, 231, 211, 255],
            outlineColor: [20, 37, 43, 245],
            outlineWidthPixels: 1.5
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'spectral-texture-crop-outline',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: cropOutlineBuffer,
            instanceCount: 4,
            widthPixels: 1.5,
            color: [255, 247, 220, 245],
            outlineColor: [35, 40, 45, 240],
            outlineWidthPixels: 1
          }),
          new SpatialAnalysisPointLayer({
            id: 'spectral-texture-selected',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            positions: marker,
            instanceCount: 1,
            radiusPixels: 7,
            color: [30, 34, 39, 255]
          })
        ];
      }
      let values: Buffer = active.outputs.values.source;
      let valueRange: readonly [number, number] = [0, 0.6];
      let colormap: SpatialAnalysisColormap = 'ylorrd';
      let palette: readonly (readonly [number, number, number, number])[] | undefined;
      if (ctx.options.layer === 'display') {
        values = active.diagnostics.transformed;
        valueRange =
          ctx.options.displayDomain === 'gpu' && active.summary
            ? [active.summary.diagnostics.gpuDomain[0], active.summary.diagnostics.gpuDomain[1]]
            : [0, 0.6];
        colormap = 'cividis';
      } else if (ctx.options.layer === 'transport') {
        values = active.diagnostics.returnedValues;
        valueRange = [0, 0.6];
        colormap = 'ylorrd';
      } else if (ctx.options.layer === 'focal') {
        const focalLayer = ctx.options.focalLayer;
        values = active.outputs.focal[focalLayer];
        const style = getSpectralLayerStyle(focalLayer, ctx.options.radius);
        valueRange = style.extent;
        colormap = style.ramp;
        palette = style.palette;
      } else if (ctx.options.layer === 'texture') {
        values = active.outputs.textureDisplay;
        const textureRange = getSpectralTextureRange(
          ctx.options.metric,
          Number(ctx.options.textureLevels),
          ctx.options.textureDisplayDomain
        );
        valueRange = textureRange.extent;
        colormap = textureRange.ramp;
      } else if (ctx.options.layer !== 'source') {
        values = active.outputs.values[ctx.options.layer];
        const style = getSpectralLayerStyle(ctx.options.layer);
        valueRange = style.extent;
        colormap = style.ramp;
      }
      const layers: Layer[] = [
        new UtmRasterLayer({
          id: 'spectral-texture-map',
          frame,
          values,
          valueFormat: 'float32',
          valueRange,
          colormap,
          ...(palette ? {palette: [...palette]} : {}),
          opacity: 0.94,
          noDataColor: [0, 0, 0, 0]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'spectral-profile-line',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: profileDisplayLine,
          instanceCount: 2,
          widthPixels: 3,
          color: [27, 231, 211, 255],
          outlineColor: [20, 37, 43, 245],
          outlineWidthPixels: 1.5
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'spectral-texture-crop-outline',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: cropOutlineBuffer,
          instanceCount: 4,
          widthPixels: 1.5,
          color: [255, 247, 220, 245],
          outlineColor: [35, 40, 45, 240],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisPointLayer({
          id: 'spectral-texture-selected',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          positions: marker,
          instanceCount: 1,
          radiusPixels: 7,
          color: [30, 34, 39, 255]
        })
      ];
      return layers;
    },
    setOption(id) {
      if (COMPILE_OPTIONS.has(id)) {
        const nextOptions = {...ctx.options};
        ctx.setLegendData('displayDomain', null);
        ctx.setStatus('Rebuilding the selected raster recipe');
        ctx.setReadout('sourceSupport', 'Acquisition support pending · graph rebuild in progress.');
        ctx.setReadout(
          'selectedEvidence',
          'Raster recipe pending · evidence updates when the graph swap commits.'
        );
        ctx.setReadout(
          'textureEvidence',
          'Texture evidence pending · evidence updates when the graph swap commits.'
        );
        ctx.setReadout(
          'focalEvidence',
          'Focal evidence pending · evidence updates when the graph swap commits.'
        );
        ctx.setReadout('displayEvidence', 'Display transform pending · graph rebuild in progress.');
        ctx.setReadout('profileEvidence', 'Transect profile pending · graph rebuild in progress.');
        ctx.setReadout(
          'transportEvidence',
          'Texture transport pending · graph rebuild in progress.'
        );
        ctx.setChart('pairChart', null);
        ctx.setChart('displayChart', null);
        ctx.setChart('profileChart', null);
        ctx.refreshTooltip();
        void graphSwap
          .request({
            build: () => compileVariant(nextOptions),
            commit: next => {
              const previous = active;
              active = next;
              previous.reader.stop();
              previous.resources.destroy();
              dirty = true;
              ctx.setReadout(
                'sourceSupport',
                `${nextOptions.date === 'before' ? 'July 13 ignition day' : 'September 21'} strict QA · ${sources[nextOptions.date].validity.reduce((sum, value) => sum + value, 0).toLocaleString()} valid crop cells / 16,384`
              );
              ctx.setStatus('');
              publishReadouts();
              ctx.requestLayers();
            },
            discard: next => next.resources.destroy()
          })
          .catch(error => ctx.setStatus(`Raster recipe rebuild failed: ${String(error)}`));
      } else if (id === 'metric' || id === 'layer' || id === 'focalLayer') {
        if (id === 'metric') writeTextureDisplay(active, ctx.options.metric);
        publishReadouts();
      }
      ctx.requestLayers();
    },
    getTooltip(event) {
      if (active.optionsKey !== getCompileKey(ctx.options)) return null;
      if (!active.summary) return null;
      const cell = getCell(event.coordinate);
      if (cell < 0) return null;
      const value = sources[ctx.options.date].nir[cell];
      const focalStatus = active.summary?.focal.status?.[cell] ?? Number.NaN;
      const textureValue =
        active.summary?.texture[cell * 7 + TEXTURE_METRICS.indexOf(ctx.options.metric)] ??
        Number.NaN;
      const displayValue = active.summary.diagnostics.transformed[cell];
      const returnedValue = active.summary.diagnostics.returnedValues[cell];
      return `Crop row ${Math.floor(cell / CROP_SIZE)}, column ${cell % CROP_SIZE} · ${ctx.options.date === 'before' ? '2021-07-13 ignition day' : '2021-09-21'}\nMeasured NIR ${Number.isFinite(value) ? value.toFixed(6) : 'no data'} reflectance\nTransformed display ${Number.isFinite(displayValue) ? displayValue.toFixed(6) : 'no data'} · returned buffer ${Number.isFinite(returnedValue) ? returnedValue.toFixed(6) : 'no data'}\n${ctx.options.metric} ${Number.isFinite(textureValue) ? textureValue.toFixed(6) : 'undefined'} · ${active.summary?.texturePairs[cell] ?? '…'} pairs\nFocal status ${Number.isFinite(Number(focalStatus)) ? focalStatus : 'pending'}`;
    },
    onClick(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return false;
      selectedCell = cell;
      writeMarker();
      publishReadouts();
      ctx.requestLayers();
      return true;
    },
    destroy() {
      active.reader.stop();
      active.resources.destroy();
      rootResources.destroy();
    }
  };

  function classifyFocalStatus(statusMask: number): number {
    if (statusMask === 0) return 0;
    if (statusMask === 1) return 1;
    if (statusMask === 2) return 2;
    if (statusMask === 4) return 3;
    if (statusMask === 8) return 4;
    if (statusMask === 16) return 5;
    return 6;
  }

  function reflectCoordinate(value: number, length: number): number {
    const period = (length - 1) * 2;
    const reflected = ((value % period) + period) % period;
    return reflected >= length ? period - reflected : reflected;
  }
}
