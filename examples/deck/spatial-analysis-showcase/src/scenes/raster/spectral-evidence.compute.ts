// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURasterEVI,
  GPURasterNBR,
  GPURasterNDWI,
  GPURasterSAVI,
  GPURasterSpectralUnmixing
} from '@luma-contributors/analysis/raster-analysis';
import {GPURasterNDVI, type GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {GraphSwap} from '../../engine/graph-swap';
import {addKernelPass} from '../../engine/mode-kernels';
import {SummaryReader} from '../../engine/summary-reader';
import {fetchBytes} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {makeDixieQuadrantIds, addDixieRegionalSupport} from './spectral-regions';
import {getGPUChangeDetectionParameterValues} from '@luma-contributors/analysis/raster-analysis';

const DATASET_ID = 'dixie-spectral';
const SOURCE_WIDTH = 250;
const CROP_ROW = 50;
const CROP_COLUMN = 101;
const CROP_SIZE = 64;
const CELL_SIZE = 60;
const CELL_COUNT = CROP_SIZE * CROP_SIZE;
const BAND_NAMES = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22'] as const;
const INDEX_NAMES = ['ndvi', 'nbr', 'evi', 'savi', 'ndwi-gao', 'ndwi-mcfeeters'] as const;
const SAVI_LEVELS = ['0', '0.5', '1'] as const;
const INDEX_OUTPUT_NAMES = [
  'ndvi',
  'nbr',
  'evi',
  'ndwi-gao',
  'ndwi-mcfeeters',
  ...SAVI_LEVELS.map(level => `savi-${level}`)
] as const;
const MODEL_NAMES = ['unconstrained', 'nonnegative', 'simplex'] as const;
const EXEMPLAR_NAMES = ['vegetation-like', 'dark low-NBR', 'bright low-NDVI'] as const;
const EXEMPLAR_CELLS = [8 * CROP_SIZE + 1, 30 * CROP_SIZE + 18, 57 * CROP_SIZE + 38];
const PROBE_CELL = 56 * CROP_SIZE + 32;

export type SpectralEvidenceOptions = {
  date: 'before' | 'after';
  product: (typeof INDEX_NAMES)[number] | 'abundance' | 'residual';
  saviL: (typeof SAVI_LEVELS)[number];
  indexEpsilon: '0.000001' | '0.01' | '0.1';
  indexClamp: 'none' | 'unit-range' | 'sensitivity';
  constraint: (typeof MODEL_NAMES)[number];
  component: '0' | '1' | '2';
  maximumIterations: '64' | '256';
  tolerance: '0.001' | '0.0001';
  regionView: 'indices' | 'distribution' | 'change';
  selectedRegion: '0' | '1' | '2' | '3';
  changeThreshold: '0.02' | '0.05' | '0.10' | '0.20';
};

type IndexBuffers = {values: Buffer; validity: Buffer};
type FitBuffers = {
  abundances: Buffer;
  residual: Buffer;
  status: Buffer;
  iterations: Buffer;
  componentMaps: [Buffer, Buffer, Buffer];
};
type CompiledFit = {graph: CompiledGPUCommandGraph<void>};
type SpectralSummary = {
  indices: Record<string, Float32Array>;
  indexValidity: Record<string, Uint32Array>;
  fits: Record<
    string,
    {abundances: Float32Array; residual: Float32Array; status: Uint32Array; iterations: Uint32Array}
  >;
};

const INDEX_LABELS: Record<(typeof INDEX_NAMES)[number], string> = {
  ndvi: 'NDVI · NIR vs red',
  nbr: 'NBR · NIR vs SWIR2',
  evi: 'EVI · blue correction',
  savi: 'SAVI',
  'ndwi-gao': 'Gao-style · NIR vs SWIR1',
  'ndwi-mcfeeters': 'McFeeters · green vs NIR'
};

const MODEL_LABELS: Record<(typeof MODEL_NAMES)[number], string> = {
  unconstrained: 'Unconstrained least squares',
  nonnegative: 'Nonnegative fit',
  simplex: 'Simplex fit · weights sum to one'
};

/** Six-band index and endmember evidence from a small, exact UTM crop of the Dixie observations. */
export async function createSpectralEvidence(
  ctx: SceneContext<SpectralEvidenceOptions>
): Promise<SceneInstance<SpectralEvidenceOptions>> {
  const dataset = ctx.datasets.get(DATASET_ID);
  ctx.setStatus('Loading calibrated six-band Sentinel observations and strict source masks');
  const [beforeBands, afterBands, beforeMaskBytes, afterMaskBytes, afterLandMaskBytes] =
    await Promise.all([
      Promise.all(BAND_NAMES.map(name => dataset.loadRaster(`${name}_before`, ctx.signal))),
      Promise.all(BAND_NAMES.map(name => dataset.loadRaster(`${name}_after`, ctx.signal))),
      fetchBytes(dataset.fileUrl('validity_before.bin'), ctx.signal),
      fetchBytes(dataset.fileUrl('validity_after.bin'), ctx.signal),
      fetchBytes(dataset.fileUrl('land_valid_after.bin'), ctx.signal)
    ]);
  ctx.signal.throwIfAborted();
  const sourceBands = {before: beforeBands, after: afterBands};
  const sourceMasks = {
    before: new Uint8Array(beforeMaskBytes),
    after: new Uint8Array(afterMaskBytes)
  };
  const afterLandMask = new Uint8Array(afterLandMaskBytes);
  const cropBands = {} as Record<
    'before' | 'after',
    Record<(typeof BAND_NAMES)[number], Float32Array>
  >;
  const validCells = {} as Record<'before' | 'after', Uint32Array>;
  const packedSpectra = {} as Record<'before' | 'after', Float32Array>;
  for (const date of ['before', 'after'] as const) {
    cropBands[date] = {} as Record<(typeof BAND_NAMES)[number], Float32Array>;
    validCells[date] = new Uint32Array(CELL_COUNT);
    packedSpectra[date] = new Float32Array(CELL_COUNT * BAND_NAMES.length);
    for (const bandName of BAND_NAMES) cropBands[date][bandName] = new Float32Array(CELL_COUNT);
    for (let row = 0; row < CROP_SIZE; row++) {
      for (let column = 0; column < CROP_SIZE; column++) {
        const cell = row * CROP_SIZE + column;
        const sourceCell = (CROP_ROW + row) * SOURCE_WIDTH + CROP_COLUMN + column;
        const valid =
          sourceMasks[date][sourceCell] !== 0 &&
          sourceBands[date].every(band => Number.isFinite(band.values[sourceCell]));
        for (let band = 0; band < BAND_NAMES.length; band++) {
          const value = valid ? sourceBands[date][band].values[sourceCell] : Number.NaN;
          cropBands[date][BAND_NAMES[band]][cell] = value;
          packedSpectra[date][cell * BAND_NAMES.length + band] = value;
        }
        validCells[date][cell] = valid ? 1 : 0;
      }
    }
  }

  const rasterSpec = dataset.rasterSpecs.nir_before;
  const bounds = rasterSpec?.boundsProjected;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 4 ||
    bounds.some(value => typeof value !== 'number')
  )
    throw new Error('Dixie spectra require numeric projected bounds');
  const minimumEasting = bounds[0] + CROP_COLUMN * CELL_SIZE;
  const maximumNorthing = bounds[3] - CROP_ROW * CELL_SIZE;
  const cropBounds: readonly [number, number, number, number] = [
    minimumEasting,
    maximumNorthing - CROP_SIZE * CELL_SIZE,
    minimumEasting + CROP_SIZE * CELL_SIZE,
    maximumNorthing
  ];
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: cropBounds,
    cellSizeM: CELL_SIZE,
    width: CROP_SIZE,
    height: CROP_SIZE
  });
  const resources = new SpatialAnalysisResources(ctx.device, 'spectral-evidence');
  const inputBuffers = {} as Record<
    'before' | 'after',
    Record<(typeof BAND_NAMES)[number], Buffer>
  >;
  const spectraBuffers = {} as Record<'before' | 'after', Buffer>;
  for (const date of ['before', 'after'] as const) {
    inputBuffers[date] = {} as Record<(typeof BAND_NAMES)[number], Buffer>;
    for (const bandName of BAND_NAMES)
      inputBuffers[date][bandName] = resources.createBuffer(
        `${date}-${bandName}`,
        cropBands[date][bandName]
      );
    spectraBuffers[date] = resources.createBuffer(`${date}-packed-spectra`, packedSpectra[date]);
  }

  // Endmembers are measured spectra at fixed, displayed pixels in the September scene.
  // Descriptive spectral labels do not identify validated materials or land-cover fractions.
  const endmembers = new Float32Array(EXEMPLAR_CELLS.length * BAND_NAMES.length);
  for (let exemplar = 0; exemplar < EXEMPLAR_CELLS.length; exemplar++) {
    const cell = EXEMPLAR_CELLS[exemplar];
    const row = Math.floor(cell / CROP_SIZE);
    const column = cell % CROP_SIZE;
    const sourceCell = (CROP_ROW + row) * SOURCE_WIDTH + CROP_COLUMN + column;
    // These fixed measured locations are used as exemplars only while their pinned land mask is valid.
    if (!afterLandMask[sourceCell] || !validCells.after[cell])
      throw new Error('A selected Dixie spectral exemplar must have valid September land support');
    for (let band = 0; band < BAND_NAMES.length; band++)
      endmembers[exemplar * BAND_NAMES.length + band] =
        packedSpectra.after[cell * BAND_NAMES.length + band];
  }
  const endmemberBuffer = resources.createBuffer('measured-endmembers', endmembers);

  const indexBuffers = {} as Record<string, IndexBuffers>;
  for (const date of ['before', 'after'] as const)
    for (const indexName of INDEX_OUTPUT_NAMES)
      indexBuffers[`${date}-${indexName}`] = {
        values: resources.createBuffer(`${date}-${indexName}`, CELL_COUNT * 4),
        validity: resources.createBuffer(`${date}-${indexName}-validity`, CELL_COUNT * 4)
      };
  const buildIndexGraph = async (
    options: SpectralEvidenceOptions
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const indexGraph = new GPUCommandGraph<void>(ctx.device, {id: 'spectral-evidence-indices'});
    const indexView = createGraphViewer(indexGraph);
    const inputBand = (
      date: 'before' | 'after',
      name: (typeof BAND_NAMES)[number]
    ): GPURasterBufferBand<'float32'> => ({
      id: `${date}-${name}`,
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: indexView(`${date}-${name}`, inputBuffers[date][name], 'float32', CELL_COUNT)
      }
    });
    for (const date of ['before', 'after'] as const) {
      const bands = Object.fromEntries(
        BAND_NAMES.map(name => [name, inputBand(date, name)])
      ) as Record<(typeof BAND_NAMES)[number], GPURasterBufferBand<'float32'>>;
      const addIndex = (
        name: (typeof INDEX_OUTPUT_NAMES)[number],
        add: (output: IndexBuffers) => void
      ) => add(indexBuffers[`${date}-${name}`]);
      addIndex('ndvi', output =>
        new GPURasterNDVI({
          id: `${date}-ndvi`,
          width: CROP_SIZE,
          height: CROP_SIZE,
          nearInfrared: bands.nir,
          red: bands.red,
          epsilon: Number(options.indexEpsilon),
          clamp:
            options.indexClamp === 'none'
              ? undefined
              : options.indexClamp === 'unit-range'
                ? [-1, 1]
                : [-0.25, 0.25],
          output: indexView(`${date}-ndvi-values`, output.values, 'float32', CELL_COUNT),
          outputValidity: indexView(`${date}-ndvi-validity`, output.validity, 'uint32', CELL_COUNT)
        }).addToGraph(indexGraph)
      );
      addIndex('nbr', output =>
        new GPURasterNBR({
          id: `${date}-nbr`,
          width: CROP_SIZE,
          height: CROP_SIZE,
          nearInfrared: bands.nir,
          shortwaveInfrared2: bands.swir22,
          epsilon: Number(options.indexEpsilon),
          clamp:
            options.indexClamp === 'none'
              ? undefined
              : options.indexClamp === 'unit-range'
                ? [-1, 1]
                : [-0.25, 0.25],
          output: indexView(`${date}-nbr-values`, output.values, 'float32', CELL_COUNT),
          outputValidity: indexView(`${date}-nbr-validity`, output.validity, 'uint32', CELL_COUNT)
        }).addToGraph(indexGraph)
      );
      addIndex('evi', output =>
        new GPURasterEVI({
          id: `${date}-evi`,
          width: CROP_SIZE,
          height: CROP_SIZE,
          nearInfrared: bands.nir,
          red: bands.red,
          blue: bands.blue,
          epsilon: Number(options.indexEpsilon),
          clamp:
            options.indexClamp === 'none'
              ? undefined
              : options.indexClamp === 'unit-range'
                ? [-1, 1]
                : [-0.25, 0.25],
          output: indexView(`${date}-evi-values`, output.values, 'float32', CELL_COUNT),
          outputValidity: indexView(`${date}-evi-validity`, output.validity, 'uint32', CELL_COUNT)
        }).addToGraph(indexGraph)
      );
      for (const soilAdjustment of SAVI_LEVELS) {
        const name = `savi-${soilAdjustment}` as (typeof INDEX_OUTPUT_NAMES)[number];
        addIndex(name, output =>
          new GPURasterSAVI({
            id: `${date}-${name}`,
            width: CROP_SIZE,
            height: CROP_SIZE,
            nearInfrared: bands.nir,
            red: bands.red,
            soilAdjustment: Number(soilAdjustment),
            epsilon: Number(options.indexEpsilon),
            clamp:
              options.indexClamp === 'none'
                ? undefined
                : options.indexClamp === 'unit-range'
                  ? [-1, 1]
                  : [-0.25, 0.25],
            output: indexView(`${date}-${name}-values`, output.values, 'float32', CELL_COUNT),
            outputValidity: indexView(
              `${date}-${name}-validity`,
              output.validity,
              'uint32',
              CELL_COUNT
            )
          }).addToGraph(indexGraph)
        );
      }
      addIndex('ndwi-gao', output =>
        new GPURasterNDWI({
          id: `${date}-ndwi-gao-adaptation`,
          method: 'gao',
          width: CROP_SIZE,
          height: CROP_SIZE,
          nearInfrared: bands.nir,
          shortwaveInfrared: bands.swir16,
          epsilon: Number(options.indexEpsilon),
          clamp:
            options.indexClamp === 'none'
              ? undefined
              : options.indexClamp === 'unit-range'
                ? [-1, 1]
                : [-0.25, 0.25],
          output: indexView(`${date}-ndwi-gao-values`, output.values, 'float32', CELL_COUNT),
          outputValidity: indexView(
            `${date}-ndwi-gao-validity`,
            output.validity,
            'uint32',
            CELL_COUNT
          )
        }).addToGraph(indexGraph)
      );
      addIndex('ndwi-mcfeeters', output =>
        new GPURasterNDWI({
          id: `${date}-ndwi-mcfeeters`,
          method: 'mcfeeters',
          width: CROP_SIZE,
          height: CROP_SIZE,
          green: bands.green,
          nearInfrared: bands.nir,
          epsilon: Number(options.indexEpsilon),
          clamp:
            options.indexClamp === 'none'
              ? undefined
              : options.indexClamp === 'unit-range'
                ? [-1, 1]
                : [-0.25, 0.25],
          output: indexView(`${date}-ndwi-mcfeeters-values`, output.values, 'float32', CELL_COUNT),
          outputValidity: indexView(
            `${date}-ndwi-mcfeeters-validity`,
            output.validity,
            'uint32',
            CELL_COUNT
          )
        }).addToGraph(indexGraph)
      );
    }
    return indexGraph.compileAsync();
  };
  let indexCompiled = resources.track(await buildIndexGraph(ctx.options));
  const indexSwap = resources.track(new GraphSwap());

  const fitBuffers = {} as Record<string, FitBuffers>;
  for (const date of ['before', 'after'] as const)
    for (const model of MODEL_NAMES) {
      fitBuffers[`${date}-${model}`] = {
        abundances: resources.createBuffer(
          `${date}-${model}-abundances`,
          CELL_COUNT * EXEMPLAR_CELLS.length * 4
        ),
        residual: resources.createBuffer(`${date}-${model}-residual`, CELL_COUNT * 4),
        status: resources.createBuffer(`${date}-${model}-status`, CELL_COUNT * 4),
        iterations: resources.createBuffer(`${date}-${model}-iterations`, CELL_COUNT * 4),
        componentMaps: [
          resources.createBuffer(`${date}-${model}-component-0`, CELL_COUNT * 4),
          resources.createBuffer(`${date}-${model}-component-1`, CELL_COUNT * 4),
          resources.createBuffer(`${date}-${model}-component-2`, CELL_COUNT * 4)
        ]
      };
    }
  let graphSerial = 0;
  const makeModelGraph = async (options: SpectralEvidenceOptions): Promise<CompiledFit> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `spectral-evidence-model-${graphSerial++}`
    });
    const view = createGraphViewer(graph);
    const endmemberView = view(
      'measured-endmembers',
      endmemberBuffer,
      'float32',
      endmembers.length
    );
    for (const date of ['before', 'after'] as const) {
      for (const model of MODEL_NAMES) {
        const buffers = fitBuffers[`${date}-${model}`];
        const abundanceView = view(
          `${date}-${model}-abundances`,
          buffers.abundances,
          'float32',
          CELL_COUNT * EXEMPLAR_CELLS.length
        );
        graph.add(
          new GPURasterSpectralUnmixing({
            id: `${date}-${model}-spectral-fit`,
            cellCount: CELL_COUNT,
            bandCount: BAND_NAMES.length,
            endmemberCount: EXEMPLAR_CELLS.length,
            values: view(
              `${date}-packed-spectra`,
              spectraBuffers[date],
              'float32',
              CELL_COUNT * BAND_NAMES.length
            ),
            endmembers: endmemberView,
            constraint: model,
            maximumIterations: Number(options.maximumIterations),
            tolerance: Number(options.tolerance),
            output: {
              abundances: abundanceView,
              residual: view(`${date}-${model}-residual`, buffers.residual, 'float32', CELL_COUNT),
              status: view(`${date}-${model}-status`, buffers.status, 'uint32', CELL_COUNT),
              iterationCounts: view(
                `${date}-${model}-iterations`,
                buffers.iterations,
                'uint32',
                CELL_COUNT
              )
            }
          })
        );
        for (let component = 0; component < EXEMPLAR_CELLS.length; component++) {
          const mapView = view(
            `${date}-${model}-component-${component}`,
            buffers.componentMaps[component],
            'float32',
            CELL_COUNT
          );
          addKernelPass(graph, {
            id: `${date}-${model}-component-map-${component}`,
            invocationCount: CELL_COUNT,
            bindings: [
              {name: 'abundances', view: abundanceView, type: 'f32', access: 'read'},
              {
                name: 'status',
                view: view(`${date}-${model}-status`, buffers.status, 'uint32', CELL_COUNT),
                type: 'u32',
                access: 'read'
              },
              {name: 'map', view: mapView, type: 'f32', access: 'read_write'}
            ],
            body: `let fitStatus = status[statusOffset + index];\nlet missing = bitcast<f32>(0x7fc00000u | (index & 0u));\nmap[mapOffset + index] = select(missing, abundances[abundancesOffset + index * ${EXEMPLAR_CELLS.length}u + ${component}u], fitStatus == 0u || fitStatus == 4u);`
          });
        }
      }
    }
    return {graph: await graph.compileAsync()};
  };
  let modelCompiled = resources.track((await makeModelGraph(ctx.options)).graph);
  ctx.setStatus('');
  const modelSwap = resources.track(new GraphSwap());
  const regionZoneLabels = makeDixieQuadrantIds();
  const regionZoneBuffer = resources.createBuffer('regional-quadrant-ids', regionZoneLabels);
  const regionAreaValues = new Float32Array(CELL_COUNT).fill(CELL_SIZE * CELL_SIZE);
  const regionAreaBuffer = resources.createBuffer('regional-cell-areas', regionAreaValues);
  const regionProbabilitiesBuffer = resources.createBuffer(
    'regional-quantile-probabilities',
    Float32Array.of(0, 0.25, 0.5, 0.75, 1)
  );
  const regionThresholdBuffer = resources.createBuffer(
    'regional-change-threshold',
    Float32Array.of(Number(ctx.options.changeThreshold))
  );
  const regionChangeParametersBuffer = resources.createBuffer(
    'regional-change-parameters',
    getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1})
  );
  const regionSlices = new Float32Array(CELL_COUNT * 2);
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    regionSlices[cell * 2] = cropBands.before.nir[cell];
    regionSlices[cell * 2 + 1] = cropBands.after.nir[cell];
  }
  const regionSlicesBuffer = resources.createBuffer('regional-nir-slices', regionSlices);
  const regionValidityBuffers = {
    before: resources.createBuffer('regional-before-validity', validCells.before),
    after: resources.createBuffer('regional-after-validity', validCells.after)
  };
  const regionalBuffers = {} as Record<
    'before' | 'after',
    {
      rowCounts: Buffer;
      validCounts: Buffer;
      medians: Buffer;
      modes: Buffer;
      uniqueCounts: Buffer;
      quantiles: Buffer;
    }
  >;
  for (const date of ['before', 'after'] as const) {
    regionalBuffers[date] = {
      rowCounts: resources.createBuffer(`regional-${date}-row-counts`, 4 * 4),
      validCounts: resources.createBuffer(`regional-${date}-valid-counts`, 4 * 4),
      medians: resources.createBuffer(`regional-${date}-medians`, 4 * 4),
      modes: resources.createBuffer(`regional-${date}-modes`, 4 * 4),
      uniqueCounts: resources.createBuffer(`regional-${date}-unique-counts`, 4 * 4),
      quantiles: resources.createBuffer(`regional-${date}-quantiles`, 4 * 5 * 4)
    };
  }
  const regionalDifferenceBuffer = resources.createBuffer(
    'regional-difference-map',
    Float32Array.from({length: CELL_COUNT}, () => Number.NaN)
  );
  const regionalChangeBuffers = {
    totalArea: resources.createBuffer('regional-total-area', 4 * 4),
    validArea: resources.createBuffer('regional-valid-area', 4 * 4),
    changedArea: resources.createBuffer('regional-changed-area', 4 * 4),
    coverage: resources.createBuffer('regional-coverage', 4 * 4),
    meanChanges: resources.createBuffer('regional-mean-changes', 4 * 4),
    validCounts: resources.createBuffer('regional-change-valid-counts', 4 * 4),
    thresholdInvalid: resources.createBuffer('regional-threshold-invalid', 4),
    stageOverflow: resources.createBuffer('regional-change-stage-status', 5 * 4)
  };
  const regionGraph = new GPUCommandGraph<void>(ctx.device, {id: 'spectral-evidence-regions'});
  const regionView = createGraphViewer(regionGraph);
  const makeRegionBand = (date: 'before' | 'after'): GPURasterBufferBand<'float32'> => ({
    id: `regional-${date}-nir`,
    format: 'float32',
    storage: {
      kind: 'buffer',
      values: regionView(`regional-${date}-nir`, inputBuffers[date].nir, 'float32', CELL_COUNT)
    },
    validity: regionView(
      `regional-${date}-validity`,
      regionValidityBuffers[date],
      'uint32',
      CELL_COUNT
    )
  });
  const regionGrid = {
    width: CROP_SIZE,
    height: CROP_SIZE,
    affine: [CELL_SIZE, 0, cropBounds[0], 0, -CELL_SIZE, cropBounds[3]] as const,
    pixelInterpretation: 'area' as const,
    coordinateReferenceSystem: {authority: 'EPSG:32610'}
  };
  const regionZones = regionView('regional-quadrant-ids', regionZoneBuffer, 'uint32', CELL_COUNT);
  const regionProbabilities = regionView(
    'regional-quantile-probabilities',
    regionProbabilitiesBuffer,
    'float32',
    5
  );
  const regionOutputs = (date: 'before' | 'after') => ({
    rowCounts: regionView(
      `regional-${date}-row-counts`,
      regionalBuffers[date].rowCounts,
      'uint32',
      4
    ),
    validCounts: regionView(
      `regional-${date}-valid-counts`,
      regionalBuffers[date].validCounts,
      'uint32',
      4
    ),
    medians: regionView(`regional-${date}-medians`, regionalBuffers[date].medians, 'float32', 4),
    modes: regionView(`regional-${date}-modes`, regionalBuffers[date].modes, 'float32', 4),
    uniqueCounts: regionView(
      `regional-${date}-unique-counts`,
      regionalBuffers[date].uniqueCounts,
      'uint32',
      4
    ),
    quantiles: regionView(
      `regional-${date}-quantiles`,
      regionalBuffers[date].quantiles,
      'float32',
      20
    )
  });
  const beforeRegionOutputs = regionOutputs('before');
  const afterRegionOutputs = regionOutputs('after');
  const regionalMedianMaps = {
    before: resources.createBuffer('regional-before-median-map', CELL_COUNT * 4),
    after: resources.createBuffer('regional-after-median-map', CELL_COUNT * 4)
  };
  const regional = addDixieRegionalSupport(regionGraph, {
    grid: regionGrid,
    before: makeRegionBand('before'),
    after: makeRegionBand('after'),
    zones: regionZones,
    probabilities: regionProbabilities,
    beforeOutputs: beforeRegionOutputs,
    afterOutputs: afterRegionOutputs,
    change: {
      slices: regionView('regional-nir-slices', regionSlicesBuffer, 'float32', CELL_COUNT * 2),
      parameters: regionView(
        'regional-change-parameters',
        regionChangeParametersBuffer,
        'float32',
        5
      ),
      sliceCount: 2
    },
    cellAreas: regionView('regional-cell-areas', regionAreaBuffer, 'float32', CELL_COUNT),
    threshold: regionView('regional-change-threshold', regionThresholdBuffer, 'float32', 1)
  });
  for (const date of ['before', 'after'] as const) {
    addKernelPass(regionGraph, {
      id: `regional-${date}-median-map`,
      invocationCount: CELL_COUNT,
      bindings: [
        {name: 'zones', view: regionZones, type: 'u32', access: 'read'},
        {
          name: 'medians',
          view: date === 'before' ? beforeRegionOutputs.medians : afterRegionOutputs.medians,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: regionView(
            `regional-${date}-median-map`,
            regionalMedianMaps[date],
            'float32',
            CELL_COUNT
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: 'let zone = zones[zonesOffset + index]; output[outputOffset + index] = medians[mediansOffset + zone];'
    });
  }
  const regionalDifferenceOutput = regionView(
    'regional-difference-map',
    regionalDifferenceBuffer,
    'float32',
    CELL_COUNT
  );
  addKernelPass(regionGraph, {
    id: 'copy-regional-difference-for-map',
    invocationCount: CELL_COUNT,
    bindings: [
      {name: 'difference', view: regional.change.outputs.difference, type: 'f32', access: 'read'},
      {name: 'output', view: regionalDifferenceOutput, type: 'f32', access: 'read_write'}
    ],
    body: 'output[outputOffset + index] = difference[differenceOffset + index];'
  });
  const regionalChangeOutputViews = {
    totalArea: regionView('regional-total-area', regionalChangeBuffers.totalArea, 'float32', 4),
    validArea: regionView('regional-valid-area', regionalChangeBuffers.validArea, 'float32', 4),
    changedArea: regionView(
      'regional-changed-area',
      regionalChangeBuffers.changedArea,
      'float32',
      4
    ),
    coverage: regionView('regional-coverage', regionalChangeBuffers.coverage, 'float32', 4),
    meanChanges: regionView(
      'regional-mean-changes',
      regionalChangeBuffers.meanChanges,
      'float32',
      4
    ),
    validCounts: regionView(
      'regional-change-valid-counts',
      regionalChangeBuffers.validCounts,
      'uint32',
      4
    ),
    thresholdInvalid: regionView(
      'regional-threshold-invalid',
      regionalChangeBuffers.thresholdInvalid,
      'uint32',
      1
    )
  };
  const regionalStageStatusOutput = regionView(
    'regional-change-stage-status',
    regionalChangeBuffers.stageOverflow,
    'uint32',
    5
  );
  for (const name of [
    'totalArea',
    'validArea',
    'changedArea',
    'coverage',
    'meanChanges',
    'validCounts'
  ] as const) {
    addKernelPass(regionGraph, {
      id: `copy-regional-${name}`,
      invocationCount: 4,
      bindings: [
        {
          name: 'source',
          view: regional.change.outputs[name],
          type: name === 'validCounts' ? 'u32' : 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: regionalChangeOutputViews[name],
          type: name === 'validCounts' ? 'u32' : 'f32',
          access: 'read_write'
        }
      ],
      body: 'output[outputOffset + index] = source[sourceOffset + index];'
    });
  }
  addKernelPass(regionGraph, {
    id: 'copy-regional-threshold-status',
    invocationCount: 1,
    bindings: [
      {
        name: 'source',
        view: regional.change.outputs.thresholdInvalidCount,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'output',
        view: regionalChangeOutputViews.thresholdInvalid,
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: 'output[outputOffset] = source[sourceOffset];'
  });
  for (let stageIndex = 0; stageIndex < regional.change.status.stages.length; stageIndex++) {
    const stageStatus = regional.change.status.stages[stageIndex].status;
    const statusView = 'overflow' in stageStatus ? stageStatus.overflow : stageStatus.invalidCount;
    if (!statusView) continue;
    addKernelPass(regionGraph, {
      id: `copy-regional-stage-status-${stageIndex}`,
      invocationCount: 1,
      bindings: [
        {name: 'source', view: statusView, type: 'u32', access: 'read'},
        {name: 'output', view: regionalStageStatusOutput, type: 'u32', access: 'read_write'}
      ],
      body: `output[outputOffset + ${stageIndex}u] = source[sourceOffset];`
    });
  }
  let regionalCompiled = resources.track(await regionGraph.compileAsync());
  let regionalDirty = true;
  let regionalReady = false;
  let regionalRevision = 0;
  let requestedRegionalRevision = -1;
  let regionalQuantiles: {
    before: Float32Array;
    after: Float32Array;
    beforeRowCounts: Uint32Array;
    beforeValidCounts: Uint32Array;
    afterRowCounts: Uint32Array;
    afterValidCounts: Uint32Array;
    beforeMedians: Float32Array;
    afterMedians: Float32Array;
    beforeModes: Float32Array;
    afterModes: Float32Array;
    beforeUniqueCounts: Uint32Array;
    afterUniqueCounts: Uint32Array;
    totalAreas: Float32Array;
    validAreas: Float32Array;
    changedAreas: Float32Array;
    coverage: Float32Array;
    meanChanges: Float32Array;
    validChangeCounts: Uint32Array;
    thresholdInvalid: number;
    stageStatus: Uint32Array;
    differences: Float32Array;
  } | null = null;
  const regionalReader = new SummaryReader(
    resources,
    'spectral-regional-readback',
    [
      ...(['before', 'after'] as const).flatMap(date => [
        {buffer: regionalBuffers[date].rowCounts, size: 16},
        {buffer: regionalBuffers[date].validCounts, size: 16},
        {buffer: regionalBuffers[date].medians, size: 16},
        {buffer: regionalBuffers[date].modes, size: 16},
        {buffer: regionalBuffers[date].uniqueCounts, size: 16},
        {buffer: regionalBuffers[date].quantiles, size: 80}
      ]),
      {buffer: regionalChangeBuffers.totalArea, size: 16},
      {buffer: regionalChangeBuffers.validArea, size: 16},
      {buffer: regionalChangeBuffers.changedArea, size: 16},
      {buffer: regionalChangeBuffers.coverage, size: 16},
      {buffer: regionalChangeBuffers.meanChanges, size: 16},
      {buffer: regionalChangeBuffers.validCounts, size: 16},
      {buffer: regionalChangeBuffers.thresholdInvalid, size: 4},
      {buffer: regionalChangeBuffers.stageOverflow, size: 20},
      {buffer: regionalDifferenceBuffer, size: CELL_COUNT * 4}
    ],
    bytes => {
      if (requestedRegionalRevision !== regionalRevision) return;
      let offset = 0;
      const readUint = (length: number) => {
        const result = new Uint32Array(bytes, offset, length);
        offset += length * 4;
        return result;
      };
      const readFloat = (length: number) => {
        const result = new Float32Array(bytes, offset, length);
        offset += length * 4;
        return result;
      };
      const beforeRowCounts = readUint(4);
      const beforeValidCounts = readUint(4);
      const beforeMedians = readFloat(4);
      const beforeModes = readFloat(4);
      const beforeUniqueCounts = readUint(4);
      const before = readFloat(20);
      const afterRowCounts = readUint(4);
      const afterValidCounts = readUint(4);
      const afterMedians = readFloat(4);
      const afterModes = readFloat(4);
      const afterUniqueCounts = readUint(4);
      const after = readFloat(20);
      const totalAreas = readFloat(4);
      const validAreas = readFloat(4);
      const changedAreas = readFloat(4);
      const coverage = readFloat(4);
      const meanChanges = readFloat(4);
      const validChangeCounts = readUint(4);
      const thresholdInvalid = readUint(1)[0];
      const stageStatus = readUint(5);
      const differences = readFloat(CELL_COUNT);
      regionalQuantiles = {
        before,
        after,
        beforeRowCounts,
        beforeValidCounts,
        afterRowCounts,
        afterValidCounts,
        beforeMedians,
        afterMedians,
        beforeModes,
        afterModes,
        beforeUniqueCounts,
        afterUniqueCounts,
        totalAreas,
        changedAreas,
        validAreas,
        coverage,
        meanChanges,
        validChangeCounts,
        thresholdInvalid,
        stageStatus,
        differences
      };
      if (thresholdInvalid !== 0 || stageStatus.some(status => status !== 0)) {
        regionalQuantiles = null;
        regionalReady = false;
        ctx.setReadout(
          'regionalEvidence',
          `Regional GPU outputs suppressed · stage overflow/status ${Array.from(stageStatus).join('/')} · invalid cutoff ${thresholdInvalid}`
        );
        ctx.setChart('regionalChart', null);
        ctx.requestLayers();
        return;
      }
      regionalReady = true;
      publishRegionalEvidence();
      ctx.requestLayers();
    }
  );
  const quadrantSegmentBuffer = resources.createBuffer(
    'regional-quadrant-boundaries',
    Float32Array.of(
      ...frame.cellToMeters(-0.5, 31.5),
      ...frame.cellToMeters(63.5, 31.5),
      ...frame.cellToMeters(31.5, -0.5),
      ...frame.cellToMeters(31.5, 63.5)
    )
  );
  const makeQuadrantOutline = (region: number): Float32Array => {
    const rowStart = region >= 2 ? 32 : 0;
    const rowEnd = rowStart + 31;
    const columnStart = region % 2 ? 32 : 0;
    const columnEnd = columnStart + 31;
    const northwest = frame.cellToMeters(columnStart - 0.5, rowStart - 0.5);
    const northeast = frame.cellToMeters(columnEnd + 0.5, rowStart - 0.5);
    const southeast = frame.cellToMeters(columnEnd + 0.5, rowEnd + 0.5);
    const southwest = frame.cellToMeters(columnStart - 0.5, rowEnd + 0.5);
    return Float32Array.of(
      ...northwest,
      ...northeast,
      ...northeast,
      ...southeast,
      ...southeast,
      ...southwest,
      ...southwest,
      ...northwest
    );
  };
  const selectedQuadrantOutline = resources.createBuffer(
    'regional-selected-quadrant-outline',
    makeQuadrantOutline(Number(ctx.options.selectedRegion))
  );
  const publishRegionalEvidence = () => {
    if (!regionalQuantiles) return;
    const region = Number(ctx.options.selectedRegion);
    const names = ['north-west', 'north-east', 'south-west', 'south-east'];
    const offset = region * 5;
    const beforeQuantiles = Array.from(regionalQuantiles.before.slice(offset, offset + 5));
    const afterQuantiles = Array.from(regionalQuantiles.after.slice(offset, offset + 5));
    const totalArea = regionalQuantiles.totalAreas[region];
    const beforeRows = regionalQuantiles.beforeRowCounts[region];
    const beforeValid = regionalQuantiles.beforeValidCounts[region];
    const afterRows = regionalQuantiles.afterRowCounts[region];
    const afterValid = regionalQuantiles.afterValidCounts[region];
    const beforeMedian = regionalQuantiles.beforeMedians[region];
    const afterMedian = regionalQuantiles.afterMedians[region];
    const beforeMode = regionalQuantiles.beforeModes[region];
    const afterMode = regionalQuantiles.afterModes[region];
    const difference = regionalQuantiles.meanChanges[region];
    const differenceValid = regionalQuantiles.validChangeCounts[region];
    ctx.setReadout(
      'regionalEvidence',
      `${names[region]} · 1,024 assigned whole cells / ${(totalArea / 1_000_000).toFixed(4)} km² · Jul support ${beforeRows} QA-retained rows / ${beforeValid} finite NIR / ${1024 - beforeRows} masked-out cells · median ${beforeMedian.toFixed(4)}, numeric sample mode ${beforeMode.toFixed(4)}, ${regionalQuantiles.beforeUniqueCounts[region]} unique values · Sep support ${afterRows} QA-retained rows / ${afterValid} finite NIR / ${1024 - afterRows} masked-out cells · median ${afterMedian.toFixed(4)}, numeric sample mode ${afterMode.toFixed(4)}, ${regionalQuantiles.afterUniqueCounts[region]} unique values · paired common support ${differenceValid}, mean Δ ${difference.toFixed(4)} · valid area ${(regionalQuantiles.validAreas[region] / 1_000_000).toFixed(4)} km² · changed area ${(regionalQuantiles.changedAreas[region] / 1_000_000).toFixed(4)} km² · coverage ${(regionalQuantiles.coverage[region] * 100).toFixed(1)}% · cutoff ${ctx.options.changeThreshold} · stage flags ${Array.from(regionalQuantiles.stageStatus).join('/')}, invalid cutoff ${regionalQuantiles.thresholdInvalid}`
    );
    ctx.setChart('regionalChart', {
      kind: 'line',
      title: `${names[region]} · NIR reflectance quantiles by acquisition`,
      xLabel: 'linear quantile probability',
      yLabel: 'NIR reflectance',
      series: [
        {label: '13 Jul 2021', x: [0, 0.25, 0.5, 0.75, 1], y: beforeQuantiles},
        {label: '21 Sep 2021', x: [0, 0.25, 0.5, 0.75, 1], y: afterQuantiles}
      ],
      description: `GPU linear quantiles for each date use that date's strict valid NIR support. Paired difference support is reported separately.`
    });
  };
  const missingModelMap = resources.createBuffer(
    'model-pending-map',
    Float32Array.from({length: CELL_COUNT}, () => Number.NaN)
  );
  let fitReady = false;

  const marker = resources.createBuffer(
    'selected-cell-marker',
    Float32Array.of(...frame.cellToMeters(32, 56))
  );
  const exemplarMarkers = EXEMPLAR_CELLS.map((cell, index) => {
    const row = Math.floor(cell / CROP_SIZE);
    const column = cell % CROP_SIZE;
    return resources.createBuffer(
      `exemplar-marker-${index}`,
      Float32Array.of(...frame.cellToMeters(column, row))
    );
  });
  const summarySources = [
    ...(['before', 'after'] as const).flatMap(date =>
      INDEX_OUTPUT_NAMES.flatMap(name => {
        const output = indexBuffers[`${date}-${name}`];
        return [
          {buffer: output.values, size: CELL_COUNT * 4},
          {buffer: output.validity, size: CELL_COUNT * 4}
        ];
      })
    ),
    ...(['before', 'after'] as const).flatMap(date =>
      MODEL_NAMES.flatMap(model => {
        const output = fitBuffers[`${date}-${model}`];
        return [
          {buffer: output.abundances, size: CELL_COUNT * EXEMPLAR_CELLS.length * 4},
          {buffer: output.residual, size: CELL_COUNT * 4},
          {buffer: output.status, size: CELL_COUNT * 4},
          {buffer: output.iterations, size: CELL_COUNT * 4}
        ];
      })
    )
  ];
  let latestSummary: SpectralSummary | null = null;
  let revision = 0;
  let requestedRevision = -1;
  let acceptedRevision = -1;
  let dirty = true;
  let selectedCell = PROBE_CELL;
  const selectedLocation = (): string => {
    const row = Math.floor(selectedCell / CROP_SIZE);
    const column = selectedCell % CROP_SIZE;
    const [longitude, latitude] = frame.cellToLngLat(column, row);
    return `${Math.abs(longitude).toFixed(3)}°W, ${latitude.toFixed(3)}°N · source row ${CROP_ROW + row}, column ${CROP_COLUMN + column}`;
  };
  function publishEvidence(): void {
    if (!latestSummary) return;
    const date = ctx.options.date;
    const displayIndexName = (name: (typeof INDEX_NAMES)[number]) =>
      name === 'savi' ? `savi-${ctx.options.saviL}` : name;
    const indexValues = INDEX_NAMES.map(
      name => latestSummary!.indices[`${date}-${displayIndexName(name)}`][selectedCell]
    );
    const indexValidity = INDEX_NAMES.map(
      name => latestSummary!.indexValidity[`${date}-${displayIndexName(name)}`][selectedCell] !== 0
    );
    const validIndexCount = INDEX_NAMES.reduce((sum, name) => {
      const values = latestSummary!.indexValidity[`${date}-${displayIndexName(name)}`];
      return sum + values.reduce((count, valid) => count + (valid ? 1 : 0), 0);
    }, 0);
    ctx.setReadout(
      'sourceSupport',
      `${date === 'before' ? '13 Jul 2021 ignition-day' : '21 Sep 2021 post-ignition'} · ${validCells[date].reduce((sum, valid) => sum + valid, 0).toLocaleString()}/${CELL_COUNT} crop cells pass the strict six-band and SCL mask · selected pixel ${selectedLocation()}`
    );
    ctx.setReadout(
      'indexEvidence',
      INDEX_NAMES.map(
        (name, index) =>
          `${name === 'savi' ? `SAVI · L = ${ctx.options.saviL}` : INDEX_LABELS[name]} ${indexValidity[index] ? indexValues[index].toFixed(3) : 'invalid denominator/support'}`
      ).join(' · ') +
        ` · GPU-valid index cells ${validIndexCount.toLocaleString()} / ${CELL_COUNT * INDEX_NAMES.length} · denominator threshold ${ctx.options.indexEpsilon} · output clamp ${ctx.options.indexClamp === 'none' ? 'none' : ctx.options.indexClamp === 'unit-range' ? '[-1, 1]' : '[-0.25, 0.25] sensitivity'}`
    );
    ctx.setChart('indexChart', {
      kind: 'bars',
      title: 'Six GPU index results at the selected pixel',
      labels: INDEX_NAMES.map(name =>
        name === 'savi' ? `SAVI L=${ctx.options.saviL}` : name.toUpperCase()
      ),
      yLabel: 'unitless index',
      values: indexValues.map((value, index) => (indexValidity[index] ? value : Number.NaN)),
      horizontal: false,
      yDomain: [-1, 1],
      formatY: value => value.toFixed(2),
      description:
        'Bars are separate GPU index values, not a time series. Invalid denominators or source support are omitted; the fixed display domain is −1 to 1.'
    });
    const exemplarSeries = EXEMPLAR_CELLS.map((cell, exemplar) => ({
      label: `${EXEMPLAR_NAMES[exemplar]} · row ${CROP_ROW + Math.floor(cell / CROP_SIZE)}, col ${CROP_COLUMN + (cell % CROP_SIZE)}`,
      x: BAND_NAMES.map((_, index) => index),
      y: Array.from(
        {length: BAND_NAMES.length},
        (_, band) => packedSpectra.after[cell * BAND_NAMES.length + band]
      ),
      points: true
    }));
    ctx.setReadout(
      'exemplars',
      EXEMPLAR_CELLS.map((cell, exemplar) => {
        const row = Math.floor(cell / CROP_SIZE);
        const column = cell % CROP_SIZE;
        const [longitude, latitude] = frame.cellToLngLat(column, row);
        return `${EXEMPLAR_NAMES[exemplar]} · ${Math.abs(longitude).toFixed(3)}°W, ${latitude.toFixed(3)}°N · source row ${CROP_ROW + row}, column ${CROP_COLUMN + column}`;
      }).join(' · ')
    );
    ctx.setChart('exemplarSpectra', {
      kind: 'line',
      title: 'Measured September spectra used as model exemplars',
      xLabel: 'Sentinel-2 band',
      yLabel: 'surface reflectance',
      xDomain: [0, BAND_NAMES.length - 1],
      yDomain: [0, 0.6],
      formatX: value => BAND_NAMES[Math.round(value)] ?? '',
      formatY: value => value.toFixed(2),
      description:
        'Each curve is the six-band source spectrum at its displayed 60 m pixel. Names describe spectral signatures only; they are not independently validated classes.',
      series: exemplarSeries
    });
    const modelKey = `${date}-${ctx.options.constraint}`;
    const fit = latestSummary.fits[modelKey];
    const statusCounts = fit.status.reduce(
      (counts, status) => {
        if (status === 0) counts.solved++;
        else if (status === 4) counts.iterationLimited++;
        else if (status === 1) counts.insufficient++;
        else counts.failed++;
        return counts;
      },
      {solved: 0, iterationLimited: 0, insufficient: 0, failed: 0}
    );
    const modelStatus = fit.status[selectedCell];
    const abundanceOffset = selectedCell * EXEMPLAR_CELLS.length;
    const weights = Array.from(
      fit.abundances.subarray(abundanceOffset, abundanceOffset + EXEMPLAR_CELLS.length)
    );
    const statusText =
      modelStatus === 0
        ? 'solved'
        : modelStatus === 1
          ? 'insufficient valid bands'
          : modelStatus === 2
            ? 'rank or non-finite failure'
            : modelStatus === 4
              ? 'iteration limit reached; feasible fit retained'
              : `unknown status ${modelStatus}`;
    ctx.setReadout(
      'modelEvidence',
      `${MODEL_LABELS[ctx.options.constraint]} · weights ${weights.map((weight, index) => `${EXEMPLAR_NAMES[index]} ${Number.isFinite(weight) ? weight.toFixed(3) : 'missing'}`).join(' · ')} · squared residual ${Number.isFinite(fit.residual[selectedCell]) ? fit.residual[selectedCell].toFixed(5) : 'missing'} reflectance² · ${statusText} · ${fit.iterations[selectedCell]} iterations · budget ${ctx.options.maximumIterations} · tolerance ${ctx.options.tolerance}`
    );
    ctx.setReadout(
      'fitStatus',
      `${MODEL_LABELS[ctx.options.constraint]} across ${CELL_COUNT.toLocaleString()} crop cells · ${statusCounts.solved.toLocaleString()} solved · ${statusCounts.iterationLimited.toLocaleString()} iteration-limited · ${statusCounts.insufficient.toLocaleString()} insufficient support · ${statusCounts.failed.toLocaleString()} failed`
    );
    const observed = Array.from(
      {length: BAND_NAMES.length},
      (_, band) => packedSpectra[date][selectedCell * BAND_NAMES.length + band]
    );
    const modelValid = modelStatus === 0 || modelStatus === 4;
    const fitted = Array.from({length: BAND_NAMES.length}, (_, band) =>
      weights.reduce(
        (sum, weight, exemplar) => sum + weight * endmembers[exemplar * BAND_NAMES.length + band],
        0
      )
    );
    ctx.setChart(
      'pixelSpectra',
      modelValid
        ? {
            kind: 'line',
            title: 'Observed spectrum and GPU mixture fit',
            xLabel: 'Sentinel-2 band',
            yLabel: 'surface reflectance',
            xDomain: [0, BAND_NAMES.length - 1],
            yDomain: [0, 0.6],
            formatX: value => BAND_NAMES[Math.round(value)] ?? '',
            formatY: value => value.toFixed(2),
            description: `Observed six-band source values and the selected GPU ${ctx.options.constraint} fit at ${selectedLocation()}. Squared residual and solver status are reported above.`,
            series: [
              {
                label: 'Observed',
                x: BAND_NAMES.map((_, index) => index),
                y: observed,
                color: 0,
                points: true
              },
              {
                label: 'Mixture fit',
                x: BAND_NAMES.map((_, index) => index),
                y: fitted,
                color: 1,
                points: true
              }
            ]
          }
        : null
    );
  }

  const summaryReader = new SummaryReader(
    resources,
    'spectral-evidence-readback',
    summarySources,
    bytes => {
      if (requestedRevision !== revision) return;
      let offset = 0;
      const readFloat = (length: number) => {
        const result = new Float32Array(bytes, offset, length);
        offset += length * 4;
        return result;
      };
      const readUint = (length: number) => {
        const result = new Uint32Array(bytes, offset, length);
        offset += length * 4;
        return result;
      };
      const indices: Record<string, Float32Array> = {};
      const indexValidity: Record<string, Uint32Array> = {};
      for (const date of ['before', 'after'] as const)
        for (const name of INDEX_OUTPUT_NAMES) {
          indices[`${date}-${name}`] = readFloat(CELL_COUNT);
          indexValidity[`${date}-${name}`] = readUint(CELL_COUNT);
        }
      const fits: SpectralSummary['fits'] = {};
      for (const date of ['before', 'after'] as const)
        for (const model of MODEL_NAMES)
          fits[`${date}-${model}`] = {
            abundances: readFloat(CELL_COUNT * EXEMPLAR_CELLS.length),
            residual: readFloat(CELL_COUNT),
            status: readUint(CELL_COUNT),
            iterations: readUint(CELL_COUNT)
          };
      latestSummary = {indices, indexValidity, fits};
      fitReady = true;
      acceptedRevision = revision;
      publishEvidence();
      ctx.requestLayers();
    }
  );

  ctx.setLegendData('cropFrame', frame);
  ctx.setLegendData('cropBounds', cropBounds);
  ctx.setReadout('sourceSupport', 'GPU products are compiling for the bounded 64 × 64 crop…');
  ctx.setReadout(
    'modelEvidence',
    'Measured exemplars are loaded; waiting for GPU fit status and weights…'
  );
  ctx.setReadout('regionalEvidence', 'Compiling GPU quadrant summaries and paired support…');

  return {
    getCompiledGraphs: () => [indexCompiled, modelCompiled, regionalCompiled],
    encode(commandEncoder) {
      if (modelSwap.pending || indexSwap.pending) {
        summaryReader.flush(commandEncoder);
        return;
      }
      if (dirty) {
        indexCompiled.encode(commandEncoder, {parameters: undefined});
        modelCompiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (regionalDirty) {
        regionalCompiled.encode(commandEncoder, {parameters: undefined});
        regionalDirty = false;
      }
      if (
        !regionalReady &&
        requestedRegionalRevision !== regionalRevision &&
        !regionalReader.isPending
      ) {
        requestedRegionalRevision = regionalRevision;
        regionalReader.request(commandEncoder);
      }
      regionalReader.flush(commandEncoder);
      if (
        acceptedRevision !== revision &&
        requestedRevision !== revision &&
        !summaryReader.isPending
      ) {
        requestedRevision = revision;
        summaryReader.request(commandEncoder);
      }
      summaryReader.flush(commandEncoder);
    },
    getLayers() {
      const date = ctx.options.date;
      const product = ctx.options.product;
      let values: Buffer;
      let valueRange: [number, number];
      let ramp: 'vik' | 'cividis';
      if (ctx.options.regionView === 'change') {
        values = regionalReady ? regionalDifferenceBuffer : missingModelMap;
        valueRange = [-0.4, 0.4];
        ctx.setLegendExtent('regionalChange', valueRange);
        ramp = 'vik';
      } else if (ctx.options.regionView === 'distribution') {
        values = regionalReady ? regionalMedianMaps[date] : missingModelMap;
        valueRange = [0, 0.4];
        ctx.setLegendExtent('regionalDistribution', valueRange);
        ramp = 'cividis';
      } else if ((INDEX_NAMES as readonly string[]).includes(product)) {
        const outputName = product === 'savi' ? `savi-${ctx.options.saviL}` : product;
        values = fitReady ? indexBuffers[`${date}-${outputName}`].values : missingModelMap;
        valueRange = [-1, 1];
        ctx.setLegendExtent('index', valueRange);
        ramp = 'vik';
      } else if (product === 'abundance') {
        const component = Number(ctx.options.component);
        values = fitReady
          ? fitBuffers[`${date}-${ctx.options.constraint}`].componentMaps[component]
          : missingModelMap;
        const fit = latestSummary?.fits[`${date}-${ctx.options.constraint}`];
        let minimum = 0;
        let maximum = 1;
        if (fit) {
          minimum = Infinity;
          maximum = -Infinity;
          for (let cell = 0; cell < CELL_COUNT; cell++) {
            if (fit.status[cell] !== 0 && fit.status[cell] !== 4) continue;
            const coefficient = fit.abundances[cell * EXEMPLAR_CELLS.length + component];
            if (!Number.isFinite(coefficient)) continue;
            minimum = Math.min(minimum, coefficient);
            maximum = Math.max(maximum, coefficient);
          }
          if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
            minimum = 0;
            maximum = 1;
          }
        }
        if (ctx.options.constraint === 'unconstrained') {
          const extent = Math.max(Math.abs(minimum), Math.abs(maximum), 1e-6);
          valueRange = [-extent, extent];
          ctx.setLegendExtent('abundance', valueRange);
          ramp = 'vik';
        } else {
          valueRange = [0, ctx.options.constraint === 'simplex' ? 1 : Math.max(maximum, 1e-6)];
          ctx.setLegendExtent('abundance', valueRange);
          ramp = 'cividis';
        }
      } else {
        values = fitReady
          ? fitBuffers[`${date}-${ctx.options.constraint}`].residual
          : missingModelMap;
        const residuals =
          latestSummary?.fits[`${date}-${ctx.options.constraint}`].residual.filter(
            Number.isFinite
          ) ?? [];
        const maximum = residuals.length ? Math.max(...residuals) : 0.1;
        valueRange = [0, Math.max(maximum, 1e-6)];
        ctx.setLegendExtent('residual', valueRange);
        ramp = 'cividis';
      }
      const exemplarColors = [
        [91, 185, 126, 255],
        [198, 111, 88, 255],
        [243, 194, 92, 255]
      ] as const;
      return [
        new UtmRasterLayer({
          id:
            ctx.options.regionView === 'change'
              ? 'spectral-evidence-regional-difference'
              : `spectral-evidence-${product}`,
          frame,
          values,
          valueFormat: 'float32',
          valueRange,
          colormap: ramp,
          opacity: 0.92,
          noDataColor: [0, 0, 0, 0]
        }),
        ...(ctx.options.regionView !== 'indices'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'spectral-evidence-quadrant-lines',
                coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
                segments: quadrantSegmentBuffer,
                instanceCount: 2,
                widthPixels: 1.4,
                color: [255, 250, 222, 240],
                outlineColor: [30, 36, 44, 220],
                outlineWidthPixels: 1
              })
            ]
          : []),
        ...(ctx.options.regionView !== 'indices'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'spectral-evidence-selected-quadrant',
                coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
                segments: selectedQuadrantOutline,
                instanceCount: 4,
                widthPixels: 3,
                color: [255, 230, 83, 255],
                outlineColor: [24, 31, 39, 255],
                outlineWidthPixels: 1
              })
            ]
          : []),
        ...(ctx.options.regionView === 'indices'
          ? [
              ...exemplarMarkers.map(
                (positions, index) =>
                  new SpatialAnalysisPointLayer({
                    id: `spectral-evidence-exemplar-${index}`,
                    coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
                    positions,
                    instanceCount: 1,
                    radiusPixels: 6,
                    color: exemplarColors[index],
                    outlineColor: [24, 31, 39, 255],
                    outlineWidthPixels: 1.5
                  })
              ),
              new SpatialAnalysisPointLayer({
                id: 'spectral-evidence-selected-cell',
                coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
                positions: marker,
                instanceCount: 1,
                radiusPixels: 6,
                color: [255, 245, 96, 255],
                outlineColor: [24, 31, 39, 255],
                outlineWidthPixels: 1.5
              })
            ]
          : [])
      ];
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [columnFraction, rowFraction] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= CROP_SIZE || row >= CROP_SIZE) return null;
      const cell = row * CROP_SIZE + column;
      if (ctx.options.regionView !== 'indices') {
        const region = (row >= 32 ? 2 : 0) + (column >= 32 ? 1 : 0);
        const names = ['north-west', 'north-east', 'south-west', 'south-east'];
        if (!regionalReady || !regionalQuantiles)
          return `${names[region]} · GPU regional support is pending or suppressed`;
        return `${names[region]} · ${ctx.options.date === 'before' ? 'July' : 'September'} median NIR ${regionalQuantiles[ctx.options.date === 'before' ? 'beforeMedians' : 'afterMedians'][region].toFixed(4)} · paired mean Δ ${regionalQuantiles.meanChanges[region].toFixed(4)} · common support ${regionalQuantiles.validChangeCounts[region]} of 1,024 cells · coverage ${(regionalQuantiles.coverage[region] * 100).toFixed(1)}%`;
      }
      const date = ctx.options.date;
      return `${date} · ${BAND_NAMES.map(name => `${name} ${cropBands[date][name][cell].toFixed(3)}`).join(' · ')}`;
    },
    onClick(event) {
      if (!event.coordinate) return false;
      const [columnFraction, rowFraction] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= CROP_SIZE || row >= CROP_SIZE) return false;
      selectedCell = row * CROP_SIZE + column;
      marker.write(Float32Array.of(...frame.cellToMeters(column, row)));
      if (latestSummary) publishEvidence();
      ctx.requestLayers();
      return true;
    },
    setOption(id) {
      if (id === 'changeThreshold') {
        regionThresholdBuffer.write(Float32Array.of(Number(ctx.options.changeThreshold)));
        regionalRevision++;
        requestedRegionalRevision = -1;
        regionalReady = false;
        regionalQuantiles = null;
        regionalDirty = true;
        ctx.setReadout(
          'regionalEvidence',
          'Updating paired support and area summaries on the GPU…'
        );
        ctx.setChart('regionalChart', null);
      } else if (id === 'selectedRegion') {
        selectedQuadrantOutline.write(makeQuadrantOutline(Number(ctx.options.selectedRegion)));
        publishRegionalEvidence();
      } else if (id === 'indexEpsilon' || id === 'indexClamp') {
        const options = {...ctx.options};
        revision++;
        requestedRevision = -1;
        acceptedRevision = -1;
        latestSummary = null;
        fitReady = false;
        ctx.setChart('indexChart', null);
        ctx.setChart('pixelSpectra', null);
        ctx.setReadout(
          'indexEvidence',
          'Compiling denominator and explicit output-clamp settings…'
        );
        ctx.setStatus('Compiling spectral index settings');
        void indexSwap
          .request({
            build: () => buildIndexGraph(options),
            commit: next => {
              const previous = indexCompiled;
              indexCompiled = resources.track(next);
              resources.release(previous);
              revision++;
              requestedRevision = -1;
              acceptedRevision = -1;
              latestSummary = null;
              dirty = true;
              ctx.setStatus('');
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Spectral index compile failed: ${String(error)}`));
      } else if (id === 'maximumIterations' || id === 'tolerance') {
        const options = {...ctx.options};
        revision++;
        requestedRevision = -1;
        acceptedRevision = -1;
        latestSummary = null;
        fitReady = false;
        ctx.setReadout('modelEvidence', 'Compiling the selected solver settings…');
        ctx.setReadout('fitStatus', 'Waiting for the newly compiled model status…');
        ctx.setChart('pixelSpectra', null);
        ctx.setStatus('Compiling the selected spectral mixture solver');
        void modelSwap
          .request({
            build: async () => (await makeModelGraph(options)).graph,
            commit: next => {
              const previous = modelCompiled;
              modelCompiled = resources.track(next);
              resources.release(previous);
              revision++;
              requestedRevision = -1;
              acceptedRevision = -1;
              latestSummary = null;
              dirty = true;
              ctx.setStatus('');
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Spectral solver compile failed: ${String(error)}`));
      } else if (latestSummary) {
        publishEvidence();
      }
      ctx.requestLayers();
    },
    destroy() {
      summaryReader.release();
      regionalReader.release();
      resources.destroy();
    }
  };
}
