// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterClosing,
  GPURasterContourClassifier,
  GPURasterContours,
  GPURasterFillNoData,
  GPURasterFillHoles,
  GPURasterMorphology,
  GPURasterOpening,
  GPURasterReconstruction,
  type GPURasterBufferBand,
  type GPURasterMetadata
} from '@luma-contributors/graph/gpu-raster';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {GraphSwap} from '../../engine/graph-swap';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {fetchBytes} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {addMissingSurfacePreparation} from './missing-surface-preparation';

const DATASET_ID = 'dixie-spectral';
const SOURCE_WIDTH = 250;
const SOURCE_ROW = 50;
const SOURCE_COLUMN = 101;
const GRID_SIZE = 64;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;
const CELL_SIZE_METERS = 60;
const CONTOUR_CELL_COUNT = (GRID_SIZE - 1) * (GRID_SIZE - 1);
const CONTOUR_CAPACITY = CONTOUR_CELL_COUNT * 2;
const FILL_METHODS = ['nearest', 'inverse-distance', 'gdal-four-quadrant'] as const;
const FILL_SCENARIOS = ['isolated', 'block', 'september'] as const;
const MORPHOLOGY_OPERATIONS = ['dilate', 'erode', 'opening', 'closing'] as const;
const RECONSTRUCTION_MODES = ['opening', 'closing'] as const;
const NIR_RANGE: readonly [number, number] = [0, 0.4];

export type MissingObservationsOptions = {
  experiment: 'held-out' | 'september-gaps';
  holdoutPattern: 'isolated' | 'block';
  mapLayer:
    | 'source'
    | 'mask'
    | 'fill'
    | 'prepared'
    | 'filled-cells'
    | 'error'
    | 'morphology'
    | 'reconstruction'
    | 'topology-fill'
    | 'contours';
  fillMethod: (typeof FILL_METHODS)[number];
  maximumDistance: '2' | '4' | '8' | '12';
  preparationFilter: 'none' | 'median';
  preparationRadius: '1' | '2';
  preparationNoData: 'ignore' | 'propagate';
  preparationBorderMode: 'clamp' | 'nodata';
  morphologyOperation: (typeof MORPHOLOGY_OPERATIONS)[number];
  morphologyRadius: '1' | '2' | '4';
  structuringElement: 'square' | 'cross';
  reconstructionMode: (typeof RECONSTRUCTION_MODES)[number];
  reconstructionConnectivity: '4' | '8';
  maximumIterations: '8' | '32' | '128';
  holeConnectivity: '4' | '8';
  contourLevel: number;
};

type FillBuffers = {
  values: Buffer;
  validity: Buffer;
  filled: Buffer;
  unresolved: Buffer;
  error: Buffer;
  preparedValues: Buffer;
  preparedValidity: Buffer;
  preparedOriginalValidity: Buffer;
  preparedFilled: Buffer;
  preparedFiltered: Buffer;
  preparedUnresolved: Buffer;
};
type MorphologyBuffers = {values: Buffer; validity: Buffer};
type FillResult = {
  values: Float32Array;
  validity: Uint32Array;
  filled: Uint32Array;
  unresolved: number;
  validCount: number;
  filledCount: number;
  targetCount: number;
  targetFilled: number;
  errorCount: number;
  rmse: number;
  mae: number;
  preparedValues: Float32Array;
  preparedValidity: Uint32Array;
  preparedOriginalValidity: Uint32Array;
  preparedFilled: Uint32Array;
  preparedFiltered: Uint32Array;
  preparedUnresolved: number;
  preparedErrorCount: number;
  preparedRmse: number;
  preparedMae: number;
};
type MissingSummary = {
  fills: Record<string, FillResult>;
  morphology: Record<
    (typeof MORPHOLOGY_OPERATIONS)[number],
    {values: Float32Array; validity: Uint32Array}
  >;
  reconstruction: {values: Float32Array; validity: Uint32Array; converged: number};
  contours: {
    count: number;
    required: number;
    overflow: number;
    cases: Uint32Array;
    segmentCounts: Uint32Array;
  };
  holes: {values: Uint32Array; converged: number; filledCount: number};
};
type ReadbackSource = {buffer: Buffer; size: number; format: 'float32' | 'uint32'; length: number};

const FILL_LABELS: Record<(typeof FILL_METHODS)[number], string> = {
  nearest: 'nearest original donor',
  'inverse-distance': 'all-donor inverse distance',
  'gdal-four-quadrant': 'four-quadrant inverse distance'
};

function holdoutAt(pattern: 'isolated' | 'block', row: number, column: number): boolean {
  if (pattern === 'block') return row >= 27 && row < 36 && column >= 27 && column < 36;
  const index = row * GRID_SIZE + column;
  return (index * 17 + 13) % 31 === 0;
}

function getFillKey(
  scenario: (typeof FILL_SCENARIOS)[number],
  method: (typeof FILL_METHODS)[number]
): string {
  return `${scenario}-${method}`;
}

function makeContourOutline(frame: ReturnType<typeof createUtmFrame>): Float32Array {
  const corners = [
    frame.localToMeters(-frame.halfWidth, frame.halfHeight),
    frame.localToMeters(frame.halfWidth, frame.halfHeight),
    frame.localToMeters(frame.halfWidth, -frame.halfHeight),
    frame.localToMeters(-frame.halfWidth, -frame.halfHeight)
  ];
  const segments = new Float32Array(16);
  for (let index = 0; index < 4; index++) {
    segments.set([...corners[index], ...corners[(index + 1) % 4]], index * 4);
  }
  return segments;
}

function readFloat32(bytes: ArrayBuffer, offset: number, length: number): Float32Array {
  return new Float32Array(bytes.slice(offset, offset + length * 4));
}

function readUint32(bytes: ArrayBuffer, offset: number, length: number): Uint32Array {
  return new Uint32Array(bytes.slice(offset, offset + length * 4));
}

/** A bounded, measured-pixel missing-data investigation on the Sentinel-2 Dixie crop. */
export async function createMissingObservations(
  ctx: SceneContext<MissingObservationsOptions>
): Promise<SceneInstance<MissingObservationsOptions>> {
  const dataset = ctx.datasets.get(DATASET_ID);
  ctx.setStatus('Loading the pinned July and September NIR observations and validity masks');
  const [julyRaster, septemberRaster, julyMaskBytes, septemberMaskBytes] = await Promise.all([
    dataset.loadRaster('nir_before', ctx.signal),
    dataset.loadRaster('nir_after', ctx.signal),
    fetchBytes(dataset.fileUrl('validity_before.bin'), ctx.signal),
    fetchBytes(dataset.fileUrl('validity_after.bin'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();

  const julySource = Float32Array.from(julyRaster.values);
  const septemberSource = Float32Array.from(septemberRaster.values);
  const julyMaskSource = new Uint8Array(julyMaskBytes);
  const septemberMaskSource = new Uint8Array(septemberMaskBytes);
  const julyValues = new Float32Array(CELL_COUNT);
  const septemberValues = new Float32Array(CELL_COUNT);
  const julyValidity = new Uint32Array(CELL_COUNT);
  const septemberValidity = new Uint32Array(CELL_COUNT);
  const isolatedHoldout = new Uint32Array(CELL_COUNT);
  const blockHoldout = new Uint32Array(CELL_COUNT);
  const isolatedObserved = Float32Array.from({length: CELL_COUNT}, () => Number.NaN);
  const blockObserved = Float32Array.from({length: CELL_COUNT}, () => Number.NaN);
  const julyDisplay = Float32Array.from({length: CELL_COUNT}, () => Number.NaN);
  const septemberDisplay = Float32Array.from({length: CELL_COUNT}, () => Number.NaN);
  // Mask colormaps test nonzero values; use finite zero for transparent background.
  const isolatedOverlay = new Float32Array(CELL_COUNT);
  const blockOverlay = new Float32Array(CELL_COUNT);
  const septemberMissingOverlay = new Float32Array(CELL_COUNT);
  let julyValidCount = 0;
  let septemberValidCount = 0;
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      const cell = row * GRID_SIZE + column;
      const source = (SOURCE_ROW + row) * SOURCE_WIDTH + SOURCE_COLUMN + column;
      const julyValid = julyMaskSource[source] !== 0 && Number.isFinite(julySource[source]);
      const septemberValid =
        septemberMaskSource[source] !== 0 && Number.isFinite(septemberSource[source]);
      if (julyValid) {
        julyValues[cell] = julySource[source];
        julyDisplay[cell] = julySource[source];
        julyValidity[cell] = 1;
        julyValidCount++;
        if (holdoutAt('isolated', row, column)) {
          isolatedHoldout[cell] = 1;
          isolatedOverlay[cell] = 1;
        } else isolatedObserved[cell] = julySource[source];
        if (holdoutAt('block', row, column)) {
          blockHoldout[cell] = 1;
          blockOverlay[cell] = 1;
        } else blockObserved[cell] = julySource[source];
      } else {
        julyValues[cell] = Number.NaN;
        julyDisplay[cell] = Number.NaN;
        isolatedObserved[cell] = Number.NaN;
        blockObserved[cell] = Number.NaN;
      }
      if (septemberValid) {
        septemberValues[cell] = septemberSource[source];
        septemberDisplay[cell] = septemberSource[source];
        septemberValidity[cell] = 1;
        septemberValidCount++;
      } else {
        septemberValues[cell] = Number.NaN;
        septemberDisplay[cell] = Number.NaN;
        septemberMissingOverlay[cell] = 1;
      }
    }
  }
  if (julyValidCount !== CELL_COUNT) {
    throw new Error('The selected July NIR crop must be fully observed for withheld-truth checks');
  }

  const rasterSpec = dataset.rasterSpecs.nir_before;
  const bounds = rasterSpec?.['boundsProjected'];
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 4 ||
    !bounds.every(value => typeof value === 'number' && Number.isFinite(value))
  ) {
    throw new Error('Dixie NIR requires numeric projected UTM bounds');
  }
  const cropWest = Number(bounds[0]) + SOURCE_COLUMN * CELL_SIZE_METERS;
  const cropNorth = Number(bounds[3]) - SOURCE_ROW * CELL_SIZE_METERS;
  const cropBounds: readonly [number, number, number, number] = [
    cropWest,
    cropNorth - GRID_SIZE * CELL_SIZE_METERS,
    cropWest + GRID_SIZE * CELL_SIZE_METERS,
    cropNorth
  ];
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: cropBounds,
    cellSizeM: CELL_SIZE_METERS,
    width: GRID_SIZE,
    height: GRID_SIZE
  });
  const grid: GPURasterMetadata = {
    width: GRID_SIZE,
    height: GRID_SIZE,
    affine: [CELL_SIZE_METERS, 0, cropWest, 0, -CELL_SIZE_METERS, cropNorth],
    pixelInterpretation: 'area',
    coordinateReferenceSystem: {authority: 'EPSG:32610'}
  };

  const resources = new SpatialAnalysisResources(ctx.device, 'missing-observations');
  const julyBuffer = resources.createBuffer('july-nir', julyValues);
  const septemberBuffer = resources.createBuffer('september-nir', septemberValues);
  const julyValidityBuffer = resources.createBuffer('july-validity', julyValidity);
  const septemberValidityBuffer = resources.createBuffer('september-validity', septemberValidity);
  const isolatedInputBuffer = resources.createBuffer('july-isolated-input', isolatedObserved);
  const blockInputBuffer = resources.createBuffer('july-block-input', blockObserved);
  const isolatedValidity = Uint32Array.from(julyValidity, (valid, index) =>
    valid && !isolatedHoldout[index] ? 1 : 0
  );
  const blockValidity = Uint32Array.from(julyValidity, (valid, index) =>
    valid && !blockHoldout[index] ? 1 : 0
  );
  const isolatedValidityBuffer = resources.createBuffer('july-isolated-validity', isolatedValidity);
  const blockValidityBuffer = resources.createBuffer('july-block-validity', blockValidity);
  const binaryValidityBuffers = {
    isolated: resources.createBuffer('topology-isolated-validity', isolatedValidity),
    block: resources.createBuffer('topology-block-validity', blockValidity),
    september: resources.createBuffer('topology-september-validity', septemberValidity)
  };
  const isolatedHoldoutBuffer = resources.createBuffer('july-isolated-holdout', isolatedHoldout);
  const blockHoldoutBuffer = resources.createBuffer('july-block-holdout', blockHoldout);
  const displayBuffers = {
    july: resources.createBuffer('july-display', julyDisplay),
    september: resources.createBuffer('september-display', septemberDisplay),
    isolatedObserved: resources.createBuffer('isolated-observed-display', isolatedObserved),
    blockObserved: resources.createBuffer('block-observed-display', blockObserved),
    isolatedOverlay: resources.createBuffer('isolated-overlay', isolatedOverlay),
    blockOverlay: resources.createBuffer('block-overlay', blockOverlay),
    septemberMissing: resources.createBuffer('september-missing-overlay', septemberMissingOverlay),
    emptyMask: resources.createBuffer('empty-mask-display', new Float32Array(CELL_COUNT)),
    empty: resources.createBuffer(
      'empty-display',
      Float32Array.from({length: CELL_COUNT}, () => Number.NaN)
    )
  };
  const fillBuffers = {} as Record<string, FillBuffers>;
  for (const scenario of FILL_SCENARIOS) {
    for (const method of FILL_METHODS) {
      const key = getFillKey(scenario, method);
      fillBuffers[key] = {
        values: resources.createBuffer(`${key}-values`, CELL_COUNT * 4),
        validity: resources.createBuffer(`${key}-validity`, CELL_COUNT * 4),
        filled: resources.createBuffer(`${key}-filled`, CELL_COUNT * 4),
        unresolved: resources.createBuffer(`${key}-unresolved`, 4),
        error: resources.createBuffer(`${key}-withheld-error`, CELL_COUNT * 4),
        preparedValues: resources.createBuffer(`${key}-prepared-values`, CELL_COUNT * 4),
        preparedValidity: resources.createBuffer(`${key}-prepared-validity`, CELL_COUNT * 4),
        preparedOriginalValidity: resources.createBuffer(
          `${key}-prepared-original-validity`,
          CELL_COUNT * 4
        ),
        preparedFilled: resources.createBuffer(`${key}-prepared-filled`, CELL_COUNT * 4),
        preparedFiltered: resources.createBuffer(`${key}-prepared-filtered`, CELL_COUNT * 4),
        preparedUnresolved: resources.createBuffer(`${key}-prepared-unresolved`, 4)
      };
    }
  }
  const morphologyBuffers = {} as Record<(typeof MORPHOLOGY_OPERATIONS)[number], MorphologyBuffers>;
  for (const operation of MORPHOLOGY_OPERATIONS) {
    morphologyBuffers[operation] = {
      values: resources.createBuffer(`morphology-${operation}-values`, CELL_COUNT * 4),
      validity: resources.createBuffer(`morphology-${operation}-validity`, CELL_COUNT * 4)
    };
  }
  const reconstructionBuffers = {
    marker: resources.createBuffer('reconstruction-marker', CELL_COUNT * 4),
    markerValidity: resources.createBuffer('reconstruction-marker-validity', CELL_COUNT * 4),
    values: resources.createBuffer('reconstruction-values', CELL_COUNT * 4),
    validity: resources.createBuffer('reconstruction-validity', CELL_COUNT * 4),
    converged: resources.createBuffer('reconstruction-converged', 4)
  };
  const holeBuffers = {
    values: resources.createBuffer('topology-hole-fill-values', CELL_COUNT * 4),
    enclosed: resources.createBuffer('topology-enclosed-gap-overlay', new Float32Array(CELL_COUNT)),
    converged: resources.createBuffer('topology-hole-fill-converged', 4)
  };
  const contourBuffers = {
    level: resources.createBuffer('contour-level', Float32Array.of(ctx.options.contourLevel)),
    cases: resources.createBuffer('contour-cases', new Uint32Array(CONTOUR_CELL_COUNT)),
    segmentCounts: resources.createBuffer(
      'contour-cell-segment-counts',
      new Uint32Array(CONTOUR_CELL_COUNT)
    ),
    vertices: resources.createBuffer('contour-vertices', CONTOUR_CAPACITY * 16),
    segments: resources.createBuffer('contour-segments', CONTOUR_CAPACITY * 16),
    count: resources.createBuffer('contour-segment-count', 4),
    required: resources.createBuffer('contour-required-count', 4),
    overflow: resources.createBuffer('contour-overflow', 4)
  };
  let selectedCell = 31 * GRID_SIZE + 31;

  const buildFillGraph = async (
    maximumDistance: number,
    preparationFilter: MissingObservationsOptions['preparationFilter'],
    preparationRadius: number,
    preparationNoData: MissingObservationsOptions['preparationNoData'],
    preparationBorderMode: MissingObservationsOptions['preparationBorderMode']
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `missing-observations-fill-${maximumDistance}`
    });
    const view = createGraphViewer(graph);
    const bands: Record<(typeof FILL_SCENARIOS)[number], GPURasterBufferBand<'float32'>> = {
      isolated: {
        id: 'july-isolated-observations',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: view('july-isolated', isolatedInputBuffer, 'float32', CELL_COUNT)
        },
        validity: view('july-isolated-validity', isolatedValidityBuffer, 'uint32', CELL_COUNT)
      },
      block: {
        id: 'july-block-observations',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: view('july-block', blockInputBuffer, 'float32', CELL_COUNT)
        },
        validity: view('july-block-validity', blockValidityBuffer, 'uint32', CELL_COUNT)
      },
      september: {
        id: 'september-real-observations',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: view('september', septemberBuffer, 'float32', CELL_COUNT)
        },
        validity: view('september-validity', septemberValidityBuffer, 'uint32', CELL_COUNT)
      }
    };
    const julyTruth = view('july-truth', julyBuffer, 'float32', CELL_COUNT);
    const holdoutViews = {
      isolated: view('july-isolated-holdout-mask', isolatedHoldoutBuffer, 'uint32', CELL_COUNT),
      block: view('july-block-holdout-mask', blockHoldoutBuffer, 'uint32', CELL_COUNT)
    };
    for (const scenario of FILL_SCENARIOS) {
      for (const method of FILL_METHODS) {
        const key = getFillKey(scenario, method);
        const output = fillBuffers[key];
        const outputViews = {
          values: view(`${key}-values`, output.values, 'float32', CELL_COUNT),
          validity: view(`${key}-validity`, output.validity, 'uint32', CELL_COUNT),
          filled: view(`${key}-filled`, output.filled, 'uint32', CELL_COUNT),
          unresolvedCount: view(`${key}-unresolved`, output.unresolved, 'uint32', 1)
        };
        graph.add(
          new GPURasterFillNoData({
            id: `fill-${key}`,
            grid,
            input: bands[scenario],
            maximumDistance,
            method,
            output: outputViews
          })
        );
        const preparationOutputs = {
          values: view(`${key}-prepared-values`, output.preparedValues, 'float32', CELL_COUNT),
          validity: view(`${key}-prepared-validity`, output.preparedValidity, 'uint32', CELL_COUNT),
          originalValidity: view(
            `${key}-prepared-original-validity`,
            output.preparedOriginalValidity,
            'uint32',
            CELL_COUNT
          ),
          filled: view(`${key}-prepared-filled`, output.preparedFilled, 'uint32', CELL_COUNT),
          filtered: view(`${key}-prepared-filtered`, output.preparedFiltered, 'uint32', CELL_COUNT),
          unresolvedCount: view(
            `${key}-prepared-unresolved`,
            output.preparedUnresolved,
            'uint32',
            1
          )
        };
        addMissingSurfacePreparation(graph, {
          id: `prepared-${key}`,
          grid,
          input: bands[scenario],
          maximumDistance,
          method,
          ...(preparationFilter === 'median'
            ? {
                filter: {
                  radius: preparationRadius,
                  borderMode: preparationBorderMode,
                  noDataPolicy: preparationNoData
                }
              }
            : {}),
          outputs: preparationOutputs
        });
        if (scenario !== 'september') {
          addKernelPass(graph, {
            id: `withheld-error-${key}`,
            invocationCount: CELL_COUNT,
            bindings: [
              {name: 'truth', view: julyTruth, type: 'f32', access: 'read'},
              {name: 'holdout', view: holdoutViews[scenario], type: 'u32', access: 'read'},
              {name: 'filledValues', view: outputViews.values, type: 'f32', access: 'read'},
              {name: 'filledValidity', view: outputViews.validity, type: 'u32', access: 'read'},
              {
                name: 'error',
                view: view(`${key}-error`, output.error, 'float32', CELL_COUNT),
                type: 'f32',
                access: 'read_write'
              }
            ],
            body: 'let valid=holdout[holdoutOffset+index]!=0u&&filledValidity[filledValidityOffset+index]!=0u;let residual=filledValues[filledValuesOffset+index]-truth[truthOffset+index];let missing=bitcast<f32>(0x7fc00000u|(index&0u));error[errorOffset+index]=select(missing,residual,valid);'
          });
        }
      }
    }
    return graph.compileAsync();
  };

  const buildMorphologyGraph = async (
    radius: number,
    structuringElement: 'square' | 'cross'
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `missing-observations-morphology-${radius}-${structuringElement}`
    });
    const view = createGraphViewer(graph);
    const input: GPURasterBufferBand<'float32'> = {
      id: 'july-original-nir',
      format: 'float32',
      storage: {kind: 'buffer', values: view('july-original', julyBuffer, 'float32', CELL_COUNT)},
      validity: view('july-original-validity', julyValidityBuffer, 'uint32', CELL_COUNT)
    };
    const outputViews = Object.fromEntries(
      MORPHOLOGY_OPERATIONS.map(operation => [
        operation,
        {
          values: view(
            `morphology-${operation}-values`,
            morphologyBuffers[operation].values,
            'float32',
            CELL_COUNT
          ),
          validity: view(
            `morphology-${operation}-validity`,
            morphologyBuffers[operation].validity,
            'uint32',
            CELL_COUNT
          )
        }
      ])
    ) as Record<
      (typeof MORPHOLOGY_OPERATIONS)[number],
      {values: GraphDataView<'float32'>; validity: GraphDataView<'uint32'>}
    >;
    const props = {
      width: GRID_SIZE,
      height: GRID_SIZE,
      radius,
      structuringElement,
      borderMode: 'clamp' as const,
      noDataPolicy: 'ignore' as const,
      input
    };
    new GPURasterMorphology({
      id: 'grayscale-dilation',
      ...props,
      mode: 'grayscale',
      operation: 'dilate',
      output: outputViews.dilate.values,
      outputValidity: outputViews.dilate.validity
    }).addToGraph(graph);
    new GPURasterMorphology({
      id: 'grayscale-erosion',
      ...props,
      mode: 'grayscale',
      operation: 'erode',
      output: outputViews.erode.values,
      outputValidity: outputViews.erode.validity
    }).addToGraph(graph);
    new GPURasterOpening({
      id: 'grayscale-opening',
      ...props,
      output: outputViews.opening.values,
      outputValidity: outputViews.opening.validity
    }).addToGraph(graph);
    new GPURasterClosing({
      id: 'grayscale-closing',
      ...props,
      output: outputViews.closing.values,
      outputValidity: outputViews.closing.validity
    }).addToGraph(graph);
    return graph.compileAsync();
  };

  const buildReconstructionGraph = async (
    mode: 'opening' | 'closing',
    connectivity: 4 | 8,
    maximumIterations: number
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `missing-observations-reconstruction-${mode}-${connectivity}-${maximumIterations}`
    });
    const view = createGraphViewer(graph);
    const input: GPURasterBufferBand<'float32'> = {
      id: 'july-original-nir',
      format: 'float32',
      storage: {kind: 'buffer', values: view('july-original', julyBuffer, 'float32', CELL_COUNT)},
      validity: view('july-original-validity', julyValidityBuffer, 'uint32', CELL_COUNT)
    };
    const markerOperation = mode === 'opening' ? 'erode' : 'dilate';
    const markerValues = view(
      'reconstruction-marker',
      reconstructionBuffers.marker,
      'float32',
      CELL_COUNT
    );
    const markerValidity = view(
      'reconstruction-marker-validity',
      reconstructionBuffers.markerValidity,
      'uint32',
      CELL_COUNT
    );
    new GPURasterMorphology({
      id: 'reconstruction-marker-morphology',
      width: GRID_SIZE,
      height: GRID_SIZE,
      radius: 2,
      structuringElement: connectivity === 8 ? 'square' : 'cross',
      borderMode: 'clamp',
      noDataPolicy: 'ignore',
      mode: 'grayscale',
      operation: markerOperation,
      input,
      output: markerValues,
      outputValidity: markerValidity
    }).addToGraph(graph);
    const reconstructionOutput = {
      values: view('reconstruction-values', reconstructionBuffers.values, 'float32', CELL_COUNT),
      validity: view(
        'reconstruction-validity',
        reconstructionBuffers.validity,
        'uint32',
        CELL_COUNT
      ),
      status: {
        converged: view('reconstruction-converged', reconstructionBuffers.converged, 'uint32', 1)
      }
    };
    graph.add(
      new GPURasterReconstruction({
        id: 'bounded-grayscale-reconstruction',
        grid,
        marker: markerValues,
        mask: input.storage.values,
        method: mode === 'opening' ? 'dilation' : 'erosion',
        connectivity,
        maximumIterations,
        output: reconstructionOutput
      })
    );
    return graph.compileAsync();
  };

  const buildContourGraph = async (): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {id: 'missing-observations-contours'});
    const view = createGraphViewer(graph);
    const level = view('contour-level', contourBuffers.level, 'float32', 1);
    const input: GPURasterBufferBand<'float32'> = {
      id: 'july-original-nir',
      format: 'float32',
      storage: {kind: 'buffer', values: view('july-original', julyBuffer, 'float32', CELL_COUNT)},
      validity: view('july-original-validity', julyValidityBuffer, 'uint32', CELL_COUNT)
    };
    const cases = view('contour-cases', contourBuffers.cases, 'uint32', CONTOUR_CELL_COUNT);
    const segmentCounts = view(
      'contour-cell-segment-counts',
      contourBuffers.segmentCounts,
      'uint32',
      CONTOUR_CELL_COUNT
    );
    new GPURasterContourClassifier({
      id: 'nir-contour-classification',
      width: GRID_SIZE,
      height: GRID_SIZE,
      input,
      level,
      cases,
      segmentCounts
    }).addToGraph(graph);
    new GPURasterContours({
      id: 'nir-contour-geometry',
      width: GRID_SIZE,
      height: GRID_SIZE,
      input,
      level,
      vertices: view(
        'contour-vertices',
        contourBuffers.vertices,
        'float32x2',
        CONTOUR_CAPACITY * 2
      ),
      segmentCount: view('contour-segment-count', contourBuffers.count, 'uint32', 1),
      requiredSegmentCount: view('contour-required-count', contourBuffers.required, 'uint32', 1),
      overflow: view('contour-overflow', contourBuffers.overflow, 'uint32', 1),
      capacity: CONTOUR_CAPACITY,
      metadata: grid
    }).addToGraph(graph);
    const contourCoordinates = view(
      'contour-coordinate-scalars',
      contourBuffers.vertices,
      'float32',
      CONTOUR_CAPACITY * 4
    );
    const segmentOutput = view(
      'contour-segment-scalars',
      contourBuffers.segments,
      'float32',
      CONTOUR_CAPACITY * 4
    );
    const [axisX, axisY, northX, northY] = frame.rotation;
    const [translateX, translateY] = frame.translation;
    addKernelPass(graph, {
      id: 'place-contours-in-utm-frame',
      invocationCount: CONTOUR_CAPACITY,
      bindings: [
        {name: 'coordinates', view: contourCoordinates, type: 'f32', access: 'read'},
        {name: 'segments', view: segmentOutput, type: 'f32', access: 'read_write'}
      ],
      body: `let x0=coordinates[coordinatesOffset+index*4u];let y0=coordinates[coordinatesOffset+index*4u+1u];let x1=coordinates[coordinatesOffset+index*4u+2u];let y1=coordinates[coordinatesOffset+index*4u+3u];let localX0=x0*${CELL_SIZE_METERS}.0-${frame.halfWidth};let localY0=${frame.halfHeight}-y0*${CELL_SIZE_METERS}.0;let localX1=x1*${CELL_SIZE_METERS}.0-${frame.halfWidth};let localY1=${frame.halfHeight}-y1*${CELL_SIZE_METERS}.0;segments[segmentsOffset+index*4u]=${translateX}+${axisX}*localX0+${northX}*localY0;segments[segmentsOffset+index*4u+1u]=${translateY}+${axisY}*localX0+${northY}*localY0;segments[segmentsOffset+index*4u+2u]=${translateX}+${axisX}*localX1+${northX}*localY1;segments[segmentsOffset+index*4u+3u]=${translateY}+${axisY}*localX1+${northY}*localY1;`
    });
    return graph.compileAsync();
  };

  const buildHoleGraph = async (
    experiment: MissingObservationsOptions['experiment'],
    holdoutPattern: MissingObservationsOptions['holdoutPattern'],
    connectivity: 4 | 8
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `missing-observations-hole-fill-${experiment}-${holdoutPattern}-${connectivity}`
    });
    const view = createGraphViewer(graph);
    const scenario = experiment === 'september-gaps' ? 'september' : holdoutPattern;
    const input = view(
      `topology-${scenario}-mask`,
      binaryValidityBuffers[scenario],
      'uint32',
      CELL_COUNT
    );
    const values = view('topology-hole-fill-values', holeBuffers.values, 'uint32', CELL_COUNT);
    const converged = view('topology-hole-fill-converged', holeBuffers.converged, 'uint32', 1);
    new GPURasterFillHoles({
      id: 'binary-support-hole-fill',
      grid,
      input,
      connectivity,
      maximumIterations: 128,
      output: {
        values,
        status: {converged}
      }
    })
      .getCommandNodes(graph)
      .forEach(node => graph.add(node));
    addKernelPass(graph, {
      id: 'topology-enclosed-missing-overlay',
      invocationCount: CELL_COUNT,
      bindings: [
        {name: 'sourceMask', view: input, type: 'u32', access: 'read'},
        {name: 'filledMask', view: values, type: 'u32', access: 'read'},
        {name: 'converged', view: converged, type: 'u32', access: 'read'},
        {
          name: 'overlay',
          view: view('topology-enclosed-gap-overlay', holeBuffers.enclosed, 'float32', CELL_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: 'let enclosed=converged[convergedOffset]!=0u&&sourceMask[sourceMaskOffset+index]==0u&&filledMask[filledMaskOffset+index]==1u;overlay[overlayOffset+index]=select(0.0,1.0,enclosed);'
    });
    return graph.compileAsync();
  };

  const readbackSources: ReadbackSource[] = [];
  for (const scenario of FILL_SCENARIOS) {
    for (const method of FILL_METHODS) {
      const key = getFillKey(scenario, method);
      const output = fillBuffers[key];
      readbackSources.push(
        {buffer: output.values, size: CELL_COUNT * 4, format: 'float32', length: CELL_COUNT},
        {buffer: output.validity, size: CELL_COUNT * 4, format: 'uint32', length: CELL_COUNT},
        {buffer: output.filled, size: CELL_COUNT * 4, format: 'uint32', length: CELL_COUNT},
        {buffer: output.unresolved, size: 4, format: 'uint32', length: 1},
        {
          buffer: output.preparedValues,
          size: CELL_COUNT * 4,
          format: 'float32',
          length: CELL_COUNT
        },
        {
          buffer: output.preparedValidity,
          size: CELL_COUNT * 4,
          format: 'uint32',
          length: CELL_COUNT
        },
        {
          buffer: output.preparedOriginalValidity,
          size: CELL_COUNT * 4,
          format: 'uint32',
          length: CELL_COUNT
        },
        {buffer: output.preparedFilled, size: CELL_COUNT * 4, format: 'uint32', length: CELL_COUNT},
        {
          buffer: output.preparedFiltered,
          size: CELL_COUNT * 4,
          format: 'uint32',
          length: CELL_COUNT
        },
        {buffer: output.preparedUnresolved, size: 4, format: 'uint32', length: 1}
      );
    }
  }
  for (const operation of MORPHOLOGY_OPERATIONS) {
    readbackSources.push(
      {
        buffer: morphologyBuffers[operation].values,
        size: CELL_COUNT * 4,
        format: 'float32',
        length: CELL_COUNT
      },
      {
        buffer: morphologyBuffers[operation].validity,
        size: CELL_COUNT * 4,
        format: 'uint32',
        length: CELL_COUNT
      }
    );
  }
  readbackSources.push(
    {
      buffer: reconstructionBuffers.values,
      size: CELL_COUNT * 4,
      format: 'float32',
      length: CELL_COUNT
    },
    {
      buffer: reconstructionBuffers.validity,
      size: CELL_COUNT * 4,
      format: 'uint32',
      length: CELL_COUNT
    },
    {buffer: reconstructionBuffers.converged, size: 4, format: 'uint32', length: 1},
    {buffer: contourBuffers.count, size: 4, format: 'uint32', length: 1},
    {buffer: contourBuffers.required, size: 4, format: 'uint32', length: 1},
    {buffer: contourBuffers.overflow, size: 4, format: 'uint32', length: 1},
    {
      buffer: contourBuffers.cases,
      size: CONTOUR_CELL_COUNT * 4,
      format: 'uint32',
      length: CONTOUR_CELL_COUNT
    },
    {
      buffer: contourBuffers.segmentCounts,
      size: CONTOUR_CELL_COUNT * 4,
      format: 'uint32',
      length: CONTOUR_CELL_COUNT
    }
  );
  readbackSources.push(
    {buffer: holeBuffers.values, size: CELL_COUNT * 4, format: 'uint32', length: CELL_COUNT},
    {buffer: holeBuffers.converged, size: 4, format: 'uint32', length: 1}
  );

  const summarize = (bytes: ArrayBuffer): MissingSummary => {
    let offset = 0;
    const fills = {} as MissingSummary['fills'];
    for (const scenario of FILL_SCENARIOS) {
      for (const method of FILL_METHODS) {
        const key = getFillKey(scenario, method);
        const values = readFloat32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const validity = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const filled = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const unresolved = readUint32(bytes, offset, 1)[0];
        offset += 4;
        const preparedValues = readFloat32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const preparedValidity = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const preparedOriginalValidity = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const preparedFilled = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const preparedFiltered = readUint32(bytes, offset, CELL_COUNT);
        offset += CELL_COUNT * 4;
        const preparedUnresolved = readUint32(bytes, offset, 1)[0];
        offset += 4;
        const holdout =
          scenario === 'isolated' ? isolatedHoldout : scenario === 'block' ? blockHoldout : null;
        let validCount = 0;
        let filledCount = 0;
        let targetCount = 0;
        let targetFilled = 0;
        let errorCount = 0;
        let squaredError = 0;
        let absoluteError = 0;
        let preparedErrorCount = 0;
        let preparedSquaredError = 0;
        let preparedAbsoluteError = 0;
        if (holdout) {
          for (let index = 0; index < CELL_COUNT; index++) {
            const isFilled = filled[index] !== 0;
            validCount += validity[index] ? 1 : 0;
            filledCount += isFilled ? 1 : 0;
            if (holdout[index]) {
              targetCount++;
              targetFilled += isFilled ? 1 : 0;
              if (validity[index] && Number.isFinite(values[index])) {
                const error = values[index] - julyValues[index];
                errorCount++;
                squaredError += error * error;
                absoluteError += Math.abs(error);
              }
              if (preparedValidity[index] && Number.isFinite(preparedValues[index])) {
                const error = preparedValues[index] - julyValues[index];
                preparedErrorCount++;
                preparedSquaredError += error * error;
                preparedAbsoluteError += Math.abs(error);
              }
            }
          }
        } else {
          for (let index = 0; index < CELL_COUNT; index++) {
            validCount += validity[index] ? 1 : 0;
            filledCount += filled[index] ? 1 : 0;
          }
        }
        fills[key] = {
          values,
          validity,
          filled,
          unresolved,
          validCount,
          filledCount,
          targetCount,
          targetFilled,
          errorCount,
          rmse: errorCount ? Math.sqrt(squaredError / errorCount) : Number.NaN,
          mae: errorCount ? absoluteError / errorCount : Number.NaN,
          preparedValues,
          preparedValidity,
          preparedOriginalValidity,
          preparedFilled,
          preparedFiltered,
          preparedUnresolved,
          preparedErrorCount,
          preparedRmse: preparedErrorCount
            ? Math.sqrt(preparedSquaredError / preparedErrorCount)
            : Number.NaN,
          preparedMae: preparedErrorCount ? preparedAbsoluteError / preparedErrorCount : Number.NaN
        };
      }
    }
    const morphology = {} as MissingSummary['morphology'];
    for (const operation of MORPHOLOGY_OPERATIONS) {
      const values = readFloat32(bytes, offset, CELL_COUNT);
      offset += CELL_COUNT * 4;
      const validity = readUint32(bytes, offset, CELL_COUNT);
      offset += CELL_COUNT * 4;
      morphology[operation] = {values, validity};
    }
    const reconstructionValues = readFloat32(bytes, offset, CELL_COUNT);
    offset += CELL_COUNT * 4;
    const reconstructionValidity = readUint32(bytes, offset, CELL_COUNT);
    offset += CELL_COUNT * 4;
    const converged = readUint32(bytes, offset, 1)[0];
    offset += 4;
    const count = readUint32(bytes, offset, 1)[0];
    offset += 4;
    const required = readUint32(bytes, offset, 1)[0];
    offset += 4;
    const overflow = readUint32(bytes, offset, 1)[0];
    offset += 4;
    const cases = readUint32(bytes, offset, CONTOUR_CELL_COUNT);
    offset += CONTOUR_CELL_COUNT * 4;
    const segmentCounts = readUint32(bytes, offset, CONTOUR_CELL_COUNT);
    offset += CONTOUR_CELL_COUNT * 4;
    const holeValues = readUint32(bytes, offset, CELL_COUNT);
    offset += CELL_COUNT * 4;
    const holeConverged = readUint32(bytes, offset, 1)[0];
    let filledHoleCount = 0;
    const topologyScenario =
      ctx.options.experiment === 'september-gaps' ? 'september' : ctx.options.holdoutPattern;
    const topologyValidity =
      topologyScenario === 'september'
        ? septemberValidity
        : topologyScenario === 'isolated'
          ? isolatedValidity
          : blockValidity;
    for (let index = 0; index < CELL_COUNT; index++) {
      if (!topologyValidity[index] && holeValues[index] === 1) filledHoleCount++;
    }
    return {
      fills,
      morphology,
      reconstruction: {values: reconstructionValues, validity: reconstructionValidity, converged},
      contours: {count, required, overflow, cases, segmentCounts},
      holes: {values: holeValues, converged: holeConverged, filledCount: filledHoleCount}
    };
  };

  function publishReadouts(): void {
    if (!latestSummary || acceptedRevision !== revision) return;
    const scenario =
      ctx.options.experiment === 'september-gaps' ? 'september' : ctx.options.holdoutPattern;
    const fill = latestSummary.fills[getFillKey(scenario, ctx.options.fillMethod)];
    const patternLabel =
      ctx.options.holdoutPattern === 'block'
        ? '9 × 9 withheld block'
        : 'deterministic isolated cells';
    const targetLabel =
      ctx.options.experiment === 'september-gaps' ? 'real September gaps' : patternLabel;
    const support =
      ctx.options.experiment === 'september-gaps' ? septemberValidCount : julyValidCount;
    ctx.setReadout(
      'observedSupport',
      `${ctx.options.experiment === 'september-gaps' ? '21 Sep 2021' : '13 Jul 2021'} · ${support.toLocaleString()} / ${CELL_COUNT.toLocaleString()} observed cells · ${ctx.options.experiment === 'september-gaps' ? `${(CELL_COUNT - support).toLocaleString()} genuine source gaps` : 'fully observed source before artificial withholding'}`
    );
    ctx.setReadout(
      'fillSupport',
      `${targetLabel} · ${fill.validCount.toLocaleString()} filled-surface cells valid · ${fill.filledCount.toLocaleString()} cells interpolated · ${fill.unresolved.toLocaleString()} unresolved within ${ctx.options.maximumDistance}-cell radius`
    );
    const preparedOriginalCount = fill.preparedOriginalValidity.reduce(
      (sum, value) => sum + (value ? 1 : 0),
      0
    );
    const preparedFilledCount = fill.preparedFilled.reduce(
      (sum, value) => sum + (value ? 1 : 0),
      0
    );
    const preparedFilteredCount = fill.preparedFiltered.reduce(
      (sum, value) => sum + (value ? 1 : 0),
      0
    );
    ctx.setReadout(
      'preparationEvidence',
      `${preparedOriginalCount.toLocaleString()} original valid · ${preparedFilledCount.toLocaleString()} filled · ${preparedFilteredCount.toLocaleString()} rank-filter changed · ${fill.preparedUnresolved.toLocaleString()} unresolved after preparation${ctx.options.experiment === 'held-out' ? ` · prepared holdout RMSE ${fill.preparedRmse.toFixed(4)} / MAE ${fill.preparedMae.toFixed(4)}` : ' · September truth unavailable'}${ctx.options.preparationFilter === 'median' ? ` · median radius ${ctx.options.preparationRadius} · ${ctx.options.preparationNoData} invalid-neighbor policy · ${ctx.options.preparationBorderMode} border` : ' · median filter off'}`
    );
    ctx.setReadout(
      'holdoutError',
      ctx.options.experiment === 'september-gaps'
        ? 'No withheld truth for September gaps; error metrics are intentionally unavailable.'
        : `${fill.errorCount.toLocaleString()} / ${fill.targetCount.toLocaleString()} withheld cells evaluated · RMSE ${fill.rmse.toFixed(4)} · MAE ${fill.mae.toFixed(4)} reflectance`
    );
    const selectedRow = Math.floor(selectedCell / GRID_SIZE);
    const selectedColumn = selectedCell % GRID_SIZE;
    const selectedWasHeldOut =
      ctx.options.experiment === 'held-out' &&
      (ctx.options.holdoutPattern === 'block'
        ? blockHoldout[selectedCell]
        : isolatedHoldout[selectedCell]);
    const observed =
      ctx.options.experiment === 'september-gaps'
        ? septemberValues[selectedCell]
        : selectedWasHeldOut
          ? Number.NaN
          : julyValues[selectedCell];
    const truth = selectedWasHeldOut ? julyValues[selectedCell] : Number.NaN;
    const selectedFill = fill.values[selectedCell];
    const selectedPrepared = fill.preparedValues[selectedCell];
    const selectedWasOriginal = fill.preparedOriginalValidity[selectedCell] !== 0;
    const selectedWasFilled = fill.preparedFilled[selectedCell] !== 0;
    const selectedWasFiltered = fill.preparedFiltered[selectedCell] !== 0;
    ctx.setReadout(
      'selectedPixel',
      `Source row ${SOURCE_ROW + selectedRow}, column ${SOURCE_COLUMN + selectedColumn} · observed ${Number.isFinite(observed) ? observed.toFixed(4) : 'missing'} · withheld July truth ${Number.isFinite(truth) ? truth.toFixed(4) : 'not available'} · ${ctx.options.fillMethod} ${Number.isFinite(selectedFill) ? selectedFill.toFixed(4) : 'unresolved'}`
    );
    ctx.setReadout(
      'preparationPixel',
      `Prepared ${Number.isFinite(selectedPrepared) ? selectedPrepared.toFixed(4) : 'unresolved'} · original-valid ${selectedWasOriginal ? 'yes' : 'no'} · filled ${selectedWasFilled ? 'yes' : 'no'} · filtered ${selectedWasFiltered ? 'yes' : 'no'} · ${ctx.options.experiment === 'september-gaps' ? 'real gap values have no truth scores' : `prepared holdout ${fill.preparedErrorCount.toLocaleString()} cells · RMSE ${fill.preparedRmse.toFixed(4)} · MAE ${fill.preparedMae.toFixed(4)}`}`
    );
    const summary = latestSummary;
    const heldoutKeys = FILL_METHODS.map(method => getFillKey(ctx.options.holdoutPattern, method));
    if (ctx.options.experiment === 'held-out') {
      ctx.setChart('fillErrorChart', {
        kind: 'bars',
        title: `${patternLabel} · withheld-truth error`,
        labels: FILL_METHODS.map(method => FILL_LABELS[method]),
        values: heldoutKeys.map(key => summary.fills[key].rmse),
        horizontal: true,
        formatY: value => value.toFixed(4),
        description: `GPU fill maps are compared with the July values deliberately withheld from ${patternLabel}. Only these removed July cells contribute to the error; no metric is computed for real September gaps.`
      });
    } else ctx.setChart('fillErrorChart', null);
    const sampleSeries = [
      ['measured July', julyValues[selectedCell]],
      ['dilate', latestSummary.morphology.dilate.values[selectedCell]],
      ['erode', latestSummary.morphology.erode.values[selectedCell]],
      ['opening', latestSummary.morphology.opening.values[selectedCell]],
      ['closing', latestSummary.morphology.closing.values[selectedCell]],
      ['reconstruction', latestSummary.reconstruction.values[selectedCell]]
    ] as const;
    ctx.setChart('structureChart', {
      kind: 'bars',
      title: 'NIR reflectance at selected cell',
      labels: sampleSeries.map(([label]) => label),
      values: sampleSeries.map(([, value]) => (Number.isFinite(value) ? value : Number.NaN)),
      horizontal: true,
      xDomain: [0, NIR_RANGE[1]],
      formatX: value => value.toFixed(3),
      description:
        'The map and chart share the selected cell. Morphology and grayscale reconstruction operate on measured July NIR; bars are GPU outputs except the directly loaded July reference.'
    });
    const reconstructionState = latestSummary.reconstruction.converged
      ? `converged within ${ctx.options.maximumIterations} rounds`
      : `not converged within ${ctx.options.maximumIterations} rounds; no partial reconstruction is published`;
    ctx.setReadout(
      'reconstructionStatus',
      `${ctx.options.reconstructionMode} by ${ctx.options.reconstructionMode === 'opening' ? 'dilation' : 'erosion'} · ${ctx.options.reconstructionConnectivity}-connected · ${reconstructionState}`
    );
    ctx.setReadout(
      'holeFillStatus',
      `Binary support mask · ${ctx.options.holeConnectivity}-connected background · ${latestSummary.holes.converged ? `${latestSummary.holes.filledCount.toLocaleString()} enclosed missing cells classified inside holes` : 'not converged; no topology result is shown'}`
    );
    const caseCounts = new Uint32Array(16);
    for (const cellCase of summary.contours.cases) caseCounts[cellCase & 15]++;
    const classifiedCells = summary.contours.segmentCounts.reduce(
      (sum, count) => sum + (count > 0 ? 1 : 0),
      0
    );
    ctx.setReadout(
      'contourEvidence',
      `July NIR ${ctx.options.contourLevel.toFixed(2)} isoline · ${latestSummary.contours.count.toLocaleString()} of ${latestSummary.contours.required.toLocaleString()} required segments · ${classifiedCells.toLocaleString()} cells cross the level · overflow ${latestSummary.contours.overflow ? 'yes' : 'no'}`
    );
    ctx.setReadout(
      'contourCases',
      `Marching-squares case totals (0–15): ${Array.from(caseCounts)
        .map((count, index) => `${index}:${count}`)
        .join(' · ')}`
    );
  }

  ctx.setStatus('Compiling bounded fill, morphology, reconstruction and contour graphs');
  const [initialFill, initialMorphology, initialReconstruction, initialContours, initialHoles] =
    await Promise.all([
      buildFillGraph(
        Number(ctx.options.maximumDistance),
        ctx.options.preparationFilter,
        Number(ctx.options.preparationRadius),
        ctx.options.preparationNoData,
        ctx.options.preparationBorderMode
      ),
      buildMorphologyGraph(Number(ctx.options.morphologyRadius), ctx.options.structuringElement),
      buildReconstructionGraph(
        ctx.options.reconstructionMode,
        Number(ctx.options.reconstructionConnectivity) as 4 | 8,
        Number(ctx.options.maximumIterations)
      ),
      buildContourGraph(),
      buildHoleGraph(
        ctx.options.experiment,
        ctx.options.holdoutPattern,
        Number(ctx.options.holeConnectivity) as 4 | 8
      )
    ]);
  ctx.signal.throwIfAborted();
  let fillCompiled = resources.track(initialFill);
  let morphologyCompiled = resources.track(initialMorphology);
  let reconstructionCompiled = resources.track(initialReconstruction);
  const contourCompiled = resources.track(initialContours);
  let holeCompiled = resources.track(initialHoles);
  const fillSwap = resources.track(new GraphSwap());
  const morphologySwap = resources.track(new GraphSwap());
  const reconstructionSwap = resources.track(new GraphSwap());
  const holeSwap = resources.track(new GraphSwap());
  let dirtyFill = true;
  let dirtyMorphology = true;
  let dirtyReconstruction = true;
  let dirtyContours = true;
  let dirtyHoles = true;
  let revision = 0;
  let requestedRevision = -1;
  let acceptedRevision = -1;
  let latestSummary: MissingSummary | null = null;

  const registrationReader = new SummaryReader(
    resources,
    'missing-observations-readback',
    readbackSources,
    bytes => {
      if (requestedRevision !== revision) return;
      latestSummary = summarize(bytes);
      acceptedRevision = revision;
      publishReadouts();
      ctx.requestLayers();
    }
  );

  const markerValues = new Float32Array(2);
  const cropOutlineValues = makeContourOutline(frame);
  const dynamicMarker = resources.createBuffer('selected-cell-position', markerValues);
  dynamicMarker.write(
    Float32Array.from(
      frame.cellToMeters(selectedCell % GRID_SIZE, Math.floor(selectedCell / GRID_SIZE))
    )
  );
  const sourceOutline = resources.createBuffer('source-window-outline', cropOutlineValues);
  ctx.setReadout('observedSupport', 'Loading source support…');
  ctx.setReadout('fillSupport', 'Bounded fill variants are compiling…');
  ctx.setReadout('holdoutError', 'Withheld-truth errors are compiling…');
  ctx.setReadout(
    'selectedPixel',
    'Select a cell on the map to inspect measured and filled values.'
  );
  ctx.setReadout('reconstructionStatus', 'Bounded reconstruction is compiling…');
  ctx.setReadout('holeFillStatus', 'Binary support-hole classification is compiling…');
  ctx.setReadout('contourEvidence', 'NIR contour geometry is compiling…');
  ctx.setReadout('contourCases', 'Cell classifications are compiling…');
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => [
      fillCompiled,
      morphologyCompiled,
      reconstructionCompiled,
      contourCompiled,
      holeCompiled
    ],
    encode(commandEncoder) {
      if (
        fillSwap.pending ||
        morphologySwap.pending ||
        reconstructionSwap.pending ||
        holeSwap.pending
      ) {
        registrationReader.flush(commandEncoder);
        return;
      }
      if (dirtyFill) {
        fillCompiled.encode(commandEncoder, {parameters: undefined});
        dirtyFill = false;
      }
      if (dirtyMorphology) {
        morphologyCompiled.encode(commandEncoder, {parameters: undefined});
        dirtyMorphology = false;
      }
      if (dirtyReconstruction) {
        reconstructionCompiled.encode(commandEncoder, {parameters: undefined});
        dirtyReconstruction = false;
      }
      if (dirtyContours) {
        contourCompiled.encode(commandEncoder, {parameters: undefined});
        dirtyContours = false;
      }
      if (dirtyHoles) {
        holeCompiled.encode(commandEncoder, {parameters: undefined});
        dirtyHoles = false;
      }
      if (
        acceptedRevision !== revision &&
        requestedRevision !== revision &&
        !registrationReader.isPending
      ) {
        requestedRevision = revision;
        registrationReader.request(commandEncoder);
      }
      registrationReader.flush(commandEncoder);
    },
    getLayers() {
      const scene = ctx.options.experiment;
      const isHeldOut = scene === 'held-out';
      const sourceValues = isHeldOut
        ? ctx.options.mapLayer === 'source'
          ? displayBuffers.july
          : ctx.options.holdoutPattern === 'isolated'
            ? displayBuffers.isolatedObserved
            : displayBuffers.blockObserved
        : displayBuffers.september;
      let mapValues: Buffer = sourceValues;
      let range: readonly [number, number] = NIR_RANGE;
      let ramp: 'cividis' | 'vik' = 'cividis';
      const layers = [];
      const fillKey = getFillKey(
        scene === 'september-gaps' ? 'september' : ctx.options.holdoutPattern,
        ctx.options.fillMethod
      );
      const fill = latestSummary?.fills[fillKey];
      if (ctx.options.mapLayer === 'fill') {
        mapValues = fill ? fillBuffers[fillKey].values : displayBuffers.empty;
      } else if (ctx.options.mapLayer === 'prepared') {
        mapValues = fill ? fillBuffers[fillKey].preparedValues : displayBuffers.empty;
      } else if (ctx.options.mapLayer === 'filled-cells') {
        mapValues = fill ? fillBuffers[fillKey].filled : displayBuffers.empty;
        range = [0, 1];
        ramp = 'cividis';
      } else if (ctx.options.mapLayer === 'error') {
        mapValues = isHeldOut && fill ? fillBuffers[fillKey].error : displayBuffers.empty;
        range = [-0.2, 0.2];
        ramp = 'vik';
      } else if (ctx.options.mapLayer === 'morphology') {
        mapValues = latestSummary
          ? morphologyBuffers[ctx.options.morphologyOperation].values
          : displayBuffers.empty;
      } else if (ctx.options.mapLayer === 'reconstruction') {
        mapValues = latestSummary?.reconstruction.converged
          ? reconstructionBuffers.values
          : displayBuffers.empty;
      } else if (ctx.options.mapLayer === 'topology-fill') {
        mapValues = latestSummary?.holes.converged
          ? holeBuffers.enclosed
          : displayBuffers.emptyMask;
        range = [0, 1];
      } else if (ctx.options.mapLayer === 'contours') {
        mapValues = displayBuffers.july;
      }
      if (ctx.options.mapLayer === 'mask') {
        layers.push(
          new UtmRasterLayer({
            id: 'missing-observations-source',
            frame,
            values: sourceValues,
            valueFormat: 'float32',
            valueRange: NIR_RANGE,
            colormap: 'cividis',
            opacity: 0.96,
            noDataColor: [0, 0, 0, 0]
          }),
          new UtmRasterLayer({
            id: 'missing-observations-mask',
            frame,
            values: isHeldOut
              ? ctx.options.holdoutPattern === 'isolated'
                ? displayBuffers.isolatedOverlay
                : displayBuffers.blockOverlay
              : displayBuffers.septemberMissing,
            valueFormat: 'float32',
            valueRange: [0, 1],
            colormap: 'mask',
            color: [224, 151, 66, 235],
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        );
      } else if (ctx.options.mapLayer === 'topology-fill') {
        layers.push(
          new UtmRasterLayer({
            id: 'missing-observations-topology-source',
            frame,
            values: sourceValues,
            valueFormat: 'float32',
            valueRange: NIR_RANGE,
            colormap: 'cividis',
            opacity: 0.96,
            noDataColor: [0, 0, 0, 0]
          }),
          new UtmRasterLayer({
            id: 'missing-observations-enclosed-gaps',
            frame,
            values: mapValues,
            valueFormat: 'float32',
            valueRange: [0, 1],
            colormap: 'mask',
            color: [224, 151, 66, 235],
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        );
      } else if (ctx.options.mapLayer === 'contours') {
        layers.push(
          new UtmRasterLayer({
            id: 'missing-observations-contour-background',
            frame,
            values: displayBuffers.july,
            valueFormat: 'float32',
            valueRange: NIR_RANGE,
            colormap: 'cividis',
            opacity: 0.96,
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'missing-observations-contour-lines',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: contourBuffers.segments,
            instanceCount: latestSummary?.contours.overflow
              ? 0
              : (latestSummary?.contours.count ?? 0),
            widthPixels: 1.6,
            color: [255, 246, 210, 245],
            outlineColor: [33, 39, 42, 235],
            outlineWidthPixels: 0.9
          })
        );
      } else {
        layers.push(
          new UtmRasterLayer({
            id: `missing-observations-${ctx.options.mapLayer}`,
            frame,
            values: mapValues,
            valueFormat: ctx.options.mapLayer === 'filled-cells' ? 'uint32' : 'float32',
            valueRange: range,
            colormap: ramp,
            opacity: 0.96,
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'missing-observations-crop-outline',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: sourceOutline,
          instanceCount: 4,
          widthPixels: 1.5,
          color: [255, 247, 220, 245],
          outlineColor: [35, 40, 45, 240],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisPointLayer({
          id: 'missing-observations-selected-cell',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          positions: dynamicMarker,
          instanceCount: 1,
          radiusPixels: 6,
          color: [30, 34, 39, 255],
          outlineColor: [255, 255, 255, 255],
          outlineWidthPixels: 1.5
        })
      );
      return layers;
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [columnValue, rowValue] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnValue);
      const row = Math.floor(rowValue);
      if (column < 0 || row < 0 || column >= GRID_SIZE || row >= GRID_SIZE) return null;
      const index = row * GRID_SIZE + column;
      const sourceValue =
        ctx.options.experiment === 'september-gaps' ? septemberValues[index] : julyValues[index];
      const output =
        latestSummary?.fills[
          getFillKey(
            ctx.options.experiment === 'september-gaps' ? 'september' : ctx.options.holdoutPattern,
            ctx.options.fillMethod
          )
        ];
      const mapped = output?.validity[index] ? output.values[index].toFixed(4) : 'unresolved';
      const prepared = output?.preparedValidity[index]
        ? output.preparedValues[index].toFixed(4)
        : 'unresolved';
      return ctx.options.mapLayer === 'prepared'
        ? `NIR source ${Number.isFinite(sourceValue) ? sourceValue.toFixed(4) : 'missing'} · prepared ${prepared} · original-valid ${output?.preparedOriginalValidity[index] ? 'yes' : 'no'} · filled ${output?.preparedFilled[index] ? 'yes' : 'no'} · filtered ${output?.preparedFiltered[index] ? 'yes' : 'no'}`
        : `NIR source ${Number.isFinite(sourceValue) ? sourceValue.toFixed(4) : 'missing'} · ${ctx.options.fillMethod} ${mapped} · July held-out truth is shown only in the error summary`;
    },
    onClick(event) {
      if (!event.coordinate) return false;
      const [columnValue, rowValue] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnValue);
      const row = Math.floor(rowValue);
      if (column < 0 || row < 0 || column >= GRID_SIZE || row >= GRID_SIZE) return false;
      selectedCell = row * GRID_SIZE + column;
      dynamicMarker.write(Float32Array.from(frame.cellToMeters(column, row)));
      if (latestSummary && acceptedRevision === revision) publishReadouts();
      ctx.requestLayers();
      return true;
    },
    setOption(id) {
      const invalidate = () => {
        revision++;
        requestedRevision = -1;
        acceptedRevision = -1;
        latestSummary = null;
        ctx.setChart('fillErrorChart', null);
        ctx.setChart('structureChart', null);
        for (const readout of [
          'observedSupport',
          'fillSupport',
          'holdoutError',
          'selectedPixel',
          'preparationEvidence',
          'preparationPixel',
          'reconstructionStatus',
          'holeFillStatus',
          'contourEvidence',
          'contourCases'
        ])
          ctx.setReadout(readout, 'Computing the selected experiment…');
      };
      if (
        id === 'maximumDistance' ||
        id === 'preparationFilter' ||
        id === 'preparationRadius' ||
        id === 'preparationNoData' ||
        id === 'preparationBorderMode'
      ) {
        const maximumDistance = Number(ctx.options.maximumDistance);
        const preparationFilter = ctx.options.preparationFilter;
        const preparationRadius = Number(ctx.options.preparationRadius);
        const preparationNoData = ctx.options.preparationNoData;
        const preparationBorderMode = ctx.options.preparationBorderMode;
        invalidate();
        ctx.setReadout('fillSupport', 'Compiling bounded fill and surface preparation…');
        void fillSwap
          .request({
            build: () =>
              buildFillGraph(
                maximumDistance,
                preparationFilter,
                preparationRadius,
                preparationNoData,
                preparationBorderMode
              ),
            commit: next => {
              const previous = fillCompiled;
              fillCompiled = resources.track(next);
              resources.release(previous);
              dirtyFill = true;
              revision++;
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Fill graph compile failed: ${String(error)}`));
      } else if (id === 'morphologyRadius' || id === 'structuringElement') {
        const radius = Number(ctx.options.morphologyRadius);
        const structuringElement = ctx.options.structuringElement;
        invalidate();
        void morphologySwap
          .request({
            build: () => buildMorphologyGraph(radius, structuringElement),
            commit: next => {
              const previous = morphologyCompiled;
              morphologyCompiled = resources.track(next);
              resources.release(previous);
              dirtyMorphology = true;
              revision++;
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Morphology graph compile failed: ${String(error)}`));
      } else if (
        id === 'reconstructionMode' ||
        id === 'reconstructionConnectivity' ||
        id === 'maximumIterations'
      ) {
        const mode = ctx.options.reconstructionMode;
        const connectivity = Number(ctx.options.reconstructionConnectivity) as 4 | 8;
        const maximumIterations = Number(ctx.options.maximumIterations);
        invalidate();
        ctx.setReadout('reconstructionStatus', 'Compiling the selected reconstruction budget…');
        void reconstructionSwap
          .request({
            build: () => buildReconstructionGraph(mode, connectivity, maximumIterations),
            commit: next => {
              const previous = reconstructionCompiled;
              reconstructionCompiled = resources.track(next);
              resources.release(previous);
              dirtyReconstruction = true;
              revision++;
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Reconstruction graph compile failed: ${String(error)}`));
      } else if (id === 'holeConnectivity' || id === 'experiment' || id === 'holdoutPattern') {
        const experiment = ctx.options.experiment;
        const holdoutPattern = ctx.options.holdoutPattern;
        const connectivity = Number(ctx.options.holeConnectivity) as 4 | 8;
        invalidate();
        ctx.setReadout(
          'holeFillStatus',
          'Compiling the selected binary support mask and connectivity…'
        );
        void holeSwap
          .request({
            build: () => buildHoleGraph(experiment, holdoutPattern, connectivity),
            commit: next => {
              const previous = holeCompiled;
              holeCompiled = resources.track(next);
              resources.release(previous);
              dirtyHoles = true;
              revision++;
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Support-hole graph compile failed: ${String(error)}`));
      } else if (id === 'contourLevel') {
        contourBuffers.level.write(Float32Array.of(ctx.options.contourLevel));
        invalidate();
        dirtyContours = true;
      } else if (latestSummary && acceptedRevision === revision) publishReadouts();
      ctx.requestLayers();
    },
    destroy() {
      registrationReader.release();
      resources.destroy();
    }
  };
}
