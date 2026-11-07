// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterRegistration,
  GPURasterReprojection,
  type GPURasterBufferBand,
  type GPURasterMetadata
} from '@luma-contributors/graph/gpu-raster';
import {
  compileProjectionProgram,
  evaluateProjectionPlan,
  GPUProjectionProgram
} from '@luma-contributors/graph/gpu-project';
import {GPU_WGS84_SEMI_MAJOR_AXIS_METERS} from '@luma-contributors/graph/gpu-spatial-analysis';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {GraphSwap} from '../../engine/graph-swap';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {fetchBytes} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {createMercatorGridFrame} from './grid-alignment-frame';
import {
  createGridAlignmentProjection,
  getGridAlignmentDestinationCenter
} from './grid-alignment-projection';

const DATASET_ID = 'dixie-spectral';
const SOURCE_WIDTH = 250;
const SOURCE_HEIGHT = 250;
const SOURCE_ROW = 0;
const SOURCE_COLUMN = 32;
const GRID_SIZE = 64;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;
const SOURCE_CELL_SIZE = 60;
const PERIODIC_SHIFT: readonly [number, number] = [4, -3];
const MASKED_SHIFT: readonly [number, number] = [3, 2];
const DATES = ['before', 'after'] as const;
const REPROJECTION_METHODS = ['nearest', 'bilinear'] as const;
const NODATA_POLICIES = ['strict', 'renormalize'] as const;

export type GridAlignmentOptions = {
  experiment: 'periodic' | 'masked-linear';
  product: 'reference' | 'moving' | 'aligned' | 'difference' | 'reprojected';
  reprojectDate: (typeof DATES)[number];
  method: (typeof REPROJECTION_METHODS)[number];
  noDataPolicy: (typeof NODATA_POLICIES)[number];
  upsampleFactor: '10' | '20' | '32';
  ambiguityTolerance: '0.0001' | '0.01' | '0.05';
  maximumShift: '4' | '8' | '12';
  minimumOverlapFraction: '0.5' | '0.7' | '0.85';
  minimumOverlapCount: '512' | '1024' | '2048';
  minimumCorrelation: '-0.5' | '0' | '0.25';
};

type RegistrationBuffers = {
  shift: Buffer;
  peak: Buffer;
  status: Buffer;
  overlap: Buffer;
  aligned: Buffer;
  difference: Buffer;
};
type ReprojectionBuffers = {values: Buffer; validity: Buffer};
type AlignmentSummary = {
  shift: Float32Array;
  peak: number;
  status: number;
  overlap: number;
  aligned: Float32Array;
  difference: Float32Array;
  reprojections: Record<string, {values: Float32Array; validity: Uint32Array}>;
};

const REGISTRATION_LABELS = {
  periodic: 'Periodic phase correlation',
  'masked-linear': 'Masked linear correlation'
} as const;

function readFloat32(bytes: ArrayBuffer): Float32Array {
  const view = new DataView(bytes);
  const values = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < values.length; index++)
    values[index] = view.getFloat32(index * 4, true);
  return values;
}

/** A measured Dixie crop, a known shift experiment and a separate CRS-aware resampling comparison. */
export async function createGridAlignment(
  ctx: SceneContext<GridAlignmentOptions>
): Promise<SceneInstance<GridAlignmentOptions>> {
  const dataset = ctx.datasets.get(DATASET_ID);
  ctx.setStatus('Loading the pinned Sentinel-2 UTM observations');
  const [beforeRaster, afterRaster, beforeMaskBytes, afterMaskBytes] = await Promise.all([
    dataset.loadRaster('nir_before', ctx.signal),
    dataset.loadRaster('nir_after', ctx.signal),
    fetchBytes(dataset.fileUrl('validity_before.bin'), ctx.signal),
    fetchBytes(dataset.fileUrl('validity_after.bin'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const beforeValues = Float32Array.from(beforeRaster.values);
  const afterValues = Float32Array.from(afterRaster.values);
  const beforeValidity = Uint32Array.from(new Uint8Array(beforeMaskBytes), value =>
    value ? 1 : 0
  );
  const afterValidity = Uint32Array.from(new Uint8Array(afterMaskBytes), value => (value ? 1 : 0));
  const reference = new Float32Array(CELL_COUNT);
  const referenceValidity = new Uint32Array(CELL_COUNT);
  for (let row = 0; row < GRID_SIZE; row++)
    for (let column = 0; column < GRID_SIZE; column++) {
      const localCell = row * GRID_SIZE + column;
      const sourceCell = (SOURCE_ROW + row) * SOURCE_WIDTH + SOURCE_COLUMN + column;
      const value = beforeValues[sourceCell];
      const valid = beforeValidity[sourceCell] !== 0 && Number.isFinite(value);
      if (!valid) throw new Error('The pinned July registration crop must be fully observed');
      reference[localCell] = value;
      referenceValidity[localCell] = 1;
    }

  const periodicMoving = new Float32Array(CELL_COUNT);
  for (let row = 0; row < GRID_SIZE; row++)
    for (let column = 0; column < GRID_SIZE; column++) {
      const sourceColumn = (column - PERIODIC_SHIFT[0] + GRID_SIZE) % GRID_SIZE;
      const sourceRow = (row - PERIODIC_SHIFT[1] + GRID_SIZE) % GRID_SIZE;
      periodicMoving[row * GRID_SIZE + column] = reference[sourceRow * GRID_SIZE + sourceColumn];
    }
  const maskedMoving = new Float32Array(CELL_COUNT);
  const maskedMovingDisplay = Float32Array.from({length: CELL_COUNT}, () => Number.NaN);
  const maskedMovingValidity = new Uint32Array(CELL_COUNT);
  for (let row = 0; row < GRID_SIZE; row++)
    for (let column = 0; column < GRID_SIZE; column++) {
      const sourceColumn = column - MASKED_SHIFT[0];
      const sourceRow = row - MASKED_SHIFT[1];
      if (sourceColumn < 0 || sourceRow < 0 || sourceColumn >= GRID_SIZE || sourceRow >= GRID_SIZE)
        continue;
      const destination = row * GRID_SIZE + column;
      const source = sourceRow * GRID_SIZE + sourceColumn;
      maskedMoving[destination] = reference[source];
      maskedMovingDisplay[destination] = reference[source];
      maskedMovingValidity[destination] = referenceValidity[source];
    }

  const rasterSpec = dataset.rasterSpecs['nir_before'];
  const projectedBounds = rasterSpec?.['boundsProjected'] as
    | readonly [number, number, number, number]
    | undefined;
  if (!projectedBounds) throw new Error('The Sentinel source requires projected UTM bounds');
  const [minimumEasting, , , maximumNorthing] = projectedBounds;
  const cropWest = minimumEasting + SOURCE_COLUMN * SOURCE_CELL_SIZE;
  const cropNorth = maximumNorthing - SOURCE_ROW * SOURCE_CELL_SIZE;
  const cropBoundsProjected: readonly [number, number, number, number] = [
    cropWest,
    cropNorth - GRID_SIZE * SOURCE_CELL_SIZE,
    cropWest + GRID_SIZE * SOURCE_CELL_SIZE,
    cropNorth
  ];
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: cropBoundsProjected,
    cellSizeM: SOURCE_CELL_SIZE,
    width: GRID_SIZE,
    height: GRID_SIZE
  });
  const destinationCenter = getGridAlignmentDestinationCenter(
    cropBoundsProjected[0],
    cropBoundsProjected[3],
    GRID_SIZE,
    SOURCE_CELL_SIZE
  );
  const destinationFrame = createMercatorGridFrame(
    frame.origin,
    destinationCenter,
    GRID_SIZE,
    SOURCE_CELL_SIZE
  );
  const activeFrame = () => (ctx.options.product === 'reprojected' ? destinationFrame : frame);
  const cropSupport = DATES.map(date => {
    const mask = date === 'before' ? beforeValidity : afterValidity;
    let count = 0;
    for (let row = 0; row < GRID_SIZE; row++)
      for (let column = 0; column < GRID_SIZE; column++)
        count += mask[(SOURCE_ROW + row) * SOURCE_WIDTH + SOURCE_COLUMN + column] ? 1 : 0;
    return [date, count] as const;
  });
  const supportBefore = cropSupport[0][1];
  const supportAfter = cropSupport[1][1];
  const sourceGrid: GPURasterMetadata = {
    width: SOURCE_WIDTH,
    height: SOURCE_HEIGHT,
    affine: [SOURCE_CELL_SIZE, 0, 0, 0, -SOURCE_CELL_SIZE, 0],
    pixelInterpretation: 'area',
    coordinateReferenceSystem: {
      authority: 'LOCAL:UTM10N-OFFSETS',
      projectionJson: {
        parentAuthority: 'EPSG:32610',
        origin: [projectedBounds[0], projectedBounds[3]],
        units: 'local-meter-offsets'
      }
    }
  };
  const destinationGrid: GPURasterMetadata = {
    width: GRID_SIZE,
    height: GRID_SIZE,
    affine: [SOURCE_CELL_SIZE, 0, -frame.halfWidth, 0, -SOURCE_CELL_SIZE, frame.halfHeight],
    pixelInterpretation: 'area',
    coordinateReferenceSystem: {
      authority: 'LOCAL:WEB-MERCATOR-OFFSETS',
      projectionJson: {
        origin: destinationCenter,
        radius: GPU_WGS84_SEMI_MAJOR_AXIS_METERS,
        units: 'local-meter-offsets',
        scaleAtOrigin: true
      }
    }
  };
  const resources = new SpatialAnalysisResources(ctx.device, 'grid-alignment');
  const cropCorners = [
    frame.localToMeters(-frame.halfWidth, frame.halfHeight),
    frame.localToMeters(frame.halfWidth, frame.halfHeight),
    frame.localToMeters(frame.halfWidth, -frame.halfHeight),
    frame.localToMeters(-frame.halfWidth, -frame.halfHeight)
  ];
  const cropOutlineValues = new Float32Array(16);
  for (let index = 0; index < 4; index++) {
    const start = cropCorners[index];
    const end = cropCorners[(index + 1) % 4];
    cropOutlineValues.set([start[0], start[1], end[0], end[1]], index * 4);
  }
  const cropOutline = resources.createBuffer('crop-outline', cropOutlineValues);
  const destinationCorners = [
    destinationFrame.localToMeters(-frame.halfWidth, frame.halfHeight),
    destinationFrame.localToMeters(frame.halfWidth, frame.halfHeight),
    destinationFrame.localToMeters(frame.halfWidth, -frame.halfHeight),
    destinationFrame.localToMeters(-frame.halfWidth, -frame.halfHeight)
  ];
  const destinationOutlineValues = new Float32Array(16);
  for (let index = 0; index < 4; index++) {
    const start = destinationCorners[index];
    const end = destinationCorners[(index + 1) % 4];
    destinationOutlineValues.set([start[0], start[1], end[0], end[1]], index * 4);
  }
  const destinationOutline = resources.createBuffer(
    'destination-outline',
    destinationOutlineValues
  );
  const markerValues = new Float32Array(2);
  let selectedCell = 32 * GRID_SIZE + 32;
  for (
    let radius = 0;
    radius < GRID_SIZE && !afterValidity[(SOURCE_ROW + 32) * SOURCE_WIDTH + SOURCE_COLUMN + 32];
    radius++
  ) {
    for (let row = Math.max(0, 32 - radius); row <= Math.min(GRID_SIZE - 1, 32 + radius); row++) {
      for (
        let column = Math.max(0, 32 - radius);
        column <= Math.min(GRID_SIZE - 1, 32 + radius);
        column++
      ) {
        const sourceCell = (SOURCE_ROW + row) * SOURCE_WIDTH + SOURCE_COLUMN + column;
        if (
          Math.max(Math.abs(row - 32), Math.abs(column - 32)) === radius &&
          afterValidity[sourceCell]
        ) {
          selectedCell = row * GRID_SIZE + column;
          radius = GRID_SIZE;
          break;
        }
      }
    }
  }
  const selectedPosition = resources.createBuffer('selected-position', markerValues);
  const updateSelectedPosition = () => {
    const row = Math.floor(selectedCell / GRID_SIZE);
    const column = selectedCell % GRID_SIZE;
    selectedPosition.write(Float32Array.from(activeFrame().cellToMeters(column, row)));
  };
  updateSelectedPosition();
  const referenceBuffer = resources.createBuffer('reference', reference);
  const referenceMaskBuffer = resources.createBuffer('reference-mask', referenceValidity);
  const periodicMovingBuffer = resources.createBuffer('periodic-moving', periodicMoving);
  const maskedMovingBuffer = resources.createBuffer('masked-moving', maskedMoving);
  const maskedMovingDisplayBuffer = resources.createBuffer(
    'masked-moving-display',
    maskedMovingDisplay
  );
  const maskedMovingMaskBuffer = resources.createBuffer('masked-moving-mask', maskedMovingValidity);
  const fullBeforeBuffer = resources.createBuffer('source-nir-before', beforeValues);
  const fullAfterBuffer = resources.createBuffer('source-nir-after', afterValues);
  const fullBeforeMaskBuffer = resources.createBuffer('source-mask-before', beforeValidity);
  const fullAfterMaskBuffer = resources.createBuffer('source-mask-after', afterValidity);
  const missingRegistrationMap = resources.createBuffer(
    'pending-alignment-map',
    Float32Array.from({length: CELL_COUNT}, () => Number.NaN)
  );
  const registration: RegistrationBuffers = {
    shift: resources.createBuffer('estimated-shift', 8),
    peak: resources.createBuffer('correlation-peak', 4),
    status: resources.createBuffer('registration-status', 4),
    overlap: resources.createBuffer('registration-overlap', Uint32Array.of(0)),
    aligned: resources.createBuffer('estimated-alignment', CELL_COUNT * 4),
    difference: resources.createBuffer('alignment-difference', CELL_COUNT * 4)
  };
  const reprojectionBuffers = {} as Record<string, ReprojectionBuffers>;
  for (const date of DATES)
    for (const method of REPROJECTION_METHODS)
      for (const noDataPolicy of NODATA_POLICIES)
        reprojectionBuffers[`${date}-${method}-${noDataPolicy}`] = {
          values: resources.createBuffer(
            `${date}-${method}-${noDataPolicy}-values`,
            CELL_COUNT * 4
          ),
          validity: resources.createBuffer(
            `${date}-${method}-${noDataPolicy}-validity`,
            CELL_COUNT * 4
          )
        };

  let graphSerial = 0;
  const buildRegistrationGraph = async (
    options: GridAlignmentOptions
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `grid-alignment-${options.experiment}-${graphSerial++}`
    });
    const view = createGraphViewer(graph);
    const mode = options.experiment;
    const movingBuffer = mode === 'periodic' ? periodicMovingBuffer : maskedMovingBuffer;
    const movingValues = view(`${mode}-moving-values`, movingBuffer, 'float32', CELL_COUNT);
    const shiftView = view('estimated-shift-pair', registration.shift, 'float32x2', 1);
    const shiftScalars = view('estimated-shift-scalars', registration.shift, 'float32', 2);
    const statusView = view('registration-status', registration.status, 'uint32', 1);
    const output = {
      shift: shiftView,
      peak: view('correlation-peak', registration.peak, 'float32', 1),
      status: statusView,
      ...(mode === 'masked-linear'
        ? {overlap: view('registration-overlap', registration.overlap, 'uint32', 1)}
        : {})
    };
    graph.add(
      new GPURasterRegistration({
        id: `registration-${mode}`,
        grid: {
          width: GRID_SIZE,
          height: GRID_SIZE,
          affine: [1, 0, 0, 0, -1, GRID_SIZE],
          pixelInterpretation: 'area'
        },
        reference: view('reference-values', referenceBuffer, 'float32', CELL_COUNT),
        moving: movingValues,
        mode,
        ...(mode === 'periodic'
          ? {
              upsampleFactor: Number(options.upsampleFactor),
              ambiguityTolerance: Number(options.ambiguityTolerance)
            }
          : {
              referenceValidity: view(
                'reference-validity',
                referenceMaskBuffer,
                'uint32',
                CELL_COUNT
              ),
              movingValidity: view('moving-validity', maskedMovingMaskBuffer, 'uint32', CELL_COUNT),
              maximumShift: [Number(options.maximumShift), Number(options.maximumShift)] as const,
              minimumOverlapFraction: Number(options.minimumOverlapFraction),
              minimumOverlapCount: Number(options.minimumOverlapCount),
              minimumCorrelation: Number(options.minimumCorrelation)
            }),
        output
      })
    );
    const aligned = view('estimated-alignment', registration.aligned, 'float32', CELL_COUNT);
    const difference = view('alignment-difference', registration.difference, 'float32', CELL_COUNT);
    const movingMask =
      mode === 'masked-linear'
        ? view('moving-validity', maskedMovingMaskBuffer, 'uint32', CELL_COUNT)
        : view('reference-validity', referenceMaskBuffer, 'uint32', CELL_COUNT);
    addKernelPass(graph, {
      id: `apply-estimated-${mode}-shift`,
      invocationCount: CELL_COUNT,
      bindings: [
        {
          name: 'reference',
          view: view('reference-values', referenceBuffer, 'float32', CELL_COUNT),
          type: 'f32',
          access: 'read'
        },
        {name: 'moving', view: movingValues, type: 'f32', access: 'read'},
        {name: 'movingValidity', view: movingMask, type: 'u32', access: 'read'},
        {name: 'shift', view: shiftScalars, type: 'f32', access: 'read'},
        {name: 'status', view: statusView, type: 'u32', access: 'read'},
        {name: 'aligned', view: aligned, type: 'f32', access: 'read_write'},
        {name: 'difference', view: difference, type: 'f32', access: 'read_write'}
      ],
      body: `let column=i32(index%${GRID_SIZE}u);let row=i32(index/${GRID_SIZE}u);let sourceColumn=f32(column)-shift[shiftOffset];let sourceRow=f32(row)-shift[shiftOffset+1u];let baseColumn=i32(floor(sourceColumn));let baseRow=i32(floor(sourceRow));let fractionColumn=sourceColumn-f32(baseColumn);let fractionRow=sourceRow-f32(baseRow);var value=0.0;var valid=status[statusOffset]==0u;for(var rowOffset=0;rowOffset<2;rowOffset++){for(var columnOffset=0;columnOffset<2;columnOffset++){let weight=select(1.0-fractionColumn,fractionColumn,columnOffset==1)*select(1.0-fractionRow,fractionRow,rowOffset==1);if(weight>0.0000001){var sampleColumn=baseColumn+columnOffset;var sampleRow=baseRow+rowOffset;${mode === 'periodic' ? `sampleColumn=(sampleColumn%${GRID_SIZE}+${GRID_SIZE})%${GRID_SIZE};sampleRow=(sampleRow%${GRID_SIZE}+${GRID_SIZE})%${GRID_SIZE};` : ''}var sampleValid=sampleColumn>=0&&sampleRow>=0&&sampleColumn<${GRID_SIZE}&&sampleRow<${GRID_SIZE};if(sampleValid){let sampleIndex=u32(sampleRow)*${GRID_SIZE}u+u32(sampleColumn);sampleValid=movingValidity[movingValidityOffset+sampleIndex]!=0u;if(sampleValid){let sampleValue=moving[movingOffset+sampleIndex];sampleValid=sampleValue==sampleValue;if(sampleValid){value+=weight*sampleValue;}}}valid=valid&&sampleValid;}}}let referenceValue=reference[referenceOffset+index];valid=valid&&referenceValue==referenceValue&&value==value;let missing=bitcast<f32>(0x7fc00000u|(index&0u));aligned[alignedOffset+index]=select(missing,value,valid);difference[differenceOffset+index]=select(missing,value-referenceValue,valid);`
    });
    return graph.compileAsync();
  };

  const reprojectionGraph = new GPUCommandGraph<void>(ctx.device, {
    id: 'grid-alignment-reprojection'
  });
  const reprojectionView = createGraphViewer(reprojectionGraph);
  const projectionPlan = createGridAlignmentProjection(
    destinationCenter,
    [projectedBounds[0], projectedBounds[3]],
    frame.halfWidth
  );
  const localToUtm = compileProjectionProgram(projectionPlan.program, {inputFormat: 'float32x2'});
  // Test-only readback makes precision-boundary investigations reproducible without
  // changing the scientific outputs or allocating diagnostic buffers in normal use.
  const includeProjectionDiagnostics = new URLSearchParams(window.location.search).has(
    'gridSupportDiagnostics'
  );
  let diagnosticCoordinates: Buffer | undefined;
  let diagnosticProjected: Buffer | undefined;
  if (includeProjectionDiagnostics) {
    const destinationCenters = new Float32Array(CELL_COUNT * 2);
    for (let cellIndex = 0; cellIndex < CELL_COUNT; cellIndex++) {
      destinationCenters[cellIndex * 2] =
        ((cellIndex % GRID_SIZE) + 0.5) * SOURCE_CELL_SIZE - frame.halfWidth;
      destinationCenters[cellIndex * 2 + 1] =
        frame.halfHeight - (Math.floor(cellIndex / GRID_SIZE) + 0.5) * SOURCE_CELL_SIZE;
    }
    const diagnosticInput = resources.createBuffer('diagnostic-centers', destinationCenters);
    diagnosticProjected = resources.createBuffer('diagnostic-projected', CELL_COUNT * 8);
    diagnosticCoordinates = resources.createBuffer('diagnostic-source-coordinates', CELL_COUNT * 8);
    const projectedView = reprojectionView(
      'diagnostic-projected',
      diagnosticProjected,
      'float32x2',
      CELL_COUNT
    );
    resources
      .track(
        new GPUProjectionProgram({
          id: 'diagnostic-projection',
          projection: localToUtm,
          positions: reprojectionView(
            'diagnostic-centers',
            diagnosticInput,
            'float32x2',
            CELL_COUNT
          ),
          output: projectedView
        })
      )
      .addToGraph(reprojectionGraph);
    addKernelPass(reprojectionGraph, {
      id: 'diagnostic-source-affine',
      invocationCount: CELL_COUNT,
      bindings: [
        {name: 'positions', view: projectedView, type: 'f32', access: 'read'},
        {
          name: 'coordinates',
          view: reprojectionView(
            'diagnostic-source-coordinates',
            diagnosticCoordinates,
            'float32x2',
            CELL_COUNT
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let eastingOffset=positions[positionsOffset+index*2u];let northingOffset=positions[positionsOffset+index*2u+1u];coordinates[coordinatesOffset+index*2u]=(-60.0*eastingOffset-0.0*northingOffset)/-3600.0-0.5;coordinates[coordinatesOffset+index*2u+1u]=(-0.0*eastingOffset+60.0*northingOffset)/-3600.0-0.5;`
    });
  }
  for (const date of DATES) {
    const valuesBuffer = date === 'before' ? fullBeforeBuffer : fullAfterBuffer;
    const validityBuffer = date === 'before' ? fullBeforeMaskBuffer : fullAfterMaskBuffer;
    const band: GPURasterBufferBand<'float32'> = {
      id: `nir-${date}`,
      format: 'float32',
      validity: reprojectionView(
        `nir-${date}-validity`,
        validityBuffer,
        'uint32',
        SOURCE_WIDTH * SOURCE_HEIGHT
      ),
      storage: {
        kind: 'buffer',
        values: reprojectionView(
          `nir-${date}-values`,
          valuesBuffer,
          'float32',
          SOURCE_WIDTH * SOURCE_HEIGHT
        )
      }
    };
    for (const method of REPROJECTION_METHODS)
      for (const noDataPolicy of NODATA_POLICIES) {
        const key = `${date}-${method}-${noDataPolicy}`;
        const output = reprojectionBuffers[key];
        const reprojection = resources.track(
          new GPURasterReprojection({
            id: `reproject-${key}`,
            sourceGrid,
            destinationGrid,
            input: band,
            inverseProjection: localToUtm,
            method,
            noDataPolicy,
            output: {
              values: reprojectionView(`${key}-values`, output.values, 'float32', CELL_COUNT),
              validity: reprojectionView(`${key}-validity`, output.validity, 'uint32', CELL_COUNT)
            }
          })
        );
        reprojectionGraph.add(reprojection);
      }
  }
  const reprojectionCompiled = resources.track(await reprojectionGraph.compileAsync());
  const initialRegistration = resources.track(await buildRegistrationGraph(ctx.options));
  let registrationCompiled = initialRegistration;
  const graphSwap = resources.track(new GraphSwap());
  let dirty = true;
  let revision = 0;
  let requestedRevision = -1;
  let acceptedRevision = -1;
  let latestSummary: AlignmentSummary | null = null;
  const registrationSources = [
    {buffer: registration.shift, size: 8},
    {buffer: registration.peak, size: 4},
    {buffer: registration.status, size: 4},
    {buffer: registration.overlap, size: 4},
    {buffer: registration.aligned, size: CELL_COUNT * 4},
    {buffer: registration.difference, size: CELL_COUNT * 4},
    ...Object.values(reprojectionBuffers).flatMap(output => [
      {buffer: output.values, size: CELL_COUNT * 4},
      {buffer: output.validity, size: CELL_COUNT * 4}
    ]),
    ...(diagnosticCoordinates && diagnosticProjected
      ? [
          {buffer: diagnosticCoordinates, size: CELL_COUNT * 8},
          {buffer: diagnosticProjected, size: CELL_COUNT * 8}
        ]
      : [])
  ];
  const reprojectKeys = DATES.flatMap(date =>
    REPROJECTION_METHODS.flatMap(method =>
      NODATA_POLICIES.map(policy => `${date}-${method}-${policy}`)
    )
  );
  const registrationReader = new SummaryReader(
    resources,
    'alignment-readback',
    registrationSources,
    bytes => {
      if (requestedRevision !== revision) return;
      const values = readFloat32(bytes);
      let offset = 0;
      const shift = values.slice(offset, offset + 2);
      offset += 2;
      const peak = values[offset++];
      const status = new Uint32Array(bytes, offset * 4, 1)[0];
      offset++;
      const overlap = new Uint32Array(bytes, offset * 4, 1)[0];
      offset++;
      const aligned = values.slice(offset, offset + CELL_COUNT);
      offset += CELL_COUNT;
      const difference = values.slice(offset, offset + CELL_COUNT);
      offset += CELL_COUNT;
      const reprojections: AlignmentSummary['reprojections'] = {};
      for (const key of reprojectKeys) {
        const outputValues = values.slice(offset, offset + CELL_COUNT);
        offset += CELL_COUNT;
        const validity = new Uint32Array(bytes, offset * 4, CELL_COUNT);
        offset += CELL_COUNT;
        reprojections[key] = {values: outputValues, validity};
      }
      latestSummary = {shift, peak, status, overlap, aligned, difference, reprojections};
      if (diagnosticCoordinates) {
        const planProjectedOffsets = new Array<number>(CELL_COUNT * 2);
        const providerProjectedOffsets = new Array<number>(CELL_COUNT * 2);
        for (let cellIndex = 0; cellIndex < CELL_COUNT; cellIndex++) {
          const localCenter: [number, number] = [
            ((cellIndex % GRID_SIZE) + 0.5) * SOURCE_CELL_SIZE - frame.halfWidth,
            frame.halfHeight - (Math.floor(cellIndex / GRID_SIZE) + 0.5) * SOURCE_CELL_SIZE
          ];
          const projected = evaluateProjectionPlan(projectionPlan.plan, localCenter);
          const providerProjected = projectionPlan.evaluateProvider(localCenter);
          planProjectedOffsets[cellIndex * 2] = projected[0];
          planProjectedOffsets[cellIndex * 2 + 1] = projected[1];
          providerProjectedOffsets[cellIndex * 2] = providerProjected[0];
          providerProjectedOffsets[cellIndex * 2 + 1] = providerProjected[1];
        }
        window.dispatchEvent(
          new CustomEvent('grid-alignment-diagnostic', {
            detail: {
              sourceCoordinates: Array.from(values.slice(offset, offset + CELL_COUNT * 2)),
              destinationCenter: Array.from(destinationCenter),
              projectedOffsets: Array.from(
                values.slice(offset + CELL_COUNT * 2, offset + CELL_COUNT * 4)
              ),
              providerProjectedOffsets,
              planProjectedOffsets,
              projectionOrigin: [projectedBounds[0], projectedBounds[3]],
              sourceGridAffine: [60, 0, 0, 0, -60, 0],
              projectionPlan: {
                patchCount: projectionPlan.patchCount,
                sampledToleranceMeters: projectionPlan.sampledTolerance,
                sampledMaximumErrorMeters: projectionPlan.maximumError
              },
              validity: Object.fromEntries(
                Object.entries(reprojections).map(([key, result]) => [
                  key,
                  Array.from(result.validity)
                ])
              )
            }
          })
        );
      }
      acceptedRevision = revision;
      publishReadouts();
      ctx.requestLayers();
    }
  );

  const supportLabel = (date: 'before' | 'after', count: number) =>
    `${date === 'before' ? '13 Jul' : '21 Sep'} ${count.toLocaleString()} / ${CELL_COUNT.toLocaleString()} valid cells`;
  function publishReadouts(): void {
    if (!latestSummary || acceptedRevision !== revision) return;
    const mode = ctx.options.experiment;
    const appliedShift = mode === 'periodic' ? PERIODIC_SHIFT : MASKED_SHIFT;
    const row = Math.floor(selectedCell / GRID_SIZE);
    const column = selectedCell % GRID_SIZE;
    const statusLabel =
      latestSummary.status === 0
        ? 'accepted'
        : latestSummary.status === 1
          ? 'flat/non-finite or unusable correlation'
          : latestSummary.status === 2
            ? 'ambiguous periodic peak'
            : latestSummary.status === 3
              ? 'insufficient masked support'
              : `status ${latestSummary.status}`;
    ctx.setReadout(
      'registrationResult',
      `${REGISTRATION_LABELS[mode]} · imposed shift [column ${appliedShift[0]}, row ${appliedShift[1]}] cells · GPU correction [${latestSummary.shift[0].toFixed(2)}, ${latestSummary.shift[1].toFixed(2)}] cells · ${statusLabel} · peak ${latestSummary.peak.toFixed(4)}${mode === 'masked-linear' ? ` · paired overlap ${latestSummary.overlap.toLocaleString()} cells` : ''}`
    );
    let differenceCount = 0;
    let maximumAbsoluteDifference = 0;
    let squaredDifferenceSum = 0;
    for (const value of latestSummary.difference) {
      if (!Number.isFinite(value)) continue;
      differenceCount++;
      maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, Math.abs(value));
      squaredDifferenceSum += value * value;
    }
    const rootMeanSquareDifference = differenceCount
      ? Math.sqrt(squaredDifferenceSum / differenceCount)
      : Number.NaN;
    ctx.setReadout(
      'alignmentDifference',
      `${differenceCount.toLocaleString()} valid aligned cells · max |difference| ${maximumAbsoluteDifference.toFixed(5)} · RMS ${rootMeanSquareDifference.toFixed(5)} reflectance`
    );
    ctx.setReadout(
      'selectedPixel',
      `Probe source row ${SOURCE_ROW + row}, column ${SOURCE_COLUMN + column} · July reference ${reference[selectedCell].toFixed(3)} · synthetic moving ${mode === 'periodic' || maskedMovingValidity[selectedCell] ? (mode === 'periodic' ? periodicMoving[selectedCell] : maskedMoving[selectedCell]).toFixed(3) : 'missing'} · maps are the measured crop and a labeled shift experiment`
    );
    ctx.setChart('shiftChart', {
      kind: 'bars',
      title: 'Known transform and GPU correction',
      labels: ['applied column', 'applied row', 'GPU column correction', 'GPU row correction'],
      values: [appliedShift[0], appliedShift[1], latestSummary.shift[0], latestSummary.shift[1]],
      horizontal: true,
      formatX: value => `${value.toFixed(1)} cells`,
      description:
        'The test shift is deliberately imposed on measured July pixels. GPU correction is read from the selected registration mode; this is not measured landscape displacement.'
    });
    const key = `${ctx.options.reprojectDate}-${ctx.options.method}-${ctx.options.noDataPolicy}`;
    const projected = latestSummary.reprojections[key];
    const projectedSupport = projected.validity.reduce((sum, value) => sum + value, 0);
    const dateSupport = ctx.options.reprojectDate === 'before' ? supportBefore : supportAfter;
    ctx.setReadout(
      'reprojectionSupport',
      `${ctx.options.reprojectDate === 'before' ? '13 Jul' : '21 Sep'} NIR · ${ctx.options.method} · ${ctx.options.noDataPolicy} · ${projectedSupport.toLocaleString()} / ${CELL_COUNT.toLocaleString()} destination cells valid · source crop ${supportLabel(ctx.options.reprojectDate, dateSupport)} · selected cell ${Number.isFinite(projected.values[selectedCell]) ? projected.values[selectedCell].toFixed(3) : 'outside valid support'} · adaptive projection: ${projectionPlan.patchCount} patch, sampled fit error ${projectionPlan.maximumError.toExponential(1)} m`
    );
    const methods = REPROJECTION_METHODS.map(method => {
      const item =
        latestSummary!.reprojections[
          `${ctx.options.reprojectDate}-${method}-${ctx.options.noDataPolicy}`
        ];
      return [method, item.validity.reduce((sum, value) => sum + value, 0)] as const;
    });
    ctx.setChart('supportChart', {
      kind: 'bars',
      title: 'Valid destination support after inverse projection',
      labels: methods.map(([method]) => method),
      values: methods.map(([, count]) => count),
      horizontal: false,
      yDomain: [0, CELL_COUNT],
      formatY: value => Math.round(value).toLocaleString(),
      description:
        'Valid destination-cell counts are GPU readbacks. The source tile is the complete 250 by 250 Sentinel window, so interpolation support is not clipped at the displayed 64 by 64 destination.'
    });
  }

  ctx.setLegendData('cropFrame', frame);
  ctx.setReadout('registrationResult', 'GPU correlation is compiling…');
  ctx.setReadout('alignmentDifference', 'GPU-aligned difference summary is compiling…');
  ctx.setReadout('reprojectionSupport', 'GPU inverse-projection resamples are compiling…');
  ctx.setStatus('Compiling registration and reprojection graphs');
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => [registrationCompiled, reprojectionCompiled],
    encode(commandEncoder) {
      if (graphSwap.pending) {
        registrationReader.flush(commandEncoder);
        return;
      }
      if (dirty) {
        registrationCompiled.encode(commandEncoder, {parameters: undefined});
        reprojectionCompiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
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
      const product = ctx.options.product;
      let values: Buffer;
      let range: readonly [number, number];
      let ramp: 'vik' | 'cividis';
      if (product === 'reference') {
        values = referenceBuffer;
        range = [0, 0.5];
        ramp = 'cividis';
      } else if (product === 'moving') {
        values =
          ctx.options.experiment === 'periodic' ? periodicMovingBuffer : maskedMovingDisplayBuffer;
        range = [0, 0.5];
        ramp = 'cividis';
      } else if (product === 'aligned') {
        values = latestSummary ? registration.aligned : missingRegistrationMap;
        range = [0, 0.5];
        ramp = 'cividis';
      } else if (product === 'difference') {
        values = latestSummary ? registration.difference : missingRegistrationMap;
        range = [-0.3, 0.3];
        ramp = 'vik';
      } else {
        const key = `${ctx.options.reprojectDate}-${ctx.options.method}-${ctx.options.noDataPolicy}`;
        values = reprojectionBuffers[key].values;
        range = [0, 0.5];
        ramp = 'cividis';
      }
      if (product === 'reprojected') {
        return [
          new UtmRasterLayer({
            id: 'grid-alignment-reprojected',
            frame: destinationFrame,
            values,
            valueFormat: 'float32',
            valueRange: range,
            colormap: ramp,
            opacity: 0.94,
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'grid-alignment-destination-outline',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: destinationOutline,
            instanceCount: 4,
            widthPixels: 1.5,
            color: [255, 247, 220, 245],
            outlineColor: [35, 40, 45, 240],
            outlineWidthPixels: 1
          }),
          new SpatialAnalysisPointLayer({
            id: 'grid-alignment-reprojected-selected-cell',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            positions: selectedPosition,
            instanceCount: 1,
            radiusPixels: 6,
            color: [30, 34, 39, 255],
            outlineColor: [255, 255, 255, 255],
            outlineWidthPixels: 1.5
          })
        ];
      }
      return [
        new UtmRasterLayer({
          id: `grid-alignment-${product}`,
          frame,
          values,
          valueFormat: 'float32',
          valueRange: range,
          colormap: ramp,
          opacity: 0.94,
          noDataColor: [0, 0, 0, 0]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'grid-alignment-source-outline',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: cropOutline,
          instanceCount: 4,
          widthPixels: 1.5,
          color: [255, 247, 220, 245],
          outlineColor: [35, 40, 45, 240],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisPointLayer({
          id: 'grid-alignment-selected-cell',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          positions: selectedPosition,
          instanceCount: 1,
          radiusPixels: 6,
          color: [30, 34, 39, 255],
          outlineColor: [255, 255, 255, 255],
          outlineWidthPixels: 1.5
        })
      ];
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [columnFraction, rowFraction] = activeFrame().lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= GRID_SIZE || row >= GRID_SIZE) return null;
      const cell = row * GRID_SIZE + column;
      if (ctx.options.product === 'reprojected') {
        const key = `${ctx.options.reprojectDate}-${ctx.options.method}-${ctx.options.noDataPolicy}`;
        const value = latestSummary?.reprojections[key].values[cell];
        return `Destination row ${row}, column ${column} · ${ctx.options.method} ${ctx.options.noDataPolicy} · NIR ${value !== undefined && Number.isFinite(value) ? value.toFixed(6) : 'outside valid support'}`;
      }
      const movingValue =
        ctx.options.experiment === 'periodic'
          ? periodicMoving[cell]
          : maskedMovingValidity[cell]
            ? maskedMoving[cell].toFixed(3)
            : 'missing';
      return `July measured reference ${reference[cell].toFixed(3)} · synthetic ${ctx.options.experiment} moving value ${movingValue} · September source validity ${afterValidity[(SOURCE_ROW + row) * SOURCE_WIDTH + SOURCE_COLUMN + column] ? 'valid' : 'missing'}`;
    },
    onClick(event) {
      if (!event.coordinate) return false;
      const [columnFraction, rowFraction] = activeFrame().lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= GRID_SIZE || row >= GRID_SIZE) return false;
      selectedCell = row * GRID_SIZE + column;
      updateSelectedPosition();
      if (latestSummary) publishReadouts();
      ctx.requestLayers();
      return true;
    },
    setOption(id) {
      if (id === 'product') updateSelectedPosition();
      if (
        id === 'experiment' ||
        id === 'upsampleFactor' ||
        id === 'ambiguityTolerance' ||
        id === 'maximumShift' ||
        id === 'minimumOverlapFraction' ||
        id === 'minimumOverlapCount' ||
        id === 'minimumCorrelation'
      ) {
        const options = {...ctx.options};
        revision++;
        requestedRevision = -1;
        acceptedRevision = -1;
        latestSummary = null;
        ctx.setReadout('registrationResult', 'Compiling selected correlation settings…');
        ctx.setReadout('alignmentDifference', 'Waiting for the selected alignment graph…');
        ctx.setChart('shiftChart', null);
        void graphSwap
          .request({
            build: async () => await buildRegistrationGraph(options),
            commit: next => {
              const previous = registrationCompiled;
              registrationCompiled = resources.track(next);
              resources.release(previous);
              revision++;
              requestedRevision = -1;
              acceptedRevision = -1;
              dirty = true;
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Alignment graph compile failed: ${String(error)}`));
      } else if (latestSummary) {
        publishReadouts();
      }
      ctx.requestLayers();
    },
    destroy() {
      registrationReader.release();
      resources.destroy();
    }
  };
}
