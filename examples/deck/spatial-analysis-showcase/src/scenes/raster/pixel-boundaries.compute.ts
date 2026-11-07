// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterConservativeRemapping, GPURasterFractionalCategoricalStatistics, GPURasterFractionalCoverage, GPURasterFractionalZonalDistribution, GPURasterFractionalZonalStatistics, makeGPURasterMetadata, type GPURasterFractionalZonalDistributionProps, type GPURasterFractionalCategoricalStatisticsProps} from '@luma-contributors/analysis/raster-analysis';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {addKernelPass} from '../../engine/mode-kernels';
import {SummaryReader} from '../../engine/summary-reader';
import {fetchBytes} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';
import {clamp} from '@math.gl/core';

const DATASET_ID = 'dixie-spectral';
const SOURCE_WIDTH = 250;
const CROP_COLUMN = 60;
const CROP_ROW = 50;
const CROP_SIZE = 128;
const CELL_SIZE = 60;
const CELL_COUNT = CROP_SIZE * CROP_SIZE;
const ZONE_COUNT = 4;
const TUPLE_CAPACITY = CELL_COUNT * ZONE_COUNT;
const DESTINATION_SIZE = 33;
const DESTINATION_CELL_SIZE = 240;
const QUANTILE_COUNT = 4;
// Source-specific diagnostic tolerance: the smallest exact query overlap is 5.04 m².
// This only classifies readback diagnostics; it never discards weights from computations.
const SUPPORT_AREA_TOLERANCE = 1;
const REMAP_PHASES = [0, 120] as const;
const NORMALIZATIONS = ['destination-area', 'covered-area', 'extensive'] as const;

export type PixelBoundariesOptions = {
  sourceDate: 'before' | 'after';
  observationSupport: 'selected-date' | 'paired-dates';
  mapLayer: 'nir' | 'zone-coverage' | 'categorical' | 'remapped-values' | 'support-fraction';
  selectedZone: '0' | '1' | '2' | '3';
  quantileMethod: 'exactextract' | 'inverse-cdf';
  probability: number;
  gridPhase: '0' | '120';
  normalization: (typeof NORMALIZATIONS)[number];
};

type RemapBuffers = {values: Buffer; coverage: Buffer; status: Buffer};

const ZONE_NAMES = ['Rectangle', 'Oblique transect', 'Interior hole', 'Overlapping query'];
const CATEGORIES = ['< 0.10', '0.10–0.20', '0.20–0.30', '≥ 0.30'];

function makeZones(): {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
} {
  // Four analyst-drawn queries, in small crop-local UTM metres. Zone 2 has an interior hole.
  const rings = [
    [
      [480, 520],
      [3420, 520],
      [3420, 3220],
      [480, 3220]
    ],
    [
      [2140, 180],
      [3940, 1800],
      [3600, 2220],
      [1800, 600]
    ],
    [
      [3820, 540],
      [6720, 540],
      [6720, 3500],
      [3820, 3500]
    ],
    [
      [4720, 1320],
      [5820, 1320],
      [5820, 2380],
      [4720, 2380]
    ],
    [
      [2820, 2680],
      [5720, 2680],
      [5720, 5580],
      [2820, 5580]
    ]
  ];
  const positions = Float32Array.from(rings.flat(1).flat());
  return {
    positions,
    featureOffsets: Uint32Array.of(0, 1, 2, 3, 4),
    polygonOffsets: Uint32Array.of(0, 1, 2, 4, 5),
    ringOffsets: Uint32Array.of(0, 4, 8, 12, 16, 20)
  };
}

/** Bounded exact fractional statistics and conservative grid remapping on paired real Sentinel NIR. */
export async function createPixelBoundaries(
  ctx: SceneContext<PixelBoundariesOptions>
): Promise<SceneInstance<PixelBoundariesOptions>> {
  const dataset = ctx.datasets.get(DATASET_ID);
  ctx.setStatus('Loading pinned Sentinel-2 near-infrared reflectance and strict validity masks');
  const [beforeRaster, afterRaster, beforeMaskBytes, afterMaskBytes] = await Promise.all([
    dataset.loadRaster('nir_before', ctx.signal),
    dataset.loadRaster('nir_after', ctx.signal),
    fetchBytes(dataset.fileUrl('validity_before.bin'), ctx.signal),
    fetchBytes(dataset.fileUrl('validity_after.bin'), ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const beforeSource = beforeRaster.values as Float32Array;
  const afterSource = afterRaster.values as Float32Array;
  const beforeMask = new Uint8Array(beforeMaskBytes);
  const afterMask = new Uint8Array(afterMaskBytes);
  const before = new Float32Array(CELL_COUNT);
  const after = new Float32Array(CELL_COUNT);
  const beforeValidity = new Uint32Array(CELL_COUNT);
  const afterValidity = new Uint32Array(CELL_COUNT);
  for (let row = 0; row < CROP_SIZE; row++) {
    for (let column = 0; column < CROP_SIZE; column++) {
      const outputIndex = row * CROP_SIZE + column;
      const sourceIndex = (CROP_ROW + row) * SOURCE_WIDTH + CROP_COLUMN + column;
      if (beforeMask[sourceIndex] && Number.isFinite(beforeSource[sourceIndex])) {
        before[outputIndex] = beforeSource[sourceIndex];
        beforeValidity[outputIndex] = 1;
      } else before[outputIndex] = Number.NaN;
      if (afterMask[sourceIndex] && Number.isFinite(afterSource[sourceIndex])) {
        after[outputIndex] = afterSource[sourceIndex];
        afterValidity[outputIndex] = 1;
      } else after[outputIndex] = Number.NaN;
    }
  }
  const rasterSpec = dataset.rasterSpecs.nir_before;
  const bounds = rasterSpec?.boundsProjected;
  if (
    !Array.isArray(bounds) ||
    bounds.length !== 4 ||
    bounds.some(value => typeof value !== 'number')
  )
    throw new Error('Dixie NIR needs numeric projected bounds');
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
  const grid = makeGPURasterMetadata({
    width: CROP_SIZE,
    height: CROP_SIZE,
    cellSize: CELL_SIZE,
    origin: [0, CROP_SIZE * CELL_SIZE]
  });
  const resources = new SpatialAnalysisResources(ctx.device, 'pixel-boundaries');
  const beforeBuffer = resources.createBuffer('nir-before', before);
  const afterBuffer = resources.createBuffer('nir-after', after);
  const beforeValidityBuffer = resources.createBuffer('validity-before', beforeValidity);
  const afterValidityBuffer = resources.createBuffer('validity-after', afterValidity);
  // Explicit projected areas are distinct from observation weights: area reporting
  // remains physical while the paired-date option changes the statistical population.
  const cellAreasBuffer = resources.createBuffer(
    'explicit-planar-cell-areas',
    new Float32Array(CELL_COUNT).fill(CELL_SIZE ** 2)
  );
  const pairedObservationWeights = Float32Array.from(beforeValidity, (valid, index) =>
    valid && afterValidity[index] ? 1 : 0
  );
  const observationWeightsBuffer = resources.createBuffer(
    'observation-weights',
    ctx.options.observationSupport === 'paired-dates'
      ? pairedObservationWeights
      : new Float32Array(CELL_COUNT).fill(1)
  );
  const categoryBefore = new Uint32Array(CELL_COUNT);
  const categoryAfter = new Uint32Array(CELL_COUNT);
  for (let index = 0; index < CELL_COUNT; index++) {
    categoryBefore[index] = beforeValidity[index]
      ? clamp(Math.floor(before[index] * 10), 0, 3)
      : 0xffffffff;
    categoryAfter[index] = afterValidity[index]
      ? clamp(Math.floor(after[index] * 10), 0, 3)
      : 0xffffffff;
  }
  const categoryBeforeBuffer = resources.createBuffer('reflectance-bin-before', categoryBefore);
  const categoryAfterBuffer = resources.createBuffer('reflectance-bin-after', categoryAfter);
  const zones = makeZones();
  const positionsBuffer = resources.createBuffer('query-positions', zones.positions);
  const featureOffsetsBuffer = resources.createBuffer('query-features', zones.featureOffsets);
  const polygonOffsetsBuffer = resources.createBuffer('query-polygons', zones.polygonOffsets);
  const ringOffsetsBuffer = resources.createBuffer('query-rings', zones.ringOffsets);
  const coverageBuffers = {
    zoneIds: resources.createBuffer('coverage-zone-ids', TUPLE_CAPACITY * 4),
    cellIds: resources.createBuffer('coverage-cell-ids', TUPLE_CAPACITY * 4),
    fractions: resources.createBuffer('coverage-fractions', TUPLE_CAPACITY * 4),
    count: resources.createBuffer('coverage-count', 4),
    requiredCount: resources.createBuffer('coverage-required', 4),
    overflow: resources.createBuffer('coverage-overflow', 4),
    zoneValidity: resources.createBuffer('coverage-zone-validity', ZONE_COUNT * 4)
  };
  const distributions = {} as Record<string, Record<string, Record<string, Buffer>>>;
  const quantityStatistics: Record<
    string,
    {totals: Buffer; means: Buffer; coveredArea: Buffer; validArea: Buffer; validity: Buffer}
  > = {};
  const categories = {} as Record<string, Record<string, Buffer>>;
  const probabilities = resources.createBuffer(
    'probabilities',
    Float32Array.of(0.25, 0.5, 0.75, 0.5)
  );
  for (const date of ['before', 'after'] as const) {
    distributions[date] = {};
    for (const method of ['exactextract', 'inverse-cdf'] as const)
      distributions[date][method] = Object.fromEntries(
        [
          'means',
          'populationVariances',
          'sampleVariances',
          'effectiveSampleSizes',
          'weightSums',
          'squaredWeightSums',
          'coveredArea',
          'validArea',
          'quantiles',
          'quantileValidity',
          'validity',
          'status'
        ].map(name => [
          name,
          resources.createBuffer(
            `distribution-${date}-${method}-${name}`,
            (name === 'quantiles' || name === 'quantileValidity'
              ? ZONE_COUNT * QUANTILE_COUNT
              : ZONE_COUNT) * 4
          )
        ])
      );
    categories[date] = Object.fromEntries(
      [
        'coverageSums',
        'classAreas',
        'proportions',
        'coveredArea',
        'validArea',
        'missingArea',
        'majority',
        'minority',
        'validity'
      ].map(name => [
        name,
        resources.createBuffer(
          `categories-${date}-${name}`,
          (name === 'coverageSums' || name === 'classAreas' || name === 'proportions'
            ? ZONE_COUNT * 4
            : ZONE_COUNT) * 4
        )
      ])
    );
  }
  const remaps: Record<string, RemapBuffers> = {};
  const southFirst = (values: Float32Array | Uint32Array): Float32Array | Uint32Array => {
    const copy =
      values.constructor === Float32Array
        ? new Float32Array(values.length)
        : new Uint32Array(values.length);
    for (let row = 0; row < CROP_SIZE; row++)
      copy.set(
        values.subarray(row * CROP_SIZE, (row + 1) * CROP_SIZE),
        (CROP_SIZE - 1 - row) * CROP_SIZE
      );
    return copy;
  };
  const remapInputs = {
    before: resources.createBuffer('remap-before-south-first', southFirst(before)),
    after: resources.createBuffer('remap-after-south-first', southFirst(after)),
    beforeValidity: resources.createBuffer(
      'remap-validity-before-south-first',
      southFirst(beforeValidity)
    ),
    afterValidity: resources.createBuffer(
      'remap-validity-after-south-first',
      southFirst(afterValidity)
    ),
    beforeTotal: resources.createBuffer(
      'remap-extensive-before',
      southFirst(Float32Array.from(before, value => value * CELL_SIZE * CELL_SIZE))
    ),
    afterTotal: resources.createBuffer(
      'remap-extensive-after',
      southFirst(Float32Array.from(after, value => value * CELL_SIZE * CELL_SIZE))
    )
  };
  const remapDestinationBuffers: Record<string, Buffer> = {};
  const remapFrames = new Map<string, ReturnType<typeof createUtmFrame>>();
  const remapCoverageDisplayBuffers: Record<string, Buffer> = {};
  const coverageDisplay = Array.from({length: ZONE_COUNT}, (_, zone) =>
    resources.createBuffer(`zone-${zone}-coverage-display`, CELL_COUNT * 4)
  );
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'pixel-boundaries'});
  const view = createGraphViewer(graph);
  const zonesView = {
    positions: view('query-positions', positionsBuffer, 'float32x2', zones.positions.length / 2),
    featureOffsets: view(
      'query-features',
      featureOffsetsBuffer,
      'uint32',
      zones.featureOffsets.length
    ),
    polygonOffsets: view(
      'query-polygons',
      polygonOffsetsBuffer,
      'uint32',
      zones.polygonOffsets.length
    ),
    ringOffsets: view('query-rings', ringOffsetsBuffer, 'uint32', zones.ringOffsets.length)
  };
  const coverage = {
    zoneIds: view('coverage-zone-ids', coverageBuffers.zoneIds, 'uint32', TUPLE_CAPACITY),
    cellIds: view('coverage-cell-ids', coverageBuffers.cellIds, 'uint32', TUPLE_CAPACITY),
    fractions: view('coverage-fractions', coverageBuffers.fractions, 'float32', TUPLE_CAPACITY),
    status: {
      count: view('coverage-count', coverageBuffers.count, 'uint32', 1),
      requiredCount: view('coverage-required', coverageBuffers.requiredCount, 'uint32', 1),
      overflow: view('coverage-overflow', coverageBuffers.overflow, 'uint32', 1)
    },
    zoneValidity: view('coverage-zone-validity', coverageBuffers.zoneValidity, 'uint32', ZONE_COUNT)
  };
  graph.add(
    new GPURasterFractionalCoverage({
      id: 'fractional-coverage',
      grid,
      zones: zonesView,
      output: coverage
    })
  );
  for (const date of ['before', 'after'] as const) {
    const inputBuffer = date === 'before' ? beforeBuffer : afterBuffer;
    const validityBuffer = date === 'before' ? beforeValidityBuffer : afterValidityBuffer;
    const input = {
      id: `nir-${date}`,
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: view(`nir-${date}`, inputBuffer, 'float32', CELL_COUNT)
      },
      validity: view(`validity-${date}`, validityBuffer, 'uint32', CELL_COUNT)
    };
    const integralBuffer = resources.createBuffer(`cell-integral-${date}`, CELL_COUNT * 4);
    const integralView = view(`cell-integral-${date}`, integralBuffer, 'float32', CELL_COUNT);
    addKernelPass(graph, {
      id: `prepare-cell-integral-${date}`,
      invocationCount: CELL_COUNT,
      bindings: [
        {name: 'reflectance', view: input.storage.values, type: 'f32', access: 'read'},
        {name: 'integral', view: integralView, type: 'f32', access: 'read_write'}
      ],
      body: `integral[integralOffset+index]=reflectance[reflectanceOffset+index]*${CELL_SIZE ** 2}.0;`
    });
    for (const quantity of ['intensive', 'extensive'] as const) {
      const identifier = `${date}-${quantity}`;
      const output = {
        totals: resources.createBuffer(`quantity-${identifier}-totals`, ZONE_COUNT * 4),
        means: resources.createBuffer(`quantity-${identifier}-means`, ZONE_COUNT * 4),
        coveredArea: resources.createBuffer(`quantity-${identifier}-covered`, ZONE_COUNT * 4),
        validArea: resources.createBuffer(`quantity-${identifier}-observed`, ZONE_COUNT * 4),
        validity: resources.createBuffer(`quantity-${identifier}-validity`, ZONE_COUNT * 4)
      };
      quantityStatistics[identifier] = output;
      graph.add(
        new GPURasterFractionalZonalStatistics({
          id: `quantity-${identifier}`,
          grid,
          zoneCount: ZONE_COUNT,
          coverage,
          quantity,
          input:
            quantity === 'intensive'
              ? input
              : {
                  ...input,
                  id: `cell-integral-${date}`,
                  storage: {kind: 'buffer', values: integralView}
                },
          output: {
            totals: view(`quantity-${identifier}-totals`, output.totals, 'float32', ZONE_COUNT),
            means: view(`quantity-${identifier}-means`, output.means, 'float32', ZONE_COUNT),
            coveredArea: view(
              `quantity-${identifier}-covered`,
              output.coveredArea,
              'float32',
              ZONE_COUNT
            ),
            validArea: view(
              `quantity-${identifier}-observed`,
              output.validArea,
              'float32',
              ZONE_COUNT
            ),
            validity: view(`quantity-${identifier}-validity`, output.validity, 'uint32', ZONE_COUNT)
          }
        })
      );
    }
    for (const method of ['exactextract', 'inverse-cdf'] as const) {
      const outputBuffers = distributions[date][method];
      const distributionViews: GPURasterFractionalZonalDistributionProps['output'] = {
        means: view(
          `distribution-${date}-${method}-means`,
          outputBuffers.means,
          'float32',
          ZONE_COUNT
        ),
        populationVariances: view(
          `distribution-${date}-${method}-populationVariances`,
          outputBuffers.populationVariances,
          'float32',
          ZONE_COUNT
        ),
        sampleVariances: view(
          `distribution-${date}-${method}-sampleVariances`,
          outputBuffers.sampleVariances,
          'float32',
          ZONE_COUNT
        ),
        effectiveSampleSizes: view(
          `distribution-${date}-${method}-effectiveSampleSizes`,
          outputBuffers.effectiveSampleSizes,
          'float32',
          ZONE_COUNT
        ),
        weightSums: view(
          `distribution-${date}-${method}-weightSums`,
          outputBuffers.weightSums,
          'float32',
          ZONE_COUNT
        ),
        squaredWeightSums: view(
          `distribution-${date}-${method}-squaredWeightSums`,
          outputBuffers.squaredWeightSums,
          'float32',
          ZONE_COUNT
        ),
        coveredArea: view(
          `distribution-${date}-${method}-coveredArea`,
          outputBuffers.coveredArea,
          'float32',
          ZONE_COUNT
        ),
        validArea: view(
          `distribution-${date}-${method}-validArea`,
          outputBuffers.validArea,
          'float32',
          ZONE_COUNT
        ),
        quantiles: view(
          `distribution-${date}-${method}-quantiles`,
          outputBuffers.quantiles,
          'float32',
          ZONE_COUNT * QUANTILE_COUNT
        ),
        quantileValidity: view(
          `distribution-${date}-${method}-quantileValidity`,
          outputBuffers.quantileValidity,
          'uint32',
          ZONE_COUNT * QUANTILE_COUNT
        ),
        validity: view(
          `distribution-${date}-${method}-validity`,
          outputBuffers.validity,
          'uint32',
          ZONE_COUNT
        ),
        status: view(
          `distribution-${date}-${method}-status`,
          outputBuffers.status,
          'uint32',
          ZONE_COUNT
        )
      };
      graph.add(
        new GPURasterFractionalZonalDistribution({
          id: `distribution-${date}-${method}`,
          grid,
          input,
          zoneCount: ZONE_COUNT,
          coverage,
          probabilities: view('quantiles-probabilities', probabilities, 'float32', QUANTILE_COUNT),
          quantileMethod: method,
          observationWeights: view(
            'observation-weights',
            observationWeightsBuffer,
            'float32',
            CELL_COUNT
          ),
          cellAreas: view('explicit-planar-cell-areas', cellAreasBuffer, 'float32', CELL_COUNT),
          output: distributionViews
        })
      );
    }
    const categoryInput = {
      id: `nir-bin-${date}`,
      format: 'uint32' as const,
      scale: 1,
      offset: 0,
      storage: {
        kind: 'buffer' as const,
        values: view(
          `nir-bin-${date}`,
          date === 'before' ? categoryBeforeBuffer : categoryAfterBuffer,
          'uint32',
          CELL_COUNT
        )
      },
      validity: view(`validity-category-${date}`, validityBuffer, 'uint32', CELL_COUNT)
    };
    const categoryOutputBuffers = categories[date];
    const categoryViews: GPURasterFractionalCategoricalStatisticsProps['output'] = {
      coverageSums: view(
        `categories-${date}-coverageSums`,
        categoryOutputBuffers.coverageSums,
        'float32',
        ZONE_COUNT * 4
      ),
      classAreas: view(
        `categories-${date}-classAreas`,
        categoryOutputBuffers.classAreas,
        'float32',
        ZONE_COUNT * 4
      ),
      proportions: view(
        `categories-${date}-proportions`,
        categoryOutputBuffers.proportions,
        'float32',
        ZONE_COUNT * 4
      ),
      coveredArea: view(
        `categories-${date}-coveredArea`,
        categoryOutputBuffers.coveredArea,
        'float32',
        ZONE_COUNT
      ),
      validArea: view(
        `categories-${date}-validArea`,
        categoryOutputBuffers.validArea,
        'float32',
        ZONE_COUNT
      ),
      missingArea: view(
        `categories-${date}-missingArea`,
        categoryOutputBuffers.missingArea,
        'float32',
        ZONE_COUNT
      ),
      majority: view(
        `categories-${date}-majority`,
        categoryOutputBuffers.majority,
        'uint32',
        ZONE_COUNT
      ),
      minority: view(
        `categories-${date}-minority`,
        categoryOutputBuffers.minority,
        'uint32',
        ZONE_COUNT
      ),
      validity: view(
        `categories-${date}-validity`,
        categoryOutputBuffers.validity,
        'uint32',
        ZONE_COUNT
      )
    };
    graph.add(
      new GPURasterFractionalCategoricalStatistics({
        id: `categorical-${date}`,
        grid,
        input: categoryInput,
        zoneCount: ZONE_COUNT,
        categoryCapacity: 4,
        coverage,
        output: categoryViews
      })
    );
  }
  const preparedWeightsByPhase = new Map<number, {indices: Buffer; areas: Buffer}>();
  const preparedWeightGraphs = new Map<number, CompiledGPUCommandGraph<void>>();
  for (const phase of REMAP_PHASES) {
    const weightStride = 25;
    const weights = {
      indices: resources.createBuffer(
        `phase-${phase}-weight-indices`,
        DESTINATION_SIZE * DESTINATION_SIZE * weightStride * 4
      ),
      areas: resources.createBuffer(
        `phase-${phase}-weight-areas`,
        DESTINATION_SIZE * DESTINATION_SIZE * weightStride * 4
      )
    };
    preparedWeightsByPhase.set(phase, weights);
    const prepareGraph = new GPUCommandGraph<void>(ctx.device, {
      id: `pixel-boundaries-weights-${phase}`
    });
    const prepareView = createGraphViewer(prepareGraph);
    const prepareOutput = {
      values: resources.createBuffer(
        `phase-${phase}-prepare-values`,
        DESTINATION_SIZE * DESTINATION_SIZE * 4
      ),
      coverage: resources.createBuffer(
        `phase-${phase}-prepare-coverage`,
        DESTINATION_SIZE * DESTINATION_SIZE * 4
      ),
      status: resources.createBuffer(
        `phase-${phase}-prepare-status`,
        DESTINATION_SIZE * DESTINATION_SIZE * 4
      )
    };
    prepareGraph.add(
      new GPURasterConservativeRemapping({
        id: `prepare-overlap-weights-${phase}`,
        source: {
          width: CROP_SIZE,
          height: CROP_SIZE,
          origin: [0, 0],
          spacing: [CELL_SIZE, CELL_SIZE]
        },
        destination: {
          width: DESTINATION_SIZE,
          height: DESTINATION_SIZE,
          origin: [-phase, -phase],
          spacing: [DESTINATION_CELL_SIZE, DESTINATION_CELL_SIZE]
        },
        values: prepareView('prepare-after-values', remapInputs.after, 'float32', CELL_COUNT),
        validity: prepareView(
          'prepare-after-validity',
          remapInputs.afterValidity,
          'uint32',
          CELL_COUNT
        ),
        preparedWeights: {
          indices: prepareView(
            `prepared-indices-${phase}`,
            weights.indices,
            'uint32',
            DESTINATION_SIZE * DESTINATION_SIZE * weightStride
          ),
          areas: prepareView(
            `prepared-areas-${phase}`,
            weights.areas,
            'float32',
            DESTINATION_SIZE * DESTINATION_SIZE * weightStride
          )
        },
        weightMode: 'prepare',
        normalization: 'destination-area',
        output: {
          values: prepareView(
            'prepare-output-values',
            prepareOutput.values,
            'float32',
            DESTINATION_SIZE * DESTINATION_SIZE
          ),
          coveredArea: prepareView(
            'prepare-output-coverage',
            prepareOutput.coverage,
            'float32',
            DESTINATION_SIZE * DESTINATION_SIZE
          ),
          status: prepareView(
            'prepare-output-status',
            prepareOutput.status,
            'uint32',
            DESTINATION_SIZE * DESTINATION_SIZE
          )
        }
      })
    );
    preparedWeightGraphs.set(phase, resources.track(await prepareGraph.compileAsync()));
  }
  for (const phase of REMAP_PHASES) {
    const destination = {
      width: DESTINATION_SIZE,
      height: DESTINATION_SIZE,
      origin: [-phase, -phase] as [number, number],
      spacing: [DESTINATION_CELL_SIZE, DESTINATION_CELL_SIZE] as [number, number]
    };
    const weightStride = 25;
    const preparedWeights = preparedWeightsByPhase.get(phase)!;
    for (const date of ['after', 'before'] as const) {
      for (const normalization of NORMALIZATIONS) {
        const key = `${phase}-${date}-${normalization}`;
        const outputs: RemapBuffers = {
          values: resources.createBuffer(
            `remap-${key}-values`,
            DESTINATION_SIZE * DESTINATION_SIZE * 4
          ),
          coverage: resources.createBuffer(
            `remap-${key}-coverage`,
            DESTINATION_SIZE * DESTINATION_SIZE * 4
          ),
          status: resources.createBuffer(
            `remap-${key}-status`,
            DESTINATION_SIZE * DESTINATION_SIZE * 4
          )
        };
        remaps[key] = outputs;
        const valueBuffer =
          normalization === 'extensive'
            ? remapInputs[date === 'before' ? 'beforeTotal' : 'afterTotal']
            : remapInputs[date];
        graph.add(
          new GPURasterConservativeRemapping({
            id: `remap-${key}`,
            source: {
              width: CROP_SIZE,
              height: CROP_SIZE,
              origin: [0, 0],
              spacing: [CELL_SIZE, CELL_SIZE]
            },
            destination,
            values: view(`remap-values-${key}`, valueBuffer, 'float32', CELL_COUNT),
            validity: view(
              `remap-validity-${key}`,
              remapInputs[date === 'before' ? 'beforeValidity' : 'afterValidity'],
              'uint32',
              CELL_COUNT
            ),
            preparedWeights: {
              indices: view(
                `weights-${phase}-indices`,
                preparedWeights.indices,
                'uint32',
                DESTINATION_SIZE * DESTINATION_SIZE * weightStride
              ),
              areas: view(
                `weights-${phase}-areas`,
                preparedWeights.areas,
                'float32',
                DESTINATION_SIZE * DESTINATION_SIZE * weightStride
              )
            },
            weightMode: 'reuse',
            normalization,
            output: {
              values: view(
                `remap-output-${key}`,
                outputs.values,
                'float32',
                DESTINATION_SIZE * DESTINATION_SIZE
              ),
              coveredArea: view(
                `remap-support-${key}`,
                outputs.coverage,
                'float32',
                DESTINATION_SIZE * DESTINATION_SIZE
              ),
              status: view(
                `remap-status-${key}`,
                outputs.status,
                'uint32',
                DESTINATION_SIZE * DESTINATION_SIZE
              )
            }
          })
        );
      }
    }
    const projectedBounds: readonly [number, number, number, number] = [
      cropBounds[0] - phase,
      cropBounds[1] - phase,
      cropBounds[0] - phase + DESTINATION_SIZE * DESTINATION_CELL_SIZE,
      cropBounds[1] - phase + DESTINATION_SIZE * DESTINATION_CELL_SIZE
    ];
    const destinationFrame = createUtmFrame(dataset.defaultOrigin, {
      boundsProjected: projectedBounds,
      cellSizeM: DESTINATION_CELL_SIZE,
      width: DESTINATION_SIZE,
      height: DESTINATION_SIZE
    });
    for (const date of ['before', 'after'] as const)
      for (const normalization of NORMALIZATIONS) {
        const key = `${phase}-${date}-${normalization}`;
        remapDestinationBuffers[key] = resources.createBuffer(
          `remap-display-${key}`,
          DESTINATION_SIZE * DESTINATION_SIZE * 4
        );
        remapCoverageDisplayBuffers[key] = resources.createBuffer(
          `remap-support-display-${key}`,
          DESTINATION_SIZE * DESTINATION_SIZE * 4
        );
      }
    remapFrames.set(`${phase}`, destinationFrame);
  }
  for (let zone = 0; zone < ZONE_COUNT; zone++) {
    const output = view(
      `zone-${zone}-coverage-display`,
      coverageDisplay[zone],
      'float32',
      CELL_COUNT
    );
    addKernelPass(graph, {
      id: `zone-${zone}-coverage-reset`,
      invocationCount: CELL_COUNT,
      bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
      body: 'var missingBits = 0x7fc00000u; output[outputOffset + index] = bitcast<f32>(missingBits);'
    });
    addKernelPass(graph, {
      id: `zone-${zone}-coverage-values`,
      invocationCount: TUPLE_CAPACITY,
      bindings: [
        {name: 'zoneIds', view: coverage.zoneIds, type: 'u32', access: 'read'},
        {name: 'cellIds', view: coverage.cellIds, type: 'u32', access: 'read'},
        {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
        {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
        {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      body: `if (overflow[overflowOffset] == 0u && index < count[countOffset] && zoneIds[zoneIdsOffset + index] == ${zone}u) { output[outputOffset + cellIds[cellIdsOffset + index]] = fractions[fractionsOffset + index]; }`
    });
  }
  for (const phase of REMAP_PHASES)
    for (const date of ['before', 'after'] as const)
      for (const normalization of NORMALIZATIONS) {
        const key = `${phase}-${date}-${normalization}`;
        const source = view(
          `remap-output-${key}`,
          remaps[key].values,
          'float32',
          DESTINATION_SIZE * DESTINATION_SIZE
        );
        const display = view(
          `remap-display-${key}`,
          remapDestinationBuffers[key],
          'float32',
          DESTINATION_SIZE * DESTINATION_SIZE
        );
        const sourceSupport = view(
          `remap-support-${key}`,
          remaps[key].coverage,
          'float32',
          DESTINATION_SIZE * DESTINATION_SIZE
        );
        const displaySupport = view(
          `remap-support-display-${key}`,
          remapCoverageDisplayBuffers[key],
          'float32',
          DESTINATION_SIZE * DESTINATION_SIZE
        );
        addKernelPass(graph, {
          id: `remap-display-north-first-${key}`,
          invocationCount: DESTINATION_SIZE * DESTINATION_SIZE,
          bindings: [
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'display', view: display, type: 'f32', access: 'read_write'},
            {name: 'sourceSupport', view: sourceSupport, type: 'f32', access: 'read'},
            {name: 'displaySupport', view: displaySupport, type: 'f32', access: 'read_write'}
          ],
          body: `let column = index % ${DESTINATION_SIZE}u; let row = index / ${DESTINATION_SIZE}u; let sourceIndex = (${DESTINATION_SIZE - 1}u - row) * ${DESTINATION_SIZE}u + column; display[displayOffset + row * ${DESTINATION_SIZE}u + column] = source[sourceOffset + sourceIndex]; displaySupport[displaySupportOffset + row * ${DESTINATION_SIZE}u + column] = sourceSupport[sourceSupportOffset + sourceIndex] / ${DESTINATION_CELL_SIZE * DESTINATION_CELL_SIZE}.0;`
        });
      }
  const compiled = resources.track(await graph.compileAsync());
  ctx.setLegendData('cropFrame', frame);
  ctx.setLegendData('cropBounds', cropBounds);
  const summarySources = [
    {buffer: coverageBuffers.count, size: 4},
    {buffer: coverageBuffers.requiredCount, size: 4},
    {buffer: coverageBuffers.overflow, size: 4},
    {buffer: coverageBuffers.zoneValidity, size: ZONE_COUNT * 4},
    {buffer: coverageBuffers.fractions, size: TUPLE_CAPACITY * 4},
    ...(['before', 'after'] as const).flatMap(date =>
      (['exactextract', 'inverse-cdf'] as const).flatMap(method => {
        const output = distributions[date][method];
        return [
          {buffer: output.means, size: ZONE_COUNT * 4},
          {buffer: output.populationVariances, size: ZONE_COUNT * 4},
          {buffer: output.sampleVariances, size: ZONE_COUNT * 4},
          {buffer: output.effectiveSampleSizes, size: ZONE_COUNT * 4},
          {buffer: output.weightSums, size: ZONE_COUNT * 4},
          {buffer: output.validArea, size: ZONE_COUNT * 4},
          {buffer: output.coveredArea, size: ZONE_COUNT * 4},
          {buffer: output.quantiles, size: ZONE_COUNT * QUANTILE_COUNT * 4},
          {buffer: output.quantileValidity, size: ZONE_COUNT * QUANTILE_COUNT * 4},
          {buffer: output.validity, size: ZONE_COUNT * 4},
          {buffer: output.status, size: ZONE_COUNT * 4}
        ];
      })
    ),
    ...(['before', 'after'] as const).flatMap(date => [
      {buffer: categories[date].classAreas, size: ZONE_COUNT * 4 * 4},
      {buffer: categories[date].proportions, size: ZONE_COUNT * 4 * 4},
      {buffer: categories[date].missingArea, size: ZONE_COUNT * 4},
      {buffer: categories[date].majority, size: ZONE_COUNT * 4},
      {buffer: categories[date].minority, size: ZONE_COUNT * 4}
    ]),
    ...Object.values(quantityStatistics).flatMap(output =>
      Object.values(output).map(buffer => ({buffer, size: ZONE_COUNT * 4}))
    ),
    ...REMAP_PHASES.flatMap(phase =>
      (['before', 'after'] as const).flatMap(date =>
        NORMALIZATIONS.flatMap(normalization => {
          const output = remaps[`${phase}-${date}-${normalization}`];
          return [
            {buffer: output.values, size: DESTINATION_SIZE * DESTINATION_SIZE * 4},
            {buffer: output.coverage, size: DESTINATION_SIZE * DESTINATION_SIZE * 4},
            {buffer: output.status, size: DESTINATION_SIZE * DESTINATION_SIZE * 4}
          ];
        })
      )
    )
  ];
  let latestSummary: {
    count: number;
    required: number;
    meaningfulCount: number;
    overflow: boolean;
    zoneValidity: Uint32Array;
    distribution: Record<
      string,
      {
        means: Float32Array;
        populationVariances: Float32Array;
        sampleVariances: Float32Array;
        effectiveSizes: Float32Array;
        weightSums: Float32Array;
        validAreas: Float32Array;
        coveredAreas: Float32Array;
        quantiles: Float32Array;
        quantileValidity: Uint32Array;
        validity: Uint32Array;
        status: Uint32Array;
      }
    >;
    category: Record<
      string,
      {
        classAreas: Float32Array;
        proportions: Float32Array;
        missingArea: Float32Array;
        majority: Uint32Array;
        minority: Uint32Array;
      }
    >;
    quantity: Record<
      string,
      {
        totals: Float32Array;
        means: Float32Array;
        coveredArea: Float32Array;
        validArea: Float32Array;
        validity: Uint32Array;
      }
    >;
    remap: Record<string, {values: Float32Array; support: Float32Array; status: Uint32Array}>;
  } | null = null;
  function displayReadouts(): void {
    if (!latestSummary) return;
    const zone = Number(ctx.options.selectedZone);
    const distribution =
      latestSummary.distribution[`${ctx.options.sourceDate}-${ctx.options.quantileMethod}`];
    ctx.setReadout(
      'coverage',
      latestSummary.overflow
        ? `Overflow: ${latestSummary.count.toLocaleString()} tuples written; ${latestSummary.required.toLocaleString()} required. Dependent summaries are invalid.`
        : `${latestSummary.count.toLocaleString()} raw GPU coverage tuples · ${latestSummary.meaningfulCount.toLocaleString()} overlaps above ${SUPPORT_AREA_TOLERANCE} m² · ${TUPLE_CAPACITY.toLocaleString()} tuple capacity · ${latestSummary.zoneValidity.reduce((sum, value) => sum + value, 0)}/4 analyst queries valid`
    );
    const zoneStatus = distribution.status[zone];
    const statsValid = !latestSummary.overflow && distribution.validity[zone] !== 0;
    const varianceValid = statsValid && (zoneStatus & 8) === 0;
    const statusLabels = [
      [1, 'incomplete coverage'],
      [2, 'invalid tuple'],
      [4, 'no valid support'],
      [8, 'insufficient sample support'],
      [16, 'non-finite result']
    ] as const;
    const statusText =
      statusLabels
        .filter(([bit]) => (zoneStatus & bit) !== 0)
        .map(([, label]) => label)
        .join(', ') || 'ok';
    ctx.setReadout(
      'zoneSummary',
      !statsValid
        ? `${ZONE_NAMES[zone]} · unavailable · status: ${statusText} · zone geometry valid: ${latestSummary.zoneValidity[zone] !== 0 ? 'yes' : 'no'}`
        : `${ZONE_NAMES[zone]} · mean ${distribution.means[zone].toFixed(3)} · population variance ${distribution.populationVariances[zone].toFixed(5)} · sample variance ${varianceValid ? distribution.sampleVariances[zone].toFixed(5) : 'insufficient support'} · effective support ${distribution.effectiveSizes[zone].toFixed(1)} cells · valid area ${(distribution.validAreas[zone] / 1e6).toFixed(6)} km² · geometric area ${(distribution.coveredAreas[zone] / 1e6).toFixed(6)} km² · ${Array.from(distribution.quantileValidity.subarray(zone * QUANTILE_COUNT, (zone + 1) * QUANTILE_COUNT)).filter(value => value !== 0).length}/4 quantiles valid · status: ${statusText} · ${ctx.options.observationSupport === 'paired-dates' ? 'paired-date' : 'selected-date'} statistical support ${((distribution.weightSums[zone] * CELL_SIZE ** 2) / 1e6).toFixed(6)} km²`
    );
    const quantileOffset = zone * QUANTILE_COUNT;
    const intensive = latestSummary.quantity[`${ctx.options.sourceDate}-intensive`];
    const extensive = latestSummary.quantity[`${ctx.options.sourceDate}-extensive`];
    ctx.setReadout(
      'quantityContract',
      intensive.validity[zone] && extensive.validity[zone]
        ? `Source-valid support · intensive integral ${intensive.totals[zone].toFixed(2)} · extensive sum ${extensive.totals[zone].toFixed(2)} reflectance·m² · difference ${(extensive.totals[zone] - intensive.totals[zone]).toFixed(4)} · area-weighted mean ${intensive.means[zone].toFixed(6)} · ${(intensive.validArea[zone] / 1e6).toFixed(6)} km² observed. Extensive inputs are GPU-derived cell integrals (reflectance × 3,600 m²); these are numerical support checks, not energy or biomass totals.`
        : 'Quantity comparison unavailable because fractional coverage or observed support is invalid.'
    );
    const quantileValues = distribution.quantiles.subarray(
      quantileOffset,
      quantileOffset + QUANTILE_COUNT
    );
    const quantilePoints = [0.25, 0.5, 0.75, ctx.options.probability]
      .map((probability, index) => ({
        probability,
        value: quantileValues[index],
        valid: distribution.quantileValidity[quantileOffset + index] !== 0
      }))
      .filter(point => point.valid)
      .sort((first, second) => first.probability - second.probability);
    ctx.setChart(
      'distributionChart',
      !statsValid || quantilePoints.length === 0
        ? null
        : {
            kind: 'line',
            title: `${ctx.options.sourceDate} NIR · ${ZONE_NAMES[zone]} weighted quantiles`,
            xLabel: 'probability',
            yLabel: 'surface reflectance',
            xDomain: [0, 1],
            yDomain: [0, 0.6],
            formatX: value => `${Math.round(value * 100)}%`,
            formatY: value => value.toFixed(2),
            description: `GPU ${ctx.options.quantileMethod} weighted quantiles from exact area fractions; invalid probabilities are omitted.`,
            series: [
              {
                label: 'Reflectance',
                x: quantilePoints.map(point => point.probability),
                y: quantilePoints.map(point => point.value),
                points: true
              }
            ]
          }
    );
    const category = latestSummary.category[ctx.options.sourceDate];
    const categoryStart = zone * 4;
    ctx.setReadout(
      'categories',
      latestSummary.overflow
        ? 'Unavailable because coverage overflowed.'
        : `${CATEGORIES.map((label, index) => `${label}: ${(category.proportions[categoryStart + index] * 100).toFixed(1)}% (${(category.classAreas[categoryStart + index] / 1e6).toFixed(3)} km²)`).join(' · ')} · missing ${(category.missingArea[zone] / 1e6).toFixed(3)} km² · majority ${CATEGORIES[category.majority[zone]] ?? 'none'} · least-present interval ${CATEGORIES[category.minority[zone]] ?? 'none'}`
    );
    const key = `${Number(ctx.options.gridPhase)}-${ctx.options.sourceDate}-${ctx.options.normalization}`;
    const remap = latestSummary.remap[key];
    let integral = 0;
    let validCells = 0;
    let validArea = 0;
    let unsupportedCells = 0;
    let failedCells = 0;
    for (let index = 0; index < remap.values.length; index++) {
      if (remap.status[index] === 1) {
        unsupportedCells++;
        continue;
      }
      if (
        remap.status[index] !== 0 ||
        !Number.isFinite(remap.values[index]) ||
        !Number.isFinite(remap.support[index])
      ) {
        failedCells++;
        continue;
      }
      validCells++;
      validArea += remap.support[index];
      integral +=
        ctx.options.normalization === 'destination-area'
          ? remap.values[index] * DESTINATION_CELL_SIZE ** 2
          : ctx.options.normalization === 'covered-area'
            ? remap.values[index] * remap.support[index]
            : remap.values[index];
    }
    ctx.setReadout(
      'remapping',
      `${DESTINATION_SIZE}×${DESTINATION_SIZE} at 240 m · phase ${ctx.options.gridPhase} m · ${ctx.options.normalization} · ${validCells.toLocaleString()} supported · ${unsupportedCells.toLocaleString()} no-overlap · ${failedCells.toLocaleString()} failed · ${(validArea / 1e6).toFixed(3)} km² valid area · ${integral.toFixed(1)} reflectance·m²`
    );
  }
  const summaryReader = new SummaryReader(
    resources,
    'pixel-boundaries-readback',
    summarySources,
    bytes => {
      if (requestedRevision !== revision) return;
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
      const count = readUint(1)[0];
      const required = readUint(1)[0];
      const overflow = readUint(1)[0] !== 0;
      const zoneValidity = readUint(ZONE_COUNT);
      const fractions = readFloat(TUPLE_CAPACITY);
      const meaningfulCount = fractions
        .subarray(0, count)
        .reduce(
          (total, fraction) =>
            total + Number(fraction * CELL_SIZE * CELL_SIZE > SUPPORT_AREA_TOLERANCE),
          0
        );
      const distribution: NonNullable<typeof latestSummary>['distribution'] = {};
      for (const date of ['before', 'after'] as const)
        for (const method of ['exactextract', 'inverse-cdf'] as const)
          distribution[`${date}-${method}`] = {
            means: readFloat(ZONE_COUNT),
            populationVariances: readFloat(ZONE_COUNT),
            sampleVariances: readFloat(ZONE_COUNT),
            effectiveSizes: readFloat(ZONE_COUNT),
            weightSums: readFloat(ZONE_COUNT),
            validAreas: readFloat(ZONE_COUNT),
            coveredAreas: readFloat(ZONE_COUNT),
            quantiles: readFloat(ZONE_COUNT * QUANTILE_COUNT),
            quantileValidity: readUint(ZONE_COUNT * QUANTILE_COUNT),
            validity: readUint(ZONE_COUNT),
            status: readUint(ZONE_COUNT)
          };
      const category: NonNullable<typeof latestSummary>['category'] = {};
      for (const date of ['before', 'after'] as const)
        category[date] = {
          classAreas: readFloat(ZONE_COUNT * 4),
          proportions: readFloat(ZONE_COUNT * 4),
          missingArea: readFloat(ZONE_COUNT),
          majority: readUint(ZONE_COUNT),
          minority: readUint(ZONE_COUNT)
        };
      const quantity: NonNullable<typeof latestSummary>['quantity'] = {};
      for (const identifier of Object.keys(quantityStatistics))
        quantity[identifier] = {
          totals: readFloat(ZONE_COUNT),
          means: readFloat(ZONE_COUNT),
          coveredArea: readFloat(ZONE_COUNT),
          validArea: readFloat(ZONE_COUNT),
          validity: readUint(ZONE_COUNT)
        };
      const remap: NonNullable<typeof latestSummary>['remap'] = {};
      for (const phase of REMAP_PHASES)
        for (const date of ['before', 'after'] as const)
          for (const normalization of NORMALIZATIONS)
            remap[`${phase}-${date}-${normalization}`] = {
              values: readFloat(DESTINATION_SIZE * DESTINATION_SIZE),
              support: readFloat(DESTINATION_SIZE * DESTINATION_SIZE),
              status: readUint(DESTINATION_SIZE * DESTINATION_SIZE)
            };
      latestSummary = {
        count,
        required,
        meaningfulCount,
        overflow,
        zoneValidity,
        distribution,
        category,
        quantity,
        remap
      };
      displayReadouts();
    }
  );
  let dirty = true;
  let revision = 0;
  let requestedRevision = -1;
  const submittedWeightPhases = new Set<number>();
  let selectedCell = 56 * CROP_SIZE + 73;
  const marker = resources.createBuffer(
    'selected-cell-marker',
    Float32Array.of(...frame.cellToMeters(73, 56))
  );
  ctx.setReadout(
    'pixel',
    `Cell 73, 56 · before ${before[selectedCell].toFixed(3)} · after ${after[selectedCell].toFixed(3)} · validity ${beforeValidity[selectedCell]}/${afterValidity[selectedCell]}`
  );
  const outlineSegments: number[] = [];
  const outlineRings = [
    [
      [480, 520],
      [3420, 520],
      [3420, 3220],
      [480, 3220]
    ],
    [
      [2140, 180],
      [3940, 1800],
      [3600, 2220],
      [1800, 600]
    ],
    [
      [3820, 540],
      [6720, 540],
      [6720, 3500],
      [3820, 3500]
    ],
    [
      [4720, 1320],
      [5820, 1320],
      [5820, 2380],
      [4720, 2380]
    ],
    [
      [2820, 2680],
      [5720, 2680],
      [5720, 5580],
      [2820, 5580]
    ]
  ];
  for (const ring of outlineRings) {
    for (let index = 0; index < ring.length; index++) {
      const first = frame.localToMeters(
        ring[index][0] - (CROP_SIZE * CELL_SIZE) / 2,
        ring[index][1] - (CROP_SIZE * CELL_SIZE) / 2
      );
      const second = frame.localToMeters(
        ring[(index + 1) % ring.length][0] - (CROP_SIZE * CELL_SIZE) / 2,
        ring[(index + 1) % ring.length][1] - (CROP_SIZE * CELL_SIZE) / 2
      );
      outlineSegments.push(first[0], first[1], second[0], second[1]);
    }
  }
  const outlineBuffer = resources.createBuffer(
    'query-outlines',
    Float32Array.from(outlineSegments)
  );
  return {
    getCompiledGraphs: () => [...preparedWeightGraphs.values(), compiled],
    encode(commandEncoder) {
      for (const phase of REMAP_PHASES) {
        if (!submittedWeightPhases.has(phase)) {
          preparedWeightGraphs.get(phase)!.encode(commandEncoder, {parameters: undefined});
          submittedWeightPhases.add(phase);
          return;
        }
      }
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (requestedRevision !== revision && !summaryReader.isPending) {
        requestedRevision = revision;
        summaryReader.request(commandEncoder);
      }
      summaryReader.flush(commandEncoder);
    },
    getLayers() {
      const selectedZone = Number(ctx.options.selectedZone);
      const gridPhase = Number(ctx.options.gridPhase) as 0 | 120;
      if (ctx.options.mapLayer === 'zone-coverage')
        return [
          new UtmRasterLayer({
            id: 'pixel-boundaries-zone-coverage',
            frame,
            values: coverageDisplay[selectedZone],
            valueFormat: 'float32',
            valueRange: [0, 1],
            colormap: 'cividis',
            opacity: 0.88,
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'pixel-boundaries-query-outlines',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: outlineBuffer,
            instanceCount: outlineSegments.length / 4,
            widthPixels: 2,
            color: [252, 242, 211, 245],
            outlineColor: [24, 31, 39, 245],
            outlineWidthPixels: 1
          }),
          new SpatialAnalysisPointLayer({
            id: 'pixel-boundaries-selected-cell',
            positions: marker,
            instanceCount: 1,
            radiusPixels: 6,
            color: [255, 245, 96, 255],
            outlineColor: [24, 31, 39, 255],
            outlineWidthPixels: 1.5
          })
        ];
      if (
        ctx.options.mapLayer === 'remapped-values' ||
        ctx.options.mapLayer === 'support-fraction'
      ) {
        const key = `${gridPhase}-${ctx.options.sourceDate}-${ctx.options.normalization}`;
        const projectedBounds: readonly [number, number, number, number] = [
          cropBounds[0] - gridPhase,
          cropBounds[1] - gridPhase,
          cropBounds[0] - gridPhase + DESTINATION_SIZE * DESTINATION_CELL_SIZE,
          cropBounds[1] - gridPhase + DESTINATION_SIZE * DESTINATION_CELL_SIZE
        ];
        const destinationFrame =
          remapFrames.get(String(gridPhase)) ??
          createUtmFrame(dataset.defaultOrigin, {
            boundsProjected: projectedBounds,
            cellSizeM: DESTINATION_CELL_SIZE,
            width: DESTINATION_SIZE,
            height: DESTINATION_SIZE
          });
        return [
          new UtmRasterLayer({
            id: `pixel-boundaries-${ctx.options.mapLayer}`,
            frame: destinationFrame,
            values:
              ctx.options.mapLayer === 'remapped-values'
                ? remapDestinationBuffers[key]
                : remapCoverageDisplayBuffers[key],
            valueFormat: 'float32',
            valueRange:
              ctx.options.mapLayer === 'remapped-values'
                ? ctx.options.normalization === 'extensive'
                  ? [0, 36000]
                  : [0, 0.6]
                : [0, 1],
            colormap: 'cividis',
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'pixel-boundaries-query-outlines',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            segments: outlineBuffer,
            instanceCount: outlineSegments.length / 4,
            widthPixels: 2,
            color: [252, 242, 211, 245],
            outlineColor: [24, 31, 39, 245],
            outlineWidthPixels: 1
          }),
          new SpatialAnalysisPointLayer({
            id: 'pixel-boundaries-selected-cell',
            positions: marker,
            instanceCount: 1,
            radiusPixels: 6,
            color: [255, 245, 96, 255],
            outlineColor: [24, 31, 39, 255],
            outlineWidthPixels: 1.5
          })
        ];
      }
      const dateBuffer = ctx.options.sourceDate === 'before' ? beforeBuffer : afterBuffer;
      const values =
        ctx.options.mapLayer === 'categorical'
          ? ctx.options.sourceDate === 'before'
            ? categoryBeforeBuffer
            : categoryAfterBuffer
          : dateBuffer;
      return [
        new UtmRasterLayer({
          id: `pixel-boundaries-${ctx.options.mapLayer}`,
          frame,
          values,
          valueFormat: ctx.options.mapLayer === 'categorical' ? 'uint32' : 'float32',
          valueRange: ctx.options.mapLayer === 'categorical' ? [0, 3] : [0, 0.6],
          noDataValue: ctx.options.mapLayer === 'categorical' ? 0xffffffff : undefined,
          colormap: ctx.options.mapLayer === 'categorical' ? 'category' : 'cividis',
          palette: [
            [50, 78, 125, 255],
            [66, 143, 150, 255],
            [224, 190, 91, 255],
            [197, 103, 66, 255]
          ],
          opacity: 0.9,
          noDataColor: [0, 0, 0, 0]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'pixel-boundaries-query-outlines',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 2,
          color: [252, 242, 211, 245],
          outlineColor: [24, 31, 39, 245],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisPointLayer({
          id: 'pixel-boundaries-selected-cell',
          positions: marker,
          instanceCount: 1,
          radiusPixels: 6,
          color: [255, 245, 96, 255],
          outlineColor: [24, 31, 39, 255],
          outlineWidthPixels: 1.5
        })
      ];
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [columnFraction, rowFraction] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= CROP_SIZE || row >= CROP_SIZE) return null;
      const index = row * CROP_SIZE + column;
      return `Cell ${column}, ${row} · before ${before[index].toFixed(3)} · after ${after[index].toFixed(3)} · validity ${beforeValidity[index]}/${afterValidity[index]}`;
    },
    onClick(event) {
      if (!event.coordinate) return false;
      const [columnFraction, rowFraction] = frame.lngLatToCell(...event.coordinate);
      const column = Math.floor(columnFraction);
      const row = Math.floor(rowFraction);
      if (column < 0 || row < 0 || column >= CROP_SIZE || row >= CROP_SIZE) return false;
      selectedCell = row * CROP_SIZE + column;
      marker.write(Float32Array.of(...frame.cellToMeters(column, row)));
      ctx.setReadout(
        'pixel',
        `Cell ${column}, ${row} · before ${before[selectedCell].toFixed(3)} · after ${after[selectedCell].toFixed(3)} · validity ${beforeValidity[selectedCell]}/${afterValidity[selectedCell]}`
      );
      ctx.requestLayers();
      return true;
    },
    setOption(id) {
      if (id === 'probability' || id === 'observationSupport') {
        if (id === 'observationSupport') {
          observationWeightsBuffer.write(
            ctx.options.observationSupport === 'paired-dates'
              ? pairedObservationWeights
              : new Float32Array(CELL_COUNT).fill(1)
          );
        }
        probabilities.write(Float32Array.of(0.25, 0.5, 0.75, ctx.options.probability));
        dirty = true;
        revision++;
        latestSummary = null;
        ctx.setChart('distributionChart', null);
        ctx.setReadout('zoneSummary', 'Updating GPU weighted distribution and support…');
      } else {
        displayReadouts();
      }
      ctx.requestLayers();
    },
    destroy() {
      summaryReader.release();
      resources.destroy();
    }
  };
}
