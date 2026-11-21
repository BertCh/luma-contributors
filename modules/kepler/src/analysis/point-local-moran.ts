// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {matchesKeplerPointWeightSpatialContext} from './point-weight-coordinates';
import {Kernel} from '@luma.gl/graph';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  GPUMask,
  GPUReduction,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPULocalMoran,
  GPULocalPermutationTest,
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  isGPUChunkedSpatialWeights,
  type GPULocalMoranOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  parseKeplerPointWeightsSettings,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS,
  MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS,
  MAXIMUM_KEPLER_PERMUTATION_DEGREE,
  type KeplerMultipleTestingMethod
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {
  getKeplerPointWeightsMaximumNeighbors,
  getKeplerPointWeightsCapacity,
  type KeplerPointWeightsBinding
} from './point-weights';
import {
  getKeplerPointWeightsRowCount,
  getKeplerPointWeightsViews,
  type KeplerPointWeightsStorage
} from './point-weights-chunked';
import {
  validateKeplerPolygonWeightsBinding,
  type KeplerPolygonWeightsBinding
} from './polygon-weights';
import {KEPLER_ANALYSIS_ROW_TILE_SIZE} from './analysis-scaling';
import {copyPointStatus} from './point-hotspots';

export type KeplerPointLocalMoranSettings = Readonly<{
  minimumNeighbors: number;
  islands: 'exclude' | 'keep';
  multipleTesting: KeplerMultipleTestingMethod;
  significanceLevel: number;
  quadrantGating: 'analytic' | 'permutation';
  permutationCount?: number;
  seed?: number;
}>;

export type KeplerPointLocalMoranOutputs = Partial<GPULocalMoranOutput> & {
  status?: {
    weightsOverflow?: GraphDataView<'uint32'>;
    weightsRequiredCount?: GraphDataView<'uint32'>;
    /** Source nulls, nonfinite coordinates and coordinates outside longitude/latitude bounds. */
    weightsInvalidCount?: GraphDataView<'uint32'>;
    /** Included finite rows with undefined inference (analytic z-score or permutation p-value). */
    invalidCount?: GraphDataView<'uint32'>;
    /** Incomplete permutation inference; explicitly zero in analytic mode. */
    inferenceOverflow?: GraphDataView<'uint32'>;
  };
};

export type KeplerPointLocalMoranProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerPointLocalMoranSettings;
  /** Explicitly packed original source rows. The adapter never aggregates or reorders points. */
  values: GraphDataView<'float32'>;
  /** Declared population intersected with source numeric validity; coordinate validity is added here. */
  mask: GraphDataView<'uint32'>;
  /** Source-wide binary geographic point topology imported into this consumer graph. */
  pointWeights: KeplerPointWeightsBinding;
  /** Supply imported views to retain source-aligned outputs beyond the compiled graph's lifetime. */
  outputs?: KeplerPointLocalMoranOutputs;
};

/**
 * Local Moran's I on original point rows using binary weights without row normalization.
 *
 * The existing contributor computes conditional-randomization moments and analytic two-sided
 * normal-approximation z-scores. The explicit permutation mode replaces p-values and gating with
 * conditional two-sided permutation inference, without sequential stopping. Quadrants 1 HH, 2 LH, 3 LL,
 * 4 HL are gated by the adjusted p-value at significanceLevel; 0 means unreported. The selected
 * population filters existing source-wide neighbors without replacing them or changing topology.
 *
 * This adapter records GPU work only. The caller compiles/submits and retains borrowed inputs.
 * After all GPU work completes, destroyParameters() releases the adapter's parameter buffer;
 * the compiled graph owns omitted outputs and scratch, while supplied outputs stay caller-owned.
 */
export type KeplerPolygonLocalMoranProps = Omit<KeplerPointLocalMoranProps, 'pointWeights'> & {
  polygonWeights: KeplerPolygonWeightsBinding;
};
type WeightedLocalMoranProps = Omit<KeplerPointLocalMoranProps, 'pointWeights'> & {
  topology: {
    source: KeplerAnalysisProvenance['source'];
    weights: KeplerPointWeightsStorage;
    mask: GraphDataView<'uint32'>;
    status: {
      overflow: GraphDataView<'uint32'>;
      requiredCount: GraphDataView<'uint32'>;
      invalidCount: GraphDataView<'uint32'>;
    };
    maximumNeighbors: number;
  };
};

export function addKeplerPointLocalMoran<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointLocalMoranProps
) {
  validatePointTopology(props);
  const {pointWeights} = props;
  const result = addWeightedLocalMoran(graph, {
    ...props,
    topology: {
      source: pointWeights.source,
      weights: pointWeights.weights,
      mask: pointWeights.coordinateMask,
      status: pointWeights.status,
      maximumNeighbors: getKeplerPointWeightsMaximumNeighbors(pointWeights.settings)
    }
  });
  return {...result, pointWeights};
}

export function addKeplerPolygonLocalMoran<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPolygonLocalMoranProps
) {
  const {polygonWeights} = props;
  validateKeplerPolygonWeightsBinding(graph, polygonWeights, props.source, props.values.length);
  const result = addWeightedLocalMoran(graph, {
    ...props,
    topology: {
      source: polygonWeights.source,
      weights: polygonWeights.weights,
      mask: polygonWeights.geometryMask,
      status: polygonWeights.status,
      maximumNeighbors: 64
    }
  });
  return {...result, polygonWeights};
}

function addWeightedLocalMoran<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WeightedLocalMoranProps
) {
  validateInputs(graph, props);
  const id = props.id ?? 'kepler-point-local-moran';
  const settings = Object.freeze({...props.settings});
  const source = Object.freeze({...props.source});
  const {values, topology} = props;
  const weights = {...topology.weights, status: topology.status};
  const rowCount = values.length;
  const provided = props.outputs ?? {};
  const localI =
    provided.localI ?? createTransientView(graph, `${id}-local-i`, 'float32', rowCount);
  const zScores =
    provided.zScores ?? createTransientView(graph, `${id}-z-scores`, 'float32', rowCount);
  const spatialLag =
    provided.spatialLag ?? createTransientView(graph, `${id}-spatial-lag`, 'float32', rowCount);
  const pValues =
    provided.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rowCount);
  const adjustedPValues =
    provided.adjustedPValues ??
    createTransientView(graph, `${id}-adjusted-p-values`, 'float32', rowCount);
  const quadrants =
    provided.quadrants ?? createTransientView(graph, `${id}-quadrants`, 'uint32', rowCount);
  const neighborCounts =
    provided.neighborCounts ??
    createTransientView(graph, `${id}-neighbor-counts`, 'uint32', rowCount);
  const globalStatistics =
    provided.globalStatistics ??
    createTransientView(
      graph,
      `${id}-global-statistics`,
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
    );
  const weightsOverflow =
    provided.status?.weightsOverflow ??
    createTransientView(graph, `${id}-weights-overflow`, 'uint32', 1);
  const weightsRequiredCount =
    provided.status?.weightsRequiredCount ??
    createTransientView(graph, `${id}-weights-required-count`, 'uint32', 1);
  const weightsInvalidCount =
    provided.status?.weightsInvalidCount ??
    createTransientView(graph, `${id}-weights-invalid-count`, 'uint32', 1);
  const invalidCount =
    provided.status?.invalidCount ??
    createTransientView(graph, `${id}-undefined-count`, 'uint32', 1);
  const inferenceOverflow =
    provided.status?.inferenceOverflow ??
    createTransientView(graph, `${id}-inference-overflow-output`, 'uint32', 1);
  const permutation = settings.quadrantGating === 'permutation';
  const ungatedQuadrants = permutation
    ? createTransientView(graph, `${id}-ungated-quadrants`, 'uint32', rowCount)
    : quadrants;
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const parameterSettings = {significanceLevel: settings.significanceLevel};
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-parameters`,
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues(parameterSettings)
  });
  let permutationParameters: GPUParameterBuffer<'uint32'> | undefined;
  try {
    const permutationSettings = permutation
      ? {
          seed: settings.seed!,
          permutationCount: settings.permutationCount!,
          significanceLevel: settings.significanceLevel
        }
      : undefined;
    if (permutationSettings)
      permutationParameters = new GPUParameterBuffer(graph.device, {
        id: `${id}-permutation-parameters`,
        format: 'uint32',
        length: GPU_PERMUTATION_PARAMETER_LENGTH,
        values: getGPUPermutationParameterValues(permutationSettings, {
          maximumPermutations: MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS
        })
      });
    const outputs = {
      localI,
      zScores,
      spatialLag,
      pValues,
      adjustedPValues,
      quadrants,
      neighborCounts,
      globalStatistics
    };
    const contributor = new GPULocalMoran({
      id,
      values,
      weights,
      mask: effectiveMask,
      parameters: parameters.importToGraph(graph),
      minimumNeighbors: settings.minimumNeighbors,
      islands: settings.islands,
      multipleTesting: permutation ? 'none' : settings.multipleTesting,
      quadrantGating: permutation ? 'none' : 'analytic',
      output: permutation
        ? {...outputs, quadrants: ungatedQuadrants, pValues: undefined, adjustedPValues: undefined}
        : outputs
    });
    graph.add(
      new GPUMask({
        id: `${id}-coordinate-validity`,
        inputs: [props.mask, topology.mask],
        output: effectiveMask
      })
    );
    graph.add(contributor);
    let inference = contributor.getInferenceMetadata(parameterSettings);
    if (permutationSettings && permutationParameters) {
      const significant = createTransientView(
        graph,
        `${id}-permutation-significant`,
        'uint32',
        rowCount
      );
      const permutationContributor = new GPULocalPermutationTest({
        id: `${id}-permutation`,
        values,
        weights,
        mask: effectiveMask,
        statistic: 'localMoran',
        alternative: 'two-sided',
        islands: settings.islands,
        multipleTesting: settings.multipleTesting,
        maximumNeighbors: topology.maximumNeighbors,
        extendedNeighborScratch: topology.maximumNeighbors > 64,
        rowTileSize: KEPLER_ANALYSIS_ROW_TILE_SIZE,
        maximumPermutations: permutationSettings.permutationCount,
        parameters: permutationParameters.importToGraph(graph),
        output: {
          exceedances: createTransientView(
            graph,
            `${id}-permutation-exceedances`,
            'uint32',
            rowCount
          ),
          pseudoPValues: pValues,
          adjustedPValues,
          significant,
          status: {overflow: inferenceOverflow}
        }
      });
      graph.add(permutationContributor);
      addQuadrantGate(graph, `${id}-permutation-gate`, ungatedQuadrants, significant, quadrants);
      const plan = permutationContributor.getInferenceMetadata(permutationSettings);
      inference = {...plan, permutationCount: plan.requestedPermutationCount};
    } else {
      addInferenceOverflowReset(graph, `${id}-inference-overflow-reset`, inferenceOverflow);
    }
    copyPointStatus(graph, `${id}-copy-overflow`, topology.status.overflow, weightsOverflow);
    copyPointStatus(
      graph,
      `${id}-copy-required-count`,
      topology.status.requiredCount,
      weightsRequiredCount
    );
    copyPointStatus(
      graph,
      `${id}-copy-coordinate-invalid-count`,
      topology.status.invalidCount,
      weightsInvalidCount
    );
    addUndefinedTally(
      graph,
      id,
      values,
      effectiveMask,
      permutation ? pValues : zScores,
      invalidCount
    );
    return {
      operation: 'local-moran' as const,
      source,
      population: props.population,
      ...(props.selectionRevision === undefined
        ? {}
        : {selectionRevision: props.selectionRevision}),
      settings,
      numericalPolicy: 'float32' as const,
      weightTransform: 'binary' as const,
      inferenceMethod: permutation
        ? ('permutation-conditional-two-sided' as const)
        : ('analytic-conditional-normal-two-sided' as const),
      inference,
      rowCount,
      values,
      effectiveMask,
      ...outputs,
      outputs,
      status: {
        weightsOverflow,
        weightsRequiredCount,
        weightsInvalidCount,
        invalidCount,
        inferenceOverflow,
        stages: [
          {stage: 'weights', status: topology.status},
          {
            stage: permutation ? 'permutation-inference' : 'analytic-inference',
            status: {invalidCount, inferenceOverflow}
          }
        ]
      },
      parameterByteLength:
        parameters.buffer.byteLength + (permutationParameters?.buffer.byteLength ?? 0),
      destroyParameters() {
        parameters.destroy();
        permutationParameters?.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    permutationParameters?.destroy();
    throw error;
  }
}

function validateInputs<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WeightedLocalMoranProps
): void {
  const {source, values, mask, topology, settings} = props;
  const rowCount = values.length;
  if (
    !source.datasetId ||
    !Number.isSafeInteger(source.revision) ||
    source.revision < 1 ||
    (props.population === 'selected'
      ? !Number.isSafeInteger(props.selectionRevision) || props.selectionRevision! < 1
      : props.population !== 'all' || props.selectionRevision !== undefined)
  )
    throw new Error('Local Moran requires current source and population provenance');
  if (
    topology.source.datasetId !== source.datasetId ||
    topology.source.revision !== source.revision
  )
    throw new Error('Point weights source revision is stale');
  if (
    rowCount < 1 ||
    rowCount >= 2 ** 24 ||
    mask.length !== rowCount ||
    topology.mask.length !== rowCount ||
    getKeplerPointWeightsRowCount(topology.weights) !== rowCount
  )
    throw new KeplerAllocationError('capacity', {
      message: 'Local Moran requires matching source rows within capacity'
    });
  if (
    !Number.isSafeInteger(settings.minimumNeighbors) ||
    settings.minimumNeighbors < 1 ||
    settings.minimumNeighbors >= 2 ** 24 ||
    (settings.islands !== 'exclude' && settings.islands !== 'keep') ||
    !['analytic', 'permutation'].includes(settings.quadrantGating) ||
    !['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].includes(
      settings.multipleTesting
    ) ||
    !(
      settings.significanceLevel > 0 &&
      settings.significanceLevel < 1 &&
      Math.fround(settings.significanceLevel) > 0 &&
      Math.fround(settings.significanceLevel) < 1
    )
  )
    throw new Error('Invalid local Moran inference settings');
  if (settings.quadrantGating === 'permutation') {
    if (
      rowCount > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATION_ROWS ||
      topology.maximumNeighbors > MAXIMUM_KEPLER_PERMUTATION_DEGREE ||
      settings.minimumNeighbors !== 1 ||
      !Number.isSafeInteger(settings.permutationCount) ||
      settings.permutationCount! < 1 ||
      settings.permutationCount! > MAXIMUM_KEPLER_LOCAL_MORAN_PERMUTATIONS ||
      !Number.isSafeInteger(settings.seed) ||
      settings.seed! < 0 ||
      settings.seed! > 2 ** 32 - 1
    )
      throw new Error('Invalid local Moran permutation bounds or inference settings');
  } else if (settings.permutationCount !== undefined || settings.seed !== undefined) {
    throw new Error('Analytic Local Moran does not accept permutation controls');
  }
  validatePackedView(values, ['float32'], 'Local Moran values');
  for (const view of [mask, topology.mask])
    validatePackedView(view, ['uint32'], 'Local Moran mask');
  for (const view of Object.values(topology.status)) {
    validatePackedView(view, ['uint32'], 'Point weights status');
    if (view.length !== 1) throw new Error('Point weights status must be scalar');
  }
  const provided = props.outputs ?? {};
  const outputViews = [
    provided.localI,
    provided.zScores,
    provided.spatialLag,
    provided.pValues,
    provided.adjustedPValues,
    provided.quadrants,
    provided.neighborCounts,
    provided.globalStatistics,
    ...Object.values(provided.status ?? {})
  ];
  for (let first = 0; first < outputViews.length; first++) {
    const firstView = outputViews[first];
    if (!firstView) continue;
    for (let second = first + 1; second < outputViews.length; second++) {
      const secondView = outputViews[second];
      if (secondView && doGraphDataViewsOverlap(firstView, secondView))
        throw new Error('Local Moran output ports must not overlap');
    }
  }
  if (provided.quadrants) {
    validatePackedView(provided.quadrants, ['uint32'], 'Local Moran quadrants');
    if (provided.quadrants.length !== rowCount)
      throw new Error('Local Moran quadrants must match source rows');
    for (const view of [
      ...outputViews,
      values,
      mask,
      topology.mask,
      ...getKeplerPointWeightsViews(topology.weights),
      ...Object.values(topology.status)
    ]) {
      if (view && view !== provided.quadrants && doGraphDataViewsOverlap(provided.quadrants, view))
        throw new Error('Local Moran quadrant output must not overlap other ports');
    }
  }
  for (const view of [
    values,
    mask,
    topology.mask,
    ...getKeplerPointWeightsViews(topology.weights),
    ...Object.values(topology.status),
    ...outputViews
  ])
    if (view && view.buffer.graph !== graph)
      throw new Error('Local Moran views must belong to the consumer graph');
  for (const view of Object.values(provided.status ?? {})) {
    validatePackedView(view, ['uint32'], 'Local Moran status');
    if (view.length !== 1) throw new Error('Local Moran status must be scalar');
  }
}

function validatePointTopology(props: KeplerPointLocalMoranProps): void {
  const {pointWeights} = props;
  const rowCount = props.values.length;
  const weightsSettings = parseKeplerPointWeightsSettings(pointWeights.settings);
  getKeplerPointWeightsCapacity(weightsSettings, rowCount);
  const information = pointWeights.weights.information;
  const context = pointWeights.weights.spatialContext;
  if (
    information?.rowCount !== rowCount ||
    information.transform !== 'binary' ||
    information.hasSelfWeights !== isGPUChunkedSpatialWeights(pointWeights.weights) ||
    information.provenance?.source !== 'GPUNeighborSearch' ||
    information.provenance.criterion !== weightsSettings.kind ||
    (weightsSettings.kind === 'knn' && information.provenance.k !== weightsSettings.k) ||
    (weightsSettings.kind === 'distance-band' &&
      !isGPUChunkedSpatialWeights(pointWeights.weights) &&
      pointWeights.weights.neighbors.length !== weightsSettings.neighborCapacity) ||
    !matchesKeplerPointWeightSpatialContext(weightsSettings, context)
  )
    throw new Error('Local Moran requires binary point weights without self edges');
}

function addUndefinedTally<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'float32'>,
  mask: GraphDataView<'uint32'>,
  inferenceValues: GraphDataView<'float32'>,
  invalidCount: GraphDataView<'uint32'>
): void {
  const flags = createTransientView(graph, `${id}-undefined-flags`, 'uint32', values.length);
  const dispatch = getBoundedDispatchLayout(
    id,
    values.length,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id: `${id}-undefined-flags`,
    resources: [
      {buffer: values, usage: 'storage-read'},
      {buffer: mask, usage: 'storage-read'},
      {buffer: inferenceValues, usage: 'storage-read'},
      {buffer: flags, usage: 'storage-read-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: `${id}-undefined-flags`,
        source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> inferenceValues: array<f32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatch, 64)}
  if (index >= ${values.length}u) { return; }
  let included = mask[${getViewElementOffset(mask)}u + index] != 0u && finite(values[${getViewElementOffset(values)}u + index]);
  let undefinedResult = !finite(inferenceValues[${getViewElementOffset(inferenceValues)}u + index]);
  flags[${getViewElementOffset(flags)}u + index] = select(0u, 1u, included && undefinedResult);
}`,
        shaderLayout: {
          bindings: [
            {name: 'values', type: 'storage', group: 0, location: 0},
            {name: 'mask', type: 'storage', group: 0, location: 1},
            {name: 'inferenceValues', type: 'storage', group: 0, location: 2},
            {name: 'flags', type: 'storage', group: 0, location: 3}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              values: getViewBinding(values, getBuffer),
              mask: getViewBinding(mask, getBuffer),
              inferenceValues: getViewBinding(inferenceValues, getBuffer),
              flags: getViewBinding(flags, getBuffer)
            },
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
  graph.add(
    new GPUReduction({
      id: `${id}-undefined-count`,
      input: flags,
      output: invalidCount,
      operation: 'sum'
    })
  );
}

function addInferenceOverflowReset<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  output: GraphDataView<'uint32'>
): void {
  graph.addComputePass({
    id,
    resources: [{buffer: output, usage: 'storage-read-write'}],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id,
        source: `@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[${getViewElementOffset(output)}u] = 0u; }`,
        shaderLayout: {bindings: [{name: 'output', type: 'storage', group: 0, location: 0}]}
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {output: getViewBinding(output, getBuffer)},
            x: 1
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

function addQuadrantGate<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'uint32'>,
  significant: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): void {
  const dispatch = getBoundedDispatchLayout(
    id,
    output.length,
    64,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  graph.addComputePass({
    id,
    resources: [
      {buffer: input, usage: 'storage-read'},
      {buffer: significant, usage: 'storage-read'},
      {buffer: output, usage: 'storage-read-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id,
        source: `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> significant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatch, 64)}
  if (index >= ${output.length}u) { return; }
  output[${getViewElementOffset(output)}u + index] = select(0u, input[${getViewElementOffset(input)}u + index], significant[${getViewElementOffset(significant)}u + index] != 0u);
}`,
        shaderLayout: {
          bindings: [
            {name: 'input', type: 'storage', group: 0, location: 0},
            {name: 'significant', type: 'storage', group: 0, location: 1},
            {name: 'output', type: 'storage', group: 0, location: 2}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              input: getViewBinding(input, getBuffer),
              significant: getViewBinding(significant, getBuffer),
              output: getViewBinding(output, getBuffer)
            },
            ...dispatch
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}
