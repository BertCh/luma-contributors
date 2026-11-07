// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma-contributors/graph';
import {
  createTransientView,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  GPUElementwise,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPUHotSpotAnalysis,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  type GPUHotSpotAnalysisRecipeProps
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {KeplerPointWeightsStorage} from './point-weights-chunked';
import type {KeplerAnalysisRecipe} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';

export type KeplerPointHotSpotsProps = KeplerAnalysisProvenance & {
  id: string;
  operation: Extract<
    KeplerAnalysisRecipe['operation'],
    {kind: 'hot-spots-points' | 'hot-spots-polygons'}
  >;
  values: GraphDataView<'float32'>;
  mask: GraphDataView<'uint32'>;
  weights: KeplerPointWeightsStorage;
  outputs?: GPUHotSpotAnalysisRecipeProps['outputs'];
};

/** Gi* on the original point rows and explicit reusable neighbors; no cell aggregation occurs. */
export function addKeplerPointHotSpots<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerPointHotSpotsProps
) {
  const {id, operation, values, mask, weights} = props;
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1 ||
    (props.population === 'selected'
      ? !Number.isSafeInteger(props.selectionRevision) || props.selectionRevision! < 1
      : props.population !== 'all' || props.selectionRevision !== undefined)
  )
    throw new Error('Point hotspots require current source and population provenance');
  const provided = props.outputs ?? {};
  const rowCount = values.length;
  const zScores =
    provided.zScores ?? createTransientView(graph, `${id}-z-scores`, 'float32', rowCount);
  const bins = provided.bins ?? createTransientView(graph, `${id}-bins`, 'sint32', rowCount);
  const pValues =
    provided.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rowCount);
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
  const invalidCount =
    provided.status?.invalidCount ??
    createTransientView(graph, `${id}-undefined-count`, 'uint32', 1);
  if (!weights.status?.overflow || !weights.status.requiredCount)
    throw new Error('Point hotspots require weights completeness diagnostics');
  copyPointStatus(graph, `${id}-weights-overflow-copy`, weights.status.overflow, weightsOverflow);
  copyPointStatus(
    graph,
    `${id}-weights-required-copy`,
    weights.status.requiredCount,
    weightsRequiredCount
  );
  const parameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-parameters`,
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues({})
  });
  try {
    const contributor = new GPUHotSpotAnalysis({
      id: `${id}-gi-star`,
      values,
      mask,
      weights,
      parameters: parameters.importToGraph(graph),
      selfWeight: operation.selfWeight,
      minimumNeighbors: operation.minimumNeighbors,
      multipleTesting: operation.multipleTesting,
      islands: operation.islands,
      output: {zScores, bins, pValues, neighborCounts, globalStatistics}
    });
    graph.add(contributor);
    addUndefinedTally(graph, id, values, mask, zScores, invalidCount);
    return {
      operation: 'hot-spots' as const,
      source: props.source,
      population: props.population,
      selectionRevision: props.selectionRevision,
      numericalPolicy: 'float32' as const,
      rowCount,
      values,
      mask,
      weights,
      zScores,
      bins,
      pValues,
      neighborCounts,
      globalStatistics,
      inference: contributor.getInferenceMetadata(),
      outputs: {zScores, bins, pValues, neighborCounts, globalStatistics},
      status: {
        weightsOverflow,
        weightsRequiredCount,
        invalidCount,
        stages: [
          {stage: 'weights', status: weights.status},
          {stage: 'minimum-neighbors', status: {invalidCount}}
        ]
      },
      parameterByteLength: parameters.buffer.byteLength,
      destroyParameters() {
        parameters.destroy();
      }
    };
  } catch (error) {
    parameters.destroy();
    throw error;
  }
}

/** Mirrors compact diagnostics without requiring copy usage on graph-owned storage. */
export function copyPointStatus<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'uint32'>,
  output: GraphDataView<'uint32'>
): void {
  if (
    input.buffer === output.buffer &&
    input.byteOffset === output.byteOffset &&
    input.length === output.length
  )
    return;
  graph.add(new GPUElementwise({id, operation: 'copy', input, output}));
}

function addUndefinedTally<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  values: GraphDataView<'float32'>,
  mask: GraphDataView<'uint32'>,
  zScores: GraphDataView<'float32'>,
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
      {buffer: zScores, usage: 'storage-read'},
      {buffer: flags, usage: 'storage-read-write'}
    ],
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: `${id}-undefined-flags`,
        source: /* wgsl */ `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> zScores: array<f32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
@compute @workgroup_size(64) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatch, 64)}
  if (index >= ${values.length}u) { return; }
  let included = mask[${getViewElementOffset(mask)}u + index] != 0u && finite(values[${getViewElementOffset(values)}u + index]);
  let undefinedResult = !finite(zScores[${getViewElementOffset(zScores)}u + index]);
  flags[${getViewElementOffset(flags)}u + index] = select(0u, 1u, included && undefinedResult);
}`,
        shaderLayout: {
          bindings: [
            {name: 'values', type: 'storage', group: 0, location: 0},
            {name: 'mask', type: 'storage', group: 0, location: 1},
            {name: 'zScores', type: 'storage', group: 0, location: 2},
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
              zScores: getViewBinding(zScores, getBuffer),
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
