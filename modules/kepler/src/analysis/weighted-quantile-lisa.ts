// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  createTransientView,
  GPUMask,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUQuantileLISA,
  isGPUChunkedSpatialWeights,
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUQuantileLISAOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
  type KeplerQuantileLISASettings
} from '../recipes/analysis-recipe';
import type {KeplerAnalysisProvenance} from './analysis-bindings';
import {copyPointStatus} from './point-hotspots';
import {
  getKeplerPointWeightsRowCount,
  getKeplerPointWeightsViews,
  type KeplerPointWeightsStorage
} from './point-weights-chunked';

export type WeightedQuantileLISAOutputs = Partial<Omit<GPUQuantileLISAOutput, 'status'>> & {
  status?: {
    /** Inference capacity/parameter failure, distinct from invalidQuantile. */
    overflow?: GraphDataView<'uint32'>;
    weightsOverflow?: GraphDataView<'uint32'>;
    weightsRequiredCount?: GraphDataView<'uint32'>;
    weightsInvalidCount?: GraphDataView<'uint32'>;
  };
};

export type WeightedQuantileLISAProps = KeplerAnalysisProvenance & {
  id?: string;
  settings: KeplerQuantileLISASettings;
  values: GraphDataView<'float32'>;
  /** Declared population intersected with numeric validity; topology validity is added here. */
  mask: GraphDataView<'uint32'>;
  /** Source-specific wrappers validate producer provenance and bind immutable source-wide topology. */
  topology: {
    source: KeplerAnalysisProvenance['source'];
    /** One CSR, or consecutive row chunks that are dispatched one at a time. */
    weights: KeplerPointWeightsStorage;
    mask: GraphDataView<'uint32'>;
    status: {
      overflow: GraphDataView<'uint32'>;
      requiredCount: GraphDataView<'uint32'>;
      invalidCount: GraphDataView<'uint32'>;
    };
    /** Scratch limit only: excess active degree reports overflow; it never truncates the CSR. */
    maximumNeighbors: number;
  };
  outputs?: WeightedQuantileLISAOutputs;
};

/**
 * Internal shared selected-quantile local join-count computation on original rows and binary CSR.
 * The contributor uses type-5 boundaries and lower-inclusive, upper-exclusive membership.
 * Missing rows are omitted before quantiling; islands remain in the quantile distribution but
 * are excluded from inference. Selection filters existing neighbors without replacing them.
 *
 * The legacy directed pseudo p-value is (min(upper, P-upper)+1)/(P+1). It is anti-conservative,
 * including the all-ties value 1/(P+1), and has no two-sided error-control guarantee. No
 * multiple-testing correction is applied. Uniform Philox sampling does not reproduce GeoDa's
 * random stream or biased endpoint sampling; this adapter does not claim GeoDa parity.
 *
 * Read invalidQuantile and status.overflow before publishing results. Caller owns submission,
 * input/output lifetimes, and destruction of this adapter's two parameter buffers after GPU
 * completion. Omitted outputs and scratch belong to the compiled graph.
 */
export function addWeightedQuantileLISA<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WeightedQuantileLISAProps
) {
  validateInputs(graph, props);
  const id = props.id ?? 'kepler-weighted-quantile-lisa';
  const settings = Object.freeze({...props.settings});
  const source = Object.freeze({...props.source});
  const {values, topology} = props;
  const rowCount = values.length;
  const provided = props.outputs ?? {};
  const membership =
    provided.membership ??
    createTransientView(graph, `${id}-membership-output`, 'uint32', rowCount);
  const observed =
    provided.observed ?? createTransientView(graph, `${id}-observed-output`, 'float32', rowCount);
  const exceedances =
    provided.exceedances ??
    createTransientView(graph, `${id}-exceedances-output`, 'uint32', rowCount);
  const pseudoPValues =
    provided.pseudoPValues ??
    createTransientView(graph, `${id}-pseudo-p-output`, 'float32', rowCount);
  const significant =
    provided.significant ??
    createTransientView(graph, `${id}-significant-output`, 'uint32', rowCount);
  const classes =
    provided.classes ?? createTransientView(graph, `${id}-classes-output`, 'uint32', rowCount);
  const boundaries =
    provided.boundaries ?? createTransientView(graph, `${id}-boundaries-output`, 'float32', 2);
  const validCount =
    provided.validCount ?? createTransientView(graph, `${id}-valid-count-output`, 'uint32', 1);
  const islandCount =
    provided.islandCount ?? createTransientView(graph, `${id}-island-count-output`, 'uint32', 1);
  const invalidQuantile =
    provided.invalidQuantile ??
    createTransientView(graph, `${id}-invalid-quantile-output`, 'uint32', 1);
  const overflow =
    provided.status?.overflow ?? createTransientView(graph, `${id}-overflow-output`, 'uint32', 1);
  const weightsOverflow =
    provided.status?.weightsOverflow ??
    createTransientView(graph, `${id}-weights-overflow-output`, 'uint32', 1);
  const weightsRequiredCount =
    provided.status?.weightsRequiredCount ??
    createTransientView(graph, `${id}-weights-required-output`, 'uint32', 1);
  const weightsInvalidCount =
    provided.status?.weightsInvalidCount ??
    createTransientView(graph, `${id}-weights-invalid-output`, 'uint32', 1);
  const effectiveMask = createTransientView(graph, `${id}-effective-mask`, 'uint32', rowCount);
  const quantileParameters = new GPUParameterBuffer(graph.device, {
    id: `${id}-quantile-parameters`,
    format: 'uint32',
    length: 2,
    values: new Uint32Array([settings.quantileCount, settings.selectedQuantile])
  });
  let parameters: GPUParameterBuffer<'uint32'> | undefined;
  try {
    parameters = new GPUParameterBuffer(graph.device, {
      id: `${id}-permutation-parameters`,
      format: 'uint32',
      length: GPU_PERMUTATION_PARAMETER_LENGTH,
      values: getGPUPermutationParameterValues(settings, {
        maximumPermutations: settings.permutationCount
      })
    });
    const outputs = {
      membership,
      observed,
      exceedances,
      pseudoPValues,
      significant,
      classes,
      boundaries,
      validCount,
      islandCount,
      invalidQuantile
    };
    const weights = {...topology.weights, status: topology.status};
    const contributor = new GPUQuantileLISA({
      id,
      values,
      mask: effectiveMask,
      weights,
      quantileParameters: quantileParameters.importToGraph(graph),
      parameters: parameters.importToGraph(graph),
      maximumPermutations: settings.permutationCount,
      maximumNeighbors: topology.maximumNeighbors,
      output: {...outputs, status: {overflow}}
    });
    graph.add(
      new GPUMask({
        id: `${id}-topology-validity`,
        inputs: [props.mask, topology.mask],
        output: effectiveMask
      })
    );
    graph.add(contributor);
    copyPointStatus(
      graph,
      `${id}-copy-weights-overflow`,
      topology.status.overflow,
      weightsOverflow
    );
    copyPointStatus(
      graph,
      `${id}-copy-weights-required`,
      topology.status.requiredCount,
      weightsRequiredCount
    );
    copyPointStatus(
      graph,
      `${id}-copy-weights-invalid`,
      topology.status.invalidCount,
      weightsInvalidCount
    );
    const ownedParameters = parameters;
    return {
      operation: 'quantile-lisa' as const,
      source,
      population: props.population,
      ...(props.selectionRevision === undefined
        ? {}
        : {selectionRevision: props.selectionRevision}),
      settings,
      numericalPolicy: 'float32' as const,
      weightTransform: 'binary' as const,
      inference: {
        method: settings.inference,
        seed: settings.seed,
        permutationCount: settings.permutationCount,
        significanceLevel: settings.significanceLevel,
        multipleTesting: settings.multipleTesting,
        randomStream: 'philox' as const,
        sampling: 'uniform-without-replacement' as const,
        tail: 'minimum-upper-strict-complement' as const,
        twoSidedErrorControl: false,
        testedRowsDefinition:
          'included non-island rows with a positive observed selected-quantile join count'
      },
      rowCount,
      values,
      effectiveMask,
      ...outputs,
      outputs,
      status: {
        overflow,
        weightsOverflow,
        weightsRequiredCount,
        weightsInvalidCount,
        stages: [
          {stage: 'weights', status: topology.status},
          {stage: 'quantile-membership', status: {invalidQuantile}},
          {stage: 'conditional-permutation', status: {overflow}}
        ]
      },
      parameterByteLength: parameters.buffer.byteLength + quantileParameters.buffer.byteLength,
      destroyParameters() {
        ownedParameters.destroy();
        quantileParameters.destroy();
      }
    };
  } catch (error) {
    parameters?.destroy();
    quantileParameters.destroy();
    throw error;
  }
}

function validateInputs<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WeightedQuantileLISAProps
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
    throw new Error('Quantile LISA requires current source and population provenance');
  if (
    topology.source.datasetId !== source.datasetId ||
    topology.source.revision !== source.revision
  )
    throw new Error('Weights source revision is stale');
  if (
    rowCount < 3 ||
    rowCount >= 2 ** 24 ||
    mask.length !== rowCount ||
    topology.mask.length !== rowCount ||
    getKeplerPointWeightsRowCount(topology.weights) !== rowCount
  )
    throw new KeplerAllocationError('capacity', {
      message: 'Quantile LISA requires matching source rows within capacity'
    });
  if (
    !Number.isSafeInteger(topology.maximumNeighbors) ||
    topology.maximumNeighbors < 1 ||
    topology.maximumNeighbors > 64
  )
    throw new Error('Quantile LISA maximum neighbors must be within 1 to 64');
  const information = topology.weights.information;
  if (
    information?.rowCount !== rowCount ||
    information.transform !== 'binary' ||
    // Streamed (chunked) weights keep one self slot per row; the kernels skip self slots.
    information.hasSelfWeights !== isGPUChunkedSpatialWeights(topology.weights)
  )
    throw new Error('Quantile LISA requires source-aligned binary weights without self edges');
  if (
    !Number.isSafeInteger(settings.quantileCount) ||
    settings.quantileCount < 2 ||
    settings.quantileCount >= rowCount ||
    !Number.isSafeInteger(settings.selectedQuantile) ||
    settings.selectedQuantile < 1 ||
    settings.selectedQuantile > settings.quantileCount ||
    !Number.isSafeInteger(settings.seed) ||
    settings.seed < 0 ||
    !Number.isSafeInteger(settings.permutationCount) ||
    settings.permutationCount < 1 ||
    settings.permutationCount > MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS ||
    settings.inference !== 'legacy-directed-pseudo' ||
    settings.multipleTesting !== 'none' ||
    settings.islands !== 'exclude' ||
    settings.quantileMethod !== 'type-5' ||
    !(
      settings.significanceLevel > 0 &&
      settings.significanceLevel < 1 &&
      Math.fround(settings.significanceLevel) > 0 &&
      Math.fround(settings.significanceLevel) < 1
    )
  )
    throw new Error('Invalid Quantile LISA settings');
  validatePackedView(values, ['float32'], 'Quantile LISA values');
  for (const view of [mask, topology.mask])
    validatePackedView(view, ['uint32'], 'Quantile LISA mask');
  for (const view of Object.values(topology.status)) {
    validatePackedView(view, ['uint32'], 'Weights status');
    if (view.length !== 1) throw new Error('Weights status must be scalar');
  }
  const provided = props.outputs ?? {};
  const outputViews = [
    provided.membership,
    provided.observed,
    provided.exceedances,
    provided.pseudoPValues,
    provided.significant,
    provided.classes,
    provided.boundaries,
    provided.validCount,
    provided.islandCount,
    provided.invalidQuantile,
    ...Object.values(provided.status ?? {})
  ];
  for (const view of [
    values,
    mask,
    topology.mask,
    ...getKeplerPointWeightsViews(topology.weights),
    ...Object.values(topology.status),
    ...outputViews
  ])
    if (view && view.buffer.graph !== graph)
      throw new Error('Quantile LISA views must belong to the consumer graph');
  for (const view of Object.values(provided.status ?? {})) {
    validatePackedView(view, ['uint32'], 'Quantile LISA status');
    if (view.length !== 1) throw new Error('Quantile LISA status must be scalar');
  }
}
