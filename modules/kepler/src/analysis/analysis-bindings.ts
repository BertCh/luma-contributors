// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getGraphVectorData,
  GraphVectorView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUClassBreaks,
  GPUVariableStandardization,
  type GPUClassBreaksProps,
  type GPUVariableStandardizationProps
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  addExcessRiskRecipe,
  addHotSpotAnalysisRecipe,
  addSpatialRateRecipe,
  type GPUExcessRiskRecipeProps,
  type GPUHotSpotAnalysisRecipeProps,
  type GPUSpatialRateRecipeProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerAnalysisPopulation, KeplerDatasetReference} from '../recipes/analysis-recipe';

export type KeplerAnalysisProvenance = {
  source: KeplerDatasetReference;
  population: KeplerAnalysisPopulation;
  selectionRevision?: number;
};

/** Appends real contributor work. Outputs are usable by later graph nodes without CPU materialization. */
export function addKeplerStandardization<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Omit<GPUVariableStandardizationProps, 'output'> &
    KeplerAnalysisProvenance & {
      output?: GPUVariableStandardizationProps['output'];
    }
) {
  const provenance = getProvenance(props, props.mask !== undefined);
  const id = props.id ?? 'kepler-standardize';
  const output = props.output ?? createOutputColumn(graph, `${id}-output`, props.values);
  graph.add(new GPUVariableStandardization({...props, id, output}));
  return {
    operation: 'standardize' as const,
    ...provenance,
    numericalPolicy: 'float32' as const,
    outputs: {values: output}
  };
}

/** Global expected rate uses the complete declared analysis population, never an implicit viewport. */
export function addKeplerExcessRisk<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUExcessRiskRecipeProps & KeplerAnalysisProvenance
) {
  const provenance = getProvenance(props, props.mask !== undefined);
  const result = addExcessRiskRecipe(graph, {...props, id: props.id ?? 'kepler-excess-risk'});
  return {
    ...result,
    operation: 'excess-risk' as const,
    ...provenance,
    numericalPolicy: 'float32' as const
  };
}

/** Caller supplies weights for exactly this population and must retain their status diagnostics. */
export function addKeplerSpatialRate<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialRateRecipeProps &
    KeplerAnalysisProvenance & {weightsSource: KeplerDatasetReference}
) {
  const provenance = getProvenance(props, props.mask !== undefined);
  if (
    props.weightsSource.datasetId !== props.source.datasetId ||
    props.weightsSource.revision !== props.source.revision
  )
    throw new Error('Spatial weights belong to a different dataset revision');
  const result = addSpatialRateRecipe(graph, {...props, id: props.id ?? 'kepler-spatial-rate'});
  return {
    ...result,
    operation: 'spatial-rate' as const,
    ...provenance,
    numericalPolicy: 'float32' as const
  };
}

/** Keep caller-owned parameters/output for UI replay and bounded legend readback. */
export function addKeplerClassification<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUClassBreaksProps & KeplerAnalysisProvenance
) {
  const provenance = getProvenance(props, props.mask !== undefined);
  graph.add(new GPUClassBreaks({...props, id: props.id ?? 'kepler-classify'}));
  return {
    operation: 'classify' as const,
    ...provenance,
    numericalPolicy:
      props.naturalBreaksMode === 'exact'
        ? ('float32-optimal' as const)
        : ('method-defined' as const),
    outputs: props.output
  };
}

/** Preserves the recipe's inference metadata, convergence/completeness ports and settings updates. */
export function addKeplerHotSpots<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerAnalysisProvenance & {analysis: GPUHotSpotAnalysisRecipeProps}
) {
  const provenance = getProvenance(props, props.analysis.source.mask !== undefined);
  const result = addHotSpotAnalysisRecipe(graph, props.analysis);
  return {
    ...result,
    operation: 'hot-spots' as const,
    ...provenance,
    numericalPolicy: 'float32' as const
  };
}

function getProvenance(
  props: KeplerAnalysisProvenance,
  hasMask: boolean
): KeplerAnalysisProvenance {
  if (
    !props.source.datasetId ||
    !Number.isSafeInteger(props.source.revision) ||
    props.source.revision < 1
  )
    throw new Error('Invalid analysis dataset reference');
  if (props.population !== 'all' && props.population !== 'selected')
    throw new Error('Invalid analysis population');
  if (props.population === 'selected') {
    if (!hasMask || !Number.isSafeInteger(props.selectionRevision) || props.selectionRevision! < 1)
      throw new Error('Selected analysis requires a mask and selection revision');
  } else if (props.selectionRevision !== undefined) {
    throw new Error('Full-population analysis cannot have a selection revision');
  }
  return {
    source: Object.freeze({...props.source}),
    population: props.population,
    ...(props.selectionRevision === undefined ? {} : {selectionRevision: props.selectionRevision})
  };
}

function createOutputColumn<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GPUVariableStandardizationProps['values']
): GPUVariableStandardizationProps['output'] {
  if (!(input instanceof GraphVectorView))
    return createTransientView(graph, id, 'float32', input.length);
  const data: GraphDataView<'float32'>[] = getGraphVectorData(input).map((chunk, index) =>
    createTransientView(graph, `${id}-${index}`, 'float32', chunk.length)
  );
  return new GraphVectorView({
    id,
    name: id,
    format: 'float32',
    length: input.length,
    valueLength: input.length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
}
