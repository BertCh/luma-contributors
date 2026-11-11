// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
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

/**
 * Preserves the recipe's inference metadata, convergence/completeness ports and settings updates.
 *
 * Host `selfWeight` values are any non-negative number, but the recipe's permutation confirmation
 * has a fixed focal weight: `'localGStar'` reproduces only `selfWeight` 1 and `'localG'` only 0.
 * A permutation request with another self weight is rejected here, before any graph work, in host
 * terms; without `permutation` every self weight is forwarded unchanged.
 */
export function addKeplerHotSpots<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KeplerAnalysisProvenance & {analysis: GPUHotSpotAnalysisRecipeProps}
) {
  const provenance = getProvenance(props, props.analysis.source.mask !== undefined);
  validateKeplerHotSpotPermutation(props.analysis);
  const result = addHotSpotAnalysisRecipe(graph, props.analysis);
  return {
    ...result,
    operation: 'hot-spots' as const,
    ...provenance,
    numericalPolicy: 'float32' as const
  };
}

/** Gi and Gi* confirmations each reproduce one focal weight; other statistics have no such rule. */
function validateKeplerHotSpotPermutation(analysis: GPUHotSpotAnalysisRecipeProps): void {
  if (!analysis.permutation) return;
  const selfWeight = analysis.selfWeight ?? 1;
  const statistic = analysis.permutation.statistic ?? (selfWeight === 0 ? 'localG' : 'localGStar');
  if (statistic !== 'localG' && statistic !== 'localGStar') return;
  const requiredSelfWeight = statistic === 'localG' ? 0 : 1;
  if (selfWeight !== requiredSelfWeight) {
    throw new Error(
      `${analysis.id ?? 'kepler-hot-spots'} analysis.selfWeight must be ${requiredSelfWeight} when analysis.permutation uses '${statistic}', not ${selfWeight}; omit permutation to map another self weight without confirmation`
    );
  }
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

/** GPUVariableStandardization requires single-chunk columns, so the default output is one transient. */
function createOutputColumn<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GPUVariableStandardizationProps['values']
): GPUVariableStandardizationProps['output'] {
  return createTransientView(graph, id, 'float32', input.length);
}
