// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expectTypeOf, it} from 'vitest';
import {
  addVisibilityGraphRecipe as addNetworkVisibilityGraphRecipe,
  type GPUNetworkVisibilityGraphCSROutput,
  type GPUNetworkVisibilityGraphCSRStatus
} from '@luma-contributors/analysis/gpu-network';
import type {GPUVisibilityGraphReferenceRecipeResult} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph';
import {
  addVisibilityGraphRecipe,
  addBuildingTessellationRecipe,
  addPatchProximityRecipe,
  type GPUVisibilityGraphProps,
  type GPUVisibilityGraphAdjacencyCSRProps,
  type GPUVisibilityGraphAdjacencyCSROutput,
  type GPUVisibilityGraphAdjacencyCSRStatus,
  type GPUBuildingTessellationProps,
  type GPUBuildingTessellationLocalProps,
  type GPUPatchProximityProps,
  type GPUPatchProximityBoundaryProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  GPURasterResistanceConnectivityProps,
  GPURasterResistanceConnectivityOutput
} from '../../../src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity';

// Compiled by the lead's test typecheck; no device or graph execution is needed.
function checkDispatchTypes(
  graph: GPUCommandGraph,
  visibility: GPUVisibilityGraphProps | GPUVisibilityGraphAdjacencyCSRProps,
  tessellation: GPUBuildingTessellationProps | GPUBuildingTessellationLocalProps,
  proximity: GPUPatchProximityProps | GPUPatchProximityBoundaryProps
) {
  const visibilityUnion = addVisibilityGraphRecipe(graph, visibility);
  const referenceVisibility = addVisibilityGraphRecipe(graph, {
    ...visibility,
    mode: 'reference'
  } as GPUVisibilityGraphProps);
  const csrVisibility = addVisibilityGraphRecipe(graph, {
    ...visibility,
    mode: 'csr'
  } as GPUVisibilityGraphAdjacencyCSRProps);
  expectTypeOf(visibilityUnion).toEqualTypeOf<typeof referenceVisibility | typeof csrVisibility>();
  expectTypeOf(referenceVisibility.outputs.adjacency).not.toBeAny();
  expectTypeOf(csrVisibility.outputs.offsets).not.toBeAny();
  const tessellationUnion = addBuildingTessellationRecipe(graph, tessellation);
  const referenceTessellation = addBuildingTessellationRecipe(graph, {
    ...tessellation,
    mode: 'reference'
  } as GPUBuildingTessellationProps & {mode: 'reference'});
  const localTessellation = addBuildingTessellationRecipe(graph, {
    ...tessellation,
    mode: 'local'
  } as GPUBuildingTessellationLocalProps);
  expectTypeOf(tessellationUnion).toEqualTypeOf<
    typeof referenceTessellation | typeof localTessellation
  >();
  expectTypeOf(referenceTessellation.intermediates.fragments).not.toBeAny();
  expectTypeOf(localTessellation.outputs.cells).not.toBeAny();
  const proximityUnion = addPatchProximityRecipe(graph, proximity);
  const referenceProximity = addPatchProximityRecipe(graph, {
    ...proximity,
    mode: 'reference'
  } as GPUPatchProximityProps);
  const boundaryProximity = addPatchProximityRecipe(graph, {
    ...proximity,
    mode: 'boundary'
  } as GPUPatchProximityBoundaryProps);
  expectTypeOf(proximityUnion).toEqualTypeOf<
    typeof referenceProximity | typeof boundaryProximity
  >();
  expectTypeOf(referenceProximity.outputs.distances).not.toBeAny();
  expectTypeOf(boundaryProximity.outputs.validity).not.toBeAny();
}

function checkLiteralTypes(
  graph: GPUCommandGraph,
  referenceVisibility: GPUVisibilityGraphProps,
  csrVisibility: GPUVisibilityGraphAdjacencyCSRProps,
  referenceTessellation: GPUBuildingTessellationProps,
  localTessellation: GPUBuildingTessellationLocalProps,
  referenceProximity: GPUPatchProximityProps,
  boundaryProximity: GPUPatchProximityBoundaryProps
) {
  expectTypeOf(
    addVisibilityGraphRecipe(graph, {...referenceVisibility, mode: 'reference'}).outputs.adjacency
  ).not.toBeAny();
  expectTypeOf(
    addVisibilityGraphRecipe(graph, {...csrVisibility, mode: 'csr'}).outputs.offsets
  ).not.toBeAny();
  expectTypeOf(
    addBuildingTessellationRecipe(graph, {...referenceTessellation, mode: 'reference'})
      .intermediates.fragments
  ).not.toBeAny();
  expectTypeOf(
    addBuildingTessellationRecipe(graph, {...localTessellation, mode: 'local'}).outputs.cells
  ).not.toBeAny();
  expectTypeOf(
    addPatchProximityRecipe(graph, {...referenceProximity, mode: 'reference'}).outputs.distances
  ).not.toBeAny();
  expectTypeOf(
    addPatchProximityRecipe(graph, {...boundaryProximity, mode: 'boundary'}).outputs.validity
  ).not.toBeAny();
}

function checkRasterDestinationTypes(
  base: Omit<GPURasterResistanceConnectivityProps, 'output'>,
  views: GPURasterResistanceConnectivityOutput
) {
  // @ts-expect-error R1 S5 requires output destinations.
  const missing: GPURasterResistanceConnectivityProps = base;
  const incomplete: GPURasterResistanceConnectivityProps = {
    ...base,
    // @ts-expect-error R1 S5 the required diagnostics, vertexCurrents and solverStatus are missing.
    output: {potentials: views.potentials}
  };
  // The retired `outputs` spelling is no longer a prop (one name, no aliases).
  const complete: GPURasterResistanceConnectivityProps = {...base, output: views};
  return {missing, incomplete, complete};
}

function checkNetworkReferenceType(graph: GPUCommandGraph, props: GPUVisibilityGraphProps) {
  const result = addNetworkVisibilityGraphRecipe(graph, {...props, mode: 'reference'});
  expectTypeOf(result.outputs.adjacency).not.toBeAny();
  expectTypeOf(result.outputs.choice).not.toBeAny();
  expectTypeOf(result).toEqualTypeOf<GPUVisibilityGraphReferenceRecipeResult>();
}

it('ENT-14 qualified adjacency and network contracts expose their respective outputs', () => {
  expectTypeOf<
    Extract<'betweenness' | 'closeness', keyof GPUVisibilityGraphAdjacencyCSROutput>
  >().toEqualTypeOf<never>();
  expectTypeOf<
    Extract<'converged' | 'approximation', keyof GPUVisibilityGraphAdjacencyCSRStatus>
  >().toEqualTypeOf<never>();
  expectTypeOf<GPUNetworkVisibilityGraphCSROutput['betweenness']>().not.toBeAny();
  expectTypeOf<GPUNetworkVisibilityGraphCSRStatus['converged']>().not.toBeAny();
  expectTypeOf(checkDispatchTypes).toBeFunction();
  expectTypeOf(checkLiteralTypes).toBeFunction();
  expectTypeOf(checkRasterDestinationTypes).toBeFunction();
  expectTypeOf(checkNetworkReferenceType).toBeFunction();
});
