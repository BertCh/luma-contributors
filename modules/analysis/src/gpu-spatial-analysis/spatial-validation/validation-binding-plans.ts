// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRegressionAllocations} from '../spatial-regression/spatial-likelihood-allocations';
import type {GPUSpatialCrossValidationProps} from './gpu-spatial-cross-validation';
import {getSpatialDesignInputs} from './spatial-validation-design';

/** All bindings of a fold builder, without declaring its generated masks or candidate partitions. */
export function getSpatialCrossValidationBindingViews(props: GPUSpatialCrossValidationProps) {
  return {
    positions: props.positions,
    blocks: !props.design || props.design.mode === 'groups' ? props.blocks : undefined,
    mask: props.mask,
    ...props.output,
    ...Object.fromEntries(
      getSpatialDesignInputs(props.design).map((view, index) => [`design input ${index}`, view])
    ),
    designDiagnostics: props.design?.diagnostics
  };
}

/** Includes generated kNNDM clustering states, candidates, and the dense NNDM distance table. */
export function getSpatialCrossValidationAllocationLengths(
  props: GPUSpatialCrossValidationProps
): Readonly<Record<string, number>> {
  const rows = props.positions.length;
  const design = props.design;
  if (!design) return {'block order': 2 * rows};
  return {
    'design state': 4 + (design.mode === 'nndm' ? rows : 0),
    ...(design.mode === 'environmental'
      ? {
          'environmental centered features': rows * design.featureCount,
          'environmental rounding zero': 1
        }
      : {}),
    ...(design.mode === 'nndm' ? {'NNDM distances': rows * rows} : {}),
    ...(design.mode === 'knndm' && !design.candidates
      ? {
          'generated kNNDM candidates': rows * (design.candidateCount ?? 8),
          'generated kNNDM centered features': rows * 2,
          'generated kNNDM rounding zero': 1
        }
      : {})
  };
}

/** Run before any composing contributor declares resources, including before OLS workspaces. */
export function validateSpatialCrossValidationBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialCrossValidationProps
): void {
  validateGraphChunkBindingRanges(graph, id, getSpatialCrossValidationBindingViews(props));
  validateRegressionAllocations(
    graph.device,
    id,
    getSpatialCrossValidationAllocationLengths(props)
  );
}
