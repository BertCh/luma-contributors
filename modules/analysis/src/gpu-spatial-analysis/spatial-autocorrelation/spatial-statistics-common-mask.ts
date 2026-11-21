// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Scalar diagnostics of the membership producer, preserved separately from inference results. */
export type GPUSpatialStatisticsMembershipStatus = {
  /** Scalar certificate: nonzero promises complete membership. */
  complete?: GraphDataView<'uint32'>;
  overflow?: GraphDataView<'uint32'>;
  candidateOverflow?: GraphDataView<'uint32'>;
  uncertainCount?: GraphDataView<'uint32'>;
  antimeridianUncertainCount?: GraphDataView<'uint32'>;
  poleUncertainCount?: GraphDataView<'uint32'>;
};

/** Inputs for a common observation-row mask shared by analytic and permutation inference. */
export type GPUSpatialStatisticsCommonMaskProps = {
  id?: string;
  rowCount: number;
  /** Scalar: nonzero means the producer promises complete observation values. */
  valuesComplete?: GraphDataView<'uint32'>;
  /** Existing observation-row membership; never a point-row truncation mask. */
  mask?: GraphDataView<'uint32'>;
  /** A false complete certificate or any nonzero membership loss suppresses the entire inference family. */
  membershipStatus?: GPUSpatialStatisticsMembershipStatus;
};

/**
 * Builds a common row mask for local statistics, permutation tests and subsequent adjustment.
 * Add nodes before either consumer, supply mask to both, and retain membershipStatus beside the
 * corrected probabilities. With incomplete values every row is masked, giving an empty tested
 * family rather than finite inference on partial means. Contents are evaluated on every encoding.
 * Reachable from the public entry as GPULocalMoran.createCommonMaskNodes.
 */
export function createSpatialStatisticsCommonMaskNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialStatisticsCommonMaskProps
): {
  nodes: readonly GPUCommandNode<Parameters>[];
  mask: GraphDataView<'uint32'>;
  complete: GraphDataView<'uint32'>;
  membershipStatus: Readonly<GPUSpatialStatisticsMembershipStatus>;
} {
  const {id = 'spatial-statistics-common-mask', rowCount, valuesComplete, mask: membership} = props;
  const membershipStatus = Object.freeze({
    complete: props.membershipStatus?.complete,
    overflow: props.membershipStatus?.overflow,
    candidateOverflow: props.membershipStatus?.candidateOverflow,
    uncertainCount: props.membershipStatus?.uncertainCount,
    antimeridianUncertainCount: props.membershipStatus?.antimeridianUncertainCount,
    poleUncertainCount: props.membershipStatus?.poleUncertainCount
  });
  const sources = Object.entries(membershipStatus).filter(
    (entry): entry is [string, GraphDataView<'uint32'>] => Boolean(entry[1])
  );
  validateCapacity(id, 'rowCount', rowCount, 1, 0xffffffff);
  validateRequirement(
    id,
    Boolean(valuesComplete || sources.length),
    'valuesComplete or membershipStatus must provide a scalar completeness source'
  );
  for (const [name, view] of [
    ...sources,
    ...(valuesComplete ? [['valuesComplete', valuesComplete] as const] : [])
  ]) {
    validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
    validateCapacity(id, `${name}.length`, view.length, 1, 1);
  }
  if (membership) {
    validateSinglePackedView(membership, ['uint32'], `${id} mask`);
    validateCapacity(id, 'mask.length', membership.length, rowCount, rowCount);
  }
  validateGraphViewsBelongToGraph(id, graph, [
    valuesComplete,
    membership,
    ...sources.map(([, view]) => view)
  ]);
  const complete = createTransientView(graph, `${id}-complete`, 'uint32', 1);
  const mask = createTransientView(graph, `${id}-mask`, 'uint32', rowCount);
  const scalarBindings: WGSLKernelBinding[] = sources.map(([name, view]) => ({
    name: name === 'complete' ? 'membershipComplete' : name,
    view,
    type: 'u32',
    access: 'read'
  }));
  if (valuesComplete)
    scalarBindings.push({
      name: 'valuesComplete',
      view: valuesComplete,
      type: 'u32',
      access: 'read'
    });
  const conditions = [
    ...(valuesComplete ? ['valuesComplete[valuesCompleteOffset] != 0u'] : []),
    ...sources.map(([name]) =>
      name === 'complete'
        ? 'membershipComplete[membershipCompleteOffset] != 0u'
        : `${name}[${name}Offset] == 0u`
    )
  ];
  return {
    mask,
    complete,
    membershipStatus,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-complete`,
        operation: 'SpatialStatisticsCommonMask',
        variant: 'complete',
        invocationCount: 1,
        bindings: [
          ...scalarBindings,
          {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
        ],
        body: `complete[completeOffset] = u32(${conditions.join(' && ')});`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mask`,
        operation: 'SpatialStatisticsCommonMask',
        variant: 'mask',
        invocationCount: rowCount,
        bindings: [
          {name: 'complete', view: complete, type: 'u32', access: 'read'},
          ...(membership
            ? [
                {
                  name: 'membership',
                  view: membership,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'mask', view: mask, type: 'u32', access: 'read_write'}
        ],
        body: `mask[maskOffset + index] = u32(complete[completeOffset] != 0u${membership ? ' && membership[membershipOffset + index] != 0u' : ''});`
      })
    ]
  };
}
