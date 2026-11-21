// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getGPUSpatialWeightsChunks,
  type GPUSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/index';
import {getWeightsLossSources} from '../spatial-weights/spatial-weights-loss';
import {isGraphDataView} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Whole/chunk-root and per-chunk loss flags; a shared scalar is read only once. */
export function getRegressionLossViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): GraphDataView<'uint32'>[] {
  return [
    ...new Set(
      [weights, ...getGPUSpatialWeightsChunks(weights).map(chunk => chunk.weights)].flatMap(
        part => {
          const sources = getWeightsLossSources(
            [{status: part.status}],
            'spatial-regression',
            true
          );
          return [...sources.overflow, ...sources.candidateOverflow];
        }
      )
    )
  ];
}

/** Positive completeness certificates have inverse polarity to loss flags. @internal */
export function getRegressionCertificateViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): GraphDataView<'uint32'>[] {
  return [
    ...new Set(
      [
        weights.status,
        ...getGPUSpatialWeightsChunks(weights).map(chunk => chunk.weights.status)
      ].flatMap(status => (status?.complete ? [status.complete] : []))
    )
  ];
}

/** Count comparisons that prove retained topology is incomplete, including root capacities. */
export function getRegressionCountGroups(weights: GPUSpatialWeights | GPUChunkedSpatialWeights) {
  const chunks = getGPUSpatialWeightsChunks(weights);
  return [
    {
      status: weights.status,
      capacity: Math.min(
        0xffffffff,
        chunks.reduce((total, chunk) => total + chunk.weights.neighbors.length, 0)
      ),
      offsets: chunks.map(chunk => chunk.weights.offsets)
    },
    ...chunks.map(chunk => ({
      status: chunk.weights.status,
      capacity: chunk.weights.neighbors.length,
      offsets: [chunk.weights.offsets]
    }))
  ]
    .flatMap(({status, capacity, offsets}) =>
      status?.requiredCount
        ? [{requiredCount: status.requiredCount, count: status.count, capacity, offsets}]
        : []
    )
    .filter(
      (group, index, groups) =>
        !groups
          .slice(0, index)
          .some(
            previous =>
              previous.requiredCount === group.requiredCount &&
              previous.count === group.count &&
              previous.capacity === group.capacity &&
              previous.offsets.length === group.offsets.length &&
              previous.offsets.every((view, position) => view === group.offsets[position])
          )
    );
}

/** Every scalar read by completeness guards, for construction validation and input ownership. */
export function getRegressionCompletenessViews(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): GraphDataView<'uint32'>[] {
  return [
    ...new Set([
      ...getRegressionLossViews(weights),
      ...getRegressionCertificateViews(weights),
      ...[weights, ...getGPUSpatialWeightsChunks(weights).map(chunk => chunk.weights)].flatMap(
        part => [
          ...(part.status?.requiredCount ? [part.status.requiredCount] : []),
          ...(part.status?.count ? [part.status.count] : [])
        ]
      )
    ])
  ];
}

/** Number of loss-fold dispatches, excluding reset and dependent-output suppression. */
export function getRegressionLossFoldCount(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): number {
  return (
    getRegressionLossViews(weights).length +
    getRegressionCertificateViews(weights).length +
    getRegressionCountGroups(weights).reduce(
      (total, group) =>
        total + (group.offsets.length === 1 ? 1 : 2 + Math.ceil(group.offsets.length / 7)),
      0
    )
  );
}

/** Suppress every dependent result after its normal writer; reset the loss fold each submission. */
export function getRegressionWeightsLossNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  includeCandidateLoss = true
): {nodes: GPUCommandNode<Parameters>[]; loss?: GraphDataView<'uint32'>} {
  const flags = includeCandidateLoss
    ? getRegressionLossViews(weights)
    : [
        ...new Set(
          [
            weights.status,
            ...getGPUSpatialWeightsChunks(weights).map(chunk => chunk.weights.status)
          ].flatMap(status => (status?.overflow ? [status.overflow] : []))
        )
      ];
  const certificates = getRegressionCertificateViews(weights);
  const countGroups = getRegressionCountGroups(weights);
  if (!flags.length && !certificates.length && !countGroups.length) return {nodes: []};
  const loss = createTransientView(graph, `${id}-weights-loss`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-weights-loss-reset`,
      operation: 'RegressionWeightsCompleteness',
      invocationCount: 1,
      bindings: [{name: 'loss', view: loss, type: 'u32', access: 'read_write'}],
      body: 'loss[lossOffset] = 0u;'
    })
  ];
  flags.forEach((flag, index) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights-loss-${index}`,
        operation: 'RegressionWeightsCompleteness',
        invocationCount: 1,
        bindings: [
          {name: 'flag', view: flag, type: 'u32', access: 'read'},
          {name: 'loss', view: loss, type: 'u32', access: 'read_write'}
        ],
        body: 'loss[lossOffset] = loss[lossOffset] | flag[flagOffset];'
      })
    )
  );
  certificates.forEach((certificate, index) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights-complete-${index}`,
        operation: 'RegressionWeightsCompleteness',
        invocationCount: 1,
        bindings: [
          {name: 'complete', view: certificate, type: 'u32', access: 'read'},
          {name: 'loss', view: loss, type: 'u32', access: 'read_write'}
        ],
        body: 'loss[lossOffset] |= select(0u, 1u, complete[completeOffset] == 0u);'
      })
    )
  );
  countGroups.forEach(({requiredCount, count, capacity, offsets}, index) => {
    const retained =
      offsets.length === 1
        ? undefined
        : createTransientView(graph, `${id}-weights-retained-${index}`, 'uint32', 1);
    if (retained) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-weights-loss-count-reset-${index}`,
          operation: 'RegressionWeightsCompleteness',
          invocationCount: 1,
          bindings: [{name: 'retained', view: retained, type: 'u32', access: 'read_write'}],
          body: 'retained[retainedOffset] = 0u;'
        })
      );
      for (let start = 0; start < offsets.length; start += 7) {
        const batch = offsets.slice(start, start + 7);
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-weights-loss-count-retained-${index}-${start}`,
            operation: 'RegressionWeightsCompleteness',
            invocationCount: 1,
            bindings: [
              ...batch.map((view, position) => ({
                name: `offsets${position}`,
                view,
                type: 'u32' as const,
                access: 'read' as const
              })),
              {name: 'retained', view: retained, type: 'u32', access: 'read_write'}
            ],
            body: `var total = retained[retainedOffset];
${batch.map((view, position) => `total += min(offsets${position}[offsets${position}Offset + ${view.length - 1}u], 0xffffffffu - total);`).join('\n')}
retained[retainedOffset] = total;`
          })
        );
      }
    }
    const retainedView = retained ?? offsets[0];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights-loss-count-${index}`,
        operation: 'RegressionWeightsCompleteness',
        invocationCount: 1,
        bindings: [
          {name: 'required', view: requiredCount, type: 'u32', access: 'read'},
          {name: 'retained', view: retainedView, type: 'u32', access: 'read'},
          ...(count
            ? [{name: 'count', view: count, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'loss', view: loss, type: 'u32', access: 'read_write'}
        ],
        body: `let incomplete = required[requiredOffset] > ${capacity}u || required[requiredOffset] > retained[retainedOffset + ${retained ? 0 : offsets[0].length - 1}u]${count ? ' || required[requiredOffset] > count[countOffset]' : ''};
loss[lossOffset] = loss[lossOffset] | select(0u, 1u, incomplete);`
      })
    );
  });
  return {nodes, loss};
}

/** Suppress dependent outputs after normal writers, using the same completeness decision. */
export function getRegressionWeightsSuppressionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights,
  output: object,
  robustOutput?: object,
  outputLengths: Readonly<Record<string, number>> = {}
): GPUCommandNode<Parameters>[] {
  const {nodes, loss} = getRegressionWeightsLossNodes(graph, id, weights);
  if (!loss) return nodes;
  for (const [group, values] of [output, robustOutput].entries()) {
    if (!values) continue;
    for (const [name, value] of Object.entries(values)) {
      if (value && typeof value === 'object' && 'buffer' in value) {
        const view = value as GraphDataView;
        if (view.format !== 'float32') continue;
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-suppress-incomplete-${group}-${name}`,
            operation: 'RegressionWeightsCompleteness',
            // Caller capacity may exceed the result length; spare words are not outputs.
            invocationCount: group === 0 ? (outputLengths[name] ?? view.length) : view.length,
            bindings: [
              {name: 'loss', view: loss, type: 'u32', access: 'read'},
              {name: 'output', view, type: 'f32', access: 'read_write'}
            ],
            declarations: GET_NAN_WGSL,
            body: 'if (loss[lossOffset] != 0u) { output[outputOffset + index] = getNaN(); }'
          })
        );
      }
    }
    const status = (
      values as {
        status?:
          | {code?: GraphDataView<'uint32'>; converged?: GraphDataView<'uint32'>}
          | GraphDataView<'uint32'>;
      }
    ).status;
    if (!status) continue;
    const entries =
      group === 1 && isGraphDataView(status) && !('code' in status)
        ? [['code', status] as const]
        : Object.entries(status);
    for (const [name, view] of entries) {
      if (!view || (name !== 'code' && name !== 'converged')) continue;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-incomplete-${group}-${name}`,
          operation: 'RegressionWeightsCompleteness',
          invocationCount: 1,
          bindings: [
            {name: 'loss', view: loss, type: 'u32', access: 'read'},
            {name: 'output', view, type: 'u32', access: 'read_write'}
          ],
          body: `if (loss[lossOffset] != 0u) { output[outputOffset] = ${name === 'code' ? 3 : 0}u; }`
        })
      );
    }
  }
  return nodes;
}
