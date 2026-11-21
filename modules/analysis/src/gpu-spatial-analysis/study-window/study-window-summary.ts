// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {COMPENSATED_SUM_WGSL} from '../../analysis-core/kernels/compensated-sum-wgsl';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_STUDY_WINDOW_FLAG_EMPTY} from '../../analysis-core/contracts/index';

/** Three (sum, correction) pairs, four bounds and one flags word per partial. @internal */
export const STUDY_WINDOW_SUMMARY_PARTIAL_STRIDE = 11;
/** An initialization invocation merges this fixed pair; no data-sized serial walk. @internal */
export const STUDY_WINDOW_SUMMARY_RINGS_PER_PARTIAL = 2;

/** Each invocation initializes or merges one pair; publication reads only the final root. */
export function getStudyWindowSummaryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rings: GraphDataView<'float32'>;
    ringStride: number;
    summary: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rings, ringStride, summary} = props;
  const ringCount = rings.length / ringStride;
  const partialCount = Math.ceil(ringCount / STUDY_WINDOW_SUMMARY_RINGS_PER_PARTIAL);
  const stride = STUDY_WINDOW_SUMMARY_PARTIAL_STRIDE;
  const partials = createTransientView(
    graph,
    `${id}-summary-partials`,
    'float32',
    stride * partialCount
  );
  const partialBinding = {
    name: 'partials',
    view: partials,
    type: 'f32' as const,
    access: 'read_write' as const
  };
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-summary-partials`,
      operation,
      variant: 'summary-partials',
      invocationCount: partialCount,
      declarations: COMPENSATED_SUM_WGSL,
      bindings: [{name: 'rings', view: rings, type: 'f32', access: 'read'}, partialBinding],
      body: `let ring = ${STUDY_WINDOW_SUMMARY_RINGS_PER_PARTIAL}u * index;
  let source = ringsOffset + ${ringStride}u * ring;
  let base = partialsOffset + ${stride}u * index;
  let accepted = rings[source + 9u] != 0.0;
  ${[0, 1, 2]
    .map(
      component => `partials[base + ${2 * component}u] = select(0.0, rings[source + ${component}u], accepted);
  partials[base + ${2 * component + 1}u] = 0.0;`
    )
    .join('\n  ')}
  let hasBounds = accepted && rings[source + 2u] > 0.0;
  partials[base + 6u] = select(3.402823466e38, rings[source + 4u], hasBounds);
  partials[base + 7u] = select(3.402823466e38, rings[source + 5u], hasBounds);
  partials[base + 8u] = select(-3.402823466e38, rings[source + 6u], hasBounds);
  partials[base + 9u] = select(-3.402823466e38, rings[source + 7u], hasBounds);
  partials[base + 10u] = select(0.0, rings[source + 3u], accepted);
  if (ring + 1u < ${ringCount}u) {
    let partner = source + ${ringStride}u;
    let partnerAccepted = rings[partner + 9u] != 0.0;
    ${[0, 1, 2]
      .map(
        component => `let total${component} = addCompensatedValue(vec2f(partials[base + ${2 * component}u], 0.0), select(0.0, rings[partner + ${component}u], partnerAccepted));
    partials[base + ${2 * component}u] = total${component}.x;
    partials[base + ${2 * component + 1}u] = total${component}.y;`
      )
      .join('\n    ')}
    if (partnerAccepted) {
      partials[base + 10u] = f32(u32(partials[base + 10u]) | u32(rings[partner + 3u]));
      if (rings[partner + 2u] > 0.0) {
        partials[base + 6u] = min(partials[base + 6u], rings[partner + 4u]);
        partials[base + 7u] = min(partials[base + 7u], rings[partner + 5u]);
        partials[base + 8u] = max(partials[base + 8u], rings[partner + 6u]);
        partials[base + 9u] = max(partials[base + 9u], rings[partner + 7u]);
      }
    }
  }`
    })
  ];
  // Disjoint pairs in each dispatch, with a resource dependency between levels. Preserve the
  // left sum/correction before adding the right sum and then its correction in source order.
  for (let distance = 1; distance < partialCount; distance *= 2) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-summary-reduce-${distance}`,
        operation,
        variant: 'summary-reduce',
        invocationCount: Math.ceil(partialCount / (2 * distance)),
        bindings: [partialBinding],
        declarations: `${COMPENSATED_SUM_WGSL}
  override SUMMARY_DISTANCE: u32;
  override SUMMARY_PARTIAL_COUNT: u32;`,
        constants: {SUMMARY_DISTANCE: distance, SUMMARY_PARTIAL_COUNT: partialCount},
        body: `let left = index * (2u * SUMMARY_DISTANCE);
  let right = left + SUMMARY_DISTANCE;
  if (right >= SUMMARY_PARTIAL_COUNT) { return; }
  let base = partialsOffset + ${stride}u * left;
  let partner = partialsOffset + ${stride}u * right;
  ${[0, 2, 4]
    .map(
      component => `let total${component} = addCompensatedValue(addCompensatedValue(
    vec2f(partials[base + ${component}u], partials[base + ${component + 1}u]), partials[partner + ${component}u]), partials[partner + ${component + 1}u]);
  partials[base + ${component}u] = total${component}.x;
  partials[base + ${component + 1}u] = total${component}.y;`
    )
    .join('\n  ')}
  partials[base + 6u] = min(partials[base + 6u], partials[partner + 6u]);
  partials[base + 7u] = min(partials[base + 7u], partials[partner + 7u]);
  partials[base + 8u] = max(partials[base + 8u], partials[partner + 8u]);
  partials[base + 9u] = max(partials[base + 9u], partials[partner + 9u]);
  partials[base + 10u] = f32(u32(partials[base + 10u]) | u32(partials[partner + 10u]));`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-summarize`,
      operation,
      variant: 'summarize',
      invocationCount: 1,
      bindings: [
        {...partialBinding, access: 'read'},
        {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}${IS_FINITE_BITS_WGSL}`,
      body: `let area = partials[partialsOffset] + partials[partialsOffset + 1u];
  var flags = u32(partials[partialsOffset + 10u]);
  if (!isFiniteBits(area) || area <= 0.0) { flags |= ${GPU_STUDY_WINDOW_FLAG_EMPTY}u; }
  var minimum = vec2f(partials[partialsOffset + 6u], partials[partialsOffset + 7u]);
  var maximum = vec2f(partials[partialsOffset + 8u], partials[partialsOffset + 9u]);
  if (any(minimum > maximum)) { minimum = vec2f(getNaN()); maximum = minimum; }
  summary[summaryOffset] = minimum.x;
  summary[summaryOffset + 1u] = minimum.y;
  summary[summaryOffset + 2u] = maximum.x;
  summary[summaryOffset + 3u] = maximum.y;
  summary[summaryOffset + 4u] = area;
  summary[summaryOffset + 5u] = partials[partialsOffset + 2u] + partials[partialsOffset + 3u];
  summary[summaryOffset + 6u] = partials[partialsOffset + 4u] + partials[partialsOffset + 5u];
  summary[summaryOffset + 7u] = f32(flags);`
    })
  );
  return nodes;
}
