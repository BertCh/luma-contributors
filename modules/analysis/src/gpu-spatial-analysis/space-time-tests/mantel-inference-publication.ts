// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SPACE_TIME_STATE} from './space-time-stopping';

/** Publish Mantel's usable-draw policy without changing the shared Knox/Jacquez summary. @internal */
export function createMantelInferencePublicationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    maximumPermutations: number;
    stopAfterExceedances?: number;
    parameters: GraphDataView<'uint32'>;
    statistics: GraphDataView<'float32'>;
    summary: GraphDataView<'float32'>;
    state?: GraphDataView<'uint32'>;
    permutationCount?: GraphDataView<'uint32'>;
    usablePermutationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    parameters,
    statistics,
    summary,
    state,
    permutationCount,
    usablePermutationCount,
    stopAfterExceedances
  } = props;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-usable-inference`,
      operation: 'GPUMantelTest',
      variant: 'usable-inference',
      bindings: [
        {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
        {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
        ...(state
          ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
          : []),
        ...(permutationCount
          ? [
              {
                name: 'used',
                view: permutationCount,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : []),
        ...(usablePermutationCount
          ? [
              {
                name: 'usable',
                view: usablePermutationCount,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      // Each summary word has an independent writer. The shared baselined scalar kernels stay intact.
      invocationCount: 12,
      declarations: `${GET_NAN_WGSL}
const MAXIMUM_PERMUTATIONS: u32 = ${props.maximumPermutations}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
      body: `let observed = statistics[statisticsOffset];
  let missing = getNaN();
  var attempts = max(1u, min(parameters[parametersOffset + 2u], MAXIMUM_PERMUTATIONS));
  ${
    state
      ? `let stopIndex = state[stateOffset + ${SPACE_TIME_STATE.stopIndex}u];
  if (stopIndex != 0u) { attempts = stopIndex; }`
      : ''
  }
  var count = 0u;
  var greater = 0u;
  var lesser = 0u;
  var center = 0.0;
  var sumDeviation = 0.0;
  if (finite(observed)) {
    for (var slot = 1u; slot <= attempts; slot++) {
      let value = statistics[statisticsOffset + slot];
      if (finite(value)) {
        if (count == 0u) { center = value; }
        count++;
        sumDeviation += value - center;
        if (value >= observed) { greater++; }
        if (value <= observed) { lesser++; }
      }
    }
  }
  ${permutationCount ? 'if (index == 0u) { used[usedOffset] = select(0u, attempts, finite(observed)); }' : ''}
  ${usablePermutationCount ? 'if (index == 0u) { usable[usableOffset] = count; }' : ''}
  // Preserve the shared layout: observed, pair count and the two Knox-only zero words.
  if (index < 4u) { return; }
  var result = missing;
  if (count > 0u) {
    let meanDeviation = sumDeviation / f32(count);
    let mean = center + meanDeviation;
    var squares = 0.0;
    for (var slot = 1u; slot <= attempts; slot++) {
      let value = statistics[statisticsOffset + slot];
      if (finite(value)) {
        let deviation = value - center - meanDeviation;
        squares += deviation * deviation;
      }
    }
    let variance = select(0.0, squares / f32(max(1u, count - 1u)), count > 1u);
    var pGreater = f32(greater + 1u) / f32(count + 1u);
    ${state && stopAfterExceedances !== undefined ? `if (stopIndex != 0u) { pGreater = min(1.0, ${stopAfterExceedances}.0 / f32(count)); }` : ''}
    let pLesser = f32(lesser + 1u) / f32(count + 1u);
    switch index {
      case 4u: { result = mean; }
      case 5u: { result = variance; }
      case 6u: { result = f32(greater); }
      case 7u: { result = f32(lesser); }
      case 8u: { result = pGreater; }
      case 9u: { result = pLesser; }
      case 10u: { result = min(1.0, 2.0 * min(pGreater, pLesser)); }
      default: { if (variance > 0.0) { result = (observed - mean) / sqrt(variance); } }
    }
  }
  summary[summaryOffset + index] = result;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-undefined-observation`,
      operation: 'GPUMantelTest',
      variant: 'undefined-observation',
      bindings: [{name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}],
      invocationCount: props.maximumPermutations + 1,
      declarations: GET_NAN_WGSL,
      body: `let observed = statistics[statisticsOffset];
  if (index > 0u && (bitcast<u32>(observed) & 0x7f800000u) == 0x7f800000u) {
    statistics[statisticsOffset + index] = getNaN();
  }`
    })
  ];
}
