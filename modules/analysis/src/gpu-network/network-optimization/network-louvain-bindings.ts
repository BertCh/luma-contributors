// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateNetworkOptimizationBindings} from './network-optimization-bindings';
import type {GPUNetworkLouvainProps} from './gpu-network-louvain';

/** Checks caller bindings, including their aligned prefixes, before either route declares resources. @internal */
export function validateNetworkLouvainBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkLouvainProps
): void {
  const views: [string, GraphDataView | undefined][] = [
    ['weights', props.weights],
    ['parameters', props.parameters],
    ['topology.offsets', props.topology?.offsets],
    ['topology.neighbors', props.topology?.neighbors],
    ['topology.weights', props.topology?.weights],
    ['topology.nodePositions', props.topology?.nodePositions],
    ['output.communities', props.output.communities],
    ['output.objectiveTrace', props.output.objectiveTrace],
    ['output.summary', props.output.summary],
    ...Object.entries(props.output.status ?? {}).map(
      ([name, view]) => [`output.status.${name}`, view] as [string, GraphDataView | undefined]
    )
  ];
  for (const [name, view] of views) {
    if (!view) continue;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      getViewBindingRange(view).size,
      'reduce the view length or its prefix within the aligned binding'
    );
  }
  validateNetworkOptimizationBindings(graph, id, Object.fromEntries(views));
}
