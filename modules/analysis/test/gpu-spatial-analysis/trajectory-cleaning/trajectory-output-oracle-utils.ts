// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Poison every resolved output immediately before the real contributor executes. */
export function poisonOperation(operation: GPUContributorOperation): GPUContributorOperation {
  return new GPUContributorOperation({
    id: `${operation.id}-poisoned`,
    inputs: operation.inputs,
    outputs: operation.outputs,
    createContributor(getView) {
      const contributor = operation.createContributor(getView);
      return {
        getCommandNodes<Parameters>(
          graph: GPUCommandGraph<Parameters>
        ): readonly GPUCommandNode<Parameters>[] {
          return [
            ...Object.entries(operation.outputs).map(([port, output]) =>
              createFillNode<Parameters>(graph, {
                id: `${operation.id}-poison-${port}`,
                operation: 'output-oracle-poison',
                view: getView(port),
                type: output.format.startsWith('float32') ? 'f32' : 'u32',
                value: output.format.startsWith('float32') ? '93.0' : '93u'
              })
            ),
            ...(contributor as GPUCommandNodeProducer<Parameters>).getCommandNodes(graph)
          ];
        },
        destroy() {
          operation.destroy();
        }
      };
    }
  });
}
