// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createRasterCountNodes} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import {createTerrainStatusPortNodes} from './terrain-processing-utils';

/** Aggregate diagnostics of the terrain 0/1/2 resolution words. @internal */
export type TerrainResolutionStatusPort = Pick<
  GPUIterationStatusPort,
  'converged' | 'iterationLimitReached'
> &
  GPUInvalidStatusPort;

/** Only present views participate in alias and graph-ownership checks. @internal */
export function getTerrainResolutionStatusViews(
  status?: TerrainResolutionStatusPort
): GraphDataView<'uint32'>[] {
  return Object.values(status ?? {}).filter((view): view is GraphDataView<'uint32'> =>
    Boolean(view)
  );
}

/**
 * Reduces per-cell states with bounded workgroup counts, never a serial domain scan. State 0 is
 * excluded/invalid, 1 is resolved and 2 is still unresolved after the available work (cycle,
 * failed dependency or exhausted budget). Excluded cells do not prevent convergence. The budget
 * flag comes from the caller's termination reason, never from unresolved cell counts.
 * @internal
 */
export function createTerrainResolutionStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    states: GraphDataView<'uint32'>;
    status?: TerrainResolutionStatusPort;
    /** Termination evidence and a uint32 boolean WGSL expression over source/sourceOffset. */
    iterationLimit: {source: GraphDataView<'uint32'>; expression: string};
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, states, status} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (status?.invalidCount) {
    nodes.push(
      ...createRasterCountNodes(graph, {
        id: `${id}-invalid-count`,
        operation,
        values: states,
        count: states.length,
        condition: 'value == 0u',
        output: status.invalidCount
      })
    );
  }
  if (status?.converged || status?.iterationLimitReached) {
    const unresolved = createTransientView(graph, `${id}-unresolved-count`, 'uint32', 1);
    nodes.push(
      ...createRasterCountNodes(graph, {
        id: `${id}-unresolved-count`,
        operation,
        values: states,
        count: states.length,
        condition: 'value == 2u',
        output: unresolved
      }),
      ...createTerrainStatusPortNodes(graph, {
        id,
        operation,
        source: unresolved,
        ports: [
          {
            name: 'converged',
            view: status.converged,
            expression: 'u32(source[sourceOffset] == 0u)'
          }
        ]
      })
    );
    if (status.iterationLimitReached) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-budget-status-ports`,
          operation,
          invocationCount: 1,
          bindings: [
            {name: 'source', view: props.iterationLimit.source, type: 'u32', access: 'read'},
            {name: 'unresolved', view: unresolved, type: 'u32', access: 'read'},
            {
              name: 'iterationLimitReached',
              view: status.iterationLimitReached,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: `iterationLimitReached[iterationLimitReachedOffset] = u32((${props.iterationLimit.expression}) != 0u && unresolved[unresolvedOffset] != 0u);`
        })
      );
    }
  }

  return nodes;
}
