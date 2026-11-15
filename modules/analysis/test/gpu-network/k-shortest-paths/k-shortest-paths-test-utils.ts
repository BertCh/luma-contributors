// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUKShortestPaths,
  type GPUKShortestPathsProps
} from '../../../src/gpu-network/k-shortest-paths/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {CPUKShortestPathsInput} from '../../../src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation';
import reference from '../../oracles/fixtures/k-shortest-paths.json';

export type Fixture = (typeof reference.cases)[number];
export function getFixtureInput(fixture: Fixture): CPUKShortestPathsInput {
  return {
    topology: {
      offsets: Uint32Array.from(fixture.offsets),
      neighbors: Uint32Array.from(fixture.neighbors),
      weights: Float32Array.from(fixture.weights)
    },
    sources: Uint32Array.of(fixture.source),
    targets: Uint32Array.of(fixture.target)
  };
}

/** Declaration-only graph, also usable as a kernel-audit builder. */
export function setupKShortestPaths(
  options: {
    nodeCount?: number;
    queryCount?: number;
    routeCount?: number;
    maximumPathLength?: number;
    candidateCapacity?: number;
    maximumIterations?: number;
    device?: Device;
  } = {}
) {
  const nodeCount = options.nodeCount ?? 700;
  const queryCount = options.queryCount ?? 2;
  const routeCount = options.routeCount ?? 4;
  const pathLength = options.maximumPathLength ?? 8;
  const graph = new GPUCommandGraph(options.device ?? createNullWebGPUDevice());
  let nextView = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `ksp-view-${nextView++}`, format, length);
  const props: GPUKShortestPathsProps = {
    topology: {
      offsets: view('uint32', nodeCount + 1),
      neighbors: view('uint32', nodeCount * 2),
      weights: view('float32', nodeCount * 2)
    },
    sources: view('uint32', queryCount),
    targets: view('uint32', queryCount),
    routeCount,
    maximumPathLength: pathLength,
    candidateCapacity: options.candidateCapacity ?? 16,
    maximumIterations: options.maximumIterations ?? 2,
    output: {
      vertices: view('uint32', queryCount * routeCount * pathLength),
      edges: view('uint32', queryCount * routeCount * pathLength),
      lengths: view('uint32', queryCount * routeCount),
      costs: view('float32', queryCount * routeCount),
      counts: view('uint32', queryCount),
      status: {
        converged: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        candidateOverflow: view('uint32', 1),
        iterationCount: view('uint32', 1)
      }
    }
  };
  return {graph, props, contributor: new GPUKShortestPaths(props)};
}
