// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getSpatialJoinCandidateNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidates';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

/** Instrument the production BVH rank traversal, without changing its selection algorithm. */
export function getCandidateRankDeclarations(rightCount: number, legacy = false): string {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const bounds = (name: string, length: number) =>
    createTransientView(graph, name, 'float32x2', length);
  const words = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  try {
    const nodes = getSpatialJoinCandidateNodes(graph, {
      id: 'rank-work',
      leftCount: 1,
      rightCount,
      leftMinima: bounds('left-minimum', 1),
      leftMaxima: bounds('left-maximum', 1),
      tree: {
        nodeMinima: bounds('tree-minimum', rightCount * 2 - 1),
        nodeMaxima: bounds('tree-maximum', rightCount * 2 - 1),
        leafIds: words('leaves', rightCount),
        internalNodeCount: rightCount - 1,
        spatialSort: true
      },
      margin: 0,
      candidateCapacity: rightCount,
      candidatePairs: createTransientView(graph, 'pairs', 'uint32x2', rightCount),
      state: words('state', 4),
      page: {
        firstSlot: 0,
        length: rightCount,
        initialize: false,
        counts: words('counts', 1),
        offsets: words('offsets', 1)
      }
    });
    const source = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('write-candidates'))
    )[0].source;
    let declarations =
      source.slice(
        source.indexOf('const FLOAT32_MAXIMUM'),
        source.indexOf('fn readCandidateRight')
      ) + source.slice(source.indexOf('fn countRightsThrough'), source.indexOf('@compute'));
    if (!declarations.includes('fn findFirstPageRight')) throw new Error('Missing rank helper');
    declarations = declarations.replace(
      'let rightRow = leafIds[',
      'probeWork[probeWorkOffset] += 1u;\n        let rightRow = leafIds['
    );
    // The benchmark's baseline differs only by the reviewed fast path.
    if (legacy) declarations = declarations.replace('if (rank == 0u) { return 0u; }', '');
    return declarations;
  } finally {
    device.destroy();
  }
}
