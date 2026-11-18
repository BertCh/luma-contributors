// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';

/** Reimports the producer's complete CSR port across main, permutation and comparison graphs. */
export function importWeightsCoreWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  core: {
    csr: {offsets: Buffer; neighbors: Buffer; weights: Buffer};
    slots: number;
    overflow: Buffer;
    candidateOverflow: Buffer;
  },
  rowCount: number
) {
  return {
    offsets: importGraphBuffer(graph, 'offsets', core.csr.offsets, 'uint32', rowCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', core.csr.neighbors, 'uint32', core.slots),
    weights: importGraphBuffer(graph, 'weights', core.csr.weights, 'float32', core.slots),
    status: {
      overflow: importGraphBuffer(graph, 'weights-overflow', core.overflow, 'uint32', 1),
      candidateOverflow: importGraphBuffer(
        graph,
        'weights-candidate-overflow',
        core.candidateOverflow,
        'uint32',
        1
      )
    }
  };
}
