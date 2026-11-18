// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../graph-buffers';

/** Reimport both producer loss causes wherever a kit graph reads its source CSR. */
export function importWeightsKitLoss(
  graph: GPUCommandGraph<void>,
  name: string,
  producer: {overflow: Buffer; candidateOverflow?: Buffer}
) {
  return {
    overflow: importGraphBuffer(graph, `${name}-overflow`, producer.overflow, 'uint32', 1),
    candidateOverflow: producer.candidateOverflow
      ? importGraphBuffer(
          graph,
          `${name}-candidate-overflow`,
          producer.candidateOverflow,
          'uint32',
          1
        )
      : undefined
  };
}
