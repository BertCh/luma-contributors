// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';
import type {GPUSpatialWeights, GPUChunkedSpatialWeights} from '../spatial-weights/index';

/** Capture maps while retaining every positive certificate and GPU resource identity. @internal */
export function snapshotRegressionWeights<Weights extends object>(weights: Weights): Weights {
  const port = snapshotWeightsFields(weights, [
    'offsets',
    'neighbors',
    'weights',
    'overflow',
    'candidateOverflow',
    'distances',
    'status',
    'information',
    'spatialContext',
    'rowCount',
    'chunks',
    'asymmetricSlots'
  ] as never[]) as unknown as GPUSpatialWeights | GPUChunkedSpatialWeights;
  const status =
    port.status &&
    snapshotWeightsFields(port.status, [
      'overflow',
      'candidateOverflow',
      'requiredCount',
      'count',
      'invalidCount',
      'approximation',
      'uncertainCount',
      'iterationCount',
      'converged',
      'iterationLimitReached',
      'complete'
    ]);
  const captured = Object.freeze({
    ...port,
    ...(status ? {status} : {}),
    ...('chunks' in port
      ? {
          chunks: Object.freeze(
            port.chunks.map(inputChunk => {
              const chunk = snapshotWeightsFields(inputChunk, ['rowStart', 'rowCount', 'weights']);
              return Object.freeze({
                ...chunk,
                weights: snapshotRegressionWeights(chunk.weights) as GPUSpatialWeights
              });
            })
          )
        }
      : {})
  });
  const ordinary = snapshotWeightsPort(captured);
  return Object.freeze({
    ...ordinary,
    ...(status ? {status} : {}),
    ...('chunks' in captured ? {chunks: captured.chunks} : {})
  }) as unknown as Weights;
}
