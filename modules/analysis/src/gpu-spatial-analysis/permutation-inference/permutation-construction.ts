// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import type {GPUChunkedSpatialWeights} from '../spatial-weights/gpu-chunked-spatial-weights';
import type {GPULocalPermutationTestOutput} from './gpu-local-permutation-test';

/** Capture local inference and recipe ports once while keeping buffer contents per frame. @internal */
export function snapshotLocalPermutationProps<
  Props extends {
    weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
    output: GPULocalPermutationTestOutput;
  }
>(props: Props, fields: readonly (keyof Props)[]): Props {
  const captured = snapshotConstructionFields(props, fields);
  const output = snapshotConstructionFields(captured.output, [
    'analysisCount',
    'exceedances',
    'pseudoPValues',
    'observed',
    'significant',
    'classes',
    'quadrants',
    'adjustedPValues',
    'permutationCounts',
    'islandCount',
    'status'
  ]);
  return Object.freeze({
    ...captured,
    weights: snapshotWeightsPort(captured.weights),
    output: Object.freeze({
      ...output,
      status: snapshotConstructionFields(output.status, ['overflow', 'invalidCount'])
    })
  });
}
