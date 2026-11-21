// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

/** @internal Capture categorical statistics metadata without cloning GPU data views. */
export function snapshotCategoricalStatisticsProps<
  Props extends {weights: GPUSpatialWeights; output: object}
>(input: Props, outputFields: readonly string[]): Props {
  const props = snapshotConstructionFields(input, [
    'id',
    'weights',
    'categories',
    'categoryCount',
    'output'
  ] as readonly (keyof Props)[]);
  return Object.freeze({
    ...props,
    weights: snapshotWeightsPort(props.weights),
    output: snapshotConstructionFields(props.output, outputFields as never[])
  });
}
