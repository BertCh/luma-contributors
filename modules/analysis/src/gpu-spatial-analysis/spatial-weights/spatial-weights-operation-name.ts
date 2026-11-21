// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotAlgebraReviewProps} from './weights-review-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** @internal Capture and validate the algebra operation once at construction. */
export function validateAndSnapshotWeightsAlgebraProps<
  Props extends {id?: string; operation: string}
>(props: Props, chunked = false): Props {
  props = snapshotAlgebraReviewProps(props);
  const accepted = [
    'union',
    'intersection',
    'difference',
    'symmetric-difference',
    'selfWeight',
    'subgraph',
    ...(chunked ? [] : ['higherOrder', 'block', 'spaceTime'])
  ];
  validateRequirement(
    props.id ?? (chunked ? 'chunked-spatial-weights-algebra' : 'spatial-weights-algebra'),
    accepted.includes(props.operation),
    `operation must be one of ${accepted.join(', ')}; received ${String(props.operation)}`
  );
  return props;
}

/** @internal Named operation forms have writers for these three status destinations only. */
export function validateWeightsOperationStatus(props: {
  id?: string;
  output: {status: object};
}): void {
  for (const [name, value] of Object.entries(props.output.status))
    if (value !== undefined)
      validateRequirement(
        props.id ?? 'spatial-weights-algebra',
        ['overflow', 'requiredCount', 'candidateOverflow'].includes(name),
        `output.status.${name} is unsupported; accepted fields are overflow, requiredCount, candidateOverflow`
      );
}
