// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

type ChangePointOutputDescriptor = {
  statistic?: unknown;
  changeSlice?: unknown;
  pValues?: unknown;
  changePointCount?: unknown;
  changePoints?: unknown;
  requiredChangePointCounts?: unknown;
  optimalCost?: unknown;
  status?: {overflow?: unknown};
};

/** Captures direct and named-operation descriptors, preserving GPU view identities. @internal */
export function snapshotChangePointProps<Props extends {output: ChangePointOutputDescriptor}>(
  props: Props
): Props {
  const captured = snapshotConstructionFields(props, [
    'id',
    'values',
    'parameters',
    'layout',
    'cellCount',
    'sliceCount',
    'cost',
    'maximumChangePoints',
    'output'
  ] as (keyof Props)[]);
  const output = snapshotConstructionFields<Props['output']>(captured.output, [
    'statistic',
    'changeSlice',
    'pValues',
    'changePointCount',
    'changePoints',
    'requiredChangePointCounts',
    'optimalCost',
    'status'
  ]);
  return Object.freeze({
    ...captured,
    output: Object.freeze({
      ...output,
      ...(output.status ? {status: snapshotConstructionFields(output.status, ['overflow'])} : {})
    })
  }) as Props;
}
