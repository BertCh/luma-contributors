// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {ClassBreaksOperationProps} from './gpu-class-breaks-operation';

/** Capture declared operation fields and port names without requiring own enumerable props. @internal */
export function snapshotClassBreaksOperationProps(
  inputProps: ClassBreaksOperationProps
): ClassBreaksOperationProps {
  const props = snapshotConstructionFields(inputProps, [
    'id',
    'values',
    'rowCount',
    'valuesFormat',
    'mask',
    'parameters',
    'maximumClassCount',
    'methods',
    'naturalBreaksBinCount',
    'naturalBreaksMode',
    'naturalBreaksExactStorage',
    'output'
  ]);
  return Object.freeze({
    ...props,
    methods: props.methods ? Object.freeze([...props.methods]) : undefined,
    output: snapshotConstructionFields(props.output, ['breaks', 'classCount', 'classCounts'])
  });
}
