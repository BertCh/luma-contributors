// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUContributorOperationOutput} from '../../analysis-core/operations/gpu-contributor-operation';
import type {GPUClassBreaksProps} from './gpu-class-breaks';
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

/** Capture direct contributor options and nested ports while preserving GPU resource identities. @internal */
export function snapshotClassBreaksProps(inputProps: GPUClassBreaksProps): GPUClassBreaksProps {
  const props = snapshotConstructionFields(inputProps, [
    'id',
    'values',
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

/** Common port declarations, with no backend-specific capacity admission. @internal */
export function getClassBreaksOperationPorts(props: ClassBreaksOperationProps) {
  const outputs: Record<string, GPUContributorOperationOutput> = {
    breaks: {value: props.output.breaks, format: 'float32', length: props.maximumClassCount + 1},
    classCount: {value: props.output.classCount, format: 'uint32', length: 1}
  };
  if (props.output.classCounts !== undefined)
    outputs['classCounts'] = {
      value: props.output.classCounts,
      format: 'uint32',
      length: props.maximumClassCount
    };
  return {
    inputs: {
      values: props.values,
      parameters: props.parameters,
      ...(props.mask !== undefined ? {mask: props.mask} : {})
    },
    outputs
  };
}
