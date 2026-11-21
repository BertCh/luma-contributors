// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUClassBreaks} from './gpu-class-breaks';
import type {GPUClassBreaksMethod} from './class-breaks-parameters';
import {
  getClassBreaksProvenance,
  validateClassBreaksInput,
  validateClassBreaksInputAssertions,
  validateClassBreaksGraphOptions
} from './class-breaks-input';
import {
  getClassBreaksOperationPorts,
  snapshotClassBreaksOperationProps
} from './class-breaks-operation-snapshot';

/** Shared graph-value props. The packed parameters select a method per frame. */
export type ClassBreaksOperationProps = {
  id?: string;
  values: string;
  /** Optional assertion on the actual values row count; inferred when omitted. */
  rowCount?: number;
  /** Optional assertion on the actual values format; inferred when omitted. */
  valuesFormat?: 'float32' | 'uint32' | 'sint32';
  mask?: string;
  parameters: string;
  maximumClassCount: number;
  /** GPU defaults omit incompatible layouts/capacities; CPU defaults run on its packed rows. */
  methods?: readonly GPUClassBreaksMethod[];
  /** Coarse graph-bin default 2048. The 2048 ceiling is GPU-only; CPU computes exact partitions. */
  naturalBreaksBinCount?: number;
  naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
  naturalBreaksExactStorage?: 'auto' | 'materialized' | 'recompute';
  output: {breaks: string; classCount: string; classCounts?: string};
};

/** Named props alias for the public GPU operation class. */
export type GPUClassBreaksOperationProps = ClassBreaksOperationProps;

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUClassBreaksOperation extends GPUContributorOperation {
  /** Capacity-compatible methods; resolved input layout is reflected in the contributor.methods. */
  readonly methods: readonly GPUClassBreaksMethod[];
  /** Resolved natural-breaks route metadata, available before lowering. */
  readonly provenance: ReturnType<typeof getClassBreaksProvenance>;
  constructor(inputProps: ClassBreaksOperationProps) {
    const props = snapshotClassBreaksOperationProps(inputProps);
    validateClassBreaksInputAssertions(props.id ?? 'class-breaks', props);
    const methods = validateClassBreaksGraphOptions(props.id ?? 'class-breaks', props);
    super({
      id: props.id ?? 'class-breaks',
      ...getClassBreaksOperationPorts(props),
      createContributor: getView => {
        const values = getView<'float32' | 'uint32' | 'sint32'>('values');
        validateClassBreaksInput(props.id ?? 'class-breaks', props, values);
        return new GPUClassBreaks({
          ...props,
          values,
          parameters: getView<'float32'>('parameters'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          output: {
            breaks: getView<'float32'>('breaks'),
            classCount: getView<'uint32'>('classCount'),
            classCounts:
              props.output.classCounts !== undefined ? getView<'uint32'>('classCounts') : undefined
          }
        });
      }
    });
    this.methods = methods;
    this.provenance = getClassBreaksProvenance({...props, methods}, 'gpu');
  }
}
