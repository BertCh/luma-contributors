// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUClassBreaks,
  getGPUClassBreaksParameterLength
} from '../../../src/dataframe-analysis/column-classification/index';
import type {KernelAuditCase} from './kernel-audit-context';

/** Both exact storage strategies and the full-data monotone solver at their declared limits. */
export const EXACT_NATURAL_BREAKS_AUDIT_CASES: readonly KernelAuditCase[] = (
  [
    {mode: 'exact', storage: undefined, rowCount: 1024},
    {mode: 'exact', storage: 'recompute', rowCount: 8192},
    {mode: 'monotone', storage: undefined, rowCount: 262144}
  ] as const
).map(({mode, storage, rowCount}) => ({
  exportName: 'GPUClassBreaks',
  name: `${mode}-natural-breaks-${storage ?? 'default'}-${rowCount}-rows`,
  build: ({graph, view, construct}) => {
    graph.add(
      construct(GPUClassBreaks, {
        values: view('float32', rowCount),
        mask: view('uint32', rowCount),
        parameters: view('float32', getGPUClassBreaksParameterLength(64)),
        maximumClassCount: 64,
        methods: ['natural-breaks'],
        naturalBreaksMode: mode,
        naturalBreaksExactStorage: storage,
        output: {
          breaks: view('float32', 65),
          classCount: view('uint32', 1),
          classCounts: view('uint32', 64)
        }
      })
    );
  }
}));
