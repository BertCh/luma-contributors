// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  selectColorScaleOutput,
  selectBivariateOutput
} from './select-classification-outputs.test-utils';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterLength} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {GPUColorScale} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {GPUBivariateClassification} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import type {KernelAuditCase} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {DATAFRAME_CASES} from '../../analysis-core/kernel-audit/cases-dataframe';
import {registerKernelAudit} from '../../analysis-core/kernel-audit/register-kernel-audit';

// DPC-110/111: parse actual generated WGSL and check bindings/resources after changing shared
// comparison helpers and adding the scaled approximate-natural-break stages. This is a host
// declaration audit; device arithmetic is pinned separately in gpu-class-breaks.spec.ts.
registerKernelAudit(
  'classification numeric fixes',
  DATAFRAME_CASES.filter(({exportName}) =>
    ['GPUClassBreaks', 'GPUColorScale', 'GPUBivariateClassification'].includes(exportName)
  )
);

const subsetCases: KernelAuditCase[] = Array.from({length: 7}, (_, index) => index + 1).flatMap(
  (subset): KernelAuditCase[] => [
    {
      exportName: 'GPUColorScale',
      name: `DPC-122 numeric helper, output subset ${subset}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUColorScale({
            values: view('float32', 16),
            mask: view('uint32', 16),
            domain: view('float32', 4),
            palette: view('uint32', 4),
            parameters: view('float32', 9),
            maximumDomainCount: 4,
            maximumPaletteCount: 4,
            output: selectColorScaleOutput({
              colors: subset & 1 ? view('uint32', 16) : undefined,
              classIndices: subset & 2 ? view('uint32', 16) : undefined,
              classCounts: subset & 4 ? view('uint32', 4) : undefined
            })
          })
        )
    },
    {
      exportName: 'GPUBivariateClassification',
      name: `DPC-122 numeric helper, output subset ${subset}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUBivariateClassification({
            valuesX: view('float32', 9),
            valuesY: view('float32', 9),
            mask: view('uint32', 9),
            breaksX: view('float32', 3),
            breaksY: view('float32', 3),
            palette: view('uint32', 4),
            parameters: view('float32', 8),
            maximumClassCount: 2,
            output: selectBivariateOutput({
              colors: subset & 1 ? view('uint32', 9) : undefined,
              classIds: subset & 2 ? view('uint32', 9) : undefined,
              classCounts: subset & 4 ? view('uint32', 4) : undefined
            })
          })
        )
    }
  ]
);
registerKernelAudit('classification output subsets', subsetCases);

it('S3: approximate natural-break stage budget rejects the newly shifted chunk boundary', () => {
  const context = createKernelAuditContext('natural-budget-review');
  const {view} = context;
  try {
    const create = (chunkCount: number) =>
      new GPUClassBreaks({
        values: createVectorView(
          `chunks-${chunkCount}`,
          'float32',
          Array.from({length: chunkCount}, () => view('float32', 1))
        ),
        methods: ['natural-breaks'],
        maximumClassCount: 18,
        naturalBreaksBinCount: 32,
        parameters: view('float32', getGPUClassBreaksParameterLength(18)),
        output: {breaks: view('float32', 19), classCount: view('uint32', 1)}
      });
    // 32 + 46*660 + 132*18 was exactly 32768; the three parallel correction stages add 3.
    expect(create(659).estimatedCommandNodeCount).toBe(32725);
    expect(create(659).estimatedCommandNodeCount + 46).toBe(32771);
    expect(() => create(660)).toThrow(/command nodes received 32771; requires \[0, 32768\]/);
  } finally {
    context.device.destroy();
  }
});
