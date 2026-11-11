// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// dev-docs/contributor-contract.md section 14: a CPU twin is exported from its family entry point
// together with the operation form of its contributor and the function that does the computation.

import {GPUCPUOperation} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import * as analysisCore from '../../../src/analysis-core/index';
import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as networkAnalysis from '../../../src/gpu-network/index';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as terrainAnalysis from '../../../src/gpu-terrain/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';

const FAMILIES: Record<string, Record<string, unknown>> = {
  'dataframe-analysis': dataframeAnalysis,
  'gpu-network': networkAnalysis,
  'gpu-spatial-analysis': spatialAnalysis,
  'gpu-terrain': terrainAnalysis,
  'raster-analysis': rasterAnalysis
};

/** Twins exported on 2026-11-11 (125); a lower count means an export line was lost. */
const MINIMUM_TWIN_COUNT = 125;

it('analysis-core exports the operation form and the CPU row helpers', () => {
  for (const name of [
    'GPUContributorOperation',
    'GPU_CONTRIBUTOR_OPERATION_LOWERING',
    'GPU_CONTRIBUTOR_OPERATION_TYPE',
    'getCPUOperationFloat32Rows',
    'getCPUOperationInt32Rows',
    'getCPUOperationUint32Rows'
  ]) {
    expect((analysisCore as Record<string, unknown>)[name], name).toBeDefined();
  }
});

it('every exported CPU twin comes with its GPU operation form and its compute function', () => {
  let twinCount = 0;
  for (const [family, exports] of Object.entries(FAMILIES)) {
    const lowerCaseNames = new Map(Object.keys(exports).map(name => [name.toLowerCase(), name]));
    for (const [name, value] of Object.entries(exports)) {
      const match = /^CPU(\w+)Operation$/.exec(name);
      if (!match) {
        continue;
      }
      twinCount++;
      const subject = match[1];
      expect(typeof value, `${family} ${name}`).toBe('function');
      expect(
        (value as {prototype: unknown}).prototype instanceof GPUCPUOperation,
        `${family} ${name} extends the published GPUCPUOperation`
      ).toBe(true);
      // `GPU<Name>GraphOperation` where `GPU<Name>Operation` was already a public name of
      // something else (contract section 14).
      const operationFormClass =
        typeof exports[`GPU${subject}Operation`] === 'function'
          ? exports[`GPU${subject}Operation`]
          : exports[`GPU${subject}GraphOperation`];
      expect(typeof operationFormClass, `${family} GPU${subject}Operation`).toBe('function');
      expect(
        (operationFormClass as {prototype: unknown}).prototype instanceof
          analysisCore.GPUContributorOperation,
        `${family} GPU${subject}Operation is an operation form`
      ).toBe(true);
      // Acronyms differ in case between class and function names (OPTICS, Optics).
      const computeName = lowerCaseNames.get(`compute${subject}oncpu`.toLowerCase());
      expect(computeName, `${family} compute${subject}OnCPU`).toBeDefined();
      expect(typeof exports[computeName as string]).toBe('function');
    }
  }
  expect(twinCount).toBeGreaterThanOrEqual(MINIMUM_TWIN_COUNT);
});
