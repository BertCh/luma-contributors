// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {COVERAGE_GEOMETRY_CASES} from './kernel-audit/cases-coverage-geometry';
import {selectAnalysisCases} from './kernel-audit/analysis-contributors';
import {auditKernelCase} from './kernel-audit/kernel-audit-runner';

it.each(selectAnalysisCases(COVERAGE_GEOMETRY_CASES))('$exportName: $name', auditCase => {
  expect(auditKernelCase(auditCase).findings).toEqual([]);
});
