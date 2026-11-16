// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {
  findUndeclaredCalls,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectUrbanWGSLIdentifiers} from '../urban-ecology/visibility-graph-wgsl-test-utils';

/** Audit complete shader modules, including helpers needed by separate reduction passes. */
export function expectSpatialAnalysisWGSL(source: string, name: string): void {
  expectUrbanWGSLIdentifiers(source, name);
  expect(findUndeclaredCalls(stripWGSLComments(source)), name).toEqual([]);
}
