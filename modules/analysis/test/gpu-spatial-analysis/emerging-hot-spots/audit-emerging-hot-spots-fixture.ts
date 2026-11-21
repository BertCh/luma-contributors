// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from './audit-emerging-hot-spots-fixture.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts/inference';
import type {GPUEmergingHotSpotsSettings} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spot-parameters';

export type AuditEmergingTrendOutputs = {
  trendS: number[];
  trendZ: number[];
  trendP: number[];
  category: number[];
  hotSliceCount: number[];
  coldSliceCount: number[];
  lastValidSlice: number[];
};
export type AuditEmergingCorrection = 'none' | 'hamed-rao' | 'yue-wang';
export type AuditEmergingStage = {
  name: string;
  scores: number[];
  sliceCount: number;
  parameters: number[];
  correction: AuditEmergingCorrection;
  maximumLag: number;
  correctionCriticalZ: number;
  finalSlice: 'require' | 'last-valid';
  expected: AuditEmergingTrendOutputs;
};
export type AuditEmergingCube = {
  name: string;
  mode: 'lattice' | 'weights';
  method: GPUMultipleTestingMethod;
  correction: AuditEmergingCorrection;
  maximumLag: number;
  correctionCriticalZ: number;
  finalSlice: 'require' | 'last-valid';
  selfWeight: number;
  unsigned: boolean;
  values: number[];
  mask: number[];
  settings: GPUEmergingHotSpotsSettings;
  parameters: number[];
  expected: AuditEmergingTrendOutputs & {
    giZScores: number[];
    pValues: number[];
    adjustedPValues: number[];
    globalStatistics: number[];
    includedCount: number[];
  };
};

/** Literal independent NumPy/SciPy output. Nulls are explicit expected NaNs. */
export const AUDIT_EMERGING = decodeOracleFixture<{
  provenance: {generator: string; numpy: string; scipy: string};
  weights: {offsets: number[]; neighbors: number[]; weights: number[]};
  cubes: AuditEmergingCube[];
  stages: AuditEmergingStage[];
  parameterCriticalValues: number[];
}>(fixtureJson);
