// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import reference from '../../oracles/fixtures/raster-focal-pairs.json';
import type {GPURasterFocalPairsProps} from '../../../src/raster-analysis/focal-pairs/index';

export const METRICS = [
  'covariance',
  'correlation',
  'slope',
  'intercept',
  'rSquared',
  'residualSumSquares'
] as const;
export type Metric = (typeof METRICS)[number];
export type ReferenceBand = {
  values: (number | null)[];
  validity?: number[];
  noDataValue?: number;
  scale?: number;
  offset?: number;
};
export type FocalPairsFixture = Omit<
  GPURasterFocalPairsProps,
  'id' | 'grid' | 'predictor' | 'response' | 'output'
> & {
  name: string;
  width: number;
  height: number;
  predictor: ReferenceBand;
  response: ReferenceBand;
  expected: Record<Metric, (number | null)[]> & {pairedCount: number[]; status: number[]};
};
export const FIXTURES = reference.cases as FocalPairsFixture[];
export const TOLERANCE = reference.provenance.tolerance;
