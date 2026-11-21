// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUOrdinaryLeastSquaresModelResult,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED,
  type GPUOrdinaryLeastSquaresReadback
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerOLSSettings} from '@luma-contributors/kepler';

export type OLSAnalysisSummary = {
  coefficients: readonly {
    name: string;
    estimate: number;
    standardError: number | null;
    tStatistic: number | null;
    pValue: number | null;
  }[];
  rowCount: number;
  /** Nominal rows minus coefficient count; not effective degrees of freedom for ridge. */
  degreesOfFreedom: number;
  /** Applied float32 penalty; NaN when the producer did not report its convention. */
  ridgeLambda: number;
  rSquared: number | null;
  adjustedRSquared: number | null;
  residualSumOfSquares: number;
  sigmaSquared: number | null;
  logLikelihood: number | null;
  aic: number | null;
  bic: number | null;
  conditioning: {
    conditionNumber: number;
    correlationConditionNumber: number;
    maximumVarianceInflationFactor: number | null;
    illConditioned: boolean;
  } | null;
  tests: Record<
    string,
    {statistic: number | null; degreesOfFreedom: number; pValue: number | null}
  > | null;
  inference: 'classical-homoskedastic' | 'penalized-descriptive' | 'unavailable';
  /** Producer convention, including ridge-approximate, for coefficient-report context. */
  inferenceMethod?: string;
};

/** The caller gates failed solver status before publishing this report. */
export function makeOLSAnalysisSummary(
  readback: GPUOrdinaryLeastSquaresReadback,
  settings: KeplerOLSSettings
): OLSAnalysisSummary {
  const model = getGPUOrdinaryLeastSquaresModelResult(readback, {
    predictorNames: settings.predictorColumns
  });
  const penalized = model.regularization?.kind === 'ridge';
  const inferenceAvailable = model.inferenceMethod !== undefined;
  const suppressInference = penalized || !inferenceAvailable;
  return {
    coefficients: model.coefficients.map(coefficient => ({
      name: coefficient.role === 'intercept' ? 'Intercept' : coefficient.name,
      estimate: coefficient.estimate,
      standardError: suppressInference ? null : finiteOrNull(coefficient.standardError),
      tStatistic: suppressInference ? null : finiteOrNull(coefficient.statistic),
      pValue:
        suppressInference || !Number.isFinite(coefficient.statistic)
          ? null
          : finiteOrNull(coefficient.pValue)
    })),
    rowCount: model.rowCount,
    degreesOfFreedom: model.degreesOfFreedom,
    ridgeLambda: model.regularization?.lambda ?? (inferenceAvailable ? 0 : NaN),
    rSquared: finiteOrNull(model.rSquared),
    adjustedRSquared: suppressInference
      ? null
      : finiteOrNull(readback.summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED]),
    residualSumOfSquares: model.residualSumOfSquares,
    sigmaSquared: suppressInference ? null : finiteOrNull(model.sigmaSquared),
    logLikelihood: suppressInference ? null : finiteOrNull(model.logLikelihood),
    aic: suppressInference ? null : finiteOrNull(model.akaikeInformationCriterion),
    bic: suppressInference ? null : finiteOrNull(model.schwarzCriterion),
    conditioning: model.conditioning
      ? {
          ...model.conditioning,
          maximumVarianceInflationFactor: suppressInference
            ? null
            : model.conditioning.maximumVarianceInflationFactor
        }
      : null,
    tests: suppressInference
      ? null
      : Object.fromEntries(
          Object.entries(model.tests).map(([name, result]) => [
            name,
            {
              statistic: finiteOrNull(result.statistic),
              degreesOfFreedom: result.degreesOfFreedom,
              pValue: Number.isFinite(result.statistic) ? finiteOrNull(result.pValue) : null
            }
          ])
        ),
    inference: penalized
      ? 'penalized-descriptive'
      : inferenceAvailable
        ? 'classical-homoskedastic'
        : 'unavailable',
    ...(model.inferenceMethod ? {inferenceMethod: model.inferenceMethod} : {})
  };
}

function finiteOrNull(value: number | null): number | null {
  return value !== null && Number.isFinite(value) ? value : null;
}
