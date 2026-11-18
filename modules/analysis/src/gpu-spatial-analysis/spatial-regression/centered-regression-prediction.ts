// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Centered slope contribution, before adding the response mean and its correction. */
export function computeCenteredRegressionContribution(
  predictors: ArrayLike<number>,
  means: ArrayLike<number>,
  slopes: ArrayLike<number>
): number {
  let contribution = 0;
  for (let column = 0; column < slopes.length; column++) {
    contribution += slopes[column] * (predictors[column] - means[column]);
  }
  return contribution;
}

/**
 * Fix the complete small correction before adding a large response mean. Parentheses alone
 * allow Metal fast math to distribute the mean into its individual terms and round them away.
 * Requires ROUNDED_DIFFERENCE_WGSL and a zero read from a buffer at run time.
 * @internal
 */
export const CENTERED_REGRESSION_PREDICTION_WGSL = /* wgsl */ `
fn getCenteredRegressionPrediction(
  responseMean: f32, centeredContribution: f32, meanCorrection: f32, runtimeZero: u32
) -> f32 {
  let correctedContribution = getRoundedDifference(centeredContribution + meanCorrection, 0.0, runtimeZero);
  return responseMean + correctedContribution;
}
`;

/**
 * Shared shader prediction primitive. Expressions use `column` as their predictor index.
 * Callers supply ROUNDED_DIFFERENCE_WGSL and a runtime zero so Metal cannot regroup the
 * centered product into raw-intercept arithmetic. The response mean and correction stay separate.
 */
export function getCenteredRegressionContributionWGSL(options: {
  predictorCount: number;
  predictor: string;
  mean: string;
  slope: string;
  runtimeZero: string;
}): string {
  return `var centeredContribution = 0.0;
for (var column = 0u; column < ${options.predictorCount}u; column++) {
  centeredContribution += (${options.slope}) * getRoundedDifference(${options.predictor}, ${options.mean}, ${options.runtimeZero});
}`;
}
