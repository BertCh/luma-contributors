// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Resolve the active coefficient layout; omitting predictorCount requires an exact-length table. */
export function getRegressionTablePredictorCount(
  id: string,
  table: Float32Array,
  stride: number,
  fittedPredictorCount?: number
): number {
  const predictorCount = fittedPredictorCount ?? table.length / stride - 2;
  if (
    !Number.isSafeInteger(predictorCount) ||
    predictorCount < 1 ||
    table.length < (predictorCount + 2) * stride
  ) {
    throw new Error(
      `${id}: table must hold (predictorCount + 2) * ${stride} values for predictorCount >= 1`
    );
  }
  return predictorCount;
}
