// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Authored expectations, independent of whichever captures the scenario happens to return. */
export const STATISTICS_STATUS_CAPTURES: Readonly<
  Record<string, Readonly<Record<string, readonly number[]>>>
> = {
  GPUInstrumentalVariablesCovariance: {status: [0]},
  GPURestrictedRegimeRegression: {status: [0]},
  GPUSpatialPanelErrorMaximumLikelihood: {'status-code': [0], diagnosticStatus: [0]},
  GPUSeeminglyUnrelatedRegression: {status: [0], spatialStatus: [0, 0]},
  GPUProbitRegression: {status: [0], diagnosticStatus: [0]},
  GPULocalCollinearity: {status: Array(24).fill(0)},
  GPUMultivariateLocalGearyPermutationTest: {},
  GPURankMarkov: {},
  // Two disjoint adjacent pairs contribute four directed fuzzy edges.
  GPUFuzzyContiguity: {
    'status-overflow': [0],
    featureStatus: [0, 0, 0, 0],
    'status-requiredCount': [4]
  },
  // 23 rows × 4 original edges + the island's two reciprocal attachment edges = 94.
  GPUIslandAttachment: {'status-overflow': [0], 'status-requiredCount': [94]},
  GPUSpatialCrossValidation: {status: [0, 0, 0]},
  GPUSpatialCrossValidatedOLS: {
    'folds-status': [0, 0, 0],
    fitStatus: [0, 0, 0],
    metricStatus: [0, 0, 0]
  },
  GPUSpatialInteraction: {status: [0]},
  // This fixture's Hausman covariance difference is not positive definite (diagnostic status 2).
  GPUSpatialPanelLagMaximumLikelihood: {status: [0], hausmanStatus: [2]}
};

type Capture = {name: string; values: ArrayLike<number>};

/** Reports missing/renamed captures as well as failed status words and wrong finite results. */
export function getStatisticsCaptureFailures(
  element: string,
  snapshots: readonly Capture[]
): string[] {
  const expected = STATISTICS_STATUS_CAPTURES[element];
  if (!expected) return [`unknown statistics scenario ${element}`];
  const failures: string[] = [];
  for (const [name, words] of Object.entries(expected)) {
    const matches = snapshots.filter(snapshot => snapshot.name === name);
    if (
      matches.length !== 1 ||
      matches[0].values.length !== words.length ||
      words.some((word, index) => matches[0].values[index] !== word)
    )
      failures.push(`${element}: ${name} must capture words ${JSON.stringify(words)}`);
  }
  if (element === 'GPURankMarkov') {
    // 3 periods × 8 rows. Input v[i] = sin(0.7*i) + i/24. Descending ordinal ranks,
    // hand sorting each group of eight (no ties), independently of the rank kernel.
    const ranks = snapshots.find(snapshot => snapshot.name === 'ranks');
    const expectedRanks = [4, 2, 0, 1, 3, 5, 6, 7, 7, 4, 2, 0, 1, 3, 5, 6, 7, 6, 4, 2, 0, 1, 3, 5];
    if (
      !ranks ||
      ranks.values.length !== expectedRanks.length ||
      expectedRanks.some((value, index) => ranks.values[index] !== value)
    )
      failures.push(`${element}: ranks differ from the independently sorted fixture`);
  }
  return failures;
}
