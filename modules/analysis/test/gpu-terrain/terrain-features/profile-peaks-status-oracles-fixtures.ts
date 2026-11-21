// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ProfilePeaksOperationProps} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';

export const PROFILE_STATUS_PORTS = [
  'count',
  'requiredCount',
  'overflow',
  'candidateOverflow',
  'converged',
  'iterationLimitReached'
] as const;

// Hand derivation: only rows 2 and 7 are local maxima with positive prominence (5 and 4).
// Both have zero parabolic displacement. Radius zero gives no suppression edges, so one
// round decides both regardless of invocation order. With zero rounds those same two
// named candidates remain undecided. Capacity affects stored count/overflow only.
export const PROFILE_STATUS_VALUES = [0, 0, 5, 0, 0, 0, 0, 4, 0, 0];
export const PROFILE_STATUS_CASES = [
  {
    name: 'one-round-capacity-one',
    maximumSuppressionIterations: 1,
    capacity: 1,
    expected: [1, 2, 1, 0, 1, 0]
  },
  {
    name: 'one-round-capacity-three',
    maximumSuppressionIterations: 1,
    capacity: 3,
    expected: [2, 2, 0, 0, 1, 0]
  },
  {
    name: 'zero-rounds-with-two-undecided-peaks',
    maximumSuppressionIterations: 0,
    capacity: 1,
    expected: [0, 0, 0, 1, 0, 1]
  }
] as const;

// Masking every sample removes both candidates. No suppression work remains even at zero budget.
export const PROFILE_EMPTY_STATUS = [0, 0, 0, 0, 1, 0];

export function createProfileStatusProps(
  testCase: (typeof PROFILE_STATUS_CASES)[number]
): ProfilePeaksOperationProps {
  return {
    values: 'values',
    validity: 'validity',
    offsets: 'offsets',
    parameters: 'parameters',
    sampleCount: 10,
    window: 2,
    minimumSide: 1,
    suppressionRadius: 0,
    maximumSuppressionIterations: testCase.maximumSuppressionIterations,
    output: {
      peaks: {
        ids: 'ids',
        capacity: testCase.capacity,
        status: {
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      },
      status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
    }
  };
}
