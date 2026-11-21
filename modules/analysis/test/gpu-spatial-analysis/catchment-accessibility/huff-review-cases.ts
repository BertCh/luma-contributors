// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Hand-computed counterexamples; the near-equal oracle uses binary64 log1p, not shader logs. */
export const HUFF_REVIEW_CASES = [
  ...[NaN, Infinity].map(demand => ({
    name: `missing demand ${demand} with an underflowed share`,
    attractiveness: [1, 2],
    weights: [1, 1],
    exponent: 152,
    demand,
    winner: 1,
    probabilities: [0, 1],
    captured: [0, 0]
  })),
  ...[1, 16].map(scale => ({
    name: `near-equal attractiveness at scale ${scale}`,
    attractiveness: [scale, scale * (1 + 2 ** -23)],
    weights: [1, 1],
    exponent: 2 ** 24,
    demand: 1,
    winner: 1,
    probabilities: [
      1 / (1 + Math.exp(2 ** 24 * Math.log1p(2 ** -23))),
      1 / (1 + Math.exp(-(2 ** 24) * Math.log1p(2 ** -23)))
    ],
    captured: [
      1 / (1 + Math.exp(2 ** 24 * Math.log1p(2 ** -23))),
      1 / (1 + Math.exp(-(2 ** 24) * Math.log1p(2 ** -23)))
    ]
  })),
  ...[1, 1024].map(scale => ({
    name: `default-exponent exact weighted tie at scale ${scale}`,
    attractiveness: [3 * scale, 2 * scale],
    weights: [2, 3],
    exponent: undefined,
    demand: 1,
    winner: 0,
    probabilities: [0.5, 0.5],
    captured: [0.5, 0.5]
  })),
  ...[1, 16].map(scale => ({
    name: `near-equal attractiveness across a power-of-two boundary at scale ${scale}`,
    attractiveness: [scale * (1 - 2 ** -24), scale],
    weights: [1, 1],
    exponent: 2 ** 25,
    demand: 1,
    winner: 1,
    probabilities: [
      1 / (1 + Math.exp(-(2 ** 25) * Math.log1p(-(2 ** -24)))),
      1 / (1 + Math.exp(2 ** 25 * Math.log1p(-(2 ** -24))))
    ],
    captured: [
      1 / (1 + Math.exp(-(2 ** 25) * Math.log1p(-(2 ** -24)))),
      1 / (1 + Math.exp(2 ** 25 * Math.log1p(-(2 ** -24))))
    ]
  })),
  ...[1, -1].map(sign => ({
    name: `representable contribution from underflowed share, sign ${sign}`,
    attractiveness: [1, 2],
    weights: [1, 1],
    exponent: 152,
    demand: sign * 2 ** 127,
    winner: 1,
    probabilities: [0, 1],
    captured: [sign * 2 ** -25, sign * 2 ** 127]
  })),
  // PTS-110: a non-finite attractiveness or weight is invalid and excluded like a NaN or
  // non-positive one, so the only remaining facility takes the whole row: p = [0, 1] by the
  // definition p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik over the valid facilities.
  ...[undefined, 2].flatMap(exponent =>
    [
      {label: 'attractiveness', attractiveness: [Infinity, 1], weights: [1, 1]},
      {label: 'attractiveness', attractiveness: [-Infinity, 1], weights: [1, 1]},
      {label: 'attractiveness', attractiveness: [NaN, 1], weights: [1, 1]},
      {label: 'weight', attractiveness: [1, 1], weights: [Infinity, 1]}
    ].map(({label, attractiveness, weights}) => ({
      name: `non-finite ${label} ${attractiveness[0]}/${weights[0]} is excluded, exponent ${exponent ?? 1}`,
      attractiveness,
      weights,
      exponent,
      demand: 1,
      winner: 1,
      probabilities: [0, 1],
      captured: [0, 1]
    }))
  )
];
