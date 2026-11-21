// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtures from './spatial-two-stage-least-squares-audit-fixtures.json';

export function createReviewInput() {
  const fixture = fixtures[0];
  return {
    predictors: Float32Array.from(fixture.predictors),
    response: Float32Array.from(fixture.response),
    weights: {
      offsets: Uint32Array.from(fixture.weights.offsets),
      neighbors: Uint32Array.from(fixture.weights.neighbors),
      weights: Float32Array.from(fixture.weights.weights)
    }
  };
}

export function createReviewKernel(prepended = false) {
  const kernel = fixtures[0].kernel;
  return {
    offsets: Uint32Array.from(kernel.offsets, value => value + Number(prepended)),
    neighbors: Uint32Array.from(prepended ? [0, ...kernel.neighbors] : kernel.neighbors),
    weights: Float32Array.from(prepended ? [1, ...kernel.weights] : kernel.weights)
  };
}

export function createAffineReviewInput(departure = false) {
  const input = createReviewInput();
  input.response = Float32Array.from(
    input.response,
    (_, row) => 1 + 0.75 * input.predictors[2 * row] - 0.5 * input.predictors[2 * row + 1]
  );
  // The first response is -0.5. Its next float32 toward +infinity is -0.5 + 2^-25.
  if (departure) input.response[0] += 2 ** -25;
  return input;
}

export function createRationalAffineReviewInput(departure = false) {
  const input = createReviewInput();
  const response = Float32Array.of(1, 3, -1, -3, 2, -2, 7, -7, 5, -5, 6, -6);
  const predictors = Float32Array.from(response, value => 7 * value);
  if (departure) response[0] += 2 ** -23;
  return {...input, predictors, response};
}

export function createConstantFittedReviewInput() {
  const input = createCycleReviewInput(64);
  input.residuals = Float32Array.from(input.residuals, value => value / 256);
  input.response = Float32Array.from(input.residuals, value => 4096 + 1 / 256 + value);
  return input;
}

export function createLongAffineReviewInput(departure = false) {
  const base = createAffineReviewInput();
  const count = 300;
  const response = Float32Array.from({length: count}, (_, row) => base.response[row % 12]);
  // Row 6 is outside floor(j * 299/255), the bounded candidate's 256 sample rows.
  // Its response is -1.5; the next float32 toward +infinity differs by 2^-23.
  if (departure) response[6] += 2 ** -23;
  return {
    response,
    predictors: Float32Array.from({length: count * 2}, (_, slot) => base.predictors[slot % 24]),
    weights: {
      offsets: Uint32Array.from({length: count + 1}, (_, row) => row),
      neighbors: Uint32Array.from({length: count}, (_, row) => (row + 1) % count),
      weights: new Float32Array(count).fill(1)
    }
  };
}

export function createCycleReviewInput(count = 16, offset = 0) {
  const predictors = Float32Array.from({length: count}, (_, row) => (row % 4 < 2 ? 1 : -1));
  const residuals = Float32Array.from(predictors, (_, row) => predictors[(row + 1) % count]);
  return {
    predictors,
    residuals,
    response: Float32Array.from(predictors, (value, row) => offset + value + residuals[row]),
    weights: {
      offsets: Uint32Array.from({length: count + 1}, (_, row) => row),
      neighbors: Uint32Array.from({length: count}, (_, row) => (row + 1) % count),
      weights: new Float32Array(count).fill(1)
    }
  };
}

export function createCompleteReviewInput(count: number, increment = 0) {
  const predictors = Float32Array.from({length: count}, (_, row) => row % 2);
  const residuals = Float32Array.from(predictors, (_, row) => (row % 4 < 2 ? 1 : -1));
  return {
    predictors,
    residuals,
    response: Float32Array.from(predictors, (value, row) => 2 + value + residuals[row]),
    weights: {
      offsets: Uint32Array.from({length: count + 1}, (_, row) => row * (count - 1)),
      neighbors: Uint32Array.from(
        Array.from({length: count}, (_, row) =>
          Array.from({length: count}, (_, neighbor) => neighbor).filter(
            neighbor => neighbor !== row
          )
        ).flat()
      ),
      weights: Float32Array.from({length: count * (count - 1)}, (_, slot) =>
        slot === 0 || slot === count - 1 ? 1 + increment : 1
      )
    }
  };
}

export function createSaturatedReviewInput() {
  return {
    predictors: Float32Array.of(1, 11, 8, 11, 1, 16, 18, 7, 2, 3),
    response: Float32Array.of(3, 10, 0, 7, 14),
    weights: {
      offsets: Uint32Array.of(0, 1, 2, 3, 4, 5),
      neighbors: Uint32Array.of(1, 2, 3, 4, 0),
      weights: new Float32Array(5).fill(1)
    }
  };
}
