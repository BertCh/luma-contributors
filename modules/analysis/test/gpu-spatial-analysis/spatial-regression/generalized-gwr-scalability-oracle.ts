// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Saturated two-level GLMs: weighted category means solve each local likelihood independently. */
export function createGeneralizedGWRReviewScene(
  rows: number,
  family: 'poisson' | 'logistic',
  paired = false
) {
  const siteSize = paired ? 16 : 8;
  const siteCount = Math.max(1, Math.floor(rows / siteSize));
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  const siteIds = new Uint32Array(rows);
  const statistics = new Float64Array(siteCount * 4);
  const width = Math.ceil(Math.sqrt(siteCount));
  for (let row = 0; row < rows; row++) {
    const site = Math.min(Math.floor(row / siteSize), siteCount - 1);
    const member = row - site * siteSize;
    const category = member % 2;
    const pair = paired ? Math.floor(site / 2) : site;
    positions[row * 2] = (pair % width) * 20 + (paired ? site % 2 : 0);
    positions[row * 2 + 1] = Math.floor(pair / width) * 20;
    siteIds[row] = site;
    predictors[row] = category === 0 ? -1 : 1;
    // First focal tile prefers narrow neighborhoods. Later tiles have matching paired populations,
    // so the reduced global trace makes the wide candidate win over the complete dataset.
    const opposed = paired && site * siteSize < 65536 && site % 2 === 1;
    const effectiveCategory = opposed ? 1 - category : category;
    const positiveCount = paired
      ? effectiveCategory === 0
        ? 1
        : 7
      : category === 0
        ? 1 + (site % 3)
        : 3 - (site % 2);
    let observed =
      family === 'poisson'
        ? paired
          ? effectiveCategory === 0
            ? 1
            : 4
          : category === 0
            ? 1 + (site % 3)
            : 4 + (site % 5)
        : Number(Math.floor(member / 2) % (paired ? 8 : 4) < positiveCount);
    if (paired && row >= 1048576)
      observed = family === 'poisson' ? 100 + (member % 4) : 1 - observed;
    response[row] = observed;
    statistics[site * 4 + category]++;
    statistics[site * 4 + 2 + category] += observed;
  }
  return {positions, predictors, response, siteIds, statistics, siteCount, family, paired};
}

export function computeGeneralizedGWRReviewExpectations(
  scene: ReturnType<typeof createGeneralizedGWRReviewScene>,
  bandwidth: number
) {
  const rows = scene.response.length;
  const coefficients = new Float64Array(rows * 2);
  const fitted = new Float64Array(rows);
  const rowDeviance = new Float64Array(rows);
  const influence = new Float64Array(rows);
  const neighborWeight = scene.paired && bandwidth > 1 ? (1 - 1 / bandwidth ** 2) ** 2 : 0;
  const link = (mean: number) =>
    scene.family === 'poisson' ? Math.log(mean) : Math.log(mean / (1 - mean));
  for (let row = 0; row < rows; row++) {
    const site = scene.siteIds[row];
    const neighbor = site % 2 === 0 ? site + 1 : site - 1;
    const weight = neighbor < scene.siteCount ? neighborWeight : 0;
    const mean = (category: number) => {
      const count =
        scene.statistics[site * 4 + category] +
        (weight ? weight * scene.statistics[neighbor * 4 + category] : 0);
      const total =
        scene.statistics[site * 4 + 2 + category] +
        (weight ? weight * scene.statistics[neighbor * 4 + 2 + category] : 0);
      return {value: total / count, count};
    };
    const negative = mean(0);
    const positive = mean(1);
    coefficients[row * 2] = (link(negative.value) + link(positive.value)) / 2;
    coefficients[row * 2 + 1] = (link(positive.value) - link(negative.value)) / 2;
    const category = scene.predictors[row] === -1 ? negative : positive;
    const expected = category.value;
    const observed = scene.response[row];
    fitted[row] = expected;
    influence[row] = 1 / category.count;
    rowDeviance[row] =
      scene.family === 'poisson'
        ? 2 * ((observed ? observed * Math.log(observed / expected) : 0) - observed + expected)
        : -2 * Math.log(observed ? expected : 1 - expected);
  }
  const score = (start = 0, end = rows) => {
    let deviance = 0;
    let trace = 0;
    for (let row = start; row < end; row++) {
      deviance += rowDeviance[row];
      trace += influence[row];
    }
    const count = end - start;
    return deviance + 2 * trace + (2 * trace * (trace + 1)) / (count - trace - 1);
  };
  return {coefficients, fitted, score, rowDeviance, influence};
}
