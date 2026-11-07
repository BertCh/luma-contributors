// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerDatasetPopulation} from '@luma-contributors/kepler';

/** The current sandbox schema cannot represent a partial stream as a distinct population. */
export class IncompleteSandboxPopulationError extends Error {
  constructor() {
    super('Unfinished streams cannot be saved or checkpointed in the current sandbox schema');
    this.name = 'IncompleteSandboxPopulationError';
  }
}

export function assertCompleteSandboxPopulations(
  datasets: readonly {population?: KeplerDatasetPopulation}[]
): void {
  if (datasets.some(dataset => dataset.population && dataset.population.state !== 'complete'))
    throw new IncompleteSandboxPopulationError();
}
