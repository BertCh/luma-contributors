// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Forecast scenes for the scoring-rule runs: Helene rainfall at Carolina stations. The day-one
// rain (log residual) is the forecast, the day-two rain is the observation.

import type {Random, World} from '../../../inputs';
import {createRainScene, type RainScene} from './rain';

export type ForecastScene = RainScene & {
  /** Day-one log-rain residual per station, 0 (the trend) where day one is missing. */
  baseline: Float32Array;
  /** Station nearest the middle of the map with both days measured: the probe's starting point. */
  defaultStation: number;
};

export function createForecastScene(random: Random, world: World, count: number): ForecastScene {
  const scene = createRainScene(random, world, count);
  const baseline = Float32Array.from(scene.previousResiduals, value =>
    Number.isFinite(value) ? value : 0
  );
  let defaultStation = 0;
  let nearest = Infinity;
  for (let station = 0; station < scene.count; station++) {
    const forecast = scene.previousResiduals[station];
    if (!Number.isFinite(forecast) || Math.abs(forecast) > 1.5) continue;
    if (Math.abs(scene.residuals[station] - forecast) > 1.2) continue;
    const distance = Math.hypot(
      scene.positions[station * 2] - scene.mapWidth * 0.5,
      scene.positions[station * 2 + 1] - 0.5
    );
    if (distance < nearest) {
      nearest = distance;
      defaultStation = station;
    }
  }
  return {...scene, baseline, defaultStation};
}

function getGaussian(random: Random): number {
  return Math.sqrt(-2 * Math.log(Math.max(random(), 1e-9))) * Math.cos(2 * Math.PI * random());
}

/**
 * `memberCount` unit-variance fields over the stations, member-major: a sum of random plane waves
 * (spatially correlated at about `correlationLength`) plus a little independent noise.
 */
export function createCorrelatedFields(
  random: Random,
  positions: Float32Array,
  memberCount: number,
  correlationLength: number
): Float32Array {
  const siteCount = positions.length / 2;
  const waveCount = 24;
  const fields = new Float32Array(memberCount * siteCount);
  for (let member = 0; member < memberCount; member++) {
    const waves = Array.from({length: waveCount}, () => ({
      x: getGaussian(random) / correlationLength,
      y: getGaussian(random) / correlationLength,
      phase: random() * 2 * Math.PI
    }));
    for (let site = 0; site < siteCount; site++) {
      let sum = 0;
      for (const wave of waves) {
        sum += Math.cos(
          wave.x * positions[site * 2] + wave.y * positions[site * 2 + 1] + wave.phase
        );
      }
      fields[member * siteCount + site] =
        0.9 * Math.sqrt(2 / waveCount) * sum + 0.43 * getGaussian(random);
    }
  }
  return fields;
}
