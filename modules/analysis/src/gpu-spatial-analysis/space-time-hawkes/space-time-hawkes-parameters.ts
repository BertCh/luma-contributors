// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUSpaceTimeHawkes} parameter view. */
export const GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH = 9;

/** Number of float32 rows in the `output.estimates` view of {@link GPUSpaceTimeHawkes}. */
export const GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH = 4;

/** Default relative parameter change below which the EM iteration stops. */
export const GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE = 1e-4;

/** Shared ordinary CPU/GPU iteration budget; longer CPU reference budgets remain explicit. */
export const SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS = 100;

/** Resolve the ordinary iteration budget identically in both execution forms. @internal */
export function resolveSpaceTimeHawkesMaximumIterations(maximumIterations?: number): number {
  return maximumIterations ?? SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS;
}

/** Per-frame settings of {@link GPUSpaceTimeHawkes}. */
export type GPUSpaceTimeHawkesSettings = {
  /** Area of the observation window, in squared position units. Positive and finite. */
  windowArea: number;
  /**
   * Length of the observation period in the unit of `times`. Event times lie in `[0, duration]`;
   * an event outside it is excluded. Positive and finite.
   */
  windowDuration: number;
  /** Parents more than this many time units before an event are not considered. Positive. */
  maximumLag: number;
  /** Parents farther than this many position units from an event are not considered. Positive. */
  maximumDistance: number;
  /**
   * Largest relative change of any estimate between two iterations that counts as converged.
   * Defaults to {@link GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE}.
   */
  tolerance?: number;
  /** Starting background rate, events per unit area and time unit. Positive. */
  initialBackground: number;
  /** Starting branching ratio, expected offspring per event. Positive. */
  initialBranchingRatio: number;
  /** Starting rate of the exponential time kernel, per time unit. Positive. */
  initialDecayRate: number;
  /** Starting standard deviation of the Gaussian offspring displacement, in position units. Positive. */
  initialSpatialStandardDeviation: number;
};

/**
 * Packs per-frame {@link GPUSpaceTimeHawkes} parameters.
 *
 * Layout (float32): `[windowArea, windowDuration, maximumLag, maximumDistance, tolerance,

 * initialBackground, initialBranchingRatio, initialDecayRate, initialSpatialStandardDeviation]`.
 *
 * @param settings Window, parent bounds, tolerance and starting values.
 * @param target Optional destination of at least {@link GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} elements.
 * @throws If a setting that must be positive is not, or a value is not finite.
 */
export function getGPUSpaceTimeHawkesParameterValues(
  settings: GPUSpaceTimeHawkesSettings,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH)
): Float32Array {
  const name = 'getGPUSpaceTimeHawkesParameterValues';
  if (target.length < GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH) {
    throw new Error(`${name} target must hold ${GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH} elements`);
  }
  const tolerance = settings.tolerance ?? GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE;
  const positive = [
    ['windowArea', settings.windowArea],
    ['windowDuration', settings.windowDuration],
    ['maximumLag', settings.maximumLag],
    ['maximumDistance', settings.maximumDistance],
    ['tolerance', tolerance],
    ['initialBackground', settings.initialBackground],
    ['initialBranchingRatio', settings.initialBranchingRatio],
    ['initialDecayRate', settings.initialDecayRate],
    ['initialSpatialStandardDeviation', settings.initialSpatialStandardDeviation]
  ] as const;
  for (const [label, value] of positive) {
    if (!(Math.fround(value) > 0) || !Number.isFinite(Math.fround(value))) {
      throw new Error(`${name} ${label} must be positive and finite`);
    }
  }
  target.set(positive.map(([, value]) => value));
  return target;
}
