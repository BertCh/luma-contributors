// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensScenarioPresentation} from './types';

/**
 * Result presentation of scenarios whose headline is not their last tagged output. This module has
 * no side effects and imports no scenario code, so a reader of a saved record selects the same
 * headline whether or not any scenario registration module was loaded. Scenario descriptors take
 * their default `presentation` from this table; new recordings also carry it in the envelope.
 */
const LENS_SCENARIO_PRESENTATIONS: Readonly<Record<string, Readonly<LensScenarioPresentation>>> =
  Object.freeze({
    'gpu-network.network-strongly-connected-components': Object.freeze({
      primaryOutputCaptureId:
        'gpu-network.network-strongly-connected-components-snapshots-strongComponents-0'
    }),
    'gpu-spatial-analysis.dot-density': Object.freeze({
      primaryOutputCaptureId: 'gpu-spatial-analysis-dot-density-snapshots-points-0'
    }),
    'gpu-spatial-analysis.geometry-validity': Object.freeze({
      primaryOutputCaptureId: 'gpu-spatial-analysis-geometry-validity-snapshots-resultParks-0'
    }),
    'gpu-spatial-analysis.kriging': Object.freeze({
      primaryOutputCaptureId: 'gpu-spatial-analysis-kriging-snapshots-rainfall-0'
    }),
    'gpu-spatial-analysis.segment-intersection': Object.freeze({
      primaryOutputCaptureId: 'gpu-spatial-analysis-segment-intersection-snapshots-intersections-0'
    }),
    'gpu-spatial-analysis.spatial-predicate-join': Object.freeze({
      primaryOutputCaptureId: 'gpu-spatial-analysis-spatial-predicate-join-snapshots-joinedTracts-0'
    })
  });

/** Returns the code-side presentation default for a scenario, without registration. @internal */
export function getLensScenarioPresentation(
  scenarioId: string
): Readonly<LensScenarioPresentation> | undefined {
  return Object.hasOwn(LENS_SCENARIO_PRESENTATIONS, scenarioId)
    ? LENS_SCENARIO_PRESENTATIONS[scenarioId]
    : undefined;
}
