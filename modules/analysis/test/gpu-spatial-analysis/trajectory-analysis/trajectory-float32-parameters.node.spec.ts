// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUTrajectoryHampelParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-hampel';
import {getGPUTrajectoryThinningParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {getGPUTrajectoryOutlierFlagsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-outlier-flags';
import {getGPUTrajectoryLocationsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations';
import {getGPUTrajectorySegmentsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments';
import {getGPUTrajectoryStaypointsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints';
import {
  getGPUTrajectoryClockParameterValues,
  getGPUTrajectoryClockWordParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/trajectory-clock-parameters';
import {
  getGPUTrajectoryPlayheadParameterValues,
  getGPUTrajectoryPlayheadWordParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/trajectory-playhead-parameters';

it('rejects finite settings that become infinite or zero when packed for the GPU', () => {
  for (const value of [1e40, Number.MAX_VALUE]) {
    for (const pack of [
      () => getGPUTrajectoryHampelParameterValues({numberOfSigmas: value}),
      () => getGPUTrajectoryThinningParameterValues({tolerance: value}),
      () => getGPUTrajectoryOutlierFlagsParameterValues({maximumSpeed: value}),
      () =>
        getGPUTrajectoryOutlierFlagsParameterValues({maximumSpeed: 1, maximumAcceleration: value}),
      () => getGPUTrajectoryLocationsParameterValues({epsilon: value}),
      () =>
        getGPUTrajectorySegmentsParameterValues({maximumGap: value, activityMinimumDuration: 1}),
      () =>
        getGPUTrajectorySegmentsParameterValues({maximumGap: 0, activityMinimumDuration: value}),
      () => getGPUTrajectoryStaypointsParameterValues({distanceThreshold: value, timeThreshold: 1}),
      () => getGPUTrajectoryStaypointsParameterValues({distanceThreshold: 1, timeThreshold: value}),
      () =>
        getGPUTrajectoryStaypointsParameterValues({
          distanceThreshold: 1,
          timeThreshold: 1,
          maximumGap: value
        }),
      () => getGPUTrajectoryClockParameterValues({start: value, step: 1}),
      () => getGPUTrajectoryClockParameterValues({start: 0, step: value}),
      () => getGPUTrajectoryClockWordParameterValues({start: 0n, step: value}),
      () => getGPUTrajectoryPlayheadParameterValues({playhead: value}),
      () => getGPUTrajectoryPlayheadParameterValues({playhead: 0, maximumGap: value}),
      () => getGPUTrajectoryPlayheadWordParameterValues({playhead: 0n, maximumGap: value})
    ])
      expect(pack).toThrow(/finite/);
  }
  expect(() => getGPUTrajectoryClockParameterValues({start: 0, step: 1e-50})).toThrow(/positive/);
  expect(() => getGPUTrajectoryLocationsParameterValues({epsilon: 1e-50})).toThrow(/positive/);
  expect(() =>
    getGPUTrajectoryStaypointsParameterValues({distanceThreshold: 1e-50, timeThreshold: 1})
  ).toThrow(/positive/);
});
