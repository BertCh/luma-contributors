// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createSpatialReplayCaptures} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/replay-captures';

test('publishes only a reported replay prefix and suppresses zero-count distributions', () => {
  const tags = {
    nullDistribution: {keySpace: 'permutation', meaning: 'pinned row', infoCapture: 'replayInfo'}
  };
  for (const count of [0, 5]) {
    const replays = createSpatialReplayCaptures(
      [
        {
          id: 'distribution',
          name: 'nullDistribution',
          elementType: 'f32',
          provenance: 'gpu-readback',
          values: Float32Array.from({length: 99}, (_, index) => index)
        },
        {
          id: 'info',
          name: 'replayInfo',
          elementType: 'f32',
          provenance: 'gpu-readback',
          values: Float32Array.of(1, count, 0)
        }
      ],
      tags
    );
    expect(replays.map(capture => Array.from(capture.values))).toEqual(
      count ? [[0, 1, 2, 3, 4]] : []
    );
  }
});
