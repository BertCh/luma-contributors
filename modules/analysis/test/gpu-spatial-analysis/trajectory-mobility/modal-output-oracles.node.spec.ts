// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeModalSplitOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation';
import {getModalFixture} from './modal-split-operation-fixture';

it('modal predictions follow literal speed classes and clear the inactive capacity row', () => {
  // Speeds 3/3=1,12/2=6, undefined (duration 0),6/2=3,40/2=20.
  const result = computeModalSplitOnCPU(getModalFixture(), {
    modeCount: 3,
    metric: 'count',
    modeSpeedUpperBoundsFloat32: [2, 10, Infinity]
  });
  expect(Array.from(result.predictedModes)).toEqual([0, 1, 0xffffffff, 1, 2, 0]);
  expect(Array.from(result.modalSplit)).toEqual([1, 1, 0, 0, 1, 1]);
});
