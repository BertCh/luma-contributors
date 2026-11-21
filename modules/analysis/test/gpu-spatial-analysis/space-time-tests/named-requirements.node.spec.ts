// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import * as contributorUtils from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {validateSpaceTimeBandEdges} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-bands';
import {validateSpaceTimeStopping} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-stopping';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('routes band constructor requirements through named helpers without changing their messages', () => {
  const requirement = vi.spyOn(contributorUtils, 'validateRequirement');
  try {
    for (const [edges, message] of [
      [[0], 'needs 2 to 17 edges'],
      [Array.from({length: 18}, (_, index) => index), 'needs 2 to 17 edges'],
      [[0, NaN], 'edges must be finite and non-negative'],
      [[-1, 1], 'edges must be finite and non-negative'],
      [[0, 1e100], 'edges must be finite and non-negative'],
      [[1, 1 + 2 ** -25], 'edges must be strictly ascending as float32']
    ] as const) {
      requirement.mockClear();
      expect(() => validateSpaceTimeBandEdges('named-bands', 'timeEdges', edges)).toThrow(
        `named-bands timeEdges ${message}`
      );
      expect(requirement).toHaveBeenCalled();
    }
    expect(() => validateSpaceTimeBandEdges('named-bands', 'timeEdges', [0, 1])).not.toThrow();
  } finally {
    requirement.mockRestore();
  }
});

it('routes stopping dependencies through named helpers without changing their rejection domains', () => {
  const requirement = vi.spyOn(contributorUtils, 'validateRequirement');
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const count = createTransientView(graph, 'count', 'uint32', 1);
    expect(() => validateSpaceTimeStopping('named-stop', undefined, 3, count)).toThrow(
      'named-stop output.permutationCount needs stopAfterExceedances'
    );
    expect(requirement).toHaveBeenCalled();
    requirement.mockClear();
    expect(() => validateSpaceTimeStopping('named-stop', 1, 3, undefined)).toThrow(
      'named-stop stopAfterExceedances needs output.permutationCount'
    );
    expect(requirement).toHaveBeenCalled();
    expect(() => validateSpaceTimeStopping('named-stop', undefined, 3, undefined)).not.toThrow();
    expect(() => validateSpaceTimeStopping('named-stop', 1, 3, count)).not.toThrow();
    expect(() => validateSpaceTimeStopping('named-stop', 0, 3, count)).toThrow(
      /named-stop stopAfterExceedances/
    );
  } finally {
    requirement.mockRestore();
    device.destroy();
  }
});
