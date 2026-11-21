// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createGridBuilderAgreementFixture} from './grid-coordinate-builder-agreement-fixture';
import {expectCompactPublicationReuse} from './compact-publisher-status-fixture';
import {GRID_COORDINATE_CASES} from './grid-coordinate-bump-fixtures';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

it('B1 constructs all four overflow kernels without malformed exponent literals', () => {
  const device = createNullWebGPUDevice();
  const testCase = GRID_COORDINATE_CASES.find(
    candidate => candidate.name === 'overflowing domain width'
  )!;
  const fixture = createGridBuilderAgreementFixture(device, testCase);
  try {
    expect(fixture.bounds).toMatchObject({format: 'float32x4', length: 1});
    const sources = getGPUCommandNodeKernelSources(fixture.nodes);
    expect(sources.filter(({source}) => source.includes('fn getCoordinate('))).toHaveLength(4);
    for (const {source} of sources) {
      expect(source).not.toMatch(/\d[eE][+-]?\d+\.\d/);
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('B3 rejects status arrays containing extra words and accepts complete publication reuse', async () => {
  for (const extraWord of [true, false]) {
    let total = 1;
    let candidateFlag = 0;
    const fixture = {
      writeInput(name: string, values: Uint32Array) {
        if (name === 'total') total = values[0];
        if (name === 'candidateFlag') candidateFlag = values[0];
      },
      async run() {
        return {
          count: extraWord ? [Math.min(total, 2), 99] : [Math.min(total, 2)],
          requiredCount: [total],
          overflow: [Number(total > 2)],
          candidateOverflow: [candidateFlag]
        };
      }
    };
    if (extraWord) await expect(expectCompactPublicationReuse(fixture)).rejects.toThrow();
    else await expect(expectCompactPublicationReuse(fixture)).resolves.toBeUndefined();
  }
});

it('B3 rejects a publisher that retains the previous overflow word on candidate loss', async () => {
  let total = 3;
  let candidateFlag = 0;
  let overflow = 1; // The device sequence's preceding truncation left this word set.
  const fixture = {
    writeInput(name: string, values: Uint32Array) {
      if (name === 'total') total = values[0];
      if (name === 'candidateFlag') candidateFlag = values[0];
    },
    async run() {
      if (!candidateFlag) overflow = Number(total > 2);
      return {
        count: [Math.min(total, 2)],
        requiredCount: [total],
        overflow: [overflow],
        candidateOverflow: [candidateFlag]
      };
    }
  };
  await expect(expectCompactPublicationReuse(fixture)).rejects.toThrow();
});
