// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createAnalysisRunIdentity} from '../../../../../examples/deck/kepler-compute/src/analysis-run-identity';

describe('analysis run identity', () => {
  test.each([
    'revision',
    'runtime',
    'cancel',
    'content',
    'unavailable'
  ] as const)('%s permanently invalidates a captured run', change => {
    const originalRuntime = {};
    let runtime = originalRuntime;
    let revision = '1';
    let current = true;
    let unavailable = false;
    const controller = new AbortController();
    const identity = createAnalysisRunIdentity({
      signal: controller.signal,
      getRuntime: () => runtime,
      getRevisionKey: () => {
        if (unavailable) throw new Error('Source is unavailable');
        return revision;
      },
      isCurrent: () => current
    });
    expect(identity.isCurrent()).toBe(true);
    if (change === 'revision') revision = '2';
    if (change === 'runtime') runtime = {};
    if (change === 'cancel') controller.abort();
    if (change === 'content') current = false;
    if (change === 'unavailable') unavailable = true;
    expect(identity.isCurrent()).toBe(false);
    runtime = originalRuntime;
    revision = '1';
    current = true;
    unavailable = false;
    expect(identity.isCurrent()).toBe(false);
  });
});
