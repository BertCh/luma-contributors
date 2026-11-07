// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {planKeplerTripAnimation} from '../src/analysis/trip-animation';

test('native absolute current time rebases once to the prepared origin independently of global domain', () => {
  // Pinned native trip-layer.js653/666 subtract domain0 from both vertices/currentTime;
  // 663 multiplies trailLength seconds by1000. Prepared vertices already use their own origin.
  const epoch = 1_700_000_000_000;
  expect(
    planKeplerTripAnimation(
      {originMilliseconds: epoch + 1000, maximumTimeErrorMilliseconds: 0},
      {
        currentTimeMilliseconds: epoch + 1500,
        domainMilliseconds: [epoch, epoch + 5000],
        trailLengthSeconds: 0.25,
        fadeTrail: false
      }
    )
  ).toEqual({
    supported: true,
    currentTime: 500,
    trailLength: 250,
    fadeTrail: false,
    originMilliseconds: epoch + 1000,
    actualTimeErrorMilliseconds: 0
  });
});

test('animation planner reports grouped/scenegraph routes and time precision without silently changing semantics', () => {
  const source = {originMilliseconds: 0, maximumTimeErrorMilliseconds: 0};
  const animation = {
    currentTimeMilliseconds: 16_777_217,
    domainMilliseconds: [0, 20_000_000] as const,
    trailLengthSeconds: 1
  };
  expect(planKeplerTripAnimation(source, animation).supported).toBe(false);
  expect(
    planKeplerTripAnimation({...source, maximumTimeErrorMilliseconds: 1}, animation)
  ).toMatchObject({supported: true, currentTime: 16_777_216, actualTimeErrorMilliseconds: 1});
  expect(planKeplerTripAnimation(source, {...animation, columnMode: 'table'})).toEqual(
    planKeplerTripAnimation(source, animation)
  );
  expect(planKeplerTripAnimation(source, {...animation, scenegraphEnabled: true})).toMatchObject({
    supported: false,
    reason: expect.stringMatching(/scenegraph/)
  });
  expect(
    planKeplerTripAnimation(source, {
      ...animation,
      currentTimeMilliseconds: 0,
      trailLengthSeconds: -1
    }).supported
  ).toBe(false);
});
