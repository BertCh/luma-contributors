// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {lineGraphOracle} from './network-line-graph-oracle';
import {getGPUNetworkLineGraphParameterValues} from '../../../src/gpu-network/network-line-graph/network-line-graph-parameters';
import {getGPUNetworkStatisticsParameterValues} from '../../../src/gpu-network/network-statistics/gpu-network-statistics';

it('rejects controls that leave their encoded range (NET-117)', () => {
  // IEEE binary32 max = (2 - 2^-23) * 2^127; adjacent integers above 2^24 are not exact.
  for (const name of [
    'turnCostPerRadian',
    'leftTurnCost',
    'rightTurnCost',
    'uTurnCost',
    'straightAngleRadians',
    'uTurnAngleRadians'
  ]) {
    expect(() => getGPUNetworkLineGraphParameterValues({[name]: 1e100}), name).toThrow();
  }
  expect(() => getGPUNetworkLineGraphParameterValues({bannedTurnCount: 16777217})).toThrow(
    /bannedTurnCount/
  );
  expect(() => getGPUNetworkLineGraphParameterValues({bannedTurnCount: 2 ** 32})).toThrow(
    /bannedTurnCount/
  );
  expect(() => getGPUNetworkLineGraphParameterValues({uTurnCost: -1e-100})).toThrow(/uTurnCost/);
  for (const name of ['straightAngleRadians', 'uTurnAngleRadians']) {
    expect(() => getGPUNetworkLineGraphParameterValues({[name]: -1})).toThrow();
  }
  expect(() => getGPUNetworkStatisticsParameterValues({resolution: 1e100})).toThrow(/resolution/);
  expect(() => getGPUNetworkStatisticsParameterValues({degreeBinWidth: 2 ** 32})).toThrow(
    /degreeBinWidth/
  );
  expect(() => getGPUNetworkStatisticsParameterValues({degreeBinWidth: NaN})).toThrow(
    /degreeBinWidth/
  );
});

it('retains neighboring encodable controls and documented flooring (NET-117)', () => {
  const maximumFloat = (2 - 2 ** -23) * 2 ** 127;
  const packed = getGPUNetworkLineGraphParameterValues({
    turnCostPerRadian: maximumFloat,
    bannedTurnCount: 16777216,
    uTurnCost: -2
  });
  expect(packed[0]).toBe(maximumFloat);
  expect(packed[3]).toBe(-2);
  expect(packed[5]).toBe(16777216);
  expect(getGPUNetworkLineGraphParameterValues({bannedTurnCount: 16777218})[5]).toBe(16777218);
  expect(getGPUNetworkLineGraphParameterValues({bannedTurnCount: 4294967040})[5]).toBe(4294967040);
  for (const uTurnCost of [-(2 ** -149), -(2 ** -140), -(2 ** -126)]) {
    const encoded = getGPUNetworkLineGraphParameterValues({uTurnCost})[3];
    // A normal negative word retains the ban even on a device that flushes subnormals.
    expect(encoded).toBeLessThanOrEqual(-(2 ** -126));
  }
  expect(getGPUNetworkStatisticsParameterValues({degreeBinWidth: 2 ** 32 - 1})[1]).toBe(4294967295);
  expect(getGPUNetworkStatisticsParameterValues({degreeBinWidth: 3.9})[1]).toBe(3);
  expect(getGPUNetworkStatisticsParameterValues({degreeBinWidth: -0.5})[1]).toBe(1);
  const words = getGPUNetworkStatisticsParameterValues({resolution: maximumFloat});
  expect(new Float32Array(words.buffer)[0]).toBe(maximumFloat);
});

it('explicit radian and turn-cost names preserve packed words and reject retired keys (UNI-19)', () => {
  const current = getGPUNetworkLineGraphParameterValues({
    turnCostPerRadian: 2,
    straightAngleRadians: 0.25,
    uTurnAngleRadians: 3,
    uTurnCost: -(2 ** -149)
  });
  expect(current[0]).toBe(2);
  expect(current[4]).toBe(0.25);
  expect(current[6]).toBe(3);
  expect(current[3]).toBe(-1);
  for (const [name, oldName] of [
    ['turnCostPerRadian', 'angleCost'],
    ['straightAngleRadians', 'straightAngle'],
    ['uTurnAngleRadians', 'uTurnAngle']
  ]) {
    expect(() => getGPUNetworkLineGraphParameterValues({[name]: 1, [oldName]: 2})).toThrow(
      new RegExp(oldName + ' was removed; use ' + name)
    );
    expect(() => getGPUNetworkLineGraphParameterValues({[name]: 1e100})).toThrow();
    expect(() => getGPUNetworkLineGraphParameterValues({[name]: -1})).toThrow();
  }
});

it('line graph oracle uses cost per radian against a hand-worked right angle (B2)', () => {
  const base = {
    offsets: Uint32Array.of(0, 1, 2, 2),
    neighbors: Uint32Array.of(1, 2),
    weights: Float32Array.of(1, 4)
  };
  const positions = [0, 0, 1, 0, 1, 1];
  const result = lineGraphOracle(base, positions, [], {turnCostPerRadian: 2}, 1);
  // 90 degrees = pi/2 radians, with 2 cost/radian and next-edge cost 4 => 4 + pi.
  expect(Number.isFinite(result.csr.weights[0])).toBe(true);
  expect(result.csr.weights[0]).toBeCloseTo(7.141592653589793, 6);
});
