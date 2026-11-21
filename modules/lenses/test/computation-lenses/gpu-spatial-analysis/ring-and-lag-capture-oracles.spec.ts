// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {test} from 'vitest';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/segment-ring-assembly.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-lag-maximum-likelihood.lens';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {captureLensScenarioRecord} from '../../../src/computation-lenses/run-lens-scenario';
import {assertRingCapture, assertLagMLCapture} from './ring-and-lag-capture-assertions';

test('ring lens capture closes every active ring and conserves coverage area after shared borders cancel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const file = await captureLensScenarioRecord(
    getLensScenario('gpu-spatial-analysis.segment-ring-assembly')!,
    device
  );
  const {record} = fromRecordedLensFile(file);
  assertRingCapture(record);
}, 120000);

test('lag ML live lens captures the estimate column with rho in coefficient row four', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const file = await captureLensScenarioRecord(
    getLensScenario('gpu-spatial-analysis.spatial-lag-maximum-likelihood')!,
    device
  );
  const {record} = fromRecordedLensFile(file);
  assertLagMLCapture(record);
}, 120000);
