// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  runCostDistanceRoundMatrix,
  runCostDistanceSerpentineDiagnostic
} from './cost-distance-round-diagnostic';

it('measures 40 parallel/controlled corridor repetitions and 40 one-shot/continued serpentines', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const corridor = await runCostDistanceRoundMatrix(device, 40);
  const serpentine = await runCostDistanceSerpentineDiagnostic(device, 40);
  // Opt-in report mode deliberately throws so the headless lead receives successful-run
  // frequencies and timings. Natural scheduling variation is never a correctness failure.
  if (import.meta.env['VITE_COST_DISTANCE_ROUND_REPORT'] === '1') {
    throw new Error(
      `Cost-distance measurement report (requested): ${JSON.stringify({corridor, serpentine})}`
    );
  }
}, 240000);
