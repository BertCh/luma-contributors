// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, onTestFinished} from 'vitest';
import {readFileSync} from 'node:fs';
import {requireTestDevice} from './require-test-device';

const SILENT_DEVICE_RETURN = /if\s*\([^)]*![a-zA-Z]*[dD]evice[^)]*\)\s*(?:{\s*)?return\s*;/;

it('detects a silent device return while allowing an explicit skip', () => {
  expect('if (!device) return;').toMatch(SILENT_DEVICE_RETURN);
  expect('if (!coreDevice || !maxDevice) { return; }').toMatch(SILENT_DEVICE_RETURN);
  expect('if (!device) context.skip("missing profile");').not.toMatch(SILENT_DEVICE_RETURN);
});

it('recording and profile correctness specs cannot silently return without a device', () => {
  for (const path of [
    '../computation-record/gpu-transfer-mask.spec.ts',
    '../computation-record/computation-record-capture.spec.ts',
    '../gpu-raster/cost-distance/gpu-cost-distance-recording.spec.ts',
    './core-device-limits.spec.ts'
  ]) {
    expect(readFileSync(new URL(path, import.meta.url), 'utf8'), path).not.toMatch(
      SILENT_DEVICE_RETURN
    );
  }
});

it('requireTestDevice returns for a device and narrows its type', () => {
  const device: {label: string} | null = {label: 'device'};
  requireTestDevice(device);
  expect(device.label).toBe('device');
});

it('requireTestDevice skips the running test outside a browser', context => {
  onTestFinished(({task}) => {
    // A skip, not a pass: the guarded spec must show up in the skip count.
    expect(task.result?.state).toBe('skip');
  });
  requireTestDevice(null);
  context.expect.unreachable('requireTestDevice must stop the test');
});
