// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());
import {
  captureCanvas,
  validateImageDimensions
} from '../../../examples/deck/kepler-compute/src/native-image-capture';

vi.mock('@kepler.gl/utils', () => ({convertToPng: vi.fn()}));

test('image capture bounds reject invalid and oversized surfaces before pixel allocation', () => {
  expect(() => validateImageDimensions(4096, 4096)).not.toThrow();
  expect(() => validateImageDimensions(8192, 2048)).not.toThrow();
  for (const [width, height] of [
    [0, 100],
    [100, -1],
    [0.5, 100],
    [NaN, 100],
    [Infinity, 100],
    [8193, 1],
    [4096, 4097]
  ]) {
    const toDataURL = vi.fn(() => 'data:image/png;base64,captured');
    expect(() => captureCanvas({width, height, toDataURL})).toThrow('16 megapixels');
    expect(toDataURL).not.toHaveBeenCalled();
  }
});

test('image capture retains actual pixels and propagates browser security failures', () => {
  const toDataURL = vi.fn(() => 'data:image/png;base64,captured');
  expect(captureCanvas({width: 10, height: 10, toDataURL})).toBe('data:image/png;base64,captured');
  expect(toDataURL).toHaveBeenCalledExactlyOnceWith('image/png');
  const securityError = new DOMException(
    'The canvas has been tainted by cross-origin data.',
    'SecurityError'
  );
  expect(() =>
    captureCanvas({
      width: 10,
      height: 10,
      toDataURL: () => {
        throw securityError;
      }
    })
  ).toThrow(securityError);
});

test('browser empty canvas output is rejected rather than published as an image', () => {
  expect(() => captureCanvas({width: 10, height: 10, toDataURL: () => 'data:,'})).toThrow(
    'could not capture'
  );
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('@kepler.gl/utils');
  vi.resetModules();
});
