// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, test, vi} from 'vitest';
import {
  ANALYSIS_DEFAULT_BUFFER_SIZE,
  ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE,
  getAnalysisMaximumRowCount,
  getAnalysisRequiredLimits
} from '../../src/analysis-core/graph-support/device-limits';
import {
  getHostRequiredLimits,
  resolveHostRequiredLimits
} from '../../../../examples/deck/kepler-compute/src/device-limits';

// Recorded Apple byte ceilings, with synthetic binding/compute limits to exercise target caps.
const APPLE_ADAPTER_LIMITS = {
  maxStorageBuffersPerShaderStage: 32,
  maxStorageBufferBindingSize: 4294967292,
  maxBufferSize: 4294967292,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024
};

afterEach(() => vi.unstubAllGlobals());

describe('analysis required device limits', () => {
  test('requests adapter buffer maxima and targets sixteen storage bindings', () => {
    expect(getAnalysisRequiredLimits(APPLE_ADAPTER_LIMITS)).toEqual({
      maxStorageBuffersPerShaderStage: 16,
      maxStorageBufferBindingSize: 4294967292,
      maxBufferSize: 4294967292
    });
    expect(
      getAnalysisRequiredLimits({...APPLE_ADAPTER_LIMITS, maxStorageBuffersPerShaderStage: 12})
    ).toMatchObject({maxStorageBuffersPerShaderStage: 12});
    expect(
      getAnalysisRequiredLimits(APPLE_ADAPTER_LIMITS, {
        requestedStorageBuffersPerShaderStage: 64
      })
    ).toMatchObject({maxStorageBuffersPerShaderStage: 32});
  });

  test('handles a missing adapter and never lowers WebGPU defaults', () => {
    expect(getAnalysisRequiredLimits()).toEqual({});
    expect(getAnalysisRequiredLimits(null)).toEqual({});
    expect(
      getAnalysisRequiredLimits(
        {
          maxStorageBuffersPerShaderStage: 0,
          maxStorageBufferBindingSize: Number.NaN,
          maxBufferSize: 1
        },
        {requestedStorageBuffersPerShaderStage: 1}
      )
    ).toEqual({
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE,
      maxBufferSize: ANALYSIS_DEFAULT_BUFFER_SIZE
    });
  });

  test('requests optional compute targets only when asked, clamped to support', () => {
    expect(
      getAnalysisRequiredLimits(APPLE_ADAPTER_LIMITS, {
        maxComputeWorkgroupStorageSize: 65536,
        maxComputeInvocationsPerWorkgroup: 512
      })
    ).toMatchObject({
      maxComputeWorkgroupStorageSize: 32768,
      maxComputeInvocationsPerWorkgroup: 512
    });
    expect(
      getAnalysisRequiredLimits(
        {
          maxStorageBuffersPerShaderStage: 8,
          maxStorageBufferBindingSize: ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE,
          maxBufferSize: ANALYSIS_DEFAULT_BUFFER_SIZE
        },
        {
          maxComputeWorkgroupStorageSize: 1,
          maxComputeInvocationsPerWorkgroup: 1
        }
      )
    ).toMatchObject({
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256
    });
  });

  test.each([
    0,
    -1,
    1.5,
    Number.NaN,
    Number.POSITIVE_INFINITY
  ])('rejects invalid request target %s with a named limit', requestedLimit => {
    for (const name of [
      'requestedStorageBuffersPerShaderStage',
      'maxComputeWorkgroupStorageSize',
      'maxComputeInvocationsPerWorkgroup'
    ]) {
      expect(() =>
        getAnalysisRequiredLimits(APPLE_ADAPTER_LIMITS, {
          [name]: requestedLimit
        })
      ).toThrow(`analysis-device-limits ${name} must be an integer in [1,`);
    }
  });
});

describe('analysis packed row capacity', () => {
  test('computes large Apple ceilings without allocating GPU resources', () => {
    expect(getAnalysisMaximumRowCount(APPLE_ADAPTER_LIMITS, 4)).toBe(1073741823);
    expect(getAnalysisMaximumRowCount(APPLE_ADAPTER_LIMITS, 8)).toBe(536870911);
    expect(getAnalysisMaximumRowCount(APPLE_ADAPTER_LIMITS, 16)).toBe(268435455);
    expect(
      getAnalysisMaximumRowCount(
        {
          maxStorageBufferBindingSize: ANALYSIS_DEFAULT_STORAGE_BUFFER_BINDING_SIZE,
          maxBufferSize: ANALYSIS_DEFAULT_BUFFER_SIZE
        },
        16
      )
    ).toBe(8388608);
  });

  test('floors whole rows and respects a smaller backing buffer or zero capacity', () => {
    expect(
      getAnalysisMaximumRowCount(
        {
          maxStorageBufferBindingSize: 128,
          maxBufferSize: 35
        },
        16
      )
    ).toBe(2);
    expect(
      getAnalysisMaximumRowCount(
        {
          maxStorageBufferBindingSize: 0,
          maxBufferSize: 256
        },
        4
      )
    ).toBe(0);
    expect(
      getAnalysisMaximumRowCount(
        {
          maxStorageBufferBindingSize: 8,
          maxBufferSize: 256
        },
        16
      )
    ).toBe(0);
  });

  test.each([
    0,
    -1,
    1.5,
    Number.NaN,
    Number.POSITIVE_INFINITY
  ])('rejects invalid bytesPerRow %s', bytesPerRow => {
    expect(() => getAnalysisMaximumRowCount(APPLE_ADAPTER_LIMITS, bytesPerRow)).toThrow(
      'analysis-device-limits bytesPerRow must be an integer in [1,'
    );
  });
});

describe('kepler analysis device request', () => {
  test('requests all adapter bindings and maximum buffer sizes', async () => {
    vi.stubGlobal('navigator', {
      gpu: {requestAdapter: async () => ({limits: APPLE_ADAPTER_LIMITS})}
    });
    expect(await resolveHostRequiredLimits()).toEqual({
      maxStorageBuffersPerShaderStage: 32,
      maxStorageBufferBindingSize: 4294967292,
      maxBufferSize: 4294967292
    });
    expect(getHostRequiredLimits()).toEqual(await resolveHostRequiredLimits());
  });

  test('clears resolved limits when no adapter is available or discovery fails', async () => {
    vi.stubGlobal('navigator', {
      gpu: {requestAdapter: async () => ({limits: APPLE_ADAPTER_LIMITS})}
    });
    await resolveHostRequiredLimits();
    vi.stubGlobal('navigator', {gpu: {requestAdapter: async () => null}});
    expect(await resolveHostRequiredLimits()).toEqual({});
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: async () => {
          throw Error('Unavailable');
        }
      }
    });
    expect(await resolveHostRequiredLimits()).toEqual({});
    vi.stubGlobal('navigator', {});
    expect(await resolveHostRequiredLimits()).toEqual({});
  });
});
