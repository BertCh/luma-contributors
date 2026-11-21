// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPURasterTexture} from '../../../src/raster-analysis/remote-sensing/gpu-raster-texture';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

it('B1 preserves exact integer boundaries for eleven bins over [0,77]', () => {
  const {device, graph, view} = createKernelAuditContext('texture-integer-boundary');
  try {
    const texture = new GPURasterTexture({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      values: view('float32', 2),
      levels: 11,
      range: [0, 77],
      radius: 1,
      offset: [1, 0],
      features: view('float32', 14),
      pairCount: view('uint32', 2)
    });
    const source = getGPUCommandNodeKernelSources(texture.getCommandNodes(graph))[0].source;
    // Exact bin width is 77/11=7. Samples 56 and 63 are bins 8 and 9: contrast 1.
    const boundaries = Array.from(source.matchAll(/value >= ([\d.eE+-]+)/g), match =>
      Number(match[1])
    );
    expect(boundaries).toEqual([7, 14, 21, 28, 35, 42, 49, 56, 63, 70]);
    expect([56, 63].map(value => boundaries.filter(boundary => value >= boundary).length)).toEqual([
      8, 9
    ]);
  } finally {
    device.destroy();
  }
});

it('quantizes against one interval formed from the rounded float32 endpoints', () => {
  const {device, graph, view} = createKernelAuditContext('texture-interval');
  try {
    const range: [number, number] = [10000000.49, 10000002.51];
    const texture = new GPURasterTexture({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      values: view('float32', 2),
      levels: 4,
      range,
      radius: 1,
      offset: [1, 0],
      features: view('float32', 14),
      pairCount: view('uint32', 2)
    });
    const source = getGPUCommandNodeKernelSources(texture.getCommandNodes(graph))[0].source;
    // At this magnitude float32 spacing is 1: endpoints become 10000000 and 10000003.
    // The bins of 10000000 and 10000002 are floor(0*4/3)=0 and floor(2*4/3)=2;
    // one directed pair has contrast (2-0)^2=4, independently of any CPU twin.
    expect(source).toContain('value >= 10000001.0');
    expect(source).toContain('value >= 10000002.0');
    expect(source).toContain('value >= 10000003.0');
    range[0] = 0;
    range[1] = 1;
    expect(getGPUCommandNodeKernelSources(texture.getCommandNodes(graph))[0].source).toBe(source);
  } finally {
    device.destroy();
  }
});

it('B3 uses stable exact bin boundaries instead of a reciprocal-sensitive quotient', () => {
  const {device, graph, view} = createKernelAuditContext('texture-boundary');
  try {
    const texture = new GPURasterTexture({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      values: view('float32', 2),
      levels: 5,
      range: [0, 25],
      radius: 1,
      offset: [1, 0],
      features: view('float32', 14),
      pairCount: view('uint32', 2)
    });
    const source = getGPUCommandNodeKernelSources(texture.getCommandNodes(graph))[0].source;
    // Hand-worked breaks are 5,10,15,20; equality with 5 enters bin 1 on every backend.
    expect(source).toContain('value >= 5.0');
    expect(source).not.toContain('getScaledQuotient(value -');
  } finally {
    device.destroy();
  }
});
