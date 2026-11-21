// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {makeSpatialFixture, readIntegers} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('ring status counts a closed square before capacity clipping and rewrites all scalars for an open chain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const endpoints = fixture.view(
      'float32',
      new Float32Array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0])
    );
    const inputCount = fixture.view('uint32', new Uint32Array([4]));
    const positions = fixture.view('float32', new Float32Array(8));
    const ringOffsets = fixture.view('uint32', new Uint32Array(2));
    const count = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const overflow = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const requiredCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const openSegmentCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const touchingSegmentCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
    fixture.graph.add(
      new GPUSegmentRingAssembly({
        endpoints: fixture.graph.createDataView(endpoints.view.buffer, {
          format: 'float32x4',
          length: 4
        }),
        count: inputCount.view,
        output: {
          positions: fixture.graph.createDataView(positions.view.buffer, {
            format: 'float32x2',
            length: 4
          }),
          ringOffsets: ringOffsets.view,
          status: {
            count: count.view,
            overflow: overflow.view,
            requiredCount: requiredCount.view,
            openSegmentCount: openSegmentCount.view,
            touchingSegmentCount: touchingSegmentCount.view
          }
        }
      })
    );
    const readStatus = async () => [
      ...(await readIntegers(count.buffer)),
      ...(await readIntegers(overflow.buffer)),
      ...(await readIntegers(requiredCount.buffer)),
      ...(await readIntegers(openSegmentCount.buffer)),
      ...(await readIntegers(touchingSegmentCount.buffer))
    ];
    // Four directed edges make exactly one closed unit square. Publication
    // requires five vertices including closure, exceeding capacity four:
    // zero written, one required, overflow 1, no open or shared endpoints.
    await fixture.run();
    expect(await readStatus()).toEqual([0, 1, 1, 0, 0]);
    // Removing the closing edge leaves a three-edge open path. All three
    // valid segments lie on no ring, and there is no ring to overflow.
    inputCount.buffer.write(new Uint32Array([3]));
    await fixture.run();
    expect(await readStatus()).toEqual([0, 0, 0, 3, 0]);
    inputCount.buffer.write(new Uint32Array([4]));
    await fixture.run();
    expect(await readStatus()).toEqual([0, 1, 1, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
