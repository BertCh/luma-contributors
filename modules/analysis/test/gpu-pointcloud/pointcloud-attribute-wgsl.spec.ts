// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  readPointCloudAttribute,
  writePointCloudAttribute
} from '@luma-contributors/analysis/pointcloud-analysis';
import {getPointCloudKernelNodes} from '../../src/pointcloud-analysis/operations/pointcloud-kernel-nodes';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeSpatialFixture, readFloats} from '../spatial-test-utils';

it('attribute expressions read the next XYZ row and write a biased vector at index two (DPC-107)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const values = fixture.view(
      'float32',
      new Float32Array([2, 3, 5, 11, 13, 17, 23, 29, 31, 41, 43, 47])
    );
    const destination = fixture.view('float32', new Float32Array(9).fill(-100));
    const nodes = getPointCloudKernelNodes(fixture.graph, {
      id: 'attribute-compound-expressions',
      length: 3,
      inputs: {values: values.view},
      outputs: {destination: destination.view},
      body: `if (index != 2u) { return; }
let value = ${readPointCloudAttribute('Values', 'index + 1u', 3)};
let bias = vec3f(2.0, 3.0, 5.0);
${writePointCloudAttribute('Destination', 'index', 'value + bias', 3)}`
    });
    fixture.graph.add({getCommandNodes: () => nodes});
    await fixture.run();
    // Row 3 is (41, 43, 47); adding (2, 3, 5) gives (43, 46, 52).
    expect(await readFloats(destination.buffer)).toEqual([
      -100, -100, -100, -100, -100, -100, 43, 46, 52
    ]);
  } finally {
    fixture.destroy();
  }
});
