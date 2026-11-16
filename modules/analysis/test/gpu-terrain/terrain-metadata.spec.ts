// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {terrainBinding} from '../../src/gpu-terrain/terrain-processing/terrain-processing-utils';
import {TERRAIN_VIEWSHED_BATCH_START_WGSL} from '../../src/gpu-terrain/terrain-analysis/terrain-viewshed-batches';
import {createTerrainSweepSectorTable} from '../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {GPUTerrainHorizon} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {createTerrainReviewFixture} from './terrain-processing/terrain-review-fixture';
import {requireTestDevice} from '../utils/require-test-device';

it('addresses parallel viewshed batches across the full uint32 range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const cases = [
    [0, 23, 37],
    [1, 23, 37],
    [123456, 100000, 333333],
    [3000000000, 2, 1000000000],
    [4294967294, 4294967294, 4294967295]
  ];
  const input = fixture.input('uint32', cases.flat()).view;
  const output = fixture.output('uint32', cases.length * 2);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: 'parallel-batches',
      operation: 'TerrainViewshedBatches',
      invocationCount: cases.length,
      bindings: [
        terrainBinding('input', input),
        terrainBinding('output', output.view, 'read_write')
      ],
      declarations: TERRAIN_VIEWSHED_BATCH_START_WGSL,
      body: `let firstPair = getViewshedBatchStart(input[inputOffset + 3u * index], input[inputOffset + 3u * index + 1u], input[inputOffset + 3u * index + 2u]);
  output[outputOffset + 2u * index] = firstPair.x;
  output[outputOffset + 2u * index + 1u] = firstPair.y;`
    })
  );
  const compiled = fixture.run();
  try {
    expect(Array.from(await output.read())).toEqual(
      cases.flatMap(([batch, size, pixels]) => {
        const pair = BigInt(batch) * BigInt(size);
        return [Number(pair / BigInt(pixels)), Number(pair % BigInt(pixels))];
      })
    );
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const kind of ['ranges', 'sweep'] as const) {
  it(`publishes independently expected parallel horizon ${kind}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    recordWGSLKernelNodes(fixture.graph);
    const grid = {
      width: 4,
      height: 4,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const directions = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
      [1, 1]
    ] as const;
    const initializer =
      kind === 'sweep'
        ? createTerrainSweepSectorTable(fixture.graph, {
            id: 'records',
            width: 4,
            height: 4,
            directions
          }).writeNode
        : new GPUTerrainHorizon({
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32',
              storage: {kind: 'buffer', values: fixture.input('float32', new Float32Array(16)).view}
            },
            parameters: fixture.input('float32', new Float32Array(8)).view,
            directionCount: 16,
            maximumRadius: 2,
            maximumStepsPerDispatch: 1,
            output: {skyViewFactor: fixture.output('float32', 16).view}
          })
            .getCommandNodes(fixture.graph)
            .find(node => node.id.endsWith('-sector-ranges'))!;
    const table = getWGSLKernelNodeRecord(fixture.graph, initializer.id)!.bindings[0].view;
    const wordsPerRecord = kind === 'sweep' ? 7 : 2;
    const recordCount = kind === 'sweep' ? 5 : 16;
    const output = fixture.output('uint32', wordsPerRecord * recordCount);
    fixture.graph.add(initializer);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'copy-records',
        operation: 'TerrainMetadata',
        invocationCount: wordsPerRecord * recordCount,
        bindings: [
          terrainBinding('input', table),
          terrainBinding('output', output.view, 'read_write')
        ],
        body: `output[outputOffset + index] = input[inputOffset + (index / ${wordsPerRecord}u) * 64u + index % ${wordsPerRecord}u];`
      })
    );
    const compiled = fixture.run();
    try {
      const floatBits = (value: number) => new Uint32Array(Float32Array.of(value).buffer)[0];
      const expected =
        kind === 'ranges'
          ? Array.from({length: 16}, (_, range) => [range, range + 1]).flat()
          : [
              [1, 0, 1, floatBits(1), floatBits(1), 0, 0],
              [0, 0, 1, floatBits(1), 0, floatBits(1), 1],
              [1, 0, 0xffffffff, floatBits(1), floatBits(-1), 0, 2],
              [0, 0, 0xffffffff, floatBits(1), 0, floatBits(-1), 3],
              [1, 65536, 1, floatBits(Math.SQRT2), floatBits(1), floatBits(1), 4]
            ].flat();
      expect(Array.from(await output.read())).toEqual(expected);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
