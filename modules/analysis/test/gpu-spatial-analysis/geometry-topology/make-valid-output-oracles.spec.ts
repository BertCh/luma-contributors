// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectOracleOutput} from './make-valid-output-oracles-assertions';
import {
  COMPONENTS,
  createOracleScene,
  POISON,
  STATUS_DESTINATIONS,
  type ViewFactory,
  type Scene,
  type StatusDestination
} from './make-valid-output-oracles-fixtures';

async function runOracle(
  scene: Scene,
  uncertainty?: StatusDestination,
  snapped = false,
  segmentCapacity = 12
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  const view: ViewFactory = (format, length, values) => {
    const buffer = createOutputBuffer(device, length * COMPONENTS[format]);
    const data = values ?? new Array<number>(length * COMPONENTS[format]).fill(POISON);
    buffer.write(format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    const result = importGraphBuffer(graph, `oracle-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result;
  };
  const fixture = createOracleScene(view, scene, uncertainty, snapped, segmentCapacity);
  graph.add(fixture.contributor);
  const compiled = graph.compile();
  try {
    // Re-poisoning and resubmitting the same compiled graph catches one-time initialization.
    for (let submission = 0; submission < 2; submission++) {
      if (submission) {
        const poisonOutputs = (value: object) => {
          for (const member of Object.values(value)) {
            if (!member || typeof member !== 'object') continue;
            if (buffers.has(member)) {
              const outputView = member as GraphDataView;
              const data = new Array<number>(
                (outputView.length * outputView.rowByteLength) / 4
              ).fill(POISON);
              buffers
                .get(outputView)!
                .write(
                  outputView.format.startsWith('float')
                    ? Float32Array.from(data)
                    : Uint32Array.from(data)
                );
            } else poisonOutputs(member);
          }
        };
        poisonOutputs(fixture.output);
        if (fixture.lattice) buffers.get(fixture.lattice)!.write(new Float32Array(4).fill(POISON));
      }
      submitGraph(device, compiled, undefined);
      const read = (value: GraphDataView) =>
        value.format.startsWith('float')
          ? readFloat32(buffers.get(value)!)
          : readUint32(buffers.get(value)!);
      if (segmentCapacity === 4) {
        // Eight square edges are required; only the first square's four fit. Directed
        // requirements remain sixteen even though only eight sides are available.
        for (const [status, expected] of [
          [fixture.statuses.noded, [4, 8, 1, 0]],
          [fixture.statuses.directed, [8, 16, 1, 0]]
        ] as const) {
          expect(
            (
              await Promise.all(
                [
                  status.count,
                  status.requiredCount,
                  status.overflow!,
                  status.candidateOverflow
                ].map(read)
              )
            ).flat()
          ).toEqual(expected);
        }
        expect(await read(fixture.statuses.top.count)).toEqual([1]);
        expect(await read(fixture.statuses.top.candidateOverflow)).toEqual([1]);
        expect(await read(fixture.statuses.top.overflow!)).toEqual([0]);
      } else await expectOracleOutput(fixture, read, scene);
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

// Every supplied optional uncertainty destination runs alone; the no-option run has exactly
// the same independently expected required outputs. Snap lattice is tested present and absent.
for (const destination of [undefined, ...STATUS_DESTINATIONS]) {
  it(`GPUMakeValid publishes poisoned square outputs with ${destination ?? 'no'} uncertainty destination`, async () => {
    await runOracle('squares', destination);
  });
  it(`GPUMakeValid publishes four uncertifiable pairs with ${destination ?? 'no'} uncertainty destination`, async () => {
    await runOracle('uncertain', destination);
  });
}
for (const scene of ['bow-tie', 'dangle', 'bridge'] as const) {
  it(`GPUMakeValid matches Shapely and source-row literals for ${scene}`, async () => {
    await runOracle(scene);
  });
}
it('GPUMakeValid publishes the exact snap lattice without changing square outputs', async () => {
  await runOracle('squares', undefined, true);
});
it('GPUMakeValid preserves retraced closing-edge provenance with snap rounding', async () => {
  await runOracle('dangle', undefined, true);
});
it('GPUMakeValid separates feature and polygon offsets for a square with a hole', async () => {
  await runOracle('hole');
});
it('GPUMakeValid reports noded and directed capacity loss with the exact square requirements', async () => {
  await runOracle('squares', undefined, false, 4);
});
