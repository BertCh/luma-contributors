// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {
  addPatchProximityRecipe,
  addLandscapeMetricsRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

const enabled =
  (import.meta as unknown as {env?: Record<string, string>}).env?.[
    'VITE_URBAN_ECOLOGY_BENCHMARK'
  ] === 'true';
it.skipIf(!enabled)(
  'sparse resistance, patch distance and landscape summary scaling',
  {timeout: 300000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (device as unknown as {handle: {queue: {onSubmittedWorkDone(): Promise<void>}}})
      .handle.queue;
    console.log(
      'URBAN_ECOLOGY_DEVICE',
      JSON.stringify({
        info: device.info,
        maximumStorageBindingBytes: device.limits.maxStorageBufferBindingSize
      })
    );
    for (const workload of [
      ...[32, 128, 256].map(size => ({kind: 'patch-proximity', size})),
      ...[32, 128, 256].map(size => ({kind: 'landscape-metrics', size}))
    ]) {
      const fixture = new RecipeTestFixture(device, `${workload.kind}-${workload.size}`);
      let compiled: CompiledGPUCommandGraph | undefined;
      try {
        const start = performance.now();
        let inputBytes = 0;
        let source;
        if (workload.kind === 'landscape-metrics') {
          const side = workload.size;
          const labels = Uint32Array.from({length: side * side}, (_, cell) =>
            cell % side < side / 2 ? 0 : 1
          );
          inputBytes = labels.byteLength + 8;
          const result = addLandscapeMetricsRecipe(fixture.graph, {
            patches: fixture.input('patches', labels, 'uint32', labels.length),
            patchClasses: fixture.input('classes', new Uint32Array([0, 1]), 'uint32', 2),
            width: side,
            height: side,
            classCount: 2,
            connectivity: 4,
            cellSize: [1, 1]
          });
          source = createTransientView(fixture.graph, 'landscape-certificate', 'float32x4', 1);
          fixture.graph.add(
            createWGSLKernelNode(fixture.graph, {
              id: 'pack-landscape-certificate',
              operation: 'benchmark-certificate',
              invocationCount: 1,
              bindings: [
                {
                  name: 'aggregation',
                  view: result.outputs.landscapeAggregation,
                  type: 'f32',
                  access: 'read'
                },
                {
                  name: 'cohesion',
                  view: result.outputs.landscapeCohesion,
                  type: 'f32',
                  access: 'read'
                },
                {name: 'landscape', view: result.outputs.landscape, type: 'f32', access: 'read'},
                {name: 'output', view: source, type: 'f32', access: 'read_write'}
              ],
              body: 'output[outputOffset]=aggregation[aggregationOffset];output[outputOffset+1u]=cohesion[cohesionOffset];output[outputOffset+2u]=landscape[landscapeOffset+1u];output[outputOffset+3u]=landscape[landscapeOffset+2u];'
            })
          );
        } else {
          const side = workload.size;
          const labels = new Uint32Array(side * side);
          for (let row = 0; row < side; row++)
            for (let column = 0; column < side; column++)
              labels[row * side + column] =
                column < side / 2 - 1 ? 0 : column > side / 2 ? 1 : 0xffffffff;
          inputBytes = labels.byteLength + 8;
          source = addPatchProximityRecipe(fixture.graph, {
            patches: fixture.input('patches', labels, 'uint32', labels.length),
            patchClasses: fixture.input('classes', new Uint32Array([0, 0]), 'uint32', 2),
            width: side,
            height: side,
            cellSize: [1, 1],
            edgeDepth: 1,
            searchRadius: 5
          }).outputs.metrics;
        }
        await queue.onSubmittedWorkDone();
        const preparationMilliseconds = performance.now() - start;
        const output = fixture.output('certificate', 'float32x4', source.length);
        fixture.graph.add(
          createWGSLKernelNode(fixture.graph, {
            id: 'capture-certificate',
            operation: 'benchmark-capture',
            invocationCount: source.length * 4,
            bindings: [
              {name: 'source', view: source, type: 'f32', access: 'read'},
              {name: 'output', view: output.view, type: 'f32', access: 'read_write'}
            ],
            body: 'output[outputOffset+index]=source[sourceOffset+index];'
          })
        );
        const compileStart = performance.now();
        compiled = fixture.graph.compile();
        const compileMilliseconds = performance.now() - compileStart;
        const submit = async () => {
          const start = performance.now();
          const encoder = device.createCommandEncoder();
          compiled!.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          await queue.onSubmittedWorkDone();
          return performance.now() - start;
        };
        const coldExecutionMilliseconds = await submit();
        const residentWallMilliseconds = [];
        for (let repeat = 0; repeat < 3; repeat++) residentWallMilliseconds.push(await submit());
        const readStart = performance.now();
        const certificate = await fixture.readFloat32(output, source.length * 4);
        const readbackMilliseconds = performance.now() - readStart;
        if (workload.kind === 'landscape-metrics') {
          const side = workload.size;
          const cells = (side * side) / 2;
          const squareSide = Math.floor(Math.sqrt(cells));
          const remainder = cells - squareSide * squareSide;
          const maximumJoins =
            2 * cells - 2 * squareSide - (remainder === 0 ? 0 : remainder <= squareSide ? 1 : 2);
          const observedJoins = (side / 2 - 1) * side + ((side - 1) * side) / 2;
          expect(certificate[0]).toBeCloseTo((100 * observedJoins) / maximumJoins, 3);
          expect(certificate[1]).toBeCloseTo(
            (100 * (1 - 1 / Math.sqrt(cells))) / (1 - 1 / side),
            3
          );
          expect(certificate[2]).toBeCloseTo(cells, 3);
          expect(certificate[3]).toBe(side * side);
        } else {
          expect(certificate[2]).toBeCloseTo(2, 5);
        }
        console.log(
          'URBAN_ECOLOGY_BENCHMARK',
          JSON.stringify({
            ...workload,
            preparationMilliseconds,
            compileMilliseconds,
            coldExecutionMilliseconds,
            residentWallMilliseconds,
            readbackMilliseconds,
            inputBytes,
            outputBytes: source.length * 16,
            transientBytes: compiled.stats.physicalTransientResourceBytes,
            logicalTransientBytes: compiled.stats.logicalTransientBytes,
            nodeCount: compiled.stats.nodeOrder.length,
            certificate
          })
        );
      } finally {
        compiled?.destroy();
        fixture.destroy();
      }
    }
  }
);
