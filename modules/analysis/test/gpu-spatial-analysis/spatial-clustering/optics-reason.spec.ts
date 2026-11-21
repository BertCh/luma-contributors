// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createOPTICSStatusFields,
  getOPTICSStatusViews,
  readOPTICSStatus
} from './optics-status-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView, GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUOPTICSOperation} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const orderingMethod of ['sequential', 'parallel'] as const) {
  for (const maximumIterations of [2, 3]) {
    for (const operationForm of [false, true]) {
      it(`reports overflow then recovers on the same ${orderingMethod} grid with budget ${maximumIterations}, operation=${operationForm}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = new WeightsRig(device);
        // Three coincident points require six directed candidates, exceeding capacity two.
        const positions = rig.input(new Float32Array(6), 'float32x2', 3);
        const ordering = rig.output('uint32', 3);
        const labels = rig.output('uint32', 3);
        const status = createOPTICSStatusFields(() => rig.output('uint32', 1));
        const reason = rig.output('uint32', 1);
        const limit = rig.output('uint32', 1);
        const settings = {
          minimumSamples: 2,
          maximumDistance: 1,
          extractionDistance: 1,
          neighborIndex: 'grid' as const,
          neighborCapacity: 2,
          maximumIterations,
          orderingMethod
        };
        const output = {
          ordering: ordering.view,
          labels: labels.view,
          coreDistances: rig.output('float32', 3).view,
          reachability: rig.output('float32', 3).view,
          predecessors: rig.output('uint32', 3).view,
          ports: {
            status: {
              ...getOPTICSStatusViews(status),
              reason: reason.view,
              iterationLimitReached: limit.view
            }
          }
        };
        const operation = new GPUOPTICSOperation({
          ...settings,
          rowCount: 3,
          positions: 'positions',
          output: {
            ordering: 'ordering',
            coreDistances: 'coreDistances',
            reachability: 'reachability',
            predecessors: 'predecessors',
            labels: 'labels',
            ports: {
              status: {
                ...createOPTICSStatusFields(name => name),
                reason: 'reason',
                iterationLimitReached: 'iterationLimitReached'
              }
            }
          }
        });
        const views: Record<string, GraphDataView> = {
          positions,
          ordering: output.ordering,
          coreDistances: output.coreDistances,
          reachability: output.reachability,
          predecessors: output.predecessors,
          labels: output.labels,
          ...getOPTICSStatusViews(status),
          reason: reason.view,
          iterationLimitReached: limit.view
        };
        rig.graph.add(
          operationForm
            ? (operation.createContributor(
                <Format extends GPUGraphVectorFormat>(port: string) =>
                  views[port] as GraphDataView<Format>
              ) as GPUOPTICS)
            : new GPUOPTICS({...settings, positions, output})
        );
        const compiled = await rig.graph.compileAsync();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(reason.buffer, 1)).toEqual([1]);
          expect(await readUint32(limit.buffer, 1)).toEqual([0]);
          expect(await readOPTICSStatus(status)).toEqual([0, 0, 0]);
          expect(await readUint32(labels.buffer, 3)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
          // One pair plus one isolated point fits exactly two candidates. Hand order: 0,1,2.
          rig.bufferOf(positions).write(Float32Array.of(0, 0, 0.5, 0, 10, 0));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(reason.buffer, 1)).toEqual([maximumIterations === 3 ? 0 : 2]);
          expect(await readUint32(limit.buffer, 1)).toEqual([maximumIterations === 3 ? 0 : 1]);
          expect(await readOPTICSStatus(status)).toEqual([
            maximumIterations,
            Number(maximumIterations === 3),
            0
          ]);
          expect(await readUint32(ordering.buffer, 3)).toEqual(
            maximumIterations === 3 ? [0, 1, 2] : [0, 1, 0xffffffff]
          );
          expect(await readUint32(labels.buffer, 3)).toEqual(
            maximumIterations === 3 ? [0, 0, 0xffffffff] : [0xffffffff, 0xffffffff, 0xffffffff]
          );
        } finally {
          compiled.destroy();
          rig.destroy();
          device.destroy();
        }
      });
    }
  }
}
