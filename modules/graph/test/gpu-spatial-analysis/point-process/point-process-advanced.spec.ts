// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './point-process-advanced-fixtures.json';

function harness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const data = format.startsWith('uint32') ? Uint32Array.from(values) : Float32Array.from(values);
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        format.endsWith('x2') ? values.length / 2 : values.length
      ),
      buffer
    };
  };
  return {
    graph,
    view,
    destroy() {
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}
for (const fixture of fixtures.xi) {
  it(`xi=${fixture.xi} predecessor=${fixture.predecessorCorrection} matches pinned sklearn hierarchy`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const state = harness(device);
    const count = fixture.positions.length;
    const output = {
      ordering: state.view('ordering', 'uint32', new Array(count).fill(0)),
      coreDistances: state.view('core', 'float32', new Array(count).fill(0)),
      reachability: state.view('reachability', 'float32', new Array(count).fill(0)),
      predecessors: state.view('predecessors', 'uint32', new Array(count).fill(0)),
      labels: state.view('labels', 'uint32', new Array(count).fill(0)),
      status: state.view('status', 'uint32', [0, 0, 0]),
      clusterIntervals: state.view('intervals', 'uint32x2', new Array(2 * count).fill(0)),
      clusterSummary: state.view('summary', 'uint32', [0, 0])
    };
    state.graph.add(
      new GPUOPTICS({
        positions: state.view('positions', 'float32x2', fixture.positions.flat()).view,
        minimumSamples: fixture.minimumSamples,
        maximumDistance: fixture.maximumDistance,
        extractionMethod: 'xi',
        xi: fixture.xi,
        minimumClusterSize: fixture.minimumClusterSize,
        predecessorCorrection: fixture.predecessorCorrection,
        frontierBatchSize: 5,
        output: {
          ordering: output.ordering.view,
          coreDistances: output.coreDistances.view,
          reachability: output.reachability.view,
          predecessors: output.predecessors.view,
          labels: output.labels.view,
          status: output.status.view,
          clusterIntervals: output.clusterIntervals.view,
          clusterSummary: output.clusterSummary.view
        }
      })
    );
    const compiled = state.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(output.ordering.buffer, count)).toEqual(fixture.ordering);
      expect(await readUint32(output.labels.buffer, count)).toEqual(
        fixture.labels.map(value => (value < 0 ? 0xffffffff : value))
      );
      expect(await readUint32(output.clusterSummary.buffer, 2)).toEqual([
        fixture.intervals.length,
        0
      ]);
      expect(
        (await readUint32(output.clusterIntervals.buffer, count * 2)).slice(
          0,
          fixture.intervals.length * 2
        )
      ).toEqual(fixture.intervals.flat());
      expect(await readUint32(output.status.buffer, 3)).toEqual([count, 1, 0]);
    } finally {
      compiled.destroy();
      state.destroy();
    }
  });
}
for (const [index, fixture] of fixtures.rank.entries()) {
  it(`global rank envelope ${index} matches independent scipy midranks including ties`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const state = harness(device);
    const count = fixture.curves[0].length;
    const lower = state.view('lower', 'float32', new Array(count).fill(0));
    const upper = state.view('upper', 'float32', new Array(count).fill(0));
    const statistics = state.view('statistics', 'float32', [0, 0, 0, 0]);
    const interval = state.view('interval', 'float32', [0, 0]);
    const status = state.view('status', 'uint32', [0]);
    state.graph.add(
      new GPUPointPatternEnvelope({
        observed: state.view('observed', 'float32', fixture.curves.at(-1)!).view,
        simulations: state.view('simulations', 'float32', fixture.curves.slice(0, -1).flat()).view,
        reference: state.view('reference', 'float32', new Array(count).fill(0)).view,
        integrationWeights: state.view('weights', 'float32', new Array(count).fill(1)).view,
        simulationCount: fixture.curves.length - 1,
        statistic: 'extreme-rank',
        significanceLevel: fixture.alpha,
        nullModel: 'Supplied exchangeable oracle curves',
        fittedParameters: 'none',
        output: {
          lower: lower.view,
          upper: upper.view,
          statistics: statistics.view,
          status: status.view,
          pValueInterval: interval.view
        }
      })
    );
    const compiled = state.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(lower.buffer, count)).toEqual(fixture.lower);
      expect(await readFloat32(upper.buffer, count)).toEqual(fixture.upper);
      const actual = await readFloat32(interval.buffer, 2);
      fixture.interval.forEach((value, row) => expect(actual[row]).toBeCloseTo(value, 6));
      expect((await readFloat32(statistics.buffer, 4))[0]).toBe(fixture.ranks.at(-1));
      expect(await readUint32(status.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
      state.destroy();
    }
  });
}
