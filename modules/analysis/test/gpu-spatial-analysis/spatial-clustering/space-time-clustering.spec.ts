// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/gpu-time-words-relative';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  clusterSpaceTimeDBSCAN,
  orderSpaceTimeOPTICS,
  type SpaceTimePoints
} from './space-time-clustering-oracle';
import fixture from '../../oracles/fixtures/space-time-clustering.json';

const NOISE = GPU_SPATIAL_CLUSTERING_NOISE;
const TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

function toGPULabels(labels: readonly number[]): number[] {
  return labels.map(label => (label < 0 ? NOISE : label));
}

type ClusteringRun = {
  labels: number[];
  rootRows: number[];
  coreFlags: number[];
  clusterCount: number;
};

/** Compiles one ST-DBSCAN graph; `run` rewrites times and parameters per frame. */
function createSpaceTimeHarness(
  device: Device,
  scene: {positions: readonly number[]; times: readonly number[]},
  options: {withTimes: boolean; gridSize?: readonly [number, number]}
) {
  const rows = scene.times.length;
  const positionsBuffer = createInputBuffer(device, Float32Array.from(scene.positions));
  const timesBuffer = createInputBuffer(device, Float32Array.from(scene.times));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'space-time-clustering-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: new Float32Array(GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH)
  });
  const outputs = {
    labels: createOutputBuffer(device, rows),
    rootRows: createOutputBuffer(device, rows),
    coreFlags: createOutputBuffer(device, rows),
    clusterCount: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'space-time-clustering-test'});
  graph.add(
    new GPUSpatialClustering({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      ...(options.withTimes
        ? {
            times: importGraphBuffer(graph, 'times', timesBuffer, 'float32', rows),
            temporalContext: TEMPORAL_CONTEXT
          }
        : {}),
      parameters: parameterBuffer.importToGraph(graph),
      gridSize: options.gridSize ?? [24, 24],
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rows),
        rootRows: importGraphBuffer(graph, 'root-rows', outputs.rootRows, 'uint32', rows),
        coreFlags: importGraphBuffer(graph, 'core-flags', outputs.coreFlags, 'uint32', rows),
        clusterCount: importGraphBuffer(graph, 'cluster-count', outputs.clusterCount, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const extent = Math.max(...scene.positions) + 1;
  const bounds = [-1, -1, extent, extent] as const;
  return {
    async run(
      settings: {epsilon: number; minimumPoints: number; timeEpsilon?: number},
      times?: readonly number[]
    ): Promise<ClusteringRun> {
      if (times) {
        timesBuffer.write(Float32Array.from(times));
      }
      parameterBuffer.write(getGPUSpatialClusteringParameterValues({bounds, ...settings}));
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(0x7f7f7f7f));
      }
      submitGraph(device, compiled, undefined);
      return {
        labels: await readUint32(outputs.labels, rows),
        rootRows: await readUint32(outputs.rootRows, rows),
        coreFlags: await readUint32(outputs.coreFlags, rows),
        clusterCount: (await readUint32(outputs.clusterCount, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [positionsBuffer, timesBuffer, ...Object.values(outputs)]) {
        buffer.destroy();
      }
      parameterBuffer.destroy();
    }
  };
}

function expectMatchesOracle(run: ClusteringRun, points: SpaceTimePoints, eps: number[]) {
  const expected = clusterSpaceTimeDBSCAN(points, eps[0], eps[1], eps[2]);
  expect(run.labels).toEqual(toGPULabels(expected.labels));
  expect(run.coreFlags).toEqual(expected.coreFlags);
  expect(run.rootRows).toEqual(toGPULabels(expected.rootRows));
  expect(run.clusterCount).toBe(Math.max(-1, ...expected.labels) + 1);
}

it('ST-DBSCAN matches st-dbscan 0.2.3 labels and the double-precision CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of fixture.dbscan) {
    const harness = createSpaceTimeHarness(device, scene, {
      withTimes: true,
      gridSize: [scene.times.length > 1000 ? 64 : 16, scene.times.length > 1000 ? 64 : 16]
    });
    const run = await harness.run({
      epsilon: scene.eps1,
      timeEpsilon: scene.eps2,
      minimumPoints: scene.minimumPoints
    });
    // The library's labels, produced by sklearn DBSCAN on its time-filtered distance matrix.
    expect(run.labels, scene.name).toEqual(toGPULabels(scene.labels));
    expect(run.clusterCount, scene.name).toBe(scene.clusterCount);
    // The independent oracle also reproduces the cores and roots.
    expectMatchesOracle(run, scene, [scene.eps1, scene.eps2, scene.minimumPoints]);
    harness.destroy();
  }
});

it('ST-DBSCAN separates clusters that overlap in space and honors the closed time window', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two spatially identical groups of four points; group B is 10 time units after group A.
  const positions: number[] = [];
  const times: number[] = [];
  for (const time of [0, 10]) {
    for (const [x, y] of [
      [5, 5],
      [6, 5],
      [5, 6],
      [6, 6]
    ]) {
      positions.push(x, y);
      times.push(time);
    }
  }
  const harness = createSpaceTimeHarness(device, {positions, times}, {withTimes: true});
  const separate = await harness.run({epsilon: 2, timeEpsilon: 9, minimumPoints: 4});
  expect(separate.labels).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  // |dt| == timeEpsilon is related: the groups merge exactly at the boundary.
  const merged = await harness.run({epsilon: 2, timeEpsilon: 10, minimumPoints: 4});
  expect(merged.labels).toEqual(new Array(8).fill(0));
  // A boundary spatial distance: (5,5)-(7,5) would be exactly epsilon; here the diagonal is
  // sqrt(2) > 1.4 so only the axis neighbors relate, which is below four points.
  const tight = await harness.run({epsilon: 1.4, timeEpsilon: 9, minimumPoints: 4});
  expect(tight.labels).toEqual(new Array(8).fill(NOISE));
  // Frame change: new times in the same graph.
  const reshuffled = await harness.run(
    {epsilon: 2, timeEpsilon: 9, minimumPoints: 4},
    [0, 0, 0, 0, 0, 0, 0, 0]
  );
  expect(reshuffled.labels).toEqual(new Array(8).fill(0));
  harness.destroy();
});

it('ST-DBSCAN with a window wider than the time span equals the same graph without times', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = fixture.dbscan[1];
  const plain = createSpaceTimeHarness(device, scene, {withTimes: false});
  const spaceTime = createSpaceTimeHarness(device, scene, {withTimes: true});
  const settings = {epsilon: scene.eps1, minimumPoints: scene.minimumPoints};
  const withoutTimes = await plain.run(settings);
  const wideWindow = await spaceTime.run({...settings, timeEpsilon: 1e6});
  expect(wideWindow).toEqual(withoutTimes);
  // The window is not vacuous: the 6-unit window changes the result on the same points.
  const narrow = await spaceTime.run({...settings, timeEpsilon: scene.eps2});
  expect(narrow.labels).not.toEqual(withoutTimes.labels);
  // Time data is ignored when the contributor has no times: parameters slot 6 does not matter.
  const ignored = await plain.run({...settings, timeEpsilon: 1});
  expect(ignored).toEqual(withoutTimes);
  plain.destroy();
  spaceTime.destroy();
});

it('ST-DBSCAN labels a point with a non-finite time as noise and never uses it as a neighbor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = [5, 5, 6, 5, 5, 6, 6, 6, 5.5, 5.5];
  const times = [0, 0, 0, 0, Number.NaN];
  const harness = createSpaceTimeHarness(device, {positions, times}, {withTimes: true});
  const run = await harness.run({epsilon: 2, timeEpsilon: 5, minimumPoints: 4});
  expect(run.labels).toEqual([0, 0, 0, 0, NOISE]);
  expect(run.coreFlags).toEqual([1, 1, 1, 1, 0]);
  expectMatchesOracle(run, {positions, times}, [2, 5, 4]);
  // With one required point a lone row would be its own core, except that its time is NaN.
  const single = await harness.run({epsilon: 2, timeEpsilon: 5, minimumPoints: 1});
  expect(single.labels).toEqual([0, 0, 0, 0, NOISE]);
  expect(single.coreFlags).toEqual([1, 1, 1, 1, 0]);
  // Five related points would be needed if the NaN row counted as a neighbor.
  const five = await harness.run({epsilon: 2, timeEpsilon: 5, minimumPoints: 5});
  expect(five.labels).toEqual(new Array(5).fill(NOISE));
  harness.destroy();
});

it('ST-DBSCAN reads Int64 millisecond words through GPUTimeWordsRelative', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = fixture.dbscan[1];
  const rows = scene.times.length;
  const origin = 1_700_000_000_000n;
  const wordsBuffer = createInputBuffer(
    device,
    Uint32Array.from(
      scene.times.flatMap(time => {
        // Absolute milliseconds; the scene time is in seconds.
        const absolute = BigInt.asUintN(64, origin + BigInt(time) * 1000n);
        return [Number(absolute & 0xffffffffn), Number(absolute >> 32n)];
      })
    )
  );
  const positionsBuffer = createInputBuffer(device, Float32Array.from(scene.positions));
  const relativeBuffer = createOutputBuffer(device, rows);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'words-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues({
      bounds: [-1, -1, 100, 100],
      epsilon: scene.eps1,
      // The context unit is the millisecond, so the window is in milliseconds.
      timeEpsilon: scene.eps2 * 1000,
      minimumPoints: scene.minimumPoints
    })
  });
  const labelsBuffer = createOutputBuffer(device, rows);
  const graph = new GPUCommandGraph(device, {id: 'space-time-words'});
  const converter = new GPUTimeWordsRelative({
    timestamps: importGraphBuffer(graph, 'words', wordsBuffer, 'uint32x2', rows),
    temporalContext: {
      unit: 'millisecond',
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    },
    origin,
    output: importGraphBuffer(graph, 'relative', relativeBuffer, 'float32', rows)
  });
  graph.add(converter);
  graph.add(
    new GPUSpatialClustering({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      times: converter.props.output,
      temporalContext: converter.outputTemporalContext,
      parameters: parameterBuffer.importToGraph(graph),
      gridSize: [16, 16],
      output: {labels: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', rows)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(labelsBuffer, rows)).toEqual(toGPULabels(scene.labels));
  compiled.destroy();
  for (const buffer of [wordsBuffer, positionsBuffer, relativeBuffer, labelsBuffer]) {
    buffer.destroy();
  }
  parameterBuffer.destroy();
});

type OpticsRun = {
  ordering: number[];
  coreDistances: number[];
  reachability: number[];
  predecessors: number[];
  labels: number[];
  status: number[];
};

async function runSpaceTimeOptics(
  device: Device,
  scene: (typeof fixture.optics)[number],
  withTimes: boolean,
  maximumTimeDistance = scene.maximumTimeDistance
): Promise<OpticsRun> {
  const rows = scene.times.length;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'space-time-optics'});
  const input = <Format extends 'float32x2' | 'float32'>(
    name: string,
    format: Format,
    values: readonly number[]
  ) => {
    const buffer = createInputBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, rows);
  };
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const ordering = output('ordering', 'uint32', rows);
  const coreDistances = output('core', 'float32', rows);
  const reachability = output('reachability', 'float32', rows);
  const predecessors = output('predecessors', 'uint32', rows);
  const labels = output('labels', 'uint32', rows);
  const status = output('status', 'uint32', 3);
  graph.add(
    new GPUOPTICS({
      positions: input('positions', 'float32x2', scene.positions),
      ...(withTimes
        ? {
            times: input('times', 'float32', scene.times),
            temporalContext: TEMPORAL_CONTEXT,
            maximumTimeDistance
          }
        : {}),
      minimumSamples: scene.minimumSamples,
      maximumDistance: scene.maximumDistance,
      extractionDistance: scene.extractionDistance,
      output: {
        ordering: ordering.view,
        coreDistances: coreDistances.view,
        reachability: reachability.view,
        predecessors: predecessors.view,
        labels: labels.view,
        status: status.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    ordering: await readUint32(ordering.buffer, rows),
    coreDistances: Array.from(await readFloat32(coreDistances.buffer, rows)),
    reachability: Array.from(await readFloat32(reachability.buffer, rows)),
    predecessors: await readUint32(predecessors.buffer, rows),
    labels: await readUint32(labels.buffer, rows),
    status: await readUint32(status.buffer, 3)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectDistancesClose(actual: number[], expected: readonly (number | null)[]) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, row) => {
    if (value === null || !Number.isFinite(value)) {
      expect(actual[row]).toBe(Infinity);
    } else {
      expect(Math.abs(actual[row] - value)).toBeLessThan(1e-5 * Math.max(1, value));
    }
  });
}

it('ST-OPTICS matches the sklearn precomputed time-filtered ordering and the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of fixture.optics) {
    const run = await runSpaceTimeOptics(device, scene, true);
    expect(run.status, scene.name).toEqual([scene.times.length, 1, 0]);
    // sklearn OPTICS on the st-clustering matrix.
    expect(run.ordering, scene.name).toEqual(scene.ordering);
    expect(run.predecessors, scene.name).toEqual(
      scene.predecessors.map(value => (value < 0 ? NOISE : value))
    );
    expect(run.labels, scene.name).toEqual(toGPULabels(scene.labels));
    expectDistancesClose(run.coreDistances, scene.coreDistances);
    expectDistancesClose(run.reachability, scene.reachability);
    // The independent CPU implementation agrees with both.
    const oracle = orderSpaceTimeOPTICS(
      scene,
      scene.maximumDistance,
      scene.maximumTimeDistance,
      scene.minimumSamples,
      scene.extractionDistance
    );
    expect(oracle.ordering, scene.name).toEqual(scene.ordering);
    expect(oracle.labels, scene.name).toEqual(scene.labels);
    expect(run.ordering, scene.name).toEqual(oracle.ordering);
    expect(run.labels, scene.name).toEqual(toGPULabels(oracle.labels));
    expectDistancesClose(run.reachability, oracle.reachability);
    expectDistancesClose(run.coreDistances, oracle.coreDistances);
  }
});

it('ST-OPTICS with a window wider than the time span equals OPTICS without times, and a narrow one does not', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = fixture.optics[1];
  const plain = await runSpaceTimeOptics(device, scene, false);
  const wide = await runSpaceTimeOptics(device, scene, true, 1e6);
  expect(wide).toEqual(plain);
  const narrow = await runSpaceTimeOptics(device, scene, true);
  expect(narrow.ordering).not.toEqual(plain.ordering);
});

it('ST-OPTICS treats a row with a non-finite time as invalid, like a non-finite position', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const base = fixture.optics[0];
  const scene = {...base, times: base.times.map((time, row) => (row === 3 ? Number.NaN : time))};
  const run = await runSpaceTimeOptics(device, scene, true);
  const oracle = orderSpaceTimeOPTICS(
    scene,
    scene.maximumDistance,
    scene.maximumTimeDistance,
    scene.minimumSamples,
    scene.extractionDistance
  );
  expect(oracle.ordering).toHaveLength(scene.times.length - 1);
  expect(run.status).toEqual([scene.times.length - 1, 1, 1]);
  expect(run.ordering.slice(0, oracle.ordering.length)).toEqual(oracle.ordering);
  expect(run.labels).toEqual(toGPULabels(oracle.labels));
  expect(run.coreDistances[3]).toBe(Infinity);
  expect(run.reachability[3]).toBe(Infinity);
});
