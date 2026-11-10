// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/dataframe-analysis/time-window-filter/time-words';
import {addClockEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  getGPUTrajectoryClockParameterValues,
  getGPUTrajectoryClockWordParameterValues,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from '../../gpu-network/network-optimization/optimization-test-harness';
import {runClosestApproach} from './closest-approach-harness';
import {runZoneEvents} from '../trajectory-zones/zone-events-harness';
import {createRingEdges} from '../trajectory-zones/zone-events-oracle';

/**
 * Track 0 moves 0 to 100 in x over one 100-unit step (the gap) with no fix between. Track 1 waits
 * at (50, 2) with a fix every 10 units, so the straight line across the gap would meet it at t=50.
 */
const GAP_POSITIONS = [0, 0, 100, 0, ...Array.from({length: 11}, () => [50, 2]).flat()];
const GAP_TIMES = [0, 100, ...Array.from({length: 11}, (_, row) => row * 10)];
const GAP_OFFSETS = [0, 2, 13];
const BUCKET_COUNT = 11;

/** One compiled clock-encounter graph run for several per-frame gap limits. */
async function runClockGaps(maximumGaps: readonly number[], wordBase?: bigint) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'clock-maximum-gap'});
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    buffers.push(buffer);
    return buffer;
  };
  const rowCount = GAP_TIMES.length;
  const isWordMode = wordBase !== undefined;
  const clock = new GPUParameterBuffer(device, {
    id: 'clock',
    format: isWordMode ? 'uint32' : 'float32',
    length: GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH
  });
  const maximumGap = new GPUParameterBuffer(device, {
    id: 'maximum-gap',
    format: 'float32',
    length: 1
  });
  const timestampBuffer = input(
    isWordMode
      ? getInt64TimeWords(BigInt64Array.from(GAP_TIMES, time => wordBase + BigInt(time)))
      : Float32Array.from(GAP_TIMES)
  );
  const out = {
    samples: output(2 * 2 * BUCKET_COUNT),
    ids: output(8),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    bucketCounts: output(8)
  };
  addClockEncounters(graph, {
    id: 'gap-clock',
    positions: importGraphBuffer(
      graph,
      'positions',
      input(Float32Array.from(GAP_POSITIONS)),
      'float32x2',
      rowCount
    ),
    timestamps: isWordMode
      ? importGraphBuffer(graph, 'timestamps', timestampBuffer, 'uint32x2', rowCount)
      : importGraphBuffer(graph, 'timestamps', timestampBuffer, 'float32', rowCount),
    trackOffsets: importGraphBuffer(
      graph,
      'offsets',
      input(Uint32Array.from(GAP_OFFSETS)),
      'uint32',
      GAP_OFFSETS.length
    ),
    clock: clock.importToGraph(graph),
    maximumGap: maximumGap.importToGraph(graph),
    bucketCount: BUCKET_COUNT,
    distance: importGraphBuffer(graph, 'distance', input(Float32Array.of(5)), 'float32', 1),
    cellSize: 40,
    bounds: [-10, -10, 200, 200],
    hitCapacity: 256,
    output: {
      pairs: {
        ids: importGraphBuffer(graph, 'o-ids', out.ids, 'uint32', 8),
        status: {
          count: importGraphBuffer(graph, 'o-count', out.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1)
        },
        bucketCounts: importGraphBuffer(graph, 'o-counts', out.bucketCounts, 'uint32', 8)
      },
      status: {
        candidateOverflow: importGraphBuffer(
          graph,
          'o-candidate-overflow',
          out.candidateOverflow,
          'uint32',
          1
        )
      },
      samples: importGraphBuffer(graph, 'o-samples', out.samples, 'float32x2', 2 * BUCKET_COUNT)
    }
  });
  const compiled = graph.compile();
  try {
    clock.write(
      isWordMode
        ? getGPUTrajectoryClockWordParameterValues({start: wordBase, step: 10})
        : getGPUTrajectoryClockParameterValues({start: 0, step: 10})
    );
    const runs: {count: number; bucketCount: number; samples: number[]}[] = [];
    for (const value of maximumGaps) {
      maximumGap.write(Float32Array.of(value));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(out.count, 1);
      runs.push({
        count,
        bucketCount: count > 0 ? (await readUint32(out.bucketCounts, 1))[0] : 0,
        samples: await readFloat32(out.samples, 2 * 2 * BUCKET_COUNT)
      });
    }
    return runs;
  } finally {
    compiled.destroy();
    clock.destroy();
    maximumGap.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const wordBase of [undefined, 1_700_000_000_000n]) {
  it(`clock resample and encounters skip a gap longer than maximumGap (${wordBase === undefined ? 'float32' : 'words'})`, async () => {
    const [noLimit, below, exactly, above] = await runClockGaps([0, 99, 100, 101], wordBase);
    // A limit of 0 means no limit: the straight line across the gap meets track 1 at t=50.
    expect(noLimit.count).toBe(1);
    expect(noLimit.bucketCount).toBe(1);
    expect(noLimit.samples[2 * 5]).toBe(50);
    // A limit below the gap: no encounter; only the fixes at the gap edges remain.
    expect(below.count).toBe(0);
    expect(below.samples[0]).toBe(0);
    expect(below.samples[2 * 10]).toBe(100);
    for (let bucket = 1; bucket < 10; bucket++) {
      expect(below.samples[2 * bucket], `bucket ${bucket}`).toBeNaN();
    }
    // A step exactly equal to the limit is not a gap, and neither is a shorter one.
    for (const run of [exactly, above]) {
      expect(run.count).toBe(1);
      expect(run.bucketCount).toBe(1);
      expect(run.samples[2 * 5]).toBe(50);
    }
  });
}

it('closest approach ignores segments longer than maximumGap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {
    distance: 5,
    cellSize: 20,
    timeCellSize: 10,
    entryCapacity: 1 << 14,
    hitCapacity: 1 << 10,
    pairCapacity: 8
  };
  const tracks = [
    {
      positions: [
        [0, 0],
        [100, 0]
      ] as [number, number][],
      times: [0, 100]
    },
    {
      positions: Array.from({length: 11}, () => [50, 2] as [number, number]),
      times: Array.from({length: 11}, (_, row) => row * 10)
    }
  ];
  for (const wordBase of [undefined, 1_700_000_000_000n]) {
    const unlimited = await runClosestApproach(device, tracks, {...options, wordBase});
    expect(unlimited.count).toBe(1);
    expect(unlimited.closestTimes[0]).toBeCloseTo(50, 3);
    const disabled = await runClosestApproach(device, tracks, {
      ...options,
      wordBase,
      maximumGap: 0
    });
    expect(disabled.count).toBe(1);
    // A step equal to the limit is not a gap.
    const atLimit = await runClosestApproach(device, tracks, {
      ...options,
      wordBase,
      maximumGap: 100
    });
    expect(atLimit.count).toBe(1);
    const skipped = await runClosestApproach(device, tracks, {
      ...options,
      wordBase,
      maximumGap: 99
    });
    expect(skipped.count).toBe(0);
    expect(skipped.entryOverflow).toBe(0);
  }
});

it('zone events report no crossing and no dwell inside a gap, for one maximumGap rule', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges = createRingEdges([[0, [10.5, 10.5, 20.5, 10.5, 20.5, 20.5, 10.5, 20.5]]]);
  const makeTracks = (word: boolean) => {
    const rows = [
      // Would-be crossings inside the 100-unit gap.
      [
        [0, 15, 0],
        [5, 15, 1],
        [25, 15, 101],
        [30, 15, 102]
      ],
      // Inside the zone on both sides of the gap: no crossing.
      [
        [12, 15, 0],
        [14, 15, 1],
        [16, 15, 101],
        [18, 15, 102]
      ],
      // Crossings on a step exactly equal to the limit.
      [
        [0, 15, 0],
        [25, 15, 10],
        [30, 15, 11]
      ]
    ];
    const flat = rows.flat();
    return {
      positions: Float32Array.from(flat.flatMap(([x, y]) => [x, y])),
      trackOffsets: [0, 4, 8, 11],
      times: word
        ? {kind: 'words' as const, values: flat.map(row => 1_700_000_000_000n + BigInt(row[2]))}
        : {kind: 'float32' as const, values: Float32Array.from(flat.map(row => row[2]))}
    };
  };
  const options = {
    zoneCount: 1,
    candidateCapacity: 256,
    trackEventCapacity: 8,
    eventCapacity: 32
  };
  for (const word of [false, true]) {
    const tracks = makeTracks(word);
    const unlimited = await runZoneEvents(device, tracks, edges, options);
    expect(unlimited.dwellTimes[0]).toBeCloseTo(50, 3);
    expect(unlimited.dwellTimes[1]).toBeCloseTo(102, 3);
    expect(unlimited.dwellTimes[2]).toBeCloseTo(4, 3);
    const disabled = await runZoneEvents(device, tracks, edges, {...options, maximumGap: 0});
    expect(disabled.dwellTimes).toEqual(unlimited.dwellTimes);
    expect(disabled.times).toEqual(unlimited.times);
    const limited = await runZoneEvents(device, tracks, edges, {...options, maximumGap: 10});
    // Track 0: both crossings are reported at the fix after the gap, and the gap adds no dwell.
    expect(limited.dwellTimes[0]).toBeCloseTo(0, 3);
    expect(limited.visitCounts[0]).toBe(1);
    const track0 = limited.tracks
      .map((track, event) => (track === 0 ? event : -1))
      .filter(event => event >= 0);
    expect(track0.map(event => limited.types[event])).toEqual([0, 1]);
    for (const event of track0) {
      expect(limited.times[event]).toBeCloseTo(101, 3);
      expect(limited.rows[event]).toBe(2);
      expect(limited.positions[event * 2]).toBeCloseTo(25, 3);
    }
    // Track 1: the observed spans inside the zone, not the 100-unit gap.
    expect(limited.dwellTimes[1]).toBeCloseTo(2, 3);
    // Track 2: its 10-unit step equals the limit, so it is interpolated as without a limit.
    expect(limited.dwellTimes[2]).toBeCloseTo(4, 3);
    const track2 = limited.tracks
      .map((track, event) => (track === 2 ? event : -1))
      .filter(event => event >= 0);
    expect(limited.times[track2[0]]).toBeCloseTo(4.2, 3);
    expect(limited.times[track2[1]]).toBeCloseTo(8.2, 3);
    // A limit above every gap leaves results as without a limit.
    const above = await runZoneEvents(device, tracks, edges, {...options, maximumGap: 200});
    expect(above.dwellTimes).toEqual(unlimited.dwellTimes);
  }
});

for (const wordBase of [undefined, 1_700_000_000_000n]) {
  it(`time-spaced resample applies the same gap rule (${wordBase === undefined ? 'float32' : 'words'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const sampleCount = 12;
    const times = [0, 100, 110];
    const run = async (maximumGap: number | undefined) => {
      const harness = createOptimizationHarness(device);
      const output = {
        samples: harness.output('float32x2', sampleCount),
        sampleTimes: harness.output('float32', sampleCount)
      };
      const result = await harness.run(
        new GPUTrajectoryResample({
          positions: harness.input('float32x2', Float32Array.of(0, 0, 100, 0, 110, 0)),
          timestamps:
            wordBase === undefined
              ? harness.input('float32', Float32Array.from(times))
              : harness.input(
                  'uint32x2',
                  getInt64TimeWords(BigInt64Array.from(times, time => wordBase + BigInt(time)))
                ),
          trackOffsets: harness.input('uint32', Uint32Array.of(0, 3)),
          sampleCount,
          spacing: 'time',
          maximumGap:
            maximumGap === undefined
              ? undefined
              : harness.input('float32', Float32Array.of(maximumGap)),
          output
        })
      );
      return {samples: result(output.samples), sampleTimes: result(output.sampleTimes)};
    };
    const unlimited = await run(undefined);
    expect(unlimited.samples[2 * 5]).toBeCloseTo(50, 3);
    for (const limit of [0, 100, 101]) {
      expect((await run(limit)).samples, `limit ${limit}`).toEqual(unlimited.samples);
    }
    const limited = await run(99);
    // Samples strictly inside the gap are absent; the fixes at its ends and after it remain.
    expect(limited.samples[0]).toBe(0);
    for (let sample = 1; sample < 10; sample++) {
      expect(limited.samples[2 * sample], `sample ${sample}`).toBeNaN();
    }
    expect(limited.samples[2 * 10]).toBeCloseTo(100, 3);
    expect(limited.samples[2 * 11]).toBeCloseTo(110, 3);
    expect(limited.sampleTimes).toEqual(unlimited.sampleTimes);
  });
}
