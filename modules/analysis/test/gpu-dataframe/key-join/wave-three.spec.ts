// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {ValueCubeRig} from '../value-cube/value-cube-rig';

it('UDF-14 gathers the first right match and compacts each matched left row once', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const compactMatchedLeftRows of [undefined, true]) {
    const rig = new ValueCubeRig(device);
    try {
      const leftKeys = rig.input(Uint32Array.of(2, 9, 1, 2), 'uint32');
      const rightKeys = rig.input(Uint32Array.of(2, 2, 1), 'uint32');
      const rightValues = rig.input(Float32Array.of(10, 20, 30), 'float32');
      const gathered = rig.output('float32', 4);
      const sums = rig.output('float32', 4);
      const rightRows = rig.output('uint32', 4);
      const matchCounts = rig.output('uint32', 4);
      const rows = compactMatchedLeftRows
        ? {
            ids: rig.output('uint32', 4),
            status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
          }
        : undefined;
      rig.run(
        new GPUKeyLookup({
          leftKeys,
          rightKeys,
          compactMatchedLeftRows,
          gather: [{column: rightValues, output: gathered}],
          aggregates: [{operation: 'sum', column: rightValues, output: sums}],
          output: {rightRows, matchCounts, rows}
        })
      );
      // Key 2 gathers 10 at right row 0 and sums 10 + 20. Key 1 uses 30 at row 2.
      // Key 9 is absent. The matched left indices are 0, 2 and 3.
      expect(await rig.readUint(rightRows)).toEqual([0, 0xffffffff, 2, 0]);
      expect(await rig.readUint(matchCounts)).toEqual([2, 0, 1, 2]);
      expect(await rig.readFloat(gathered)).toEqual([10, NaN, 30, 10]);
      expect(await rig.readFloat(sums)).toEqual([30, NaN, 30, 30]);
      if (rows) {
        expect(await rig.readUint(rows.ids)).toEqual([0, 2, 3, 0xffffffff]);
        expect(await rig.readUint(rows.status.count)).toEqual([3]);
        expect(await rig.readUint(rows.status.overflow)).toEqual([0]);
      }
    } finally {
      rig.destroy();
    }
  }
});

it('UDF-06 unrelated right keys and groups retain singleton results with independent scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const values of [[1], [1, 2 ** 100]]) {
    const rig = new ValueCubeRig(device);
    try {
      const column = rig.input(Float32Array.from(values), 'float32');
      const keys = rig.input(
        Uint32Array.from(values, (_, index) => index + 1),
        'uint32'
      );
      const sums = rig.output('float32', 1);
      const means = rig.output('float32', 1);
      const matchCounts = rig.output('uint32', 1);
      const groupSums = rig.output('float32', values.length);
      const groupMeans = rig.output('float32', values.length);
      rig.run(
        new GPUKeyLookup({
          scaleMode: 'per-group',
          leftKeys: rig.input(Uint32Array.of(1), 'uint32'),
          rightKeys: keys,
          aggregates: [
            {operation: 'sum', column, output: sums},
            {operation: 'mean', column, output: means}
          ],
          output: {matchCounts}
        }),
        new GPUGroupStatistics({
          scaleMode: 'per-group',
          keys,
          columns: [
            {
              values: column,
              statistics: ['sum', 'mean'],
              output: {sumValues: groupSums, means: groupMeans}
            }
          ],
          output: {
            keys: rig.output('uint32', values.length),
            counts: rig.output('uint32', values.length),
            status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
          }
        })
      );
      // Every occupied key has exactly one value; unrelated keys cannot change its result.
      expect(await rig.readUint(matchCounts)).toEqual([1]);
      expect(await rig.readFloat(sums)).toEqual([1]);
      expect(await rig.readFloat(means)).toEqual([1]);
      expect(await rig.readFloat(groupSums)).toEqual(values);
      expect(await rig.readFloat(groupMeans)).toEqual(values);
    } finally {
      rig.destroy();
    }
  }
});

it('W3-01 subnormal means round once at final spacing in join and group', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  try {
    const keys = rig.input(Uint32Array.of(1, 1, 1), 'uint32');
    const column = rig.input(
      Float32Array.of(2 ** -126, -(2 ** -126), 12582914 * 2 ** -149),
      'float32'
    );
    const joined = rig.output('float32', 1);
    const grouped = rig.output('float32', 1);
    rig.run(
      new GPUKeyLookup({
        leftKeys: rig.input(Uint32Array.of(1), 'uint32'),
        rightKeys: keys,
        aggregates: [{operation: 'mean', column, output: joined}],
        output: {}
      }),
      new GPUGroupStatistics({
        keys,
        columns: [{values: column, statistics: ['mean'], output: {means: grouped}}],
        output: {
          keys: rig.output('uint32', 1),
          counts: rig.output('uint32', 1),
          status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
        }
      })
    );
    // 12582914 / 3 = 4194304 + 2/3 subnormal units, nearest word = 0x00400001.
    expect(Array.from(await rig.readWords(joined))).toEqual([0x00400001]);
    expect(Array.from(await rig.readWords(grouped))).toEqual([0x00400001]);
  } finally {
    rig.destroy();
  }
});

it('CMP-01 and CMP-03 separate uncertified misses from sparse zero count fills', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const overflow of [0, 1]) {
    const rig = new ValueCubeRig(device);
    try {
      const keys = rig.input(Uint32Array.of(20), 'uint32');
      const complete = rig.output('uint32', 1);
      const candidateOverflow = rig.output('uint32', 1);
      const matched = rig.output('uint32', 2);
      const counts = rig.output('uint32', 2);
      rig.run(
        new GPUKeyLookup({
          leftKeys: rig.input(Uint32Array.of(10, 20), 'uint32'),
          rightKeys: keys,
          rightTable: {keys, status: {overflow: rig.input(Uint32Array.of(overflow), 'uint32')}},
          gather: [
            {column: rig.input(Uint32Array.of(3), 'uint32'), output: counts, unmatchedValue: 0}
          ],
          output: {matched, status: {complete, candidateOverflow}}
        })
      );
      expect(await rig.readUint(counts)).toEqual([0, 3]);
      expect(await rig.readUint(matched)).toEqual([0, 1]);
      expect(await rig.readUint(complete)).toEqual([1 - overflow]);
      expect(await rig.readUint(candidateOverflow)).toEqual([overflow]);
    } finally {
      rig.destroy();
    }
  }
});

it('FLT-15 both production count finishes publish exact word 0x01000001', async () => {
  const {createWGSLKernelNode} = await import(
    '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['count', 'count-distinct'] as const) {
    const rig = new ValueCubeRig(device);
    const output = rig.output('uint32', 1);
    const producer = new GPUKeyLookup({
      leftKeys: rig.input(Uint32Array.of(1), 'uint32'),
      rightKeys: rig.input(Uint32Array.of(1, 1), 'uint32'),
      aggregates: [{operation, column: rig.input(Float32Array.of(1, 2), 'float32'), output}],
      output: {}
    });
    const finish = producer.getCommandNodes(rig.graph).find(node => node.id.endsWith('-finish'))!;
    try {
      for (const [resourceIndex, data] of [
        [0, Uint32Array.of(0)],
        [1, Uint32Array.of(0x01000001, 0)]
      ] as const) {
        const resource = finish.resources![resourceIndex];
        if (!('buffer' in resource)) throw new Error('count publication buffer required');
        const source = rig.input(data, 'uint32');
        rig.graph.add(
          createWGSLKernelNode(rig.graph, {
            id: `inject-count-${resourceIndex}`,
            operation: 'count-regression',
            variant: 'inject',
            invocationCount: data.length,
            bindings: [
              {name: 'source', view: source, type: 'u32', access: 'read'},
              {
                name: 'destination',
                view: resource.buffer as typeof source,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
          })
        );
      }
      rig.graph.add(finish);
      rig.run();
      // 2^24+1 is exactly representable in uint32; f32 would round to 2^24.
      expect(await rig.readUint(output)).toEqual([0x01000001]);
    } finally {
      rig.destroy();
    }
  }
});

it('CMP-01 grouped capacity loss differs from a genuine missing key', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [1, 2]) {
    const rig = new ValueCubeRig(device);
    try {
      const keys = rig.output('uint32', capacity);
      const overflow = rig.output('uint32', 1);
      const complete = rig.output('uint32', 1);
      const matched = rig.output('uint32', 2);
      rig.run(
        new GPUGroupStatistics({
          keys: rig.input(Uint32Array.of(10, 20), 'uint32'),
          columns: [],
          output: {
            keys,
            counts: rig.output('uint32', capacity),
            status: {count: rig.output('uint32', 1), overflow}
          }
        }),
        new GPUKeyLookup({
          leftKeys: rig.input(Uint32Array.of(20, 30), 'uint32'),
          rightKeys: keys,
          rightTable: {keys, status: {overflow}},
          output: {matched, status: {complete}}
        })
      );
      // 20 was observed upstream; 30 is a genuine miss. Capacity 1 retains only key 10.
      expect(await rig.readUint(matched)).toEqual(capacity === 1 ? [0, 0] : [1, 0]);
      expect(await rig.readUint(complete)).toEqual(capacity === 1 ? [0] : [1]);
    } finally {
      rig.destroy();
    }
  }
});

it('B1 B2 compact and table statuses forward inherited loss independently of final overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [inheritedLoss, capacity] of [
    [1, 2],
    [0, 1]
  ] as const) {
    const rig = new ValueCubeRig(device);
    try {
      const keys = rig.input(Uint32Array.of(1), 'uint32');
      const complete = rig.output('uint32', 1);
      const tableLoss = rig.output('uint32', 1);
      const compactLoss = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      rig.run(
        new GPUKeyLookup({
          compactMatchedLeftRows: true,
          leftKeys: rig.input(Uint32Array.of(1, 1), 'uint32'),
          rightKeys: keys,
          rightTable: {
            keys,
            status: {
              overflow: rig.input(Uint32Array.of(0), 'uint32'),
              candidateOverflow: rig.input(Uint32Array.of(inheritedLoss), 'uint32')
            }
          },
          output: {
            status: {complete, candidateOverflow: tableLoss},
            rows: {
              ids: rig.output('uint32', capacity),
              status: {count: rig.output('uint32', 1), overflow, candidateOverflow: compactLoss}
            }
          }
        })
      );
      expect(await rig.readUint(complete)).toEqual([0]);
      expect(await rig.readUint(tableLoss)).toEqual([inheritedLoss]);
      expect(await rig.readUint(compactLoss)).toEqual([inheritedLoss]);
      expect(await rig.readUint(overflow)).toEqual([capacity === 1 ? 1 : 0]);
    } finally {
      rig.destroy();
    }
  }
});

it.each([
  undefined,
  2
])('B3 local headroom survives an unrelated fourth row (group tile=%s)', async scratchTileSize => {
  const {GPUValueCube} = await import('../../../src/dataframe-analysis/value-cube/gpu-value-cube');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const independent of [false, true])
    for (const append of [false, true]) {
      const rig = new ValueCubeRig(device);
      try {
        const values = [2 ** 92, -(2 ** 92), 1, ...(append ? [0] : [])];
        const keys = rig.input(
          Uint32Array.from(values, (_, index) => (index === 3 ? 2 : 1)),
          'uint32'
        );
        const column = rig.input(Float32Array.from(values), 'float32');
        const joinSum = rig.output('float32', 1);
        const joinMean = rig.output('float32', 1);
        const matchCounts = rig.output('uint32', 1);
        const groupSum = rig.output('float32', 2);
        const groupMean = rig.output('float32', 2);
        const cubeSum = rig.output('float32', 2);
        const cubeMean = rig.output('float32', 2);
        rig.run(
          new GPUKeyLookup({
            scaleMode: independent ? 'per-group' : undefined,
            leftKeys: rig.input(Uint32Array.of(1), 'uint32'),
            rightKeys: keys,
            aggregates: [
              {operation: 'sum', column, output: joinSum},
              {operation: 'mean', column, output: joinMean}
            ],
            output: {matchCounts}
          }),
          new GPUGroupStatistics({
            scaleMode: independent ? 'per-group' : undefined,
            scratchTileSize,
            keys,
            columns: [
              {
                values: column,
                statistics: ['sum', 'mean'],
                output: {sumValues: groupSum, means: groupMean}
              }
            ],
            output: {
              keys: rig.output('uint32', 2),
              counts: rig.output('uint32', 2),
              status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
            }
          }),
          new GPUValueCube({
            scaleMode: independent ? 'per-bin' : undefined,
            cellCount: 1,
            periodCount: 2,
            cells: rig.input(new Uint32Array(values.length), 'uint32'),
            periods: rig.input(
              Uint32Array.from(values, (_, index) => (index === 3 ? 1 : 0)),
              'uint32'
            ),
            values: column,
            output: {sums: cubeSum, means: cubeMean}
          })
        );
        // Cancellation leaves exactly 1 in a 3-term group; local h=2 gives unit 1.
        // Shared h=3 after the fourth row gives unit 2: ties-to-even quantizes 1 to 0.
        const expectedSum = independent || !append ? 1 : 0;
        const expectedMean = expectedSum ? 0.3333333432674408 : 0;
        expect(await rig.readUint(matchCounts)).toEqual([3]);
        expect(await rig.readFloat(joinSum)).toEqual([expectedSum]);
        expect(await rig.readFloat(joinMean)).toEqual([expectedMean]);
        for (const output of [groupSum, cubeSum])
          expect((await rig.readFloat(output))[0]).toBe(expectedSum);
        for (const output of [groupMean, cubeMean])
          expect((await rig.readFloat(output))[0]).toBe(expectedMean);
      } finally {
        rig.destroy();
      }
    }
});

it('B4 inherited masks exclude rows in group and cube publication', async () => {
  const {GPUValueCube} = await import('../../../src/dataframe-analysis/value-cube/gpu-value-cube');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  try {
    const mask = rig.input(Uint32Array.of(1, 0), 'uint32');
    const values = rig.input(Float32Array.of(2, 99), 'float32');
    const groupCounts = rig.output('uint32', 1);
    const groupMeans = rig.output('float32', 1);
    const cubeCounts = rig.output('uint32', 1);
    const cubeMeans = rig.output('float32', 1);
    const groupProps = Object.assign(
      Object.create({
        get mask() {
          return mask;
        }
      }),
      {
        keys: rig.input(Uint32Array.of(0, 0), 'uint32'),
        columns: [{values, statistics: ['mean'], output: {means: groupMeans}}],
        output: {
          keys: rig.output('uint32', 1),
          counts: groupCounts,
          status: {count: rig.output('uint32', 1), overflow: rig.output('uint32', 1)}
        }
      }
    );
    const cubeProps = Object.assign(
      Object.create({
        get mask() {
          return mask;
        }
      }),
      {
        cellCount: 1,
        periodCount: 1,
        cells: rig.input(Uint32Array.of(0, 0), 'uint32'),
        periods: rig.input(Uint32Array.of(0, 0), 'uint32'),
        values,
        output: {counts: cubeCounts, means: cubeMeans}
      }
    );
    rig.run(new GPUGroupStatistics(groupProps), new GPUValueCube(cubeProps));
    expect(await rig.readUint(groupCounts)).toEqual([1]);
    expect(await rig.readUint(cubeCounts)).toEqual([1]);
    expect(await rig.readFloat(groupMeans)).toEqual([2]);
    expect(await rig.readFloat(cubeMeans)).toEqual([2]);
  } finally {
    rig.destroy();
  }
});
