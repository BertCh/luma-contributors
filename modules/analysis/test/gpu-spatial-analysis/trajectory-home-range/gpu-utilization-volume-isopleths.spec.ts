// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUUtilizationVolumeIsopleths,
  getGPUUtilizationVolumeIsoplethsParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths';
import {
  createUtilizationPrefixNode,
  createUtilizationOffsetNode,
  getUtilizationPrefixLayout
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths-kernels';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readUint32, getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';
import {getUtilizationReservedIdentifiers} from './utilization-isopleths-wgsl-check';
import {
  createNonuniformUtilizationDensity,
  NONUNIFORM_MASS_RELATIVE_TOLERANCE,
  UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT,
  validateUtilizationHeadlessCellCount
} from './utilization-isopleths-test-utils';

async function compareLegacyPrefixes(length: number, byteOffset: number) {
  validateUtilizationHeadlessCellCount(length);
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, `isopleth-legacy-prefix-${length}-${byteOffset}`);
  // A broad exponent range exposes float32 addition association changes.
  const values = Float32Array.from({length}, (_, index) =>
    Math.fround(
      (1 + ((Math.floor(index / 2) * 7919) % 127) / 127) * 2 ** ((Math.floor(index / 2) % 37) - 18)
    )
  );
  const input = createShiftedInput(harness, 'values', values, byteOffset);
  const layout = getUtilizationPrefixLayout(length);
  const actual = createShiftedOutput(harness, 'actual-prefix', 'float32', length, byteOffset);
  const expected = createShiftedOutput(harness, 'legacy-prefix', 'float32', length, byteOffset);
  const actualTotal = createShiftedOutput(harness, 'actual-total', 'float32', 1, byteOffset);
  const expectedTotal = createShiftedOutput(harness, 'legacy-total', 'float32', 1, byteOffset);
  const upperPrefix = createTransientView(
    harness.graph,
    'upper-prefix',
    'float32',
    layout.upperLength
  );
  let levelInput = input;
  for (const [level, levelLength] of layout.lengths.entries()) {
    const prefix =
      level === 0
        ? actual.view
        : harness.graph.createDataView(upperPrefix.buffer, {
            format: 'float32',
            length: levelLength,
            byteOffset: layout.upperOffsets[level - 1] * 4
          });
    const totals =
      level === layout.lengths.length - 1
        ? actualTotal.view
        : createTransientView(
            harness.graph,
            `actual-totals-${level}`,
            'float32',
            Math.ceil(levelLength / 256)
          );
    harness.graph.add(
      createUtilizationPrefixNode(harness.graph, {
        id: `actual-level-${level}`,
        input: levelInput,
        prefix,
        totals
      })
    );
    levelInput = totals;
  }
  harness.graph.add(
    createUtilizationOffsetNode(harness.graph, {
      id: 'actual-offsets',
      localPrefix: actual.view,
      upperPrefix,
      upperOffsets: layout.upperOffsets
    })
  );
  // Frozen legacy route: three bounded serial stages, followed by the old single expression.
  const legacyPrefixes: GraphDataView<'float32'>[] = [expected.view];
  levelInput = input;
  for (const [level, levelLength] of layout.lengths.entries()) {
    const prefix =
      level === 0
        ? expected.view
        : createTransientView(harness.graph, `legacy-prefix-${level}`, 'float32', levelLength);
    if (level > 0) legacyPrefixes.push(prefix);
    const totals =
      level === 2
        ? expectedTotal.view
        : createTransientView(
            harness.graph,
            `legacy-totals-${level}`,
            'float32',
            Math.ceil(levelLength / 256)
          );
    harness.graph.add(
      createWGSLKernelNode(harness.graph, {
        id: `legacy-level-${level}`,
        operation: 'LegacyUtilizationPrefix',
        invocationCount: Math.ceil(levelLength / 256),
        bindings: [
          {name: 'values', view: levelInput, type: 'f32', access: 'read'},
          {name: 'prefix', view: prefix, type: 'f32', access: 'read_write'},
          {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
        ],
        body: `var sum = 0.0;
        let end = min((index + 1u) * 256u, ${levelLength}u);
        for (var row = index * 256u; row < end; row = row + 1u) {
          prefix[prefixOffset + row] = sum;
          sum = sum + values[valuesOffset + row];
        }
        totals[totalsOffset + index] = sum;`
      })
    );
    levelInput = totals;
  }
  harness.graph.add(
    createWGSLKernelNode(harness.graph, {
      id: 'legacy-offsets',
      operation: 'LegacyUtilizationPrefix',
      invocationCount: length,
      bindings: [
        {name: 'localPrefix', view: legacyPrefixes[0], type: 'f32', access: 'read_write'},
        {name: 'blockPrefix', view: legacyPrefixes[1], type: 'f32', access: 'read'},
        {name: 'groupPrefix', view: legacyPrefixes[2], type: 'f32', access: 'read'}
      ],
      body: `let block = index / 256u;
      localPrefix[localPrefixOffset + index] = localPrefix[localPrefixOffset + index] +
        blockPrefix[blockPrefixOffset + block] + groupPrefix[groupPrefixOffset + block / 256u];`
    })
  );
  // Compare on the GPU so the 2^24-cell cases do not build enormous JavaScript arrays.
  const mismatches = harness.output('mismatches', 'uint32', 2);
  harness.graph.add(
    createWGSLKernelNode(harness.graph, {
      id: 'compare-prefix-bits',
      operation: 'CompareUtilizationPrefixBits',
      invocationCount: length,
      bindings: [
        {name: 'actualBits', view: actual.view, type: 'u32', access: 'read'},
        {name: 'expectedBits', view: expected.view, type: 'u32', access: 'read'},
        {name: 'actualTotalBits', view: actualTotal.view, type: 'u32', access: 'read'},
        {name: 'expectedTotalBits', view: expectedTotal.view, type: 'u32', access: 'read'},
        {name: 'mismatchCounts', view: mismatches.view, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `if (actualBits[actualBitsOffset + index] != expectedBits[expectedBitsOffset + index]) {
      atomicAdd(&mismatchCounts[mismatchCountsOffset], 1u);
    }
    if (index == 0u && actualTotalBits[actualTotalBitsOffset] != expectedTotalBits[expectedTotalBitsOffset]) {
      atomicAdd(&mismatchCounts[mismatchCountsOffset + 1u], 1u);
    }`
    })
  );
  validateGeneratedIdentifiers(harness);
  await harness.run(async () => {
    expect(await readUint32(mismatches.buffer)).toEqual([0, 0]);
  });
}

for (const byteOffset of [0, 4, 252, 256]) {
  for (const length of [255, 256, 257, 65535, 65536, 65537, 70001, 2 ** 24 - 1, 2 ** 24]) {
    it(`bit-compares the frozen prefix tree at ${length} cells, offset ${byteOffset}, with mixed exponents and ties`, async () => {
      await compareLegacyPrefixes(length, byteOffset);
    }, 180000);
  }
}

it.each([
  0, 4, 252, 256
])('matches an independent stable binary64 oracle at byte offset %i', async byteOffset => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, 'isopleth-oracle');
  const values = Float32Array.from({length: 1027}, (_, index) => (index % 19) / 16);
  values[3] = NaN;
  values[4] = -1;
  values[5] = Infinity;
  const density = createShiftedInput(harness, 'density', values, byteOffset);
  const parameters = createShiftedInput(
    harness,
    'parameters',
    getGPUUtilizationVolumeIsoplethsParameterValues({level: 0.75}),
    byteOffset
  );
  const volume = createShiftedOutput(harness, 'volume', 'float32', values.length, byteOffset);
  const inside = createShiftedOutput(harness, 'inside', 'uint32', values.length, byteOffset);
  const totalMass = createShiftedOutput(harness, 'total', 'float32', 1, byteOffset);
  harness.graph.add(
    new GPUUtilizationVolumeIsopleths({
      density,
      parameters,
      output: {volume: volume.view, inside: inside.view, totalMass: totalMass.view}
    })
  );
  validateGeneratedIdentifiers(harness);
  await harness.run(async () => {
    const actualVolume = await readFloat32Rows(volume.buffer, byteOffset, values.length);
    const actualInside = await readUint32Rows(inside.buffer, byteOffset, values.length);
    const order = Array.from(values, (value, cell) => ({
      cell,
      value: value > 0 && value < 3.4e38 ? value : 0
    })).sort((first, second) => second.value - first.value || first.cell - second.cell);
    const total = order.reduce((sum, row) => sum + row.value, 0);
    expect((await readFloat32Rows(totalMass.buffer, byteOffset, 1))[0]).toBe(total);
    let preceding = 0;
    for (const {cell, value} of order) {
      expect(actualInside[cell]).toBe(value > 0 && preceding / total < 0.75 ? 1 : 0);
      preceding += value;
      expect(actualVolume[cell]).toBeCloseTo(value > 0 ? preceding / total : 1, 6);
    }
  });
});

// Every caller field uses its own buffer, allowing any valid storage-aligned prefix offset.
function createShiftedInput(
  harness: GraphHarness,
  id: string,
  values: Float32Array,
  byteOffset: number
) {
  const paddedValues = byteOffset === 0 ? values : new Float32Array(values.length + byteOffset / 4);
  if (byteOffset !== 0) paddedValues.set(values, byteOffset / 4);
  const entire = harness.input(id, paddedValues, 'float32', paddedValues.length);
  return harness.graph.createDataView(entire.buffer, {
    format: 'float32',
    length: values.length,
    byteOffset
  });
}

function createShiftedOutput<Format extends 'float32' | 'uint32'>(
  harness: GraphHarness,
  id: string,
  format: Format,
  length: number,
  byteOffset: number
) {
  const output = harness.output(id, format, length + byteOffset / 4);
  return {
    buffer: output.buffer,
    view: harness.graph.createDataView(output.view.buffer, {format, length, byteOffset})
  };
}

async function readFloat32Rows(buffer: Buffer, byteOffset: number, length: number) {
  const bytes = await buffer.readAsync(byteOffset, length * 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, length);
}

async function readUint32Rows(buffer: Buffer, byteOffset: number, length: number) {
  const bytes = await buffer.readAsync(byteOffset, length * 4);
  return new Uint32Array(bytes.buffer, bytes.byteOffset, length);
}

function validateGeneratedIdentifiers(harness: GraphHarness) {
  const nodes = (harness.graph as unknown as {nodes: unknown[]}).nodes;
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    expect(getUtilizationReservedIdentifiers(kernel.source), kernel.id).toEqual([]);
  }
}

async function checkNonuniformSurface(
  device: Device,
  length: number,
  byteOffset: number,
  recordResults = false
) {
  validateUtilizationHeadlessCellCount(length);
  const harness = new GraphHarness(device, `isopleth-large-${length}-${byteOffset}`);
  if (recordResults) {
    console.info(
      'isopleth fixed scalability selection',
      JSON.stringify({
        cellCount: length,
        byteOffset,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        maxBufferSize: device.limits.maxBufferSize
      })
    );
  }
  const {density, node} = createNonuniformUtilizationDensity(harness.graph, length, byteOffset);
  harness.graph.add(node);
  const parameters = createShiftedInput(
    harness,
    'parameters',
    getGPUUtilizationVolumeIsoplethsParameterValues({level: 0.75}),
    byteOffset
  );
  const volume = createShiftedOutput(harness, 'volume', 'float32', length, byteOffset);
  const inside = createShiftedOutput(harness, 'inside', 'uint32', length, byteOffset);
  const totalMass = createShiftedOutput(harness, 'total', 'float32', 1, byteOffset);
  harness.graph.add(
    new GPUUtilizationVolumeIsopleths({
      density,
      parameters,
      output: {volume: volume.view, inside: inside.view, totalMass: totalMass.view}
    })
  );
  validateGeneratedIdentifiers(harness);
  await harness.run(async () => {
    // Closed form, independent of the GPU prefix tree: all even cells (density 2) precede all
    // odd cells (density 1), and stable ties preserve each parity's original row order.
    const highCount = Math.ceil(length / 2);
    const total = length + highCount;
    const actualTotal = (await readFloat32Rows(totalMass.buffer, byteOffset, 1))[0];
    // The tree rounds intermediate totals. For 2^24 + 1 cells the root adds 25165824 + 1,
    // giving 25165824, while one rounding of the exact mass gives 25165826. Keep the oracle
    // independent (closed-form mass) and check the float32 error instead of promising one rounding.
    const massDiagnostic = JSON.stringify({
      cellCount: length,
      actualTotal,
      exactTotal: total,
      relativeError: Math.abs(actualTotal / total - 1)
    });
    expect(Number.isFinite(actualTotal), massDiagnostic).toBe(true);
    expect(actualTotal, massDiagnostic).toBeGreaterThan(0);
    expect(Math.abs(actualTotal / total - 1), massDiagnostic).toBeLessThan(
      NONUNIFORM_MASS_RELATIVE_TOLERANCE
    );
    // This even, workgroup-aligned fixture has exactly representable block and root totals.
    if (length === UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT) expect(actualTotal).toBe(25166208);
    const ranks = [
      ...new Set([
        0,
        1,
        254,
        255,
        256,
        257,
        65534,
        65535,
        65536,
        65537,
        highCount - 1,
        highCount,
        highCount + 1,
        2 ** 24 - 2,
        2 ** 24 - 1,
        2 ** 24,
        2 ** 24 + 1,
        2 ** 24 + 254,
        2 ** 24 + 255,
        2 ** 24 + 256,
        length - 2,
        length - 1
      ])
    ]
      .filter(rank => rank >= 0 && rank < length)
      .sort((first, second) => first - second);
    const boundaries: {rank: number; cell: number; volume: number; inside: number}[] = [];
    let previousVolume = 0;
    for (const rank of ranks) {
      const high = rank < highCount;
      const cell = high ? 2 * rank : 2 * (rank - highCount) + 1;
      const preceding = high ? 2 * rank : highCount + rank;
      const value = high ? 2 : 1;
      const actualVolume = (await readFloat32Rows(volume.buffer, byteOffset + cell * 4, 1))[0];
      const actualInside = (await readUint32Rows(inside.buffer, byteOffset + cell * 4, 1))[0];
      expect(
        Math.abs(actualVolume - (preceding + value) / total),
        `rank ${rank}, cell ${cell}`
      ).toBeLessThan(3e-7);
      expect(actualVolume).toBeGreaterThanOrEqual(previousVolume);
      if (Math.abs(preceding / total - 0.75) > 5e-7) {
        expect(actualInside).toBe(preceding / total < 0.75 ? 1 : 0);
      }
      previousVolume = actualVolume;
      boundaries.push({rank, cell, volume: actualVolume, inside: actualInside});
    }
    if (recordResults) {
      console.info(
        'isopleth fixed scalability results',
        JSON.stringify({
          cellCount: length,
          byteOffset,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxBufferSize: device.limits.maxBufferSize,
          prefixLevels: getUtilizationPrefixLayout(length).lengths.length,
          totalMass: actualTotal,
          boundaries
        })
      );
    }
  });
}

for (const byteOffset of [0, 4, 252, 256]) {
  for (const additionalCells of [1, 255, 256, 257]) {
    const length = 2 ** 24 + additionalCells;
    it(`checks nonuniform root contributions and stable ties at ${length} cells, offset ${byteOffset}`, async context => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      if (
        length * 4 + (byteOffset % 256) > device.limits.maxStorageBufferBindingSize ||
        length * 4 + byteOffset > device.limits.maxBufferSize
      ) {
        context.skip('requested boundary exceeds the device binding or buffer limit');
      }
      await checkNonuniformSurface(device, length, byteOffset);
    }, 180000);
  }
}

it('records mass and boundaries at the fixed old-limit-plus-one-workgroup size', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await checkNonuniformSurface(device, UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT, 0, true);
}, 180000);

it.each([
  {transition: 65537, tail: 1, exactMass: 25165826, treeMass: 25165824},
  {transition: 65665, tail: 257, exactMass: 25166210, treeMass: 25166208}
])('retains fixed-order rounding at the new root: $treeMass from exact mass $exactMass', async ({
  transition,
  tail,
  exactMass,
  treeMass
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = new GraphHarness(device, `isopleth-root-rounding-${tail}`);
  // Compact version of the affected upper-tree walk: 128 high-density groups sum to 2^24,
  // then the mixed group, 127 low-density groups, and the final partial group's total.
  const values = Float32Array.from([2 ** 24, transition, ...Array<number>(127).fill(65536), tail]);
  expect(Array.from(values).reduce((sum, value) => sum + value, 0)).toBe(exactMass);
  const input = harness.input('root-terms', values, 'float32', values.length);
  const prefix = harness.output('root-prefix', 'float32', values.length);
  const totalMass = harness.output('root-total', 'float32', 1);
  harness.graph.add(
    createUtilizationPrefixNode(harness.graph, {
      id: 'root-rounding',
      input,
      prefix: prefix.view,
      totals: totalMass.view
    })
  );
  validateGeneratedIdentifiers(harness);
  await harness.run(async () => {
    expect((await readFloat32Rows(totalMass.buffer, 0, 1))[0]).toBe(treeMass);
    expect((await readFloat32Rows(prefix.buffer, (values.length - 1) * 4, 1))[0]).toBe(
      tail === 1 ? 25165824 : 25165952
    );
  });
});
