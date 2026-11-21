// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUColumnProfile} from '../../src/dataframe-analysis/column-profile/gpu-column-profile';
import {GPU_COLUMN_PROFILE_STATISTIC_COUNT} from '../../src/dataframe-analysis/column-profile/column-profile-parameters';
import {GPUVariableStandardization} from '../../src/dataframe-analysis/variable-standardization/gpu-variable-standardization';
import {GPUInequality} from '../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {GPUValueCube} from '../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {GPUTemporalReduction} from '../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction';
import {GPUVariogramModelFit} from '../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {GPUKeyLookup} from '../../src/dataframe-analysis/key-join/gpu-key-join';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `field-${serial++}`, format, length);
  return {device, graph, view};
}

it('retains integer count scratch and accepts packed columns above 2^24', () => {
  const {device, graph, view} = createFixture();
  try {
    const rowCount = 2 ** 24 + 1;
    const values = view('float32', rowCount);
    const profile = new GPUColumnProfile({
      columns: [{values}],
      histogramBinCount: 2,
      hyperLogLogPrecision: 4,
      output: {
        statistics: view('float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT),
        counts: view('uint32', 2)
      }
    });
    const sources = getGPUCommandNodeKernelSources(profile.getCommandNodes(graph));
    for (const {source, constants, id} of sources)
      expect(auditWGSLModule(source, constants), id).toEqual([]);
    const merge = sources.find(({id}) => id.endsWith('-moments-merge'))!.source;
    expect(merge).toContain('count: u32');
    expect(merge).toContain('nullCount: u32');
    expect(merge).toContain('finiteCount: u32');
    expect(merge).toContain('var<storage, read> partials: array<u32>');
    expect(() =>
      new GPUVariableStandardization({
        id: 'wide-standardization',
        values,
        output: view('float32', rowCount),
        method: 'z-score'
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('column profiling reuses shader and pipeline families as the column count grows', () => {
  const build = (columnCount: number) => {
    const {device, graph, view} = createFixture();
    try {
      const nodes = new GPUColumnProfile({
        columns: Array.from({length: columnCount}, () => ({values: view('float32', 4097)})),
        histogramBinCount: 8,
        hyperLogLogPrecision: 4,
        output: {statistics: view('float32', columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT)}
      }).getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const {source, constants, id} of sources)
        expect(auditWGSLModule(source, constants), id).toEqual([]);
      return {
        count: sources.length,
        distinct: getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size
      };
    } finally {
      device.destroy();
    }
  };
  const small = build(1);
  const large = build(8);
  expect(large.count).toBeGreaterThan(small.count);
  expect(large.distinct).toBe(small.distinct);
});

it.each([
  'inequality',
  'key-join',
  'profile'
] as const)('%s rejects aligned output overlaps at construction', kind => {
  const {device, graph, view} = createFixture();
  try {
    const storage = graph.createTransientBuffer({
      id: 'output-storage',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const first = graph.createDataView(storage, {format: 'float32', length: 1});
    // Logical rows are disjoint, but the second binding includes the first row's alignment prefix.
    const second = graph.createDataView(storage, {format: 'float32', length: 1, byteOffset: 4});
    const construct = () => {
      if (kind === 'inequality')
        return new GPUInequality({
          values: view('float32', 2),
          zoneIds: view('uint32', 2),
          zoneCount: 1,
          parameters: view('float32', 4),
          output: {gini: first, theilT: second}
        });
      if (kind === 'key-join')
        return new GPUKeyLookup({
          leftKeys: view('uint32', 1),
          rightKeys: view('uint32', 2),
          output: {},
          gather: [
            {column: view('float32', 2), output: first},
            {column: view('float32', 2), output: second}
          ]
        });
      return new GPUColumnProfile({
        columns: [{values: view('float32', 2)}],
        histogramBinCount: 1,
        output: {
          statistics: graph.createDataView(storage, {
            format: 'float32',
            length: GPU_COLUMN_PROFILE_STATISTIC_COUNT
          }),
          counts: graph.createDataView(storage, {format: 'uint32', length: 2, byteOffset: 128})
        }
      });
    };
    expect(construct).toThrow(/aligned binding.*overlap/);
  } finally {
    device.destroy();
  }
});

it.each([
  'inequality',
  'key-join',
  'profile',
  'standardization'
] as const)('%s preflights the actual aligned input range without graph mutation', kind => {
  const {device, graph, view} = createFixture();
  try {
    const storage = graph.createTransientBuffer({
      id: 'input-storage',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    // Four logical bytes require a 256-byte binding because of the alignment prefix.
    const values = graph.createDataView(storage, {format: 'float32', length: 1, byteOffset: 252});
    const producer =
      kind === 'inequality'
        ? new GPUInequality({
            values,
            zoneIds: view('uint32', 1),
            zoneCount: 1,
            parameters: view('float32', 4),
            output: {gini: view('float32', 1)}
          })
        : kind === 'key-join'
          ? new GPUKeyLookup({
              leftKeys: view('uint32', 1),
              rightKeys: view('uint32', 1),
              output: {},
              gather: [{column: values, output: view('float32', 1)}]
            })
          : kind === 'profile'
            ? new GPUColumnProfile({
                columns: [{values}],
                histogramBinCount: 1,
                hyperLogLogPrecision: 4,
                output: {statistics: view('float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT)}
              })
            : new GPUVariableStandardization({values, output: view('float32', 1), method: 'mad'});
    Object.assign(device.limits, {maxStorageBufferBindingSize: 128});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declareView = vi.spyOn(graph, 'createDataView');
    const declareNode = vi.spyOn(graph, 'add');
    expect(() => producer.getCommandNodes(graph)).toThrow(/256 bytes.*limit 128/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declareView).not.toHaveBeenCalled();
    expect(declareNode).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('preflights derived key-join sum scratch before declaring graph resources', () => {
  const {device, graph, view} = createFixture();
  try {
    const producer = new GPUKeyLookup({
      leftKeys: view('uint32', 1),
      rightKeys: view('uint32', 40),
      output: {},
      aggregates: [{operation: 'sum', column: view('float32', 40), output: view('float32', 1)}]
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 256});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => producer.getCommandNodes(graph)).toThrow(/sumWords scratch.*320 bytes.*limit 256/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('preflights profile register scratch before declaring graph resources', () => {
  const {device, graph, view} = createFixture();
  try {
    const producer = new GPUColumnProfile({
      columns: [{values: view('float32', 2)}],
      histogramBinCount: 1,
      output: {statistics: view('float32', GPU_COLUMN_PROFILE_STATISTIC_COUNT)}
    });
    Object.assign(device.limits, {maxStorageBufferBindingSize: 256});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => producer.getCommandNodes(graph)).toThrow(
      /registers scratch.*16384 bytes.*limit 256/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('preflights scratch allocation against maxBufferSize before mutating the graph', () => {
  const {device, graph, view} = createFixture();
  try {
    const producer = new GPUKeyLookup({
      leftKeys: view('uint32', 1),
      rightKeys: view('uint32', 40),
      output: {},
      aggregates: [{operation: 'mean', column: view('float32', 40), output: view('float32', 1)}]
    });
    Object.assign(device.limits, {maxBufferSize: 256});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => producer.getCommandNodes(graph)).toThrow(/sumWords scratch byteLength.*256/);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('rejects key-join and inequality metadata above the published sort boundary at construction', () => {
  const {device, view} = createFixture();
  try {
    // No backing allocations: declaration-only metadata with sufficiently large logical device limits.
    Object.assign(device.limits, {maxBufferSize: 2 ** 36, maxStorageBufferBindingSize: 2 ** 36});
    for (const rowCount of [0x80000000, 0x80000001]) {
      const values = view('float32', rowCount);
      const keys = view('uint32', rowCount);
      const constructInequality = () =>
        new GPUInequality({
          values,
          zoneIds: keys,
          zoneCount: 1,
          parameters: view('float32', 4),
          output: {gini: view('float32', 1)}
        });
      const constructJoin = () =>
        new GPUKeyLookup({leftKeys: view('uint32', 1), rightKeys: keys, output: {}});
      if (rowCount === 0x80000000) {
        expect(constructInequality).not.toThrow();
        expect(constructJoin).not.toThrow();
      } else {
        expect(constructInequality).toThrow(/values length.*2147483648/);
        expect(constructJoin).toThrow(/rightKeys length.*2147483648/);
      }
    }
  } finally {
    device.destroy();
  }
});

it.each([
  0,
  1,
  0xffffffc0,
  0xffffffc1,
  0xffffffff,
  0x100000000,
  NaN,
  Infinity
])('validates row and bin metadata %s before shader interpolation', rows => {
  const {device, view} = createFixture();
  const metadata = <Format extends 'float32' | 'uint32' | 'float32x4'>(
    format: Format,
    length: number
  ) =>
    new GraphDataView(view('uint32', 1).buffer, {
      format,
      length,
      byteOffset: 0,
      byteStride: format === 'float32x4' ? 16 : 4,
      rowByteLength: format === 'float32x4' ? 16 : 4
    });
  try {
    const constructors = [
      {
        minimum: 1,
        maximum: 0xffffffff,
        build: () =>
          new GPUValueCube({
            cells: metadata('uint32', rows),
            periods: metadata('uint32', rows),
            values: metadata('float32', rows),
            cellCount: 1,
            periodCount: 1,
            output: {counts: view('uint32', 1)}
          })
      },
      {
        minimum: 1,
        maximum: 0xffffffff,
        build: () =>
          new GPUTemporalReduction({
            cellIds: metadata('uint32', rows),
            timestamps: metadata('float32', rows),
            values: metadata('float32', rows),
            parameters: view('float32', 2),
            cellCount: 1,
            bucketCount: 1,
            output: {
              counts: view('uint32', 1),
              first: view('float32', 1),
              last: view('float32', 1),
              minimums: view('float32', 1),
              maximums: view('float32', 1),
              occupiedSlots: {
                ids: view('uint32', 1),
                status: {count: view('uint32', 1), overflow: view('uint32', 1)}
              }
            }
          })
      },
      {
        minimum: 0,
        maximum: 0xffffffc0,
        build: () =>
          new GPUVariogramModelFit({
            distances: metadata('float32', rows),
            semivariances: metadata('float32', rows),
            pairCounts: metadata('uint32', rows),
            models: ['exponential'],
            output: {
              parameters: metadata('float32x4', 1),
              diagnostics: metadata('float32x4', 1),
              status: view('uint32', 1)
            }
          })
      }
    ];
    for (const {minimum, maximum, build} of constructors) {
      if (Number.isInteger(rows) && rows >= minimum && rows <= maximum) expect(build).not.toThrow();
      else
        expect(build).toThrow(
          /(?:value-cube cells|temporal-reduction values|variogram-model-fit distances) length must be an integer in \[[01], (?:4294967295|4294967232)\]/
        );
    }
  } finally {
    device.destroy();
  }
});
