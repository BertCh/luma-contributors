// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {describe, expect, it, vi} from 'vitest';
import {MOB07_HOME_RANGE_RECURSION_AUDIT_CASES} from '../../analysis-core/kernel-audit/cases-mob07-home-range-recursion';
import {
  GPUUtilizationVolumeIsopleths,
  GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS
} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths';
import {getUtilizationPrefixLayout} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths-kernels';
import {
  createKernelAuditContext,
  type KernelAuditContext
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

import {getUtilizationReservedIdentifiers} from './utilization-isopleths-wgsl-check';
import {
  createNonuniformUtilizationDensity,
  NONUNIFORM_MASS_RELATIVE_TOLERANCE,
  UTILIZATION_HEADLESS_MAXIMUM_CELL_COUNT,
  UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT,
  validateUtilizationHeadlessCellCount
} from './utilization-isopleths-test-utils';

function createProps(context: KernelAuditContext, length: number) {
  return {
    density: context.view('float32', length),
    parameters: context.view('float32', 4),
    output: {
      volume: context.view('float32', length),
      inside: context.view('uint32', length),
      totalMass: context.view('float32', 1)
    }
  };
}

function getDeclarationCounts(context: KernelAuditContext) {
  const graph = context.graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
  return [graph.buffers.size, graph.nodes.length];
}

describe('recursive utilization prefixes', () => {
  it('retains three levels through 2^24 cells and adds a bounded fourth level beyond it', () => {
    expect(getUtilizationPrefixLayout(2).lengths).toEqual([2, 1, 1]);
    expect(getUtilizationPrefixLayout(2 ** 24).lengths).toEqual([2 ** 24, 65536, 256]);
    expect(getUtilizationPrefixLayout(2 ** 24 + 1).lengths).toEqual([2 ** 24 + 1, 65537, 257, 2]);
    expect(
      getUtilizationPrefixLayout(GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS).lengths
    ).toEqual([2 ** 31, 2 ** 23, 32768, 128]);
  });

  it('keeps shader and pipeline counts constant as prefix levels grow and bounds all nodes', () => {
    const context = createKernelAuditContext('isopleth-recursion');
    try {
      Object.defineProperty(context.device, 'limits', {
        value: {
          ...context.device.limits,
          maxBufferSize: 2 ** 34,
          maxStorageBufferBindingSize: 2 ** 34
        }
      });
      let expectedKeys: Set<string> | undefined;
      for (const length of [70000, 2 ** 24, 2 ** 24 + 1, 2 ** 25, 2 ** 31]) {
        const contributor = new GPUUtilizationVolumeIsopleths({
          id: `surface-${length}`,
          ...createProps(context, length)
        });
        const nodes = contributor.getCommandNodes(context.graph);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        const treeNodes = nodes.filter(node =>
          new RegExp(`^surface-${length}-(level-\\d+|offsets)$`).test(node.id)
        );
        expect(treeNodes).toHaveLength(getUtilizationPrefixLayout(length).lengths.length + 1);
        const keys = getGPUCommandNodePipelineKeys(treeNodes, {includeShaderLayout: true});
        expect(keys.size).toBe(2);
        if (expectedKeys) expect(keys).toEqual(expectedKeys);
        expectedKeys = keys;
        for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
          expect(getUtilizationReservedIdentifiers(kernel.source), kernel.id).toEqual([]);
          const executableSource = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
          expect(executableSource).not.toMatch(/\b(?:NaN|Infinity|nan|inf)\b/);
          for (const literal of executableSource.matchAll(/\b(0x[0-9a-f]+|\d+)([ui])\b/gi)) {
            expect(BigInt(literal[1]), `${kernel.id}: ${literal[0]}`).toBeLessThanOrEqual(
              literal[2].toLowerCase() === 'u' ? 0xffffffffn : 0x7fffffffn
            );
          }
          for (const literal of executableSource.matchAll(
            /\b(\d+\.\d*(?:e[+-]?\d+)?|\d+e[+-]?\d+)f?\b/gi
          )) {
            expect(
              Number.isFinite(Math.fround(Number(literal[1]))),
              `${kernel.id}: ${literal[0]}`
            ).toBe(true);
          }
          for (const value of Object.values(kernel.constants))
            expect(Number.isFinite(value)).toBe(true);
        }
      }
    } finally {
      context.device.destroy();
    }
  });

  it('rejects every overlapping pair of writable output fields at construction', () => {
    const context = createKernelAuditContext('isopleth-overlap');
    try {
      for (const fields of [
        ['volume', 'inside'],
        ['volume', 'totalMass'],
        ['inside', 'totalMass']
      ] as const) {
        const props = createProps(context, 4);
        const shared = context.buffer(32);
        const first = context.graph.createDataView(shared, {
          format: fields[0] === 'inside' ? 'uint32' : 'float32',
          length: 4
        });
        for (const byteOffset of [12, 16]) {
          const second = context.graph.createDataView(shared, {
            format: fields[1] === 'inside' ? 'uint32' : 'float32',
            length: fields[1] === 'totalMass' ? 1 : 4,
            byteOffset
          });
          const construct = () =>
            new GPUUtilizationVolumeIsopleths({
              ...props,
              output: {...props.output, [fields[0]]: first, [fields[1]]: second}
            });
          if (byteOffset === 12 || fields[1] === 'inside') {
            expect(construct).toThrow(/utilization-volume-isopleths.*overlap/);
          } else {
            // totalMass is published separately; only logical overlap is forbidden.
            expect(construct).not.toThrow();
          }
        }
      }
      const props = createProps(context, 4);
      const shared = context.buffer(512);
      expect(() =>
        context.graph.add(
          new GPUUtilizationVolumeIsopleths({
            ...props,
            output: {
              ...props.output,
              volume: context.graph.createDataView(shared, {format: 'float32', length: 4}),
              totalMass: context.graph.createDataView(shared, {
                format: 'float32',
                length: 1,
                byteOffset: 256
              })
            }
          })
        )
      ).not.toThrow();
    } finally {
      context.device.destroy();
    }
  });

  it('checks actual aligned input and output ranges before declaring any scratch or node', () => {
    for (const field of ['density', 'parameters', 'volume', 'inside', 'totalMass'] as const) {
      const context = createKernelAuditContext(`isopleth-range-${field}`);
      try {
        const props = createProps(context, 4);
        const shared = context.buffer(512);
        const shifted = context.graph.createDataView(shared, {
          format: field === 'inside' ? 'uint32' : 'float32',
          length: field === 'totalMass' ? 2 : 4,
          byteOffset: 252
        });
        const contributor = new GPUUtilizationVolumeIsopleths(
          field === 'density' || field === 'parameters'
            ? {...props, [field]: shifted}
            : {...props, output: {...props.output, [field]: shifted}}
        );
        Object.defineProperty(context.device, 'limits', {
          value: {...context.device.limits, maxStorageBufferBindingSize: 256}
        });
        const before = getDeclarationCounts(context);
        expect(() => contributor.getCommandNodes(context.graph)).toThrow(
          new RegExp(`${field}(?: chunk 0)? needs (260|268) bytes.*limit 256`)
        );
        expect(getDeclarationCounts(context)).toEqual(before);
      } finally {
        context.device.destroy();
      }
    }
  });

  it('preflights padded prefix scratch and rejects above the published sort maximum', () => {
    const context = createKernelAuditContext('isopleth-scratch');
    try {
      const contributor = new GPUUtilizationVolumeIsopleths(createProps(context, 2));
      Object.defineProperty(context.device, 'limits', {
        value: {...context.device.limits, maxStorageBufferBindingSize: 256}
      });
      const before = getDeclarationCounts(context);
      expect(() => contributor.getCommandNodes(context.graph)).toThrow(
        /upper prefixes needs 260 bytes.*limit 256/
      );
      expect(getDeclarationCounts(context)).toEqual(before);
      Object.defineProperty(context.device, 'limits', {
        value: {...context.device.limits, maxBufferSize: 2 ** 34}
      });
      expect(() => new GPUUtilizationVolumeIsopleths(createProps(context, 2 ** 31 + 1))).toThrow(
        /cellCount.*2147483648/
      );
    } finally {
      context.device.destroy();
    }
  });
});

it('rejects physical-buffer aliases imported through different graph handles', () => {
  const context = createKernelAuditContext('isopleth-physical-alias');
  const buffer = context.device.createBuffer({byteLength: 32, usage: Buffer.STORAGE});
  try {
    const first = context.graph.importBuffer(
      {id: 'first-output', byteLength: 32, usage: Buffer.STORAGE},
      buffer
    );
    const second = context.graph.importBuffer(
      {id: 'second-output', byteLength: 32, usage: Buffer.STORAGE},
      buffer
    );
    const props = createProps(context, 4);
    expect(
      () =>
        new GPUUtilizationVolumeIsopleths({
          ...props,
          output: {
            ...props.output,
            volume: context.graph.createDataView(first, {format: 'float32', length: 4}),
            inside: context.graph.createDataView(second, {
              format: 'uint32',
              length: 4,
              byteOffset: 16
            })
          }
        })
    ).toThrow(/aligned binding byte ranges must not overlap/);
  } finally {
    buffer.destroy();
    context.device.destroy();
  }
});

it('keeps prefix workgroups sized to their own level instead of the largest level', () => {
  const context = createKernelAuditContext('isopleth-dispatch-shapes');
  try {
    const contributor = new GPUUtilizationVolumeIsopleths(createProps(context, 2 ** 24 + 1));
    const nodes = contributor
      .getCommandNodes(context.graph)
      .filter(node => /-level-\d+$/.test(node.id));
    expect(nodes.map(node => node.workload?.maximumInvocationCount)).toEqual([65537, 257, 2, 1]);
    expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual([257, 2, 1, 1]);
  } finally {
    context.device.destroy();
  }
});

it('builds the registered isopleth audit at both binding and buffer constrained maxima', () => {
  for (const [bindingLimit, bufferLimit] of [
    [2 ** 27, 2 ** 28],
    [2 ** 28, 2 ** 27]
  ]) {
    const context = createKernelAuditContext('isopleth-registered-audit');
    try {
      Object.defineProperty(context.device, 'limits', {
        value: {
          ...context.device.limits,
          maxStorageBufferBindingSize: bindingLimit,
          maxBufferSize: bufferLimit
        }
      });
      const auditCase = MOB07_HOME_RANGE_RECURSION_AUDIT_CASES.find(
        entry => entry.exportName === 'GPUUtilizationVolumeIsopleths'
      );
      expect(auditCase).toBeDefined();
      expect(() => auditCase!.build(context)).not.toThrow();
      expect(getDeclarationCounts(context)[1]).toBeGreaterThan(0);
      const graph = context.graph as unknown as {buffers: Map<string, {byteLength: number}>};
      expect(Math.max(...Array.from(graph.buffers.values(), buffer => buffer.byteLength))).toBe(
        Math.min(bindingLimit, bufferLimit)
      );
    } finally {
      context.device.destroy();
    }
  }
});

it('checks forbidden tokens against the complete W3C table and permits sample', () => {
  for (const identifier of [
    'target',
    'active',
    'filter',
    'module',
    'meta',
    'co_await',
    'reinterpret_cast',
    'yield'
  ]) {
    expect(getUtilizationReservedIdentifiers(`let ${identifier} = 0u;`)).toEqual([identifier]);
  }
  expect(getUtilizationReservedIdentifiers('let sample = 0u;')).toEqual([]);
  expect(getUtilizationReservedIdentifiers('// target\nlet targetIndex = 0u;')).toEqual([]);
});

it('reproduces both reported mass deltas with float32 rounding at each addition', () => {
  for (const {transition, tail, exactMass, treeMass, upperMass} of [
    {transition: 65537, tail: 1, exactMass: 25165826, treeMass: 25165824, upperMass: 25165824},
    {transition: 65665, tail: 257, exactMass: 25166210, treeMass: 25166208, upperMass: 25165952}
  ]) {
    // Every integer here is representable as float32. The first inexact addition loses one:
    // 2^24 + an odd transition rounds down. The root's odd tail loses another.
    let upper = Math.fround(2 ** 24 + transition);
    expect(upper).toBe(2 ** 24 + transition - 1);
    for (let group = 0; group < 127; group++) upper = Math.fround(upper + 65536);
    expect(upper).toBe(upperMass);
    const roundedRoot = Math.fround(upper + tail);
    expect(roundedRoot).toBe(treeMass);
    expect(exactMass).toBe(2 ** 24 + transition + 127 * 65536 + tail);
    expect(Math.fround(exactMass) - roundedRoot).toBe(2);
    expect(Math.abs(roundedRoot / exactMass - 1)).toBeLessThan(NONUNIFORM_MASS_RELATIVE_TOLERANCE);
  }
});

it('declares the largest fixture without allocating a row-sized CPU array', () => {
  const context = createKernelAuditContext('isopleth-device-fixture');
  const createValues = vi.spyOn(Float32Array, 'from').mockImplementation(() => {
    throw new Error('row-sized CPU fixture allocation');
  });
  try {
    Object.defineProperty(context.device, 'limits', {
      value: {
        ...context.device.limits,
        maxBufferSize: 2 ** 34,
        maxStorageBufferBindingSize: 2 ** 34
      }
    });
    const length = GPU_UTILIZATION_VOLUME_ISOPLETHS_MAXIMUM_CELLS;
    const {density, node} = createNonuniformUtilizationDensity(context.graph, length, 0);
    expect(density.length).toBe(length);
    expect(node.workload?.maximumInvocationCount).toBe(length);
    const kernels = getGPUCommandNodeKernelSources([node]);
    expect(kernels).toHaveLength(1);
    expect(getUtilizationReservedIdentifiers(kernels[0].source)).toEqual([]);
    expect(auditWGSLModule(kernels[0].source, kernels[0].constants)).toEqual([]);
    expect(createValues).not.toHaveBeenCalled();
  } finally {
    createValues.mockRestore();
    context.device.destroy();
  }
});

it('rejects device-sized headless fixtures independently of advertised GPU capacities', () => {
  expect(UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT).toBe(16777472);
  expect(UTILIZATION_HEADLESS_MAXIMUM_CELL_COUNT).toBe(16777473);
  for (const length of [
    255,
    65537,
    UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT,
    UTILIZATION_HEADLESS_MAXIMUM_CELL_COUNT
  ]) {
    expect(() => validateUtilizationHeadlessCellCount(length)).not.toThrow();
  }
  for (const length of [16777474, 2 ** 25, 1073741823, 2 ** 31, NaN, Infinity]) {
    expect(() => validateUtilizationHeadlessCellCount(length)).toThrow(
      /^isopleth-headless cellCount/
    );
  }
});

it('bounds the fixed scalability mass exactly and rules out a rounding-sized 100% error', () => {
  // At the new fixed size, sorted high-density cells fill 32768 blocks plus half a block.
  // Root terms are exactly representable, so the float32 tree must produce 25,166,208.
  let firstRoot = 0;
  for (let group = 0; group < 128; group++) firstRoot = Math.fround(firstRoot + 131072);
  firstRoot = Math.fround(firstRoot + 65664);
  for (let group = 0; group < 127; group++) firstRoot = Math.fround(firstRoot + 65536);
  expect(firstRoot).toBe(25165952);
  expect(Math.fround(firstRoot + 256)).toBe(25166208);
  expect(UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT * 1.5).toBe(25166208);

  // The abandoned 4-GiB fixture selected 1,073,741,823 rows. Even at that size its integer
  // 1/2-density root has only normal float32 rounding (one unit), not a relative error of 1.
  let oversizedRoot = 0;
  for (let region = 0; region < 32; region++) oversizedRoot = Math.fround(oversizedRoot + 33554432);
  for (let region = 0; region < 31; region++) oversizedRoot = Math.fround(oversizedRoot + 16777216);
  oversizedRoot = Math.fround(oversizedRoot + 16777215);
  expect(oversizedRoot).toBe(1610612736);
  const oversizedExactMass = 1073741823 + 536870912;
  expect(Math.abs(oversizedRoot / oversizedExactMass - 1)).toBeLessThan(
    NONUNIFORM_MASS_RELATIVE_TOLERANCE
  );
  expect(Math.abs(0 / oversizedExactMass - 1)).toBe(1);
});
