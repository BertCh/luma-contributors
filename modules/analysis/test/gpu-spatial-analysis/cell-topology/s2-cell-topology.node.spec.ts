// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getS2ChildIndex, getS2IndexFromCell} from '@math.gl/dggs';
import {build} from 'esbuild';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUCellTopology,
  getCellTopologyStride,
  type GPUCellTopologyOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-topology';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from '../../../src/testing/index';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getS2TopologyCell,
  getS2TopologyRowOnCPU,
  S2_TOPOLOGY_REFERENCE
} from './s2-topology-oracle';

it('keeps Node-only DGGS reference dependencies out of the browser dependency graph', async () => {
  const bundle = await build({
    entryPoints: [new URL('./s2-cell-topology.spec.ts', import.meta.url).pathname],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    packages: 'external',
    write: false,
    metafile: true,
    logLevel: 'silent'
  });
  const dependencies = Object.values(bundle.metafile!.inputs).flatMap(input =>
    input.imports.filter(dependency => dependency.external).map(dependency => dependency.path)
  );
  // Adding math.gl's aggregate entry to a running browser suite discovers extra decoder/CJS
  // dependencies. Vite can re-optimize and reload while luma.gl initializes its WebGPU wrapper.
  expect(dependencies.filter(path => path === '@math.gl/dggs')).toEqual([]);
});

it('cross-checks the browser-safe key and hierarchy oracle against math.gl at every level', () => {
  for (let level = 0; level <= 30; level++) {
    const size = 2 ** level;
    for (let face = 0; face < 6; face++)
      for (const [column, row] of [
        [0, 0],
        [size - 1, size - 1],
        [Math.floor(size / 2), Math.floor(size / 2)],
        [Math.floor(size * 0.73), Math.floor(size * 0.18)]
      ]) {
        const expected = getS2IndexFromCell({face, ij: [column, row], level});
        const cell = getS2TopologyCell({face, ij: [column, row], level});
        expect(cell).toBe(expected);
        for (const resolution of new Set([0, Math.floor(level / 2), level])) {
          const scale = 2 ** (level - resolution);
          const parent = getS2IndexFromCell({
            face,
            ij: [Math.floor(column / scale), Math.floor(row / scale)],
            level: resolution
          });
          expect(getS2TopologyRowOnCPU({type: 'parent', resolution}, cell).cells).toEqual([parent]);
        }
        if (level < 30) {
          const children = [0, 1, 2, 3].map(slot => getS2ChildIndex(expected, slot));
          expect(
            getS2TopologyRowOnCPU(
              {type: 'children', inputResolution: level, resolution: level + 1},
              cell
            ).cells
          ).toEqual(children);
        }
      }
  }
});

it('bounds every pinned low-level S2 disk/ring by its declared stride through radius 8', () => {
  for (const {cell, level} of S2_TOPOLOGY_REFERENCE) {
    if (level > 2) continue;
    for (let radius = 0; radius <= 8; radius++)
      for (const type of ['disk', 'ring'] as const) {
        const operation = {type, k: radius};
        const expected = getS2TopologyRowOnCPU(operation, cell);
        expect(expected.cells.length).toBeLessThanOrEqual(getCellTopologyStride('s2', operation));
        expect(new Set(expected.cells).size).toBe(expected.cells.length);
      }
  }
  expect(getCellTopologyStride('s2', {type: 'disk', k: 8})).toBe(145);
  expect(getCellTopologyStride('s2', {type: 'ring', k: 8})).toBe(32);
  expect(getCellTopologyStride('s2', {type: 'disk', k: 0})).toBe(1);
  expect(getCellTopologyStride('s2', {type: 'ring', k: 0})).toBe(1);
  expect(getCellTopologyStride('s2', {type: 'children', resolution: 30, inputResolution: 24})).toBe(
    4096
  );
  expect(() =>
    getCellTopologyStride('s2', {type: 'children', resolution: 30, inputResolution: 23})
  ).toThrow(/stride/);
  expect(() => getCellTopologyStride('s2', {type: 'parent', resolution: 31})).toThrow(/resolution/);
});

it('fine-level interior neighborhoods attain the flat-grid stride on all six faces', () => {
  for (const level of [26, 30])
    for (let face = 0; face < 6; face++) {
      const cell = getS2IndexFromCell({
        face,
        ij: [2 ** (level - 1), 2 ** (level - 1)],
        level
      });
      expect(getS2TopologyRowOnCPU({type: 'disk', k: 8}, cell).cells.length).toBe(145);
      expect(getS2TopologyRowOnCPU({type: 'ring', k: 8}, cell).cells.length).toBe(32);
    }
});

const ROOT = 1n << 60n;
const LEAF = getS2IndexFromCell({face: 4, ij: [2 ** 30 - 1, 2 ** 29], level: 30});
for (const [operation, keys] of [
  [{type: 'disk', k: 0}, [ROOT, LEAF, 0n, 2n]],
  [{type: 'disk', k: 2}, [ROOT, S2_TOPOLOGY_REFERENCE.find(row => row.level === 2)!.cell]],
  [{type: 'disk', k: 8}, [ROOT]],
  [{type: 'ring', k: 8}, [ROOT]],
  [{type: 'ring', k: 1}, [S2_TOPOLOGY_REFERENCE.find(row => row.level === 30)!.cell, ROOT]],
  [{type: 'parent', resolution: 0}, [ROOT, LEAF, 0n]],
  [{type: 'parent', resolution: 30}, [LEAF, ROOT]],
  [{type: 'children', resolution: 2, inputResolution: 0}, [ROOT, LEAF, 0n]],
  [{type: 'children', resolution: 30, inputResolution: 30}, [LEAF, ROOT]]
] as const) {
  it(`constructs and parses S2 ${JSON.stringify(operation)} with exact-key reference rows`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const stride = getCellTopologyStride('s2', operation);
      const cells = createTransientView(graph, 'cells', 'uint32x2', keys.length);
      const output = {
        cells: createTransientView(graph, 'output', 'uint32x2', keys.length * stride),
        counts: createTransientView(graph, 'counts', 'uint32', keys.length),
        distances:
          operation.type === 'disk'
            ? createTransientView(graph, 'distances', 'uint32', keys.length * stride)
            : undefined
      };
      const contributor = new GPUCellTopology({family: 's2', operation, cells, output});
      const nodes = contributor.getCommandNodes(graph);
      const [kernel] = getGPUCommandNodeKernelSources(nodes);
      expect(() => new WgslReflect(kernel!.source)).not.toThrow();
      expect(auditWGSLModule(kernel!.source, kernel!.constants)).toEqual([]);
      // These rows also feed the headless oracle spec. Parsing checks the actual assembled
      // integer shader without relying on CPU-interpreter support for dggs operations.
      for (const key of keys) {
        const expected = getS2TopologyRowOnCPU(operation, key);
        expect(expected.cells.length).toBeLessThanOrEqual(stride);
      }
    } finally {
      device.destroy();
    }
  }, 60_000);
}

it('validates S2 resolution/depth, word order, and expanded storage limits', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const create = (operation: GPUCellTopologyOperation, wordOrder?: 'high-low') => {
      const rows = 120_000;
      sequence++;
      const stride = getCellTopologyStride('s2', operation);
      return new GPUCellTopology({
        id: 's2-limit',
        family: 's2',
        operation,
        wordOrder,
        cells: createTransientView(graph, `cells-${sequence}`, 'uint32x2', rows),
        output: {cells: createTransientView(graph, `output-${sequence}`, 'uint32x2', rows * stride)}
      });
    };
    expect(() => create({type: 'disk', k: 8}).getCommandNodes(graph)).toThrow(
      /s2-limit.*output.cells.*134217728/
    );
    expect(() => create({type: 'disk', k: 7}).getCommandNodes(graph)).not.toThrow();
    expect(() => create({type: 'parent', resolution: 0}, 'wrong' as 'high-low')).toThrow(
      /s2-limit.*wordOrder/
    );
    expect(() => create({type: 'children', resolution: 29, inputResolution: 30})).toThrow(/below/);
  } finally {
    device.destroy();
  }
});
