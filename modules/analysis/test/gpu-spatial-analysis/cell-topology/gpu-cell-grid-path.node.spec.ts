// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellGridPath} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectCellStorageScopes} from '../cell-indexing/cell-dispatch-audit';
import {getS2TopologyCell} from './s2-topology-oracle';
import {quadbinTileToCell, splitCellKey} from '../cell-aggregation/cell-aggregation-oracle';

function getPathKernel(family: 'quadbin' | 's2', stride: number, highLow: boolean, rows: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `path-${sequence++}`, format, length);
  const nodes = new GPUCellGridPath({
    family,
    wordOrder: highLow ? 'high-low' : 'little-endian',
    origins: view('uint32x2', rows),
    destinations: view('uint32x2', rows),
    mask: view('uint32', rows),
    maximumPathLength: stride,
    output: {
      cells: view('uint32x2', rows * stride),
      counts: view('uint32', rows),
      distances: view('uint32', rows)
    }
  }).getCommandNodes(graph);
  expectCellStorageScopes(nodes);
  const kernel = getGPUCommandNodeKernelSources(nodes).find(
    item => item.operation === 'GPUCellGridPath'
  )!;
  expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  const words =
    kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ??
    [];
  expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
  device.destroy();
  return kernel;
}

it('Quadbin path source includes tile helpers without undeclared neighborhood dependencies', () => {
  const kernel = getPathKernel('quadbin', 3, false, 3);
  expect(kernel.source).not.toMatch(/\bCELL_TOPOLOGY_(CAPACITY|K)\b/);
  expect(kernel.source).not.toContain('var<private> cellTopologyCells');
});

it.each([
  'quadbin',
  's2'
] as const)('%s generated path stores preserve complete independent key records in both word orders', family => {
  const same =
    family === 's2'
      ? getS2TopologyCell({face: 5, ij: [0, 0], level: 0})
      : quadbinTileToCell(2, 2, 26);
  const stepCell = (step: number) =>
    family === 's2'
      ? getS2TopologyCell({face: 3, ij: [10000 + step, 10000], level: 30})
      : quadbinTileToCell(10000 + step, 10000, 26);
  for (const stride of [1, 3, 5])
    for (const highLow of [false, true]) {
      const kernel = getPathKernel(family, stride, highLow, 3);
      const expected = Array.from({length: 3 * stride}, (_, index) => {
        const row = Math.floor(index / stride);
        const slot = index % stride;
        if (row === 0) return slot === 0 ? same : 0n;
        if (row === 1 && slot <= 3) return stepCell(slot);
        return 0n;
      });
      const encode = (cells: bigint[], useHighLow: boolean) =>
        Uint32Array.from(
          cells.flatMap(cell => {
            const [low, high] = splitCellKey(cell);
            return useHighLow ? [high, low] : [low, high];
          })
        );
      // Evaluate the actual generated address expressions. These small integer addresses are
      // exact in JavaScript; key words remain uint32, without the WGSL interpreter's float32
      // vector-constructor conversions. Expected keys and their logical record order are independent.
      const stores = [...kernel.source.matchAll(/outCells\[([^\]\n]+)\] = result\.([xy]);/g)];
      expect(stores).toHaveLength(2);
      const cells = new Uint32Array(3 * stride * 2 + 2).fill(0xdeadbeef);
      for (let index = 0; index < 3; index++)
        for (let slot = 0; slot < stride; slot++) {
          const [low, high] = splitCellKey(expected[index * stride + slot]);
          for (const [, expression, component] of stores) {
            const arithmetic = expression.replace(/(\d+)u\b/g, '$1');
            expect(arithmetic.replace(/\b(?:outCellsOffset|index|slot)\b/g, '')).toMatch(
              /^[\d\s+*()]+$/
            );
            const getAddress = new Function(
              'index',
              'slot',
              'outCellsOffset',
              `return ${arithmetic};`
            );
            cells[getAddress(index, slot, 1)] = component === 'x' ? high : low;
          }
        }
      expect([...cells]).toEqual([0xdeadbeef, ...encode(expected, highLow), 0xdeadbeef]);
    }
});

it('grid-path missing-output validation names the output prop and requirement', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const cells = createTransientView(graph, 'cells', 'uint32x2', 1);
    expect(
      () =>
        new GPUCellGridPath({
          id: 'checked-path',
          family: 'h3',
          origins: cells,
          destinations: cells,
          output: {}
        })
    ).toThrow(/^checked-path output needs distances or cells$/);
  } finally {
    device.destroy();
  }
});
