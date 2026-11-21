// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test, vi} from 'vitest';
import {scanSource} from '../../../../examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {
  addShowcaseCellTableColumnsNode,
  addShowcaseCertifiedSumCountsNode
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/cells/cell-pyramid-sums';
import * as kernels from '../../../../examples/deck/spatial-analysis-showcase/src/engine/mode-kernels';

test('DGGS-102: actual showcase sum selection reads loss words while count selection ignores sum saturation', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const counts = createTransientView(graph, 'counts', 'uint32', 4);
  const sumValues = createTransientView(graph, 'sums', 'float32', 4);
  const status = {
    count: createTransientView(graph, 'count', 'uint32', 1),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    candidateOverflow: createTransientView(graph, 'candidate-overflow', 'uint32', 1),
    sumOverflow: createTransientView(graph, 'sum-overflow', 'uint32', 1)
  };
  const output = {
    values: createTransientView(graph, 'values', 'float32', 4),
    mask: createTransientView(graph, 'mask', 'uint32', 4)
  };
  const spy = vi.spyOn(kernels, 'addKernelPass').mockImplementation(() => {});
  try {
    addShowcaseCellTableColumnsNode(graph, 'sums', {counts, sumValues, status}, output);
    const sum = spy.mock.calls[0][1];
    expect(sum.bindings.map(binding => binding.view)).toContain(status.sumOverflow);
    expect(sum.body).toContain('index < rowCount[rowCountOffset]');
    expect(sum.body).toContain('complete &&');
    expect(sum.body).toContain('var value = 0.0;');
    addShowcaseCellTableColumnsNode(graph, 'counts', {counts, status}, output);
    const count = spy.mock.calls[1][1];
    for (const variant of [sum, count])
      expect(scanSource(variant.body, 'selector.wgsl').hits).toEqual([]);
    expect(count.bindings.map(binding => binding.view)).not.toContain(status.sumOverflow);
    expect(count.bindings.map(binding => binding.view)).toContain(status.candidateOverflow);
    addShowcaseCertifiedSumCountsNode(graph, 'sum-style', counts, status, output.mask);
    expect(spy.mock.calls[2][1].bindings.map(binding => binding.view)).toContain(
      status.sumOverflow
    );
  } finally {
    spy.mockRestore();
  }
});

test('DGGS-102 explorer actual totals publisher refuses equal saturated sums and recovers independently of counts', async () => {
  const {readFileSync} = await import('node:fs');
  const {transpileModule, ScriptTarget} = await import('typescript');
  const source = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/cell-pyramid-mode.ts',
    'utf8'
  );
  const start = source.indexOf('        const copy = bytes.slice();');
  const end = source.indexOf('      } catch {', start);
  const helpersStart = source.indexOf('    const sumWords =');
  const helpersEnd = source.indexOf('    // Compares the finest table', helpersStart);
  expect(start).toBeGreaterThan(0);
  const values = new Map<string, string>();
  const readout = (key: string) => ({setValue: (value: string) => values.set(key, value)});
  const body = transpileModule(
    `${source.slice(helpersStart, helpersEnd)} function publish(bytes) {${source.slice(start, end)}}`,
    {compilerOptions: {target: ScriptTarget.ES2022}}
  ).outputText;
  const publish = new Function(
    'rollupCellsReadout',
    'rollupCountReadout',
    'rollupSumReadout',
    `const capacity=1, TOTALS_HEADER_BYTES=36, GPU_CELL_DEFAULT_SUM_SCALE=65536, parentResolution=1, FINEST_RESOLUTION=17; const formatCount=String; ${body}; return publish;`
  )(readout('cells'), readout('counts'), readout('sums'));
  const words = new Uint32Array(15);
  words.set([1, 0, 6, 1, 0, 0, 0, 1, 1, 3, 0xffffffff, 0x7fffffff, 3, 0xffffffff, 0x7fffffff]);
  // Both retained integer sums equal 2^63-1, but true totals exceed it: equality cannot certify them.
  for (const [fineLoss, parentLoss] of [
    [1, 0],
    [0, 1],
    [1, 1]
  ]) {
    words[7] = fineLoss;
    words[8] = parentLoss;
    publish(new Uint8Array(words.buffer));
    expect(values.get('sums')).toMatch(/unavailable.*saturation/);
    expect(values.get('counts')).toMatch(/conserved/);
  }
  words[7] = 0;
  words[8] = 0;
  words[10] = 65536;
  words[11] = 0;
  words[13] = 65536;
  words[14] = 0;
  publish(new Uint8Array(words.buffer));
  expect(values.get('sums')).toBe('1.0 → 1.0 min (bit-exact)');
  for (const [name, offset] of [
    ['finestSumOverflowBuffer', 28],
    ['rollupSumOverflowBuffer', 32]
  ])
    expect(source).toContain(`[${name}, 4, ${offset}]`);
  expect(source).toContain('rollupColorByMeanValue ? rollupSumCountsBuffer : rollupCountsBuffer');
});

test('S3 explorer density styling keeps counts while mean styling uses certified counts', async () => {
  const {readFileSync} = await import('node:fs');
  const {transpileModule, ScriptTarget} = await import('typescript');
  const source = readFileSync(
    'examples/deck/spatial-analysis-explorer/modes/cell-pyramid-mode.ts',
    'utf8'
  );
  const first = source.indexOf('      getLayers(): Layer[] {'),
    last = source.indexOf('      destroy()', first);
  const javascript = transpileModule(
    `const instance={${source.slice(first, last)}};return instance;`,
    {compilerOptions: {target: ScriptTarget.ES2022}}
  ).outputText;
  const counts = {},
    certified = {};
  const scope = new Proxy<Record<string, unknown>>(
    {
      rollupEnabled: true,
      rollupColorByMeanValue: false,
      rollupCountsBuffer: counts,
      rollupSumCountsBuffer: certified,
      QuadbinCellLayer: class {
        constructor(props: object) {
          return props;
        }
      }
    },
    {
      has: () => true,
      get: (object, key: string | symbol) =>
        typeof key === 'symbol' ? undefined : key in object ? object[key] : {}
    }
  );
  const instance = new Function('scope', `with(scope){${javascript}}`)(scope);
  expect(instance.getLayers()[0].countsEven).toBe(counts);
  scope['rollupColorByMeanValue'] = true;
  expect(instance.getLayers()[0].countsEven).toBe(certified);
});
