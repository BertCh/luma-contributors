// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {expect, it} from 'vitest';
import {getRegionalizationWeightsReadout} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/regression/regionalization-readout';
import {GPU_REGION_PARTITION_EVALUATION_LAYOUT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';

const source = readFileSync(
  'examples/deck/spatial-analysis-showcase/src/scenes/regression/regionalization.compute.ts',
  'utf8'
);

function getReaders(text: string): {name: string; bindings: string; callback: string}[] {
  const file = ts.createSourceFile(
    'regionalization.compute.ts',
    text,
    ts.ScriptTarget.Latest,
    true
  );
  const readers: {name: string; bindings: string; callback: string}[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText(file) === 'createNamedReader' &&
      ts.isStringLiteral(node.arguments[1]) &&
      ['regions-csr', 'regions-results'].includes(node.arguments[1].text)
    )
      readers.push({
        name: node.arguments[1].text,
        bindings: node.arguments[2].getText(file),
        callback: node.arguments[3].getText(file)
      });
    ts.forEachChild(node, visit);
  };
  visit(file);
  expect(readers).toHaveLength(2);
  return readers;
}

function getCallbacks(text: string): string[] {
  return getReaders(text).map(reader => reader.callback);
}

function assertReadouts(text: string): void {
  const definitions = getReaders(text);
  const names = [
    ...new Set(
      definitions.flatMap(reader =>
        [...reader.bindings.matchAll(/buffer:\s*(\w+)/g)].map(match => match[1])
      )
    )
  ];
  type Readback = {u32: Uint32Array; f32: Float32Array};
  const buffers: Record<string, Readback> = Object.fromEntries(
    names.map(name => [name, {u32: new Uint32Array(16), f32: new Float32Array(16)}])
  );
  // Two adjacent rows, two singleton regions, one tree edge and one applied cut.
  buffers['offsetsBuffer'].u32 = Uint32Array.of(0, 1, 2);
  buffers['neighborsBuffer'].u32 = Uint32Array.of(1, 0);
  for (const name of ['labelsBuffer', 'kLabelsBuffer']) buffers[name].u32 = Uint32Array.of(0, 1);
  buffers['regionCountBuffer'].u32 = Uint32Array.of(2);
  buffers['edgeCountBuffer'].u32 = Uint32Array.of(1);
  buffers['edgeIdsBuffer'].u32 = Uint32Array.of(0);
  buffers['edgeCostsBuffer'].f32 = Float32Array.of(4);
  buffers['cutEdgesBuffer'].u32 = Uint32Array.of(0);
  buffers['cutGainsBuffer'].f32 = Float32Array.of(4);
  buffers['componentLabelsBuffer'].u32 = Uint32Array.of(0, 0);
  buffers['kIterationCountBuffer'].u32 = Uint32Array.of(2);
  buffers['kConvergedBuffer'].u32 = Uint32Array.of(1);
  // Attributes [0,2], singleton labels [0,1]: within=0, between=total=2,
  // sizes=1, every neighbor crosses the boundary, pseudo-F undefined for n=k.
  for (const name of ['summaryBuffer', 'kSummaryBuffer'])
    buffers[name].f32 = Float32Array.of(2, 0, 2, 2, 1, 1, 1, 0, NaN);
  const readouts = new Map<string, unknown>();
  let requests = 0;
  const helpers = ts
    .createSourceFile('scene.ts', text, ts.ScriptTarget.Latest, true)
    .statements.filter(
      statement =>
        ts.isFunctionDeclaration(statement) &&
        ['countPieces', 'assignRegionColors'].includes(statement.name?.text ?? '')
    )
    .map(statement => statement.getText())
    .join('\n');
  const javascript = ts.transpileModule(
    `
    let weightsComplete = true, treeEdges = 7, adjacency = [], csrReady = false;
    let latestLabels, latestKLabels, costRange;
    const rowCount = 2, capacity = 2, stepCount = 1, resources = {};
    const centroids = Float32Array.of(0, 0, 1, 0);
    const REGION_PALETTE = [0, 1], GPU_SKATER_NO_CUT = 0xffffffff;
    const {${names.join(',')}} = buffers;
    const sink = {write: () => {}};
    const slotSegmentsBuffer = sink, cutMaskBuffer = sink, slotCostBuffer = sink;
    const regionColorsBuffer = sink, clusterColorsBuffer = sink;
    const formatInteger = String, formatNumber = (value, places) => value.toFixed(places);
    const getQuantile = values => values[0];
    ${helpers}
    const csrReader = createNamedReader(resources, 'regions-csr', ${definitions[0].bindings}, ${definitions[0].callback});
    const reader = createNamedReader(resources, 'regions-results', ${definitions[1].bindings}, ${definitions[1].callback});
    return {readers: [csrReader, reader], complete: () => weightsComplete, treeEdges: () => treeEdges};
  `,
    {compilerOptions: {target: ts.ScriptTarget.ES2022}}
  ).outputText;
  const createNamedReader = (
    _resources: unknown,
    _name: string,
    bindings: {name: string; buffer: Readback}[],
    callback: (get: (name: string) => Readback) => void
  ) => ({
    markStale() {},
    invoke: () =>
      callback(name => {
        const binding = bindings.find(binding => binding.name === name);
        if (!binding) throw new Error(`missing actual reader binding: ${name}`);
        return binding.buffer;
      })
  });
  const run = new Function(
    'buffers',
    'ctx',
    'createNamedReader',
    'getRegionalizationWeightsReadout',
    'GPU_REGION_PARTITION_EVALUATION_LAYOUT',
    javascript
  )(
    buffers,
    {
      options: {regions: 2},
      setReadout: (name: string, value: unknown) => readouts.set(name, value),
      requestLayers: () => requests++,
      setLegendExtent() {}
    },
    createNamedReader,
    getRegionalizationWeightsReadout,
    GPU_REGION_PARTITION_EVALUATION_LAYOUT
  ) as {readers: {invoke(): void}[]; complete(): boolean; treeEdges(): number};
  for (const [index, reader] of run.readers.entries()) {
    for (const [overflow, candidateOverflow] of [
      [0, 1],
      [1, 0],
      [0, 0]
    ] as const) {
      readouts.clear();
      requests = 0;
      buffers['overflowBuffer'].u32[0] = overflow;
      buffers['candidateOverflowBuffer'].u32[0] = candidateOverflow;
      if (overflow || candidateOverflow) {
        reader.invoke();
        expect(run.complete()).toBe(false);
        expect(run.treeEdges()).toBe(0);
        expect(requests).toBe(1);
        expect([...readouts.keys()].sort()).toEqual([
          'boundary',
          'capacity',
          'explained',
          'gains',
          'pieces',
          'pseudoF',
          'regions',
          'sizes',
          'tree',
          'within'
        ]);
        expect(readouts.get('capacity')).toBe(
          'Incomplete weights: raise neighbor or pair capacity'
        );
        for (const [name, value] of readouts)
          if (name !== 'capacity') expect(value, name).toBe('Unavailable (incomplete weights)');
      } else {
        reader.invoke();
        expect(run.complete()).toBe(true);
        expect(requests).toBe(1);
        if (index === 1) {
          expect(readouts.get('capacity')).toBe('ok');
          expect(readouts.get('regions')).toBe('2 (target 2)');
          expect(readouts.get('pieces')).toBe(
            'SKATER 2 pieces for 2 regions; k-means 2 pieces for 2 clusters'
          );
          expect(readouts.get('tree')).toBe('1 edges in 1 tree');
          expect(readouts.get('gains')).toBe('4');
          expect(readouts.get('within')).toBe('SKATER 0 vs k-means 0 (of 2)');
          expect(readouts.get('sizes')).toBe('SKATER 1 to 1 tracts; k-means 1 to 1');
          expect(readouts.get('boundary')).toBe(
            'SKATER 100.0% vs k-means 100.0% of neighbour links'
          );
          expect(readouts.get('explained')).toBe(
            'SKATER 100.0% vs k-means 100.0% of variance (0.0 percentage points, k-means minus SKATER)'
          );
          expect(readouts.get('pseudoF')).toBe('SKATER NaN vs k-means NaN');
          expect(readouts.get('kmeans')).toBe('2 iterations, converged');
          expect(run.treeEdges()).toBe(1);
        }
      }
    }
  }
}

it('H6 both actual callbacks suppress either loss word and allow a complete retry', () => {
  assertReadouts(source);
});

it('R5 rejects a permanently unavailable clean capacity label', () => {
  const assignment = "ctx.setReadout('capacity', availability.capacity);";
  const position = source.lastIndexOf(assignment);
  const mutated =
    source.slice(0, position) +
    source
      .slice(position)
      .replace(assignment, "ctx.setReadout('capacity', 'unavailable forever');");
  expect(() => assertReadouts(mutated)).toThrow();
});

it('R5 rejects candidate-loss readers bound to the final-loss buffer', () => {
  const mutated = source.replaceAll(
    "{name: 'candidateOverflow', buffer: candidateOverflowBuffer, bytes: 4}",
    "{name: 'candidateOverflow', buffer: overflowBuffer, bytes: 4}"
  );
  expect(mutated).not.toBe(source);
  expect(() => assertReadouts(mutated)).toThrow();
});

it('H6 rejects ignoring candidate loss in either actual callback', () => {
  const assignment = 'weightsComplete = availability.complete;';
  for (const callback of getCallbacks(source)) {
    expect(callback).toContain(assignment);
    const mutated = source.replace(
      callback,
      callback.replace(assignment, "weightsComplete = get('overflow').u32[0] === 0;")
    );
    // The older helper-only regression still accepts this source: the two helper calls and
    // every helper return value are unchanged. Executing the callback rejects it.
    expect([...mutated.matchAll(/getRegionalizationWeightsReadout\(/g)]).toHaveLength(2);
    expect(getRegionalizationWeightsReadout(0, 1).complete).toBe(false);
    expect(() => assertReadouts(mutated)).toThrow();
  }
});
