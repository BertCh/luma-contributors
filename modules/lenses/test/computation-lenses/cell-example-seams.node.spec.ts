// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import typescript from 'typescript';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {createGlobalCoordinateFixture} from '../../../analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinate-fixture';
import {addExampleCellConsumers} from './cell-example-consumers-fixture';

const paths = [
  'examples/deck/spatial-analysis-showcase/src/scenes/cells/h3-aggregation.compute.ts',
  'examples/deck/spatial-analysis-explorer/modes/cells-mode.ts'
];

test.each(
  paths
)('B2/B3 %s guards table overflow and computes ring counts after the outline', path => {
  const source = readFileSync(path, 'utf8');
  expect(
    source
      .replace(/\s+/g, ' ')
      .includes('if (tabular && tableCells && tableCounts && tableCount && tableOverflow)')
  ).toBe(true);
  const start = source.indexOf('const buildOutline =');
  const end = source.indexOf('const compiled =', start);
  const declaration = source.slice(start, end);
  expect(declaration.indexOf('addKernelPass(graph, ringCountsPass)')).toBeGreaterThan(
    declaration.indexOf('new GPUCellSetOutline(')
  );
});

test.each(
  paths
)('CPL-12 %s keeps counted producer status in every outline/compaction declaration', path => {
  const source = typescript.createSourceFile(
    path,
    readFileSync(path, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  let checked = 0;
  const visit = (node: typescript.Node): void => {
    if (
      typescript.isNewExpression(node) &&
      ['GPUCellSetOutline', 'GPUCellCompaction'].includes(node.expression.getText(source))
    ) {
      const props = node.arguments![0];
      expect(typescript.isObjectLiteralExpression(props)).toBe(true);
      if (!typescript.isObjectLiteralExpression(props)) throw new Error('expected literal props');
      const sourceProperty = props.properties.find(
        property => property.name?.getText(source) === 'source'
      );
      expect(
        sourceProperty,
        `${node.expression.getText(source)} must retain producer status`
      ).toBeDefined();
      expect(sourceProperty?.getText(source)).toContain('overflow:');
      if (node.expression.getText(source) === 'GPUCellCompaction') {
        expect(sourceProperty?.getText(source)).toContain('candidateOverflow:');
        if (props.getText(source).includes("type: 'compact'")) {
          const output = props.properties.find(
            property => property.name?.getText(source) === 'output'
          );
          expect(output?.getText(source)).toContain('candidateOverflow:');
        }
      }
      checked++;
    }
    typescript.forEachChild(node, visit);
  };
  visit(source);
  expect(checked).toBe(path.includes('showcase') ? 3 : 2);
});

test('CPL-12 showcase actual summary refuses candidate/final loss and recovers with aligned mask rows', () => {
  const source = readFileSync(paths[0], 'utf8');
  const start = source.indexOf('  const onZonesSummary =');
  const end = source.indexOf('  // --- Selection pipeline', start);
  expect(start).toBeGreaterThan(0);
  expect(end).toBeGreaterThan(start);
  const javascript = typescript.transpileModule(source.slice(start, end), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, string>();
  const publish = new Function(
    'ctx',
    `const zonesPipeline={serial:1}, destroyed=false, COVER_CAPACITY=1, areas={names:['area']}, formatCount=String;let coverAreaCells, coverComplete;${javascript};return onZonesSummary;`
  )({
    setReadout: (name: string, value: string) => readouts.set(name, value)
  });
  const words = Uint32Array.of(1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0);
  for (const [coverOverflow, coverCandidateOverflow, compactOverflow] of [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ]) {
    words[1] = coverOverflow;
    words[4] = compactOverflow;
    words[10] = coverCandidateOverflow;
    words[11] = coverOverflow || coverCandidateOverflow;
    publish(1, words.buffer);
    expect(readouts.get('roundTrip')).toBe('round trip unavailable: source cells were lost');
    expect(readouts.get('core')).toBe('1 core / 0 border (100% core)');
  }
  words[1] = words[4] = words[10] = words[11] = 0;
  publish(1, words.buffer);
  expect(readouts.get('roundTrip')).toBe(
    'uncompact gives back 1 cells: identical count to the cover'
  );
});

test('B4 incomplete cover cannot assert that the missing second area has no cells', () => {
  const source = readFileSync(paths[0], 'utf8');
  const start = source.indexOf('  const onZonesSummary =');
  const end = source.indexOf('  // --- Selection pipeline', start);
  const javascript = typescript.transpileModule(source.slice(start, end), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, string>();
  const publish = new Function(
    'ctx',
    `const zonesPipeline={serial:1}, destroyed=false, COVER_CAPACITY=1, areas={names:['retained','lost']}, formatCount=String;let coverAreaCells, coverComplete;${javascript};return onZonesSummary;`
  )({setReadout: (name: string, value: string) => readouts.set(name, value)});
  const words = Uint32Array.of(1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0);
  publish(1, words.buffer);
  expect(readouts.get('cover')).not.toContain('areas get no cell');
  expect(readouts.get('cover')).toContain('incomplete');
  words[10] = words[11] = 0;
  publish(1, words.buffer);
  expect(readouts.get('cover')).toContain('1 of 2 areas get no cell');
});

test('B4 missing selected cell and disk totals remain unavailable until the table is complete', () => {
  const source = readFileSync(paths[0], 'utf8');
  const start = source.indexOf('  const getTableLookup =');
  const end = source.indexOf('  // --- Build orchestration', start);
  const javascript = typescript.transpileModule(source.slice(start, end), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const readouts = new Map<string, string>();
  const snapshot = {
    cells: Uint32Array.of(1, 2),
    counts: Uint32Array.of(9),
    rows: 1,
    complete: false
  };
  const publish = new Function(
    'ctx',
    'tableSnapshot',
    `let tableLookup=null;const activeSelection={stride:1,operation:'disk'},lastSelectionPipeline=activeSelection,lastSelectionWords=Uint32Array.of(3,4,1,3,4);const formatCount=String,formatCellKey=(low,high)=>low+':'+high,packKeyWords=formatCellKey;${javascript};return showSelectionSummary;`
  )({setReadout: (name: string, value: string) => readouts.set(name, value)}, snapshot);
  publish();
  expect(readouts.get('selected')).not.toContain('no observations');
  expect(readouts.get('selection')).toContain('incomplete');
  snapshot.complete = true;
  publish();
  expect(readouts.get('selected')).toContain('no observations');
  expect(readouts.get('selection')).toContain('0 observations');
});

test('B4 missing area tooltip qualifies zero as retained until a complete cover arrives', () => {
  const source = readFileSync(paths[0], 'utf8');
  const start = source.indexOf('  const getHoverText =');
  const end = source.indexOf('  const instance:', start);
  const javascript = typescript.transpileModule(source.slice(start, end), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const state = {coverComplete: false};
  const render = new Function(
    'state',
    `const areas={names:['retained','lost']},areaObservationCounts=[0,4,7],FAMILIES={quadbin:{tabular:true}},ctx={options:{family:'quadbin'}},coverAreaCells=[1,0],zonesPipeline={},findArea=()=>1,formatCount=String;${javascript.replaceAll('coverComplete', 'state.coverComplete')};return getHoverText;`
  )(state);
  expect(render(0, 0)).toContain('0 retained cover cells (incomplete cover)');
  state.coverComplete = true;
  expect(render(0, 0)).toContain('; 0 cover cells');
});

test.each([
  'showcase',
  'explorer'
] as const)('CPL-12 actual %s constructor declarations lower with full status identities', example => {
  const device = createNullWebGPUDevice();
  const fixture = createGlobalCoordinateFixture(device);
  try {
    const consumers = addExampleCellConsumers(fixture, example, 16, true);
    expect(consumers.contributors).toHaveLength(example === 'showcase' ? 3 : 2);
    for (const contributor of consumers.contributors) {
      expect(contributor.props.source?.status.overflow).toBeDefined();
    }
    const compact = consumers.contributors[1];
    expect(compact.props.source?.status.overflow).toBe(consumers.sourceOverflow.view);
    expect(compact.props.source?.status.candidateOverflow).toBe(
      consumers.sourceCandidateOverflow.view
    );
    if (example === 'showcase') {
      const uncompact = consumers.contributors[2];
      expect(uncompact.props.source?.status.overflow).toBe(consumers.compactOverflow.view);
      expect(uncompact.props.source?.status.candidateOverflow).toBe(
        consumers.compactCandidateOverflow.view
      );
    }
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
