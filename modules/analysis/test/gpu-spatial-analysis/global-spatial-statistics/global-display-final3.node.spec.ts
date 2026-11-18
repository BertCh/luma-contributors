// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  formatGlobalJoinReadout,
  getGlobalJoinTooltip,
  getGlobalQuadrantCounts,
  getGlobalQuadrantLegend
} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation-display';
import ts from 'typescript';

const COMPUTE =
  'examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation.compute.ts';
it('final3 the actual observed join readout preserves half joins and symmetric integers', () => {
  const source = readFileSync(COMPUTE, 'utf8');
  expect(/formatGlobalJoinReadout\(\s*snapshot.weightsComplete/.test(source)).toBe(true);
  const render = (counts: number[]) => formatGlobalJoinReadout(true, counts, 0, 0);
  expect(render([0, 5, 3])).toBe('BB 0 (E 0), BW 2.5 (E 0), WW 1.5');
  expect(render([2, 4, 2])).toBe('BB 1 (E 0), BW 2 (E 0), WW 1');
});

for (const [value, complete, joinClass, expected] of [
  [NaN, true, 0xffffffff, 'No data'],
  [4, false, 0xffffffff, 'Unavailable (incomplete weights)'],
  [4, true, 1, 'Black'],
  [1, true, 0, 'White']
] as const)
  it(`final3 actual join tooltip: ${value}/${complete} -> ${expected}`, () => {
    const source = readFileSync(COMPUTE, 'utf8');
    const body = /if \(o.display === 'binary'\) \{([^]*?)\n    \}/.exec(source)![1];
    const result = Function(
      'xValue',
      'snapshot',
      'row',
      'getPercentile',
      'geography',
      'o',
      'BLACK_JOIN',
      'WHITE_JOIN',
      'rank',
      'xRow',
      'getGlobalJoinTooltip',
      body
    )(
      value,
      {weightsComplete: complete, joinClasses: Uint32Array.of(joinClass)},
      0,
      () => 2.5,
      {getVariable: () => [value]},
      {variable: 'diabetes', joinPercentile: 80},
      [0, 0, 0, 255],
      [255, 255, 255, 255],
      {},
      {},
      getGlobalJoinTooltip
    );
    expect(result[0].value).toBe(expected);
  });

it('final3 the actual quadrant legend counts unavailable rows separately from islands', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/weights/global-autocorrelation.scene.ts',
    'utf8'
  );
  const body = /case 'quadrant': \{([^]*?)\n    \}/.exec(source)![1];
  const js = ts.transpile(`function run(data,getQuadrantColors,getGlobalQuadrantLegend){${body}}`);
  const render = (codes: number[]) =>
    Function(
      'data',
      'getQuadrantColors',
      'getGlobalQuadrantLegend',
      `${js};return run(data,getQuadrantColors,getGlobalQuadrantLegend)`
    )(
      {quadrantCounts: getGlobalQuadrantCounts(codes, 0)},
      () => ({highHigh: [], lowHigh: [], lowLow: [], highLow: []}),
      getGlobalQuadrantLegend
    )[0];
  const legend = render([0, 0]);
  expect(
    legend.entries.find((entry: {label: string}) => entry.label === 'Unavailable quadrant')?.count
  ).toBe(2);
  expect(legend.note).toContain('Islands: 0');
  const recovered = render([2, 4]);
  expect(
    recovered.entries.find((entry: {label: string}) => entry.label === 'Low-High')?.count
  ).toBe(1);
  expect(
    recovered.entries.find((entry: {label: string}) => entry.label === 'High-Low')?.count
  ).toBe(1);
  expect(recovered.entries.at(-1).count).toBe(0);
});
