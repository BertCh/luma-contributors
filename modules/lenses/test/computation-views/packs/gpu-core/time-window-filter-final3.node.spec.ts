// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {expect, test} from 'vitest';
import {fromRecordedLensFile} from '../../../../src/computation-lenses/registry';
import {buildDecodedLensTwin} from '../../../../src/computation-views/twin/static-twin';
import {buildGPUCoreArrayResultModel} from '../../../../src/computation-views/packs/gpu-core/array-result-model';
import '../../../../src/computation-views/packs/gpu-core/index';
import {createViewContext} from '../../../../src/computation-views/registry/view-registry';

/** Read the actual authored capture metadata without running its GPU/dataset loader. */
function getAuthoredSpecs(): Record<
  string,
  {meaning?: string; role: 'input' | 'output'; keySpace: string}
> {
  const source = readFileSync(
    new URL(
      '../../../../src/computation-lenses/scenarios/gpu-dataframe/time-window-filter.run.ts',
      import.meta.url
    ),
    'utf8'
  );
  const file = ts.createSourceFile(
    'time-window-filter.run.ts',
    source,
    ts.ScriptTarget.Latest,
    true
  );
  let specs: ts.ObjectLiteralExpression | undefined;
  const visit = (node: ts.Node) => {
    if (
      ts.isPropertyAssignment(node) &&
      node.name.getText(file) === 'specs' &&
      ts.isObjectLiteralExpression(node.initializer)
    )
      specs = node.initializer;
    ts.forEachChild(node, visit);
  };
  visit(file);
  expect(specs).toBeDefined();
  return new Function('example', `return (${specs!.getText(file)});`)({});
}

test.each([
  'full',
  'thumbnail'
] as const)('final3 H2 decoded time-filter %s twin distinguishes count 588 from first row 513', size => {
  const decoded = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../../docs/lens-records/gpu-dataframe.time-window-filter.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  );
  const authored = getAuthoredSpecs();
  for (const capture of decoded.record.captures!.snapshots) {
    const spec = authored[capture.name];
    if (spec) {
      capture.meaning = spec.meaning;
      capture.role = spec.role;
      capture.keySpace = spec.keySpace;
    }
  }
  const captures = decoded.record.captures!.snapshots;
  const count = captures.find(capture => capture.name === 'count')!;
  const mask = captures.find(capture => capture.name === 'outputMask')!;
  const rows = captures.find(capture => capture.name === 'rows')!;
  expect(Number(count.values[0])).toBe(588);
  expect(Array.from(mask.values).filter(value => value !== 0)).toHaveLength(588);
  expect(Number(rows.values[0])).toBe(513);
  const twin = buildDecodedLensTwin(
    decoded,
    {scenario: decoded.scenarioId, rung: 'result'},
    {size}
  );
  expect(twin.keyNumber).toMatchObject({label: 'count', value: '588'});
  expect(twin.figure.html).toContain('data-panel="rows"');
  const model = buildGPUCoreArrayResultModel(createViewContext(decoded.record, {options: {size}}));
  const compact = model.panels.find(panel => panel.capture.name === 'rows');
  expect(compact?.values).toEqual(Array.from({length: 588}, (_, index) => 513 + index));
});
