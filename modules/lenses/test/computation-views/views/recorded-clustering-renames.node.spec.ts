// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import type {ViewPackData} from '../../../src/computation-views/registry/view-registration';
import {buildFormulaCardModel} from '../../../src/computation-views/views/formula-card';
import {getBoundNode} from '../../../src/computation-views/views/record-queries';
import {buildScopeDiagramModel} from '../../../src/computation-views/views/scope-diagram';
import {resolveValueScale} from '../../../src/computation-views/views/value-strip';
import {createDataFramePackData} from '../../../src/gpu-dataframe/computation-pack/pack';
import {createSpatialPackData} from '../../../src/gpu-spatial-analysis/computation-pack/pack';
import {makeContext} from '../fixtures/v4-record';

function readRecordedScenario(scenario: string) {
  return fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../../docs/lens-records/${scenario}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
}

function bindRecordedOperator(
  scenario: string,
  operator: string,
  packs: readonly ViewPackData[] = [createSpatialPackData()]
) {
  const record = readRecordedScenario(scenario);
  const node = record.graphs
    .flatMap(graph => graph.nodes)
    .find(candidate => candidate.op === operator && candidate.formula === undefined);
  if (!node) throw new Error(`${scenario} has no ${operator} node without an inline formula`);
  return makeContext({
    record,
    bindings: {node: node.id},
    packs
  });
}

test.each([
  'group-geometry',
  'group-convex-hull'
])('finds the DBSCAN formula and scope inside the committed %s recording', scenario => {
  const context = bindRecordedOperator(`gpu-spatial-analysis.${scenario}`, 'GPUDBSCAN');
  const formula = buildFormulaCardModel(context);
  expect(formula.formulaId).toBe('dbscan-core-test');
  expect(formula.symbols).toContainEqual(
    expect.objectContaining({id: 'minimumSamples', role: 'parameter'})
  );
  expect(buildScopeDiagramModel(context)).toMatchObject({
    op: 'GPUDBSCAN',
    scope: 'focal',
    reduction: 'count'
  });
});

test('finds the HDBSCAN scope while preserving the recorded node operator', () => {
  const context = bindRecordedOperator(
    'gpu-spatial-analysis.hierarchical-density-clustering',
    'GPUHDBSCAN'
  );
  expect(buildScopeDiagramModel(context)).toMatchObject({
    op: 'GPUHDBSCAN',
    scope: 'pairwise',
    reduction: 'select'
  });
});

test.each([
  'spatial-clustering',
  'hierarchical-density-clustering'
])('gives the archived %s cluster labels a nominal scale', scenario => {
  const context = makeContext({
    record: readRecordedScenario(`gpu-spatial-analysis.${scenario}`),
    packs: [createSpatialPackData()]
  });
  expect(resolveValueScale(context, 'labels')).toMatchObject({
    scale: {variableKind: {level: 'nominal'}},
    levelDeclared: true
  });
});

test('finds standalone key lookup and pycnophylactic scope descriptors', () => {
  const lookup = makeContext({
    record: readRecordedScenario('gpu-dataframe.key-join'),
    packs: [createDataFramePackData()]
  });
  expect(buildScopeDiagramModel(lookup)).toMatchObject({
    op: 'GPUKeyLookup',
    scope: 'pairwise',
    reduction: 'select'
  });

  const interpolation = makeContext({
    record: readRecordedScenario('gpu-spatial-analysis.pycnophylactic'),
    packs: [createSpatialPackData()]
  });
  expect(buildScopeDiagramModel(interpolation)).toMatchObject({
    op: 'GPUPycnophylacticInterpolation',
    scope: 'solver',
    reduction: 'solve'
  });
});

test('selects the key lookup formula for a nested recorded key lookup', () => {
  const context = bindRecordedOperator(
    'gpu-spatial-analysis.population-support-alignment',
    'GPUKeyLookup',
    [createSpatialPackData(), createDataFramePackData()]
  );
  expect(buildFormulaCardModel(context).formulaId).toBe('gpu-dataframe:GPUKeyJoin');
});

test('selects the requested DBSCAN formula through the explicit recorded operator option', () => {
  const context = makeContext({
    record: readRecordedScenario('gpu-spatial-analysis.group-convex-hull'),
    options: {op: 'GPUDBSCAN'},
    packs: [createSpatialPackData()]
  });
  expect(buildFormulaCardModel(context).formulaId).toBe('dbscan-core-test');
});

test('prefers the exact legacy operator key within each pack and keeps pack ordering', () => {
  const context = bindRecordedOperator('gpu-spatial-analysis.group-convex-hull', 'GPUDBSCAN');
  // Recreate an older recording's operator so the exact key and the current fallback differ.
  const legacyOperator = 'GPUSpatialClustering';
  getBoundNode(context)!.node.op = legacyOperator;
  const current = createSpatialPackData();
  const recorded: ViewPackData = {
    id: 'archived-host-pack',
    operators: {[legacyOperator]: {scope: 'global', reduction: 'histogram'}},
    formulas: {
      [legacyOperator]: {...context.record.refs.formulas[0], id: 'archived-dbscan-formula'}
    }
  };
  const both: ViewPackData = {
    ...current,
    operators: {...current.operators, ...recorded.operators},
    formulas: {...current.formulas, ...recorded.formulas}
  };
  for (const packs of [[recorded], [both]]) {
    expect(buildScopeDiagramModel({...context, packs})).toMatchObject({
      op: legacyOperator,
      scope: 'global',
      reduction: 'histogram'
    });
    expect(buildFormulaCardModel({...context, packs}).formulaId).toBe('archived-dbscan-formula');
  }
  expect(buildScopeDiagramModel({...context, packs: [current, recorded]}).scope).toBe('focal');
  expect(buildFormulaCardModel({...context, packs: [current, recorded]}).formulaId).toBe(
    'dbscan-core-test'
  );
});
