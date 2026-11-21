// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import typescript from 'typescript';
import {expect, it} from 'vitest';

const {collectCatalogIdentities} = createRequire(import.meta.url)(
  '../../../../../scripts/contributor-contract-catalog.mjs'
) as {
  collectCatalogIdentities: (
    trees: typescript.SourceFile[],
    resolveModule: (file: string, specifier: string) => string
  ) => Set<string>;
};

it('retains all nine public identities when source-owned route metadata cannot be evaluated', () => {
  const sourcePath = new URL(
    '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts',
    import.meta.url
  );
  const tree = typescript.createSourceFile(
    sourcePath.pathname,
    readFileSync(sourcePath, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  const callbacks: typescript.ArrowFunction[] = [];
  const routeIdentityDeclarations: typescript.VariableDeclaration[] = [];
  function visit(node: typescript.Node): void {
    if (
      typescript.isArrowFunction(node) &&
      node.parameters[0]?.name.getText(tree) === 'descriptor' &&
      node.getText(tree).includes('OPERATOR_ROUTE_DETAILS')
    )
      callbacks.push(node);
    if (
      typescript.isVariableDeclaration(node) &&
      node.name.getText(tree) === 'ROUTED_OPERATOR_IDENTITIES'
    )
      routeIdentityDeclarations.push(node);
    typescript.forEachChild(node, visit);
  }
  visit(tree);
  const callback = callbacks[0];
  const routeIdentities = routeIdentityDeclarations[0];
  if (!callback || !routeIdentities) throw new Error('missing catalog enrichment declarations');
  // These nine real producer identities are the independent registration expectation from the
  // forwarded checker report. Route metadata may be unknown to a bounded static evaluator;
  // it never changes the identity of the producer being enriched.
  const names = [
    'GPUCellPyramid',
    'GPURankMarkov',
    'GPUGeographicDistribution',
    'GPUNeighborSearch',
    'GPUSpatialInteraction',
    'GPUSpatialJoinPrepared',
    'GPUSpatialPredicateJoin',
    'GPUSpatialWeightsTraces',
    'GPUIslandAttachment'
  ];
  const packageName = '@luma-contributors/analysis/gpu-spatial-analysis';
  const records = names.map(exportName => ({package: packageName, exportName}));
  const fixture = typescript.createSourceFile(
    '/catalog/analysis-operator-catalog.ts',
    `function resolveOperatorDescriptor(descriptor) { return descriptor; }
     const SPATIAL_OPERATION_FORMS = {};
     const PREPARED_TILED_FORM_RELATIONSHIPS = {};
     const OPERATOR_FORM_RELATIONSHIPS = {};
     const RECIPE_OPERATION_FORMS = {};
     const OPERATION_FORM_CONTRACTS = {};
     const OPERATOR_OUTPUT_MODES = {};
     const OPERATOR_ROUTE_DETAILS = {};
     const OPERATOR_ROUTES = {${names
       .map(name => `${JSON.stringify(`${packageName}:${name}`)}: unavailableSourceRoute`)
       .join(',')}};
     const ${routeIdentities.getText(tree)};
     export const ALL_OPERATORS = ${JSON.stringify(records)}.map(${callback.getText(tree)});`,
    typescript.ScriptTarget.Latest,
    true
  );
  expect([...collectCatalogIdentities([fixture], () => '')].sort()).toEqual(
    names.map(name => `${packageName}:${name}`).sort()
  );
});
