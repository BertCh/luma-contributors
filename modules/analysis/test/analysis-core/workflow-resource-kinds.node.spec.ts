// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';
import typeScript from 'typescript';
import {expect, it} from 'vitest';
import {
  formatCount,
  formatResourceBytes,
  formatResourceCount
} from '../../../../website/src/lens/format';
import type {WorkflowRecord} from '../../../../website/src/components/meta/pages/workflows/workflows';

function loadWorkflowRecord(bytesKind: string | undefined) {
  const file = fileURLToPath(
    new URL('../../../../website/src/components/meta/pages/workflows/workflows.ts', import.meta.url)
  );
  const source = typeScript.transpileModule(readFileSync(file, 'utf8'), {
    compilerOptions: {
      module: typeScript.ModuleKind.CommonJS,
      target: typeScript.ScriptTarget.ES2022
    }
  }).outputText;
  const exports = {} as {
    LOCAL_MORAN_RECORD: WorkflowRecord;
  };
  runInNewContext(source, {
    exports,
    require: (specifier: string) => {
      if (specifier.endsWith('.json'))
        return {
          recordedEdgeCount: 3,
          recordedNodeCount: 2,
          nodes: [
            {kind: 'compute', count: 1},
            {kind: 'copy', count: 1}
          ],
          title: 'fixture',
          resources: {count: 2, totalBytes: 16, inventoryKnown: true, totalBytesKind: bytesKind}
        };
      return {KEPLER_CURATED_ANALYSES: []};
    }
  });
  return exports.LOCAL_MORAN_RECORD;
}

it('decodes workflow JSON byte-kind strings and conservatively labels unknown precision', () => {
  for (const kind of [
    'logical',
    'lower-bound',
    'estimated',
    'unspecified',
    'unavailable',
    undefined
  ]) {
    const record = loadWorkflowRecord(kind);
    expect(record.bytesKind).toBe(kind);
    expect(record.bytes).toBe(16);
    expect(record.inventoryKnown).toBe(true);
  }
  expect(loadWorkflowRecord('future-kind').bytesKind).toBe('unspecified');
});

it('X2 labels one compute and one copy as two graph nodes', () => {
  const file = fileURLToPath(
    new URL('../../../../website/src/components/meta/pages/workflows/index.tsx', import.meta.url)
  );
  const text = readFileSync(file, 'utf8');
  const source = typeScript.createSourceFile(
    file,
    text,
    typeScript.ScriptTarget.Latest,
    true,
    typeScript.ScriptKind.TSX
  );
  const formatter = source.statements.find(
    statement =>
      typeScript.isFunctionDeclaration(statement) && statement.name?.text === 'describeRecord'
  );
  if (!formatter) throw new Error('workflow page must retain its record formatter');
  const compiled = typeScript.transpileModule(formatter.getText(source), {
    compilerOptions: {target: typeScript.ScriptTarget.ES2022}
  }).outputText;
  const description = runInNewContext(`${compiled}\ndescribeRecord(record);`, {
    record: loadWorkflowRecord('logical'),
    formatCount,
    formatResourceBytes,
    formatResourceCount
  });
  expect(description).toBe('2 graph nodes, 3 dependencies, 2 buffers of 16 B');
  expect(text).toContain("label: 'graph nodes they compiled to'");
});
