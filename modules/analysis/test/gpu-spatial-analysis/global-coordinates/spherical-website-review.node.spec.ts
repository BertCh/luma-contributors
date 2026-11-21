// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import * as typescript from 'typescript';
import {expect, it} from 'vitest';
import {
  getSphericalOverlayStatusLabel,
  selectSphericalOverlayLayers
} from '../../../../../website/src/gpu/runs/entries/cells-index-join/spherical-polygon-overlay-status';

// Execute the actual website callbacks without loading its unrelated DOM/rendering framework.
// Only slider snapping is stubbed; status, draw selection and readout code are production code.
function getWebsiteCallbacks() {
  const source = readFileSync(
    new URL(
      '../../../../../website/src/gpu/runs/entries/cells-index-join/spherical-polygon-overlay.ts',
      import.meta.url
    ),
    'utf8'
  );
  const syntax = typescript.createSourceFile(
    'website.ts',
    source,
    typescript.ScriptTarget.Latest,
    true
  );
  const callbacks: Record<string, string> = {};
  const visit = (node: typescript.Node) => {
    if (typescript.isPropertyAssignment(node) && node.name.getText(syntax) === 'draw')
      callbacks['draw'] = node.initializer.getText(syntax);
    if (
      typescript.isMethodDeclaration(node) &&
      ['resolve', 'afterRun'].includes(node.name.getText(syntax))
    ) {
      const name = node.name.getText(syntax);
      callbacks[name] = node.getText(syntax).startsWith('async ')
        ? 'async function ' + node.getText(syntax).slice(6)
        : 'function ' + node.getText(syntax);
    }
    typescript.forEachChild(node, visit);
  };
  visit(syntax);
  const script = `const snapToParameter=(_parameter,value)=>value; const LONGITUDE_PARAMETER={},LATITUDE_PARAMETER={},OPERATIONS=['intersection'],FILL_COLUMNS=160,FILL_ROWS=90;
exports.callbacks={draw:${callbacks['draw']},resolve:${callbacks['resolve']},afterRun:${callbacks['afterRun']}};`;
  const code = typescript.transpileModule(script, {
    compilerOptions: {module: typescript.ModuleKind.CommonJS}
  }).outputText;
  const result: {callbacks?: Record<string, (...args: unknown[]) => unknown>} = {};
  new Function('exports', 'getSphericalOverlayStatusLabel', 'selectSphericalOverlayLayers', code)(
    result,
    getSphericalOverlayStatusLabel,
    selectSphericalOverlayLayers
  );
  return result.callbacks!;
}

it.each([
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 0]
])('website consumers honor count=%s overflow=%s candidate=%s', async (count, overflow, candidate) => {
  const functions = getWebsiteCallbacks();
  const scalar = (value: number) => ({
    values: Uint32Array.of(value),
    buffer: {readAsync: async () => new Uint8Array(Uint32Array.of(value).buffer)}
  });
  const outputs = {
    result: {tag: 'result'},
    fill: {tag: 'fill'},
    operandOutlines: [{tag: 'left'}, {tag: 'right'}],
    count: scalar(count),
    overflow: scalar(overflow),
    candidateOverflow: scalar(candidate),
    featureOffsets: {values: Uint32Array.of(0, 1)},
    polygonOffsets: {values: Uint32Array.of(0, 1)}
  };
  const extra = await functions['afterRun']({outputs});
  const layers = functions['draw']({outputs, extra, theme: {ink: [1, 1, 1]}}) as {
    segments?: unknown;
    values?: unknown;
  }[];
  const readout = (
    functions['resolve'](
      {x: 0, y: 0},
      {
        inputs: {},
        outputs,
        values: {operation: 0},
        peek: (value: {values: Uint32Array}) => value.values
      }
    ) as {readout: string}
  ).readout;
  const failure = overflow !== 0 || candidate !== 0;
  expect(
    layers.filter(layer => layer.segments === outputs.result || layer.values === outputs.fill)
  ).toHaveLength(failure ? 0 : 2);
  for (const outline of outputs.operandOutlines)
    expect(layers.some(layer => layer.segments === outline)).toBe(true);
  if (failure) expect(readout).toContain('incomplete');
  else expect(readout).toBe('intersection · 1 ring (0 holes)');
});
