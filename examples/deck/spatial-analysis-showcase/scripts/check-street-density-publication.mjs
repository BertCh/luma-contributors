// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {transform} from 'esbuild';
import typescript from 'typescript';

// Execute the actual production summary callback with its external lifecycle services mocked.
// This verifies publication timing without a device; the companion browser check verifies pixels.
const source = await readFile(new URL('../src/scenes/points/street-density.compute.ts', import.meta.url), 'utf8');
const parsed = typescript.createSourceFile('street-density.compute.ts', source, typescript.ScriptTarget.Latest, true);
let callbackSource;
function visitNode(node) {
  if (typescript.isFunctionDeclaration(node) && node.name?.text === 'handlePolygonSummary') callbackSource = node.getText(parsed);
  typescript.forEachChild(node, visitNode);
}
visitNode(parsed);
assert.ok(callbackSource);
const compiled = await transform(callbackSource, {loader: 'ts', format: 'cjs'});
const cases = [
  {name: 'active first summary', view: 'polygons', polygonSet: 'areas', destroyed: false, expected: 1},
  {name: 'inactive polygon set', view: 'polygons', polygonSet: 'tracts', destroyed: false, expected: 0},
  {name: 'density view', view: 'density', polygonSet: 'areas', destroyed: false, expected: 0},
  {name: 'destroyed scene', view: 'polygons', polygonSet: 'areas', destroyed: true, expected: 0}
];
for (const scenario of cases) {
  let publications = 0;
  let redraws = 0;
  const state = {featureCount: 1, values: {}, ready: false, inside: 0};
  const bytes = new ArrayBuffer(24);
  new Float32Array(bytes).set([2000, 3, 4, 5]);
  new Uint32Array(bytes)[4] = 2;
  const context = {options: scenario, requestLayers: () => {assert.equal(state.ready, true); redraws++;}};
  const invoke = new Function('ctx', 'destroyed', 'NO_DATA_VALUE', 'publishPolygons', `${compiled.code}\nreturn handlePolygonSummary;`);
  invoke(context, scenario.destroyed, 0xffffffff, () => {publications++;})('areas', state, bytes);
  assert.equal(publications, scenario.expected, scenario.name);
  assert.equal(redraws, scenario.expected, scenario.name);
  if (!scenario.destroyed) {
    assert.equal(state.values.length[0], 2);
    assert.equal(state.inside, 2000);
  }
}
console.log(`Street polygon publication: ${cases.length} lifecycle cases passed`);
