// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import * as h3 from 'h3-js';

const sourcePath = fileURLToPath(new URL('../src/scenes/cells/cell-neighborhoods-geography.ts', import.meta.url));
const compiled = await build({entryPoints: [sourcePath], bundle: true, packages: 'external', platform: 'node', format: 'cjs', write: false});
const helperModule = {exports: {}};
new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(createRequire(import.meta.url), helperModule, helperModule.exports);
const helpers = helperModule.exports;

// GPU reference capture: ORD → MDW at resolution 4. The production helpers receive these
// exact bare-hex identities as low-word/high-word uint32 pairs.
const firstCell = '8427593ffffffff';
const secondCell = '842664dffffffff';
const packKeys = keys => {
  const words = new Uint32Array(keys.length * 2);
  keys.forEach((key, row) => {
    const value = BigInt(`0x${key}`);
    words[row * 2] = Number(value & 0xffffffffn);
    words[row * 2 + 1] = Number(value >> 32n);
  });
  return words;
};

const firstCellVertices = h3.cellToVertexes(firstCell);
const secondCellVertices = h3.cellToVertexes(secondCell);
const gpuVertexWords = new Uint32Array(12 * 2);
gpuVertexWords.set(packKeys(firstCellVertices), 0);
gpuVertexWords.set(packKeys(secondCellVertices), 12);
const sharedKeys = helpers.sharedVertexKeys(gpuVertexWords, 0, 1);
const expectedSharedKeys = firstCellVertices.filter(vertex => secondCellVertices.includes(vertex)).sort();
assert.deepEqual(sharedKeys, expectedSharedKeys);
assert.equal(sharedKeys.length, 2);
const decodedPoints = helpers.makeVertexPoints(sharedKeys);
assert.equal(decodedPoints.length, 2);
for (let index = 0; index < sharedKeys.length; index++) {
  const [expectedLatitude, expectedLongitude] = h3.vertexToLatLng(expectedSharedKeys[index]);
  const [longitude, latitude] = decodedPoints[index];
  assert.ok(Math.abs(latitude - expectedLatitude) < 1e-7 && Math.abs(longitude - expectedLongitude) < 1e-7);
  assert.ok(longitude > -88 && longitude < -87 && latitude > 41 && latitude < 43, 'decoded ORD/MDW vertex must be in Chicago, not Finland');
}

const firstEdge = h3.cellsToDirectedEdge(firstCell, secondCell);
const reverseEdge = h3.cellsToDirectedEdge(secondCell, firstCell);
const gpuEdgeWords = new Uint32Array(12 * 2);
gpuEdgeWords.set(packKeys([firstEdge]), 0);
gpuEdgeWords.set(packKeys([reverseEdge]), 12);
const edgePair = helpers.findOpposedEdgePair(gpuEdgeWords, firstCell, secondCell);
assert.equal(edgePair, `0x${firstEdge} ↔ 0x${reverseEdge}`);
assert.equal(h3.getDirectedEdgeOrigin(firstEdge), firstCell);
assert.equal(h3.getDirectedEdgeDestination(firstEdge), secondCell);
assert.equal(h3.getDirectedEdgeOrigin(reverseEdge), secondCell);
assert.equal(h3.getDirectedEdgeDestination(reverseEdge), firstCell);

console.log('Cell display-helper smoke check passed: GPU-format keys decode to ORD/MDW boundary vertices and opposed directed edges.');
