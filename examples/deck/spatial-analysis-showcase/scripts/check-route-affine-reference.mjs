// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const validationUrl = new URL('../../../../dev-docs/research/showcase-overhaul-2026-10-25/cta-geometry-validation.json', import.meta.url);
const validation = JSON.parse(await readFile(validationUrl, 'utf8'));
const source = validation.inputs.source;
const translated = new Float32Array(source.length * 2);
const recovered = new Float32Array(source.length * 2);
const eastShift = 64;
const northShift = -32;
let sourceLength = 0;
let translatedLength = 0;
let maximumRecoveryError = 0;

for (let row = 0; row < source.length; row++) {
  translated[row * 2] = Math.fround(source[row][0] + eastShift);
  translated[row * 2 + 1] = Math.fround(source[row][1] + northShift);
  recovered[row * 2] = Math.fround(translated[row * 2] - eastShift);
  recovered[row * 2 + 1] = Math.fround(translated[row * 2 + 1] - northShift);
  maximumRecoveryError = Math.max(
    maximumRecoveryError,
    Math.hypot(recovered[row * 2] - source[row][0], recovered[row * 2 + 1] - source[row][1])
  );
  if (row > 0) {
    sourceLength += Math.hypot(source[row][0] - source[row - 1][0], source[row][1] - source[row - 1][1]);
    translatedLength += Math.hypot(
      translated[row * 2] - translated[(row - 1) * 2],
      translated[row * 2 + 1] - translated[(row - 1) * 2 + 1]
    );
  }
}

assert.equal(translated.length, source.length * 2);
assert.equal(recovered.length, source.length * 2);
assert.ok(maximumRecoveryError < 0.0001, `inverse error too large: ${maximumRecoveryError}`);
assert.ok(Math.abs(sourceLength - validation.expected.sourceLengthMeters) < 0.01);
assert.ok(Math.abs(translatedLength - sourceLength) < 0.001);

console.log(JSON.stringify({
  translationMeters: [eastShift, northShift],
  sourceRows: source.length,
  sourceLengthMeters: sourceLength,
  translatedLengthMeters: translatedLength,
  maximumFloat32RecoveryErrorMeters: maximumRecoveryError
}, null, 2));
