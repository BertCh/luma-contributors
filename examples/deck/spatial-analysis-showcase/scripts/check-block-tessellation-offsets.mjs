// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import typescript from 'typescript';

const sourcePath = new URL('../src/scenes/geometry/block-relationships.compute.ts', import.meta.url);
const sourceText = await readFile(sourcePath, 'utf8');
const sourceFile = typescript.createSourceFile(sourcePath.pathname, sourceText, typescript.ScriptTarget.Latest, true);
const requiredNames = new Set(['TESS_RING_CAPACITY', 'TESS_VERTEX_CAPACITY']);
const declarations = sourceFile.statements
  .filter(typescript.isVariableStatement)
  .filter(statement =>
    statement.declarationList.declarations.some(
      declaration => typescript.isIdentifier(declaration.name) && requiredNames.has(declaration.name.text)
    )
  )
  .map(statement => statement.getText(sourceFile));
const validator = sourceFile.statements.find(
  statement => typescript.isFunctionDeclaration(statement) && statement.name?.text === 'isValidTessellationGeometry'
);
assert.equal(declarations.length, requiredNames.size, 'Read the production validator capacities');
assert.ok(validator, 'Find the production tessellation geometry validator');

async function loadValidator(functionSource) {
  const extractedSource = `${declarations.join('\n')}\n${functionSource}\nexport {isValidTessellationGeometry};`;
  const javascript = typescript.transpileModule(extractedSource, {
    compilerOptions: {module: typescript.ModuleKind.ESNext, target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(javascript).toString('base64')}`;
  return (await import(moduleUrl)).isValidTessellationGeometry;
}

const isValidTessellationGeometry = await loadValidator(validator.getText(sourceFile));

const validMultiRingPositions = Float32Array.from([
  0, 0, 10, 0, 10, 10, 0, 10,
  2, 2, 2, 8, 8, 8, 8, 2
]);
const featureOffsets = Uint32Array.of(0, 1);
const polygonOffsets = Uint32Array.of(0, 2);
const ringOffsets = Uint32Array.of(0, 4, 8);
assert.equal(
  isValidTessellationGeometry(
    validMultiRingPositions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    1,
    2
  ),
  true,
  'A valid shell and hole use a nonempty occupied vertex prefix'
);
assert.equal(
  isValidTessellationGeometry(
    validMultiRingPositions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    1,
    1
  ),
  false,
  'The reported required ring count must match the occupied topology'
);

const staleCursorFunction = validator
  .getText(sourceFile)
  .replace(
    '\n  previousOffset = 0;\n  for (let ring = 0; ring < ringCount; ring++) {',
    '\n  for (let ring = 0; ring < ringCount; ring++) {'
  );
assert.notEqual(staleCursorFunction, validator.getText(sourceFile), 'Locate the ring cursor reset');
const isValidWithStaleCursor = await loadValidator(staleCursorFunction);
assert.equal(
  isValidWithStaleCursor(
    validMultiRingPositions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    1,
    2
  ),
  false,
  'The multi-ring fixture reproduces rejection when the prior cursor is not reset'
);

assert.equal(
  isValidTessellationGeometry(
    validMultiRingPositions,
    featureOffsets,
    polygonOffsets,
    Uint32Array.of(0, 4, 3),
    1,
    2
  ),
  false,
  'Decreasing ring offsets are rejected'
);
assert.equal(
  isValidTessellationGeometry(
    validMultiRingPositions,
    featureOffsets,
    polygonOffsets,
    Uint32Array.of(0, 2, 8),
    1,
    2
  ),
  false,
  'Rings with fewer than three vertices are rejected'
);
const nonfinitePositions = validMultiRingPositions.slice();
nonfinitePositions[10] = Number.NaN;
assert.equal(
  isValidTessellationGeometry(
    nonfinitePositions,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    1,
    2
  ),
  false,
  'Nonfinite occupied coordinates are rejected'
);

console.log('Checked production tessellation geometry validator against valid and malformed offsets.');
