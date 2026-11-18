// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {validateWGSLIdentifier} from '../../src/analysis-core/kernels/wgsl-identifiers';
import {GPU_GRID_COORDINATE_WGSL} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  createGridCoordinateFixture,
  createGridCoordinateReaderProbe,
  createRuntimeGridCoordinateDiagnostic,
  getGridCoordinateAgreement,
  getFloat32GridCoordinate,
  GRID_COORDINATE_CASES,
  GRID_COORDINATE_READERS,
  GRID_PROBE_COMPLETED
} from '../utils/grid-coordinate-bump-fixtures';

it('every binding name declared in the grid fixture is a valid WGSL identifier', () => {
  const source = typescript.createSourceFile(
    'grid-coordinate-bump-fixtures.ts',
    readFileSync(new URL('../utils/grid-coordinate-bump-fixtures.ts', import.meta.url), 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  const names: string[] = [];
  function collectNames(node: typescript.Node): void {
    if (typescript.isPropertyAssignment(node) && node.name.getText(source) === 'name') {
      expect(
        typescript.isStringLiteral(node.initializer),
        'fixture binding names must be statically reviewable'
      ).toBe(true);
      if (typescript.isStringLiteral(node.initializer)) names.push(node.initializer.text);
    }
    typescript.forEachChild(node, collectNames);
  }
  function findBindings(node: typescript.Node): void {
    if (typescript.isPropertyAssignment(node) && node.name.getText(source) === 'bindings')
      collectNames(node.initializer);
    else typescript.forEachChild(node, findBindings);
  }
  findBindings(source);
  expect(names.length).toBeGreaterThan(0);
  for (const name of names)
    validateWGSLIdentifier('grid-coordinate-fixtures', 'binding name', name);
});

it.each([
  [180, 180, 180, true, 'reader == builder == model'],
  [0, 0, 180, true, 'reader == builder != model'],
  [180, 0, 180, false, 'reader != builder']
] as const)('overflow agreement distinguishes reader=%s builder=%s model=%s', (reader, builder, model, matches, outcome) => {
  const agreement = getGridCoordinateAgreement(reader, builder, model);
  expect(agreement.matchesBuilder).toBe(matches);
  expect(agreement.outcome).toBe(outcome);
  expect(agreement.message).toBe(
    `${outcome}: reader=${reader}, builder=${builder}, model=${model}`
  );
});

it('the extreme-domain float32 branch remains 180; a flushed runtime reciprocal is a different evaluation', () => {
  const minimum = Math.fround(-3e38);
  const maximum = Math.fround(3e38);
  expect(maximum).toBeGreaterThan(Math.fround(Math.fround(3.402823466e38) + minimum));
  expect(minimum / maximum).toBe(-1);
  expect(maximum / maximum).toBe(1);
  expect(getFloat32GridCoordinate(0, minimum, maximum, 360)).toBe(180);
  const reciprocal = Math.fround(1 / maximum);
  expect(reciprocal).toBeGreaterThan(0);
  expect(reciprocal).toBeLessThan(2 ** -126); // subnormal, vulnerable to reciprocal/flush evaluation
});

it.each(
  GRID_COORDINATE_READERS
)('%s probe uses the same literal bounds and size source as its reader', reader => {
  const device = createNullWebGPUDevice();
  const testCase = GRID_COORDINATE_CASES[4];
  const fixture = createGridCoordinateFixture(device, reader, testCase);
  try {
    const readerSource = getGPUCommandNodeKernelSources(fixture.readerNodes).find(kernel =>
      kernel.source.includes('fn getCoordinate(')
    )!.source;
    const [probe] = getGPUCommandNodeKernelSources([
      createGridCoordinateReaderProbe(fixture, testCase, readerSource),
      createRuntimeGridCoordinateDiagnostic(fixture, testCase, readerSource)
    ]);
    expect(probe.source).toContain(GPU_GRID_COORDINATE_WGSL);
    expect(probe.source).toContain('const DOMAIN_MINIMUM: f32 = -3.0000000054977558e+38');
    expect(probe.source).toContain('const DOMAIN_MAXIMUM: f32 = 3.0000000054977558e+38');
    expect(probe.source).toContain(
      'getCoordinate(positions[positionsOffset], DOMAIN_MINIMUM, DOMAIN_MAXIMUM, columns)'
    );
    expect(probe.source).not.toContain('parameters[');
    expect(probe.source).toContain(`coordinate[coordinateOffset + 1u] = ${GRID_PROBE_COMPLETED}u`);
    if (reader === 'flow')
      expect(probe.source).toContain('clamp(activeGridSize[activeGridSizeOffset], 1u, 361u)');
    else expect(probe.source).toContain('let columns = GRID_COLUMNS');
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
