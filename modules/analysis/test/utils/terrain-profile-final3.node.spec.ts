// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {expect, test} from 'vitest';
import {computeProfilePeaksOnCPU} from '../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';

type ProfileChart = {
  series: {kind: string; x: number[]; y: ArrayLike<number>; radius?: number[]}[];
  caption: string;
};

function getPeakDescriptions(path: string): string[] {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const file = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true);
  const literal = (object: ts.ObjectLiteralExpression, name: string): string | undefined => {
    const field = object.properties.find(
      property => ts.isPropertyAssignment(property) && property.name.getText(file) === name
    );
    return field && ts.isPropertyAssignment(field) && ts.isStringLiteralLike(field.initializer)
      ? field.initializer.text
      : undefined;
  };
  let root: ts.Node = file;
  if (path.includes('/catalog/')) {
    let entry: ts.ObjectLiteralExpression | undefined;
    const find = (node: ts.Node) => {
      if (ts.isObjectLiteralExpression(node) && literal(node, 'id') === 'profile-peaks')
        entry = node;
      ts.forEachChild(node, find);
    };
    find(file);
    expect(entry).toBeDefined();
    root = entry!;
  }
  const descriptions: string[] = [];
  const visit = (node: ts.Node) => {
    if (
      path.includes('/catalog/') &&
      ts.isObjectLiteralExpression(node) &&
      ['prominence', 'peakMask'].includes(literal(node, 'name') ?? '')
    )
      descriptions.push(literal(node, 'description')!);
    if (
      !path.includes('/catalog/') &&
      ts.isCallExpression(node) &&
      node.arguments[0] &&
      ts.isStringLiteralLike(node.arguments[0]) &&
      ['profilePeakProminence', 'profilePeakMask'].includes(node.arguments[0].text) &&
      node.arguments[2] &&
      ts.isObjectLiteralExpression(node.arguments[2])
    )
      descriptions.push(literal(node.arguments[2], 'meaning')!);
    ts.forEachChild(node, visit);
  };
  visit(root);
  expect(descriptions).toHaveLength(2);
  return descriptions;
}

/** Execute the authored callback with its chart/buffer bindings, without a GPU or DOM. */
function createProfileCallback(length: number) {
  const source = readFileSync(
    new URL(
      '../../../../examples/deck/spatial-analysis-explorer/modes/terrain-features-mode.ts',
      import.meta.url
    ),
    'utf8'
  );
  const file = ts.createSourceFile(
    'terrain-features-mode.ts',
    source,
    ts.ScriptTarget.Latest,
    true
  );
  let callback: ts.FunctionDeclaration | undefined;
  const visit = (node: ts.Node) => {
    if (ts.isFunctionDeclaration(node) && node.name?.text === 'updateProfile') callback = node;
    ts.forEachChild(node, visit);
  };
  visit(file);
  expect(callback).toBeDefined();
  const javascript = ts.transpileModule(callback!.getText(file), {
    compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None}
  }).outputText;
  let chart: ProfileChart | undefined,
    readout = '',
    markers = new Float32Array();
  const bindings = {
    PROFILE_PEAK_CAPACITY: 64,
    profileStart: [0, 0],
    profileEnd: [length - 1, 0],
    cell: 1,
    profileChart: {
      update: (value: ProfileChart) => {
        chart = value;
      }
    },
    profilePeakMarkers: {
      write: (values: Float32Array) => {
        markers = values.slice();
      }
    },
    readouts: {
      profile: {
        setValue: (value: string) => {
          readout = value;
        }
      }
    },
    context: {updateLayers: () => {}}
  };
  const callbackState = new Function(
    'bindings',
    `const {PROFILE_PEAK_CAPACITY, profileStart, profileEnd, cell, profileChart, profilePeakMarkers, readouts, context} = bindings; let profileSamples; let profilePeakCount = 0; ${javascript}; return {updateProfile, markerCount: () => profilePeakCount};`
  )(bindings) as {
    updateProfile: (
      count: number,
      converged: boolean,
      values: Float32Array,
      prominence: Float32Array,
      indices: Float32Array,
      heights: Float32Array,
      mask: Uint32Array
    ) => void;
    markerCount: () => number;
  };
  return {...callbackState, chart: () => chart!, readout: () => readout, markers: () => markers};
}

test.each([
  [96, 4, 8, true],
  [350, 5, 69, false]
] as const)('final3 H1 explorer counts kept peaks independently of marker capacity (%s samples)', (length, spacing, expected, incomplete) => {
  const values = Float32Array.from({length}, (_, index) =>
    index % spacing === 0 ? (length === 96 ? index : 100) : 0
  );
  const result = computeProfilePeaksOnCPU(
    values,
    new Uint32Array([0, length]),
    new Float32Array([5]),
    {window: 4, suppressionRadius: 4, maximumSuppressionIterations: 16, capacity: length}
  );
  expect(result.count[0]).toBe(expected);
  const callback = createProfileCallback(length);
  callback.updateProfile(
    length,
    Boolean(result.converged[0]),
    values,
    result.prominence,
    result.refinedIndex,
    result.refinedValue,
    result.peakMask
  );
  expect(callback.chart().caption).toContain(`${expected} peaks`);
  expect(callback.readout()).toContain(`${expected} peaks`);
  expect(callback.chart().caption.includes('incomplete')).toBe(incomplete);
  expect(callback.readout().includes('incomplete')).toBe(incomplete);
  const peaks = callback.chart().series.find(series => series.kind === 'points')!;
  expect(peaks.x).toHaveLength(expected);
  expect(Array.from(peaks.y)).toHaveLength(expected);
  expect(peaks.radius).toHaveLength(expected);
  for (const value of [...peaks.x, ...Array.from(peaks.y), ...peaks.radius!])
    expect(Number.isFinite(value)).toBe(true);
  expect(callback.markerCount()).toBe(Math.min(64, expected));
  expect(callback.markers()).toHaveLength(Math.min(64, expected) * 2);
  for (const value of callback.markers()) expect(Number.isFinite(value)).toBe(true);
});

test('final3 H4 authored peak descriptions cover all producer states', () => {
  const values = Float32Array.from({length: 96}, (_, index) => (index % 4 === 0 ? index : 0));
  const result = computeProfilePeaksOnCPU(values, new Uint32Array([0, 96]), new Float32Array([5]), {
    window: 4,
    suppressionRadius: 4,
    maximumSuppressionIterations: 16,
    capacity: 96
  });
  expect(Array.from(result.peakMask).filter(mask => mask === 1)).toHaveLength(8);
  expect(Array.from(result.peakMask).filter(mask => mask === 0xffffffff)).toHaveLength(5);
  for (let row = 0; row < values.length; row++) {
    if (result.peakMask[row] === 1) expect(Number.isFinite(result.prominence[row])).toBe(true);
    else if (result.peakMask[row] === 0xffffffff) expect(result.prominence[row]).toBe(Infinity);
    else expect(result.prominence[row]).toBeNaN();
  }
  for (const path of [
    '../../../../website/src/catalog/entries/terrain-relief-light.ts',
    '../../../lenses/src/computation-lenses/scenarios/gpu-terrain/profile-peaks.lens.ts'
  ]) {
    for (const description of getPeakDescriptions(path)) {
      expect(description).toMatch(/1 (?:= |means )?kept/);
      expect(description).toMatch(/0 (?:= |means )?absent/);
      expect(description).toMatch(/0xffffffff (?:= |means )?undecided/);
      expect(description).toContain('+Infinity');
    }
  }
});
