// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {Float32CoveringPredicate} from './s2-covering-float32-predicate';

it.each([
  false,
  true
])('exact packed vertex survives f32 plane rounding (reciprocal=%s)', reciprocal => {
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains.ts',
      import.meta.url
    ),
    'utf8'
  );
  const getBody = (name: string) => {
    const body = source.match(
      new RegExp(`fn ${name}\\([^]*?\\) -> (?:vec3f|bool) \\{([^]*?)\\n\\}`)
    )?.[1];
    expect(body, name).toBeDefined();
    return body!;
  };
  // Execute the actual control flow, with explicit f32 vector operations. The general
  // WGSL interpreter retains enough intermediate precision to mask this device case.
  const edgeBody = getBody('polygonEdgeNormal').replace(
    'secondDirection - select(-firstDirection, firstDirection, dot(firstDirection, secondDirection) >= 0.0)',
    'subtract(secondDirection, select(negate(firstDirection), firstDirection, dot(firstDirection, secondDirection) >= 0.0))'
  );
  const arithmetic = new Float32CoveringPredicate(reciprocal, true);
  const round = Math.fround;
  const subtract = (first: number[], second: number[]) =>
    first.map((value, index) => round(value - second[index]));
  const onArc: (first: number[], second: number[], point: number[], margin: number) => boolean =
    new Function(
      'normalize',
      'cross',
      'dot',
      'select',
      'subtract',
      'negate',
      'abs',
      'sin',
      'min',
      'distance',
      `function polygonEdgeNormal(first,second){${edgeBody}}
return function(first,second,point,margin){${getBody('polygonOnArc')}};`
    )(
      arithmetic.normalize.bind(arithmetic),
      arithmetic.cross.bind(arithmetic),
      arithmetic.dot.bind(arithmetic),
      (first: number[], second: number[], condition: boolean) => (condition ? second : first),
      subtract,
      (point: number[]) => point.map(value => -value),
      Math.abs,
      (angle: number) => round(Math.sin(angle)),
      Math.min,
      (first: number[], second: number[]) => arithmetic.length(subtract(first, second))
    );
  const vertices = [
    [3, -11, 9],
    [-13, -9, 14],
    [-12, -9, -2]
  ].map(point => point.map(value => round(value / Math.hypot(...point))));
  const query = [...vertices[0]];
  // Identical packed endpoint/query bytes imply zero endpoint chord distance. Both
  // incident planes may nevertheless round to 2^-23 = 1.1920928955078125e-7 > 1e-7.
  for (const [first, second] of [
    [vertices[0], vertices[1]],
    [vertices[2], vertices[0]]
  ])
    expect(onArc(first, second, query, round(1e-7))).toBe(true);
});
