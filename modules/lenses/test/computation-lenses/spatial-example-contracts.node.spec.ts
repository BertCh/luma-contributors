// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPUStreamlinesParameterValues,
  getGPUParticleAdvectionParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {
  computeNearestFeatureJoinOnCPU,
  getGPUDotDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {runExample as clustering} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-clustering.example';
import {runExample as outline} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/cluster-and-outline-recipe.example';
import {runExample as geometry} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/group-geometry.example';
import {runExample as hull} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/group-convex-hull.example';

it.each([
  ['clustering', clustering],
  ['outline', outline],
  ['geometry', geometry],
  ['hull', hull]
] as const)('%s forwards JavaScript typed bounds to the central packer', async (_name, run) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph<undefined>(device);
  const writes = vi.spyOn(GPUParameterBuffer.prototype, 'write');
  try {
    await run(graph, {
      observations: createTransientView(graph, 'observations', 'float32x2', 4),
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      // JavaScript callers can supply an array view even though the TS example names a tuple.
      bounds: new Float32Array([100, 200, 120, 220]) as unknown as readonly [
        number,
        number,
        number,
        number
      ]
    });
    const packed = writes.mock.calls.map(call => Array.from(call[0]));
    expect(packed).toContainEqual([100, 200, 120, 220, 250, 6, 0, 0]);
  } finally {
    writes.mockRestore();
    device.destroy();
  }
});

it('preserves signed field, viewport and seed-grid placement steps', () => {
  const fieldPlacement = {origin: [100, 220], cellSize: [10, -10]} as const;
  const outputPlacement = {origin: [120, 220], cellSize: [-2, -2]} as const;
  // Outer corners and signed steps are fixed literal placements, not min/max bounds.
  expect(
    Array.from(
      getGPUParticleAdvectionParameterValues({fieldPlacement, timeStep: 1}, [2, 2]).slice(0, 4)
    )
  ).toEqual([100, 220, 10, -10]);
  expect(
    Array.from(
      getGPULineIntegralConvolutionParameterValues({fieldPlacement, outputPlacement}).slice(0, 8)
    )
  ).toEqual([100, 220, 10, -10, 120, 220, -2, -2]);
  expect(
    Array.from(
      getGPULineIntegralConvolutionParameterValues({
        fieldPlacement,
        outputPlacement,
        localOrigin: outputPlacement.origin
      }).slice(0, 8)
    )
  ).toEqual([-20, 0, 10, -10, 0, 0, -2, -2]);
  expect(
    Array.from(
      getGPUStreamlinesParameterValues({
        fieldPlacement,
        gridPlacement: {origin: [120, 220], cellSize: [2, 2]},
        stepLength: 1
      })
    )
  ).toEqual([100, 220, 10, -10, 120, 220, 2, 2, 0.5, 0.5, 1, 0]);
});

it('nearest feature tie chooses source row rather than caller ID', () => {
  // Both distances are exactly 1; row zero carries the larger caller identifier.
  const result = computeNearestFeatureJoinOnCPU(
    {kind: 'points', positions: Float32Array.of(0, 0)},
    {kind: 'points', positions: Float32Array.of(-1, 0, 1, 0), sourceIds: Uint32Array.of(99, 1)}
  );
  expect(Array.from(result.neighborRows)).toEqual([0]);
  expect(Array.from(result.neighborIds)).toEqual([99]);
  expect(Array.from(result.neighborDistances)).toEqual([1]);
});

it('NAM-03 new tie controls, defaults and lens choices use lowest-row', () => {
  for (const path of [
    'examples/deck/spatial-analysis-explorer/modes/nearest-join-mode.ts',
    'examples/deck/spatial-analysis-showcase/src/scenes/joins/nearest-facility.scene.ts',
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/nearest-feature-join.lens.ts',
    'modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/joins.ts'
  ]) {
    const source = readFileSync(path, 'utf8');
    expect(source.includes("'lowest-row'"), path).toBe(true);
    expect(/(?:value:|default:|ties:|alternatives:)\s*(?:\[)?'lowest-id'/.test(source), path).toBe(
      false
    );
  }
});

it('NAM-04 dot-density consumers author named mask placement in every parameter write', () => {
  for (const path of [
    'examples/deck/spatial-analysis-explorer/modes/dot-density-mode.ts',
    'examples/deck/spatial-analysis-showcase/src/scenes/interpolation/dot-density.compute.ts',
    'examples/deck/spatial-analysis-showcase/src/scenes/interpolation/dot-density.scene.ts'
  ]) {
    const source = readFileSync(path, 'utf8');
    expect(source.includes('maskPlacement'), path).toBe(true);
    expect(source.includes('maskExtent'), path).toBe(false);
    const calls = [...source.matchAll(/getGPUDotDensityParameterValues\(\s*\{([^}]*)\}/g)];
    expect(calls.length, path).toBeGreaterThan(0);
    for (const call of calls) {
      const parsed = typescript.createSourceFile(
        'packet.ts',
        `const settings = {${call[1]}};`,
        typescript.ScriptTarget.Latest,
        true
      );
      const statement = parsed.statements[0];
      if (!typescript.isVariableStatement(statement))
        throw new Error('expected settings declaration');
      const initializer = statement.declarationList.declarations[0].initializer;
      if (!initializer || !typescript.isObjectLiteralExpression(initializer))
        throw new Error('expected settings object');
      expect(
        initializer.properties.some(property => property.name?.getText(parsed) === 'maskPlacement'),
        `${path}: ${call[0]}`
      ).toBe(true);
    }
    expect(
      calls.some(call => /seed\s*:\s*(?:randomSeed|seed)\s*\+\s*1/.test(call[1])),
      `${path} seed+1`
    ).toBe(true);
  }
});

it('S1 displayed mask placement retains tuple literals', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/interpolation/dot-density.scene.ts',
    'utf8'
  );
  expect(source).toMatch(/const maskPlacement = \{origin: .*cellSize: .*\} as const;/);
});

it('B6 displayed nearest snippet uses the canonical tie spelling', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/joins/nearest-facility.scene.ts',
    'utf8'
  );
  const start = source.indexOf('snippet: state =>');
  const end = source.indexOf('\n  about:', start);
  const expression = source
    .slice(start + 'snippet: '.length, end)
    .trim()
    .replace(/,$/, '');
  const render = new Function(`return (${expression});`)();
  const snippet = render({
    mode: 'neighbors',
    ties: 'lowest-row',
    featureSet: 'hospital',
    spatialSort: true
  });
  expect(snippet).toContain("ties: 'lowest-row'");
  expect(snippet).not.toContain("ties: 'lowest-id'");
});

it('B7 GPU mask fixture includes disabled-mask refusal rather than only accepted candidates', () => {
  const source = readFileSync(
    'modules/lenses/test/computation-lenses/spatial-example-aliases.spec.ts',
    'utf8'
  );
  const negative = source.indexOf('maskWeights.buffer.write(new Float32Array(6))');
  expect(negative).toBeGreaterThan(0);
  expect(source.slice(negative)).toContain(
    'expect(await readUint32(failed.buffer, 1)).toEqual([8])'
  );
});

it('NAM-04 named mask placement selects the southward source row', () => {
  // Asymmetric bounds [100,140] x [170,230], 2 columns x 3 rows; row 0 is north.
  // Point [130,180] lies in column 1, row 2, hence source mask row 2*2+1 = 5.
  const maskPlacement = {origin: [100, 230], cellSize: [20, -20]} as const;
  for (const seed of [7, 8]) {
    const named = getGPUDotDensityParameterValues({seed, dotsPerUnit: 0.5, maskPlacement});
    const packed = new Float32Array(named.buffer, named.byteOffset, named.length);
    expect(Array.from(packed.slice(2, 6))).toEqual([100, 230, 20, -20]);
    const column = Math.floor((130 - packed[2]) / packed[4]);
    const row = Math.floor((180 - packed[3]) / packed[5]);
    expect(row * 2 + column).toBe(5);
  }
  const legacySettings = {seed: 7, maskPlacement, maskExtent: [100, 170, 20, 20]};
  expect(() => getGPUDotDensityParameterValues(legacySettings)).toThrow(
    /maskExtent.*maskPlacement/
  );
});
