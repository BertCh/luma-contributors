// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {
  CPUPolygonTriangulationOperation,
  computePolygonTriangulationOnCPU,
  type CPUPolygonTriangulationInput
} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {
  packOperationPolygons,
  getTriangulationOperationProps
} from './polygon-triangulation-operation-cases';
import {EARCUT_FIXTURE} from './polygon-triangulation-fixture';
import type {SearchPoint} from './polygon-triangulation-search-cases';
import {checkTriangulation} from './polygon-triangulation-monotone-cases';

it('final2 default triangulation budget matches the published guide', () => {
  const guide = readFileSync('docs/api/gpu-spatial-analysis/operators-geometry.md', 'utf8');
  const match = guide.match(
    /positive uint32 `maximumWork` per polygon \(default\s+([\d,]+) elementary steps\)/
  )!;
  expect(Number(match[1].replaceAll(',', ''))).toBe(GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK);
});

it('final2 Earcut scans vacated hole queue slots at the 215/216 work boundary', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
      ],
      [
        [1, 1],
        [1, 3],
        [3, 3],
        [3, 1]
      ],
      [
        [6, 6],
        [6, 8],
        [8, 8],
        [8, 6]
      ]
    ]
  ]);
  const props = {...getTriangulationOperationProps(input), largePolygonMethod: 'none' as const};
  for (const maximumWork of [215, 216]) {
    const result = computePolygonTriangulationOnCPU(input, {...props, maximumWork});
    expect([...result.valid, ...result.triangleCount], `budget=${maximumWork}`).toEqual(
      maximumWork === 215 ? [0, 0] : [1, 14]
    );
    if (maximumWork === 215)
      expect(Array.from(result.indices)).toEqual(new Array(props.indexCapacity).fill(0));
  }
});

async function executeOperation(
  input: CPUPolygonTriangulationInput,
  props: ConstructorParameters<typeof CPUPolygonTriangulationOperation>[0]
) {
  const vector = (values: Float32Array | Uint32Array, format: 'float32x2' | 'uint32') => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  const operation = new CPUPolygonTriangulationOperation(props);
  props.largePolygonMethod = 'none';
  props.monotoneMaximumVertices = 1;
  return operation.execute({
    inputs: {
      positions: vector(input.positions, 'float32x2'),
      featureOffsets: vector(input.featureOffsets, 'uint32'),
      polygonOffsets: vector(input.polygonOffsets, 'uint32'),
      ringOffsets: vector(input.ringOffsets, 'uint32')
    }
  });
}

it.each([
  'function',
  'operation'
] as const)('final2 CPU %s monotone route rescues wavy_grid_holes within the selected Earcut budget', async form => {
  const fixture = EARCUT_FIXTURE.find(entry => entry.name === 'wavy_grid_holes')!;
  const input = packOperationPolygons([fixture.rings as SearchPoint[][]]);
  const props = {
    ...getTriangulationOperationProps(input),
    maximumWork: 1000,
    largePolygonMethod: 'monotone' as const
  };
  const result =
    form === 'function'
      ? computePolygonTriangulationOnCPU(input, props)
      : await executeOperation(input, props);
  // Monotone keeps collinear boundary vertices: 220 vertices + 2 * 25 holes - 2 = 268.
  expect([
    ...(result['valid'] as Uint32Array),
    ...(result['triangleCount'] as Uint32Array)
  ]).toEqual([1, 268]);
  const indices = result['indices'] as Uint32Array;
  const triangles: [number, number, number][] = Array.from({length: 268}, (_, triangle) => [
    indices[3 * triangle],
    indices[3 * triangle + 1],
    indices[3 * triangle + 2]
  ]);
  const check = checkTriangulation(fixture.rings, triangles);
  expect(check.boundaryMatches).toBe(true);
  expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
});

for (const form of ['function', 'operation'] as const) {
  it.each([
    'square',
    'regular-4096'
  ] as const)(`final3 CPU ${form} preserves a %s shell when layout skips a collinear hole`, async shape => {
    const shell: SearchPoint[] =
      shape === 'square'
        ? [
            [-4, -4],
            [4, -4],
            [4, 4],
            [-4, 4]
          ]
        : Array.from({length: 4096}, (_, vertex) => {
            const angle = (2 * Math.PI * vertex) / 4096;
            return [Math.fround(4 * Math.cos(angle)), Math.fround(4 * Math.sin(angle))];
          });
    const collinearHole: SearchPoint[] = [
      [-1, 0],
      [0, 0],
      [1, 0]
    ];
    const count = shell.length - 2;
    for (const skippedHole of [false, true]) {
      const input = packOperationPolygons([[shell, ...(skippedHole ? [collinearHole] : [])]]);
      const props = {
        ...getTriangulationOperationProps(input),
        ...(shape === 'square' ? {monotoneMinimumVertices: 1, maximumWork: 1} : {})
      };
      const result =
        form === 'function'
          ? computePolygonTriangulationOnCPU(input, props)
          : await executeOperation(input, props);
      expect(Array.from(result['triangleOffsets'] as Uint32Array)).toEqual([0, count]);
      expect(
        [...(result['valid'] as Uint32Array), ...(result['triangleCount'] as Uint32Array)],
        `skippedHole=${skippedHole}`
      ).toEqual([1, count]);
      const indices = result['indices'] as Uint32Array;
      const triangles: [number, number, number][] = Array.from({length: count}, (_, triangle) => [
        indices[3 * triangle],
        indices[3 * triangle + 1],
        indices[3 * triangle + 2]
      ]);
      expect(triangles.flat().every(row => row < shell.length)).toBe(true);
      const check = checkTriangulation([shell], triangles);
      expect(check.boundaryMatches).toBe(true);
      expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
    }
  });

  it(`final3 CPU ${form} retains original rows after skipping a hole before a live hole`, async () => {
    const shell: SearchPoint[] = [
      [-4, -4],
      [4, -4],
      [4, 4],
      [-4, 4]
    ];
    const skipped: SearchPoint[] = [
      [-1, 0],
      [0, 0],
      [1, 0]
    ];
    const hole: SearchPoint[] = [
      [1, -1],
      [1, 1],
      [2, 1],
      [2, -1]
    ];
    const input = packOperationPolygons([[shell, skipped, hole]]);
    const props = {
      ...getTriangulationOperationProps(input),
      monotoneMinimumVertices: 1,
      maximumWork: 1
    };
    const result =
      form === 'function'
        ? computePolygonTriangulationOnCPU(input, props)
        : await executeOperation(input, props);
    expect(Array.from(result['triangleOffsets'] as Uint32Array)).toEqual([0, 8]);
    expect([
      ...(result['valid'] as Uint32Array),
      ...(result['triangleCount'] as Uint32Array)
    ]).toEqual([1, 8]);
    const indices = result['indices'] as Uint32Array;
    const sourceRows = [0, 1, 2, 3, 7, 8, 9, 10];
    const triangles: [number, number, number][] = Array.from({length: 8}, (_, triangle) => [
      indices[3 * triangle],
      indices[3 * triangle + 1],
      indices[3 * triangle + 2]
    ]);
    expect(triangles.flat().every(row => sourceRows.includes(row))).toBe(true);
    const local = triangles.map(
      triangle => triangle.map(row => sourceRows.indexOf(row)) as [number, number, number]
    );
    const check = checkTriangulation([shell, hole], local);
    expect(check.boundaryMatches).toBe(true);
    expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
  });
}

it('final3 skipped rings still consume raw monotone capacity', () => {
  const input = packOperationPolygons([
    [
      [
        [-4, -4],
        [4, -4],
        [4, 4],
        [-4, 4]
      ],
      [
        [-1, 0],
        [0, 0],
        [1, 0]
      ]
    ]
  ]);
  for (const monotoneMaximumVertices of [4, 7]) {
    const result = computePolygonTriangulationOnCPU(input, {
      ...getTriangulationOperationProps(input),
      monotoneMinimumVertices: 1,
      maximumWork: 1,
      monotoneMaximumVertices
    });
    expect([...result.valid, ...result.triangleCount]).toEqual(
      monotoneMaximumVertices === 4 ? [0, 0] : [1, 2]
    );
  }
});

it('final2 the selected monotone threshold takes precedence over a successful Earcut walk', () => {
  const fixture = EARCUT_FIXTURE.find(entry => entry.name === 'wavy_grid_holes')!;
  const input = packOperationPolygons([fixture.rings as SearchPoint[][]]);
  const props = {...getTriangulationOperationProps(input), monotoneMinimumVertices: 1};
  const earcut = computePolygonTriangulationOnCPU(input, {...props, largePolygonMethod: 'none'});
  const monotone = computePolygonTriangulationOnCPU(input, {
    ...props,
    largePolygonMethod: 'monotone'
  });
  expect([...earcut.valid, ...earcut.triangleCount]).toEqual([1, 204]);
  expect([...monotone.valid, ...monotone.triangleCount]).toEqual([1, 268]);
});

it.each([
  'function',
  'operation'
] as const)('final2 CPU %s monotone rescue uses the shared raw-vertex capacity', async form => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
    ],
    [
      [
        [2, 0],
        [3, 0],
        [2, 1]
      ]
    ]
  ]);
  const props = {
    ...getTriangulationOperationProps(input),
    maximumWork: 1,
    largePolygonMethod: 'monotone' as const,
    monotoneMaximumVertices: 3
  };
  const result =
    form === 'function'
      ? computePolygonTriangulationOnCPU(input, props)
      : await executeOperation(input, props);
  expect(Array.from(result['valid'] as Uint32Array)).toEqual([1, 0]);
  expect(Array.from(result['triangleCount'] as Uint32Array)).toEqual([1, 0]);
  const indices = result['indices'] as Uint32Array;
  expect(Array.from(indices.slice(3, 6))).toEqual([3, 3, 3]);
  expect(Array.from(indices.slice(6))).toEqual(new Array(props.indexCapacity - 6).fill(0));
});
