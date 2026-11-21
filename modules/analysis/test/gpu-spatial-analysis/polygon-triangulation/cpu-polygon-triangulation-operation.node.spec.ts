// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  CPUPolygonTriangulationOperation,
  computePolygonTriangulationOnCPU,
  type CPUPolygonTriangulationInput
} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {GPUPolygonTriangulationOperation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {GPUPolygonTriangulation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  OPERATION_POLYGONS,
  createPaddingTriangulationInput,
  CLAMPED_RING_END_CASES,
  createClampedRingEndInput,
  getTriangulationOperationProps,
  packOperationPolygons
} from './polygon-triangulation-operation-cases';
import {
  getTilingDefect,
  getTriangleAreaSum,
  type SearchPoint
} from './polygon-triangulation-search-cases';
import touchingFixture from '../../oracles/fixtures/polygon-triangulation-touching-rings.json';
import {EARCUT_FIXTURE} from './polygon-triangulation-fixture';

for (const form of ['function', 'operation'] as const) {
  it(`final1 CPU triangulation ${form} honors maximumWork on wavy_grid_holes`, async () => {
    const wavy = EARCUT_FIXTURE.find(entry => entry.name === 'wavy_grid_holes')!;
    const input = packOperationPolygons([wavy.rings as SearchPoint[][]]);
    const props = {
      ...getTriangulationOperationProps(input),
      maximumWork: 1000,
      largePolygonMethod: 'none' as const
    };
    const vector = (values: Float32Array | Uint32Array, format: 'float32x2' | 'uint32') => {
      const length = values.length / (format === 'float32x2' ? 2 : 1);
      return {format, length, chunks: [{values, length}], pack: () => values};
    };
    const result =
      form === 'function'
        ? computePolygonTriangulationOnCPU(input, props)
        : await new CPUPolygonTriangulationOperation(props).execute({
            inputs: {
              positions: vector(input.positions, 'float32x2'),
              featureOffsets: vector(input.featureOffsets, 'uint32'),
              polygonOffsets: vector(input.polygonOffsets, 'uint32'),
              ringOffsets: vector(input.ringOffsets, 'uint32')
            }
          });
    expect([
      ...(result['valid'] as Uint32Array),
      ...(result['triangleCount'] as Uint32Array)
    ]).toEqual([0, 0]);
    expect(Array.from(result['indices'] as Uint32Array)).toEqual(
      new Array(props.indexCapacity).fill(0)
    );
  });
}

it('final1 both triangulation operations reject zero maximumWork at construction', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
    ]
  ]);
  const props = {...getTriangulationOperationProps(input), maximumWork: 0};
  for (const Operation of [CPUPolygonTriangulationOperation, GPUPolygonTriangulationOperation]) {
    expect(() => new Operation(props)).toThrow(/maximumWork/);
  }
  expect(() => computePolygonTriangulationOnCPU(input, props)).toThrow(/maximumWork/);
});

it.each([
  1, 2
])('final1 Earcut work budget %s resets for each triangle and discards partial output', maximumWork => {
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
  const result = computePolygonTriangulationOnCPU(input, {
    indexCapacity: 6,
    maximumWork,
    largePolygonMethod: 'none'
  });
  expect(Array.from(result.valid)).toEqual(maximumWork === 1 ? [0, 0] : [1, 1]);
  expect(Array.from(result.triangleCount)).toEqual(maximumWork === 1 ? [0, 0] : [1, 1]);
  if (maximumWork === 1) expect(Array.from(result.indices)).toEqual([0, 0, 0, 3, 3, 3]);
});

it('CORE-08 GPU operation retains optional output ports after caller mutation', () => {
  const context = createKernelAuditContext('triangulation-operation-snapshot');
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
    ]
  ]);
  const props = getTriangulationOperationProps(input, 3);
  const operation = new GPUPolygonTriangulationOperation(props);
  delete props.output.triangleCount;
  delete props.output.triangleOffsets;
  try {
    const views = {
      positions: context.view('float32x2', 3),
      featureOffsets: context.view('uint32', 2),
      polygonOffsets: context.view('uint32', 2),
      ringOffsets: context.view('uint32', 2),
      indices: context.view('uint32', 3),
      valid: context.view('uint32', 1),
      triangleCount: context.view('uint32', 1),
      triangleOffsets: context.view('uint32', 2)
    };
    const contributor = operation.createContributor(
      port => views[port as keyof typeof views] as never
    );
    expect(contributor).toBeInstanceOf(GPUPolygonTriangulation);
    const triangulation = contributor as GPUPolygonTriangulation;
    expect(triangulation.props.output.triangleCount).toBe(views.triangleCount);
    expect(triangulation.props.output.triangleOffsets).toBe(views.triangleOffsets);
    expect(triangulation.props.positions).toBe(views.positions);
    expect(operation.outputs['triangleCount'].value).toBe('triangleCount');
    const sources = getGPUCommandNodeKernelSources(triangulation.getCommandNodes(context.graph));
    expect(
      sources.some(kernel =>
        kernel.source.includes('triangleCount[triangleCountOffset + polygon] = written;')
      )
    ).toBe(true);
  } finally {
    operation.destroy();
    context.device.destroy();
  }
});

it('CORE-08 GPU operation retains getter-backed options and context until lowering', () => {
  const context = createKernelAuditContext('triangulation-operation-options');
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [1, 0],
        [0, 1]
      ]
    ]
  ]);
  const props = {
    ...getTriangulationOperationProps(input, 3),
    maximumWork: 100,
    spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}
  };
  let maximumWork = 100;
  Object.defineProperty(props, 'maximumWork', {enumerable: false, get: () => maximumWork});
  const operation = new GPUPolygonTriangulationOperation(props);
  maximumWork = 0;
  Reflect.set(props.spatialContext, 'coordinateSpace', 'longitude-latitude');
  try {
    const views = {
      positions: context.view('float32x2', 3),
      featureOffsets: context.view('uint32', 2),
      polygonOffsets: context.view('uint32', 2),
      ringOffsets: context.view('uint32', 2),
      indices: context.view('uint32', 3),
      valid: context.view('uint32', 1),
      triangleCount: context.view('uint32', 1),
      triangleOffsets: context.view('uint32', 2)
    };
    const contributor = operation.createContributor(
      port => views[port as keyof typeof views] as never
    ) as GPUPolygonTriangulation;
    expect(contributor.props.maximumWork).toBe(100);
    expect(contributor.spatialContext.coordinateSpace).toBe('planar');
  } finally {
    operation.destroy();
    context.device.destroy();
  }
});

for (const mutation of ['polygonCount', 'indexCapacity', 'output-context'] as const) {
  it(`CORE-08 CPU operation retains ${mutation} after caller mutation`, async () => {
    const input = packOperationPolygons([
      [
        [
          [0, 0],
          [1, 0],
          [0, 1]
        ]
      ]
    ]);
    const props = {
      ...getTriangulationOperationProps(input, 3),
      spatialContext: {coordinateSpace: 'planar' as const, metric: 'native' as const}
    };
    const operation = new CPUPolygonTriangulationOperation(props);
    if (mutation === 'polygonCount') props.polygonCount = 2;
    else if (mutation === 'indexCapacity') props.indexCapacity = 6;
    else {
      delete props.output.triangleCount;
      props.output.indices = 'changed-indices';
      Reflect.set(props.spatialContext, 'coordinateSpace', 'longitude-latitude');
    }
    const vector = (values: Float32Array | Uint32Array, format: 'float32x2' | 'uint32') => {
      const length = values.length / (format === 'float32x2' ? 2 : 1);
      return {format, length, chunks: [{values, length}], pack: () => values};
    };
    const result = await operation.execute({
      inputs: {
        positions: vector(input.positions, 'float32x2'),
        featureOffsets: vector(input.featureOffsets, 'uint32'),
        polygonOffsets: vector(input.polygonOffsets, 'uint32'),
        ringOffsets: vector(input.ringOffsets, 'uint32')
      }
    });
    expect(result['triangleCount']).toEqual(Uint32Array.of(1));
    expect(result['triangleOffsets']).toEqual(Uint32Array.of(0, 1));
    expect(result['valid']).toEqual(Uint32Array.of(1));
    expect(result['indices']).toBeInstanceOf(Uint32Array);
    expect(Array.from(result['indices'] as Uint32Array).sort()).toEqual([0, 1, 2]);
  });
}

function checkPolygonTilings(polygons: SearchPoint[][][], expectedAreas?: number[]): void {
  const input = packOperationPolygons(polygons);
  const props = getTriangulationOperationProps(input);
  const result = computePolygonTriangulationOnCPU(input, props);
  let vertexStart = 0;
  for (let polygon = 0; polygon < polygons.length; polygon++) {
    expect(result.valid[polygon], `polygon ${polygon}`).toBe(1);
    const triangles: number[][] = [];
    const start = result.triangleOffsets[polygon] * 3;
    for (let triangle = 0; triangle < result.triangleCount[polygon]; triangle++) {
      triangles.push(
        Array.from(
          result.indices.subarray(start + 3 * triangle, start + 3 * triangle + 3),
          vertex => vertex - vertexStart
        )
      );
    }
    // An independent boundary-chain/coverage invariant, not a second Earcut implementation.
    expect(getTilingDefect(polygons[polygon], triangles), `polygon ${polygon}`).toBeUndefined();
    if (expectedAreas)
      expect(getTriangleAreaSum(polygons[polygon], triangles)).toBe(expectedAreas[polygon]);
    vertexStart += polygons[polygon].flat().length;
  }
}

it('tiles a concave shell with a hole, another polygon and a hole touching an edge', () => {
  // Hand areas: (10*5 + 4*3) - 2*2, shoelace area 12, 100 - 3.
  checkPolygonTilings(OPERATION_POLYGONS, [58, 12, 97]);
});

it('tiles every committed Shapely-valid touching-ring scene and preserves its Shapely area', () => {
  const fixture = touchingFixture as unknown as {
    review: {rings: SearchPoint[][]; area: number}[];
    families: {rings: SearchPoint[][]; area: number}[];
  };
  const scenes = [...fixture.review, ...fixture.families];
  expect(scenes.length).toBe(1059);
  checkPolygonTilings(
    scenes.map(scene => scene.rings),
    scenes.map(scene => scene.area)
  );
});

it('keeps exact orientation on finite coordinates across a span the GPU cannot certify', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [2 ** 100, 0],
        [0, 2 ** -100]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 3});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleCount)).toEqual([1]);
  // Unique triangle, positive orientation, and area 1/2 regardless of clipping order.
  expect([...result.indices].sort((first, second) => first - second)).toEqual([0, 1, 2]);
});

it('keeps empty and collapsed polygons empty and pads failed slices with their first vertex', () => {
  const input = packOperationPolygons([
    [],
    [
      [
        [0, 0],
        [1, 0],
        [2, 0]
      ]
    ],
    [
      [
        [10, 0],
        [14, 0],
        [14, 4],
        [10, 4]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 3});
  expect(Array.from(result.valid)).toEqual([1, 1, 0]);
  expect(Array.from(result.triangleCount)).toEqual([0, 0, 0]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 0, 0, 2]);
  expect(Array.from(result.indices)).toEqual([3, 3, 3]);
});

it('keeps spare capacity zero, repeated closing vertices and degenerate holes out of real counts', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0]
      ],
      [
        [1, 1],
        [2, 1],
        [3, 1]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 30});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 3]);
  expect(Array.from(result.triangleCount)).toEqual([2]);
  expect(Array.from(result.indices.subarray(6))).toEqual(new Array(24).fill(0));
});

it('rejects inconsistent arrays with the argument name and flags a nonfinite live ring', () => {
  const input = packOperationPolygons(OPERATION_POLYGONS);
  const options = getTriangulationOperationProps(input);
  const malformed: [keyof CPUPolygonTriangulationInput, Float32Array | Uint32Array][] = [
    ['positions', new Float32Array(3)],
    ['featureOffsets', new Uint32Array()],
    ['polygonOffsets', Uint32Array.of(0, 2, 1)],
    ['ringOffsets', Uint32Array.of(0, 4, 2)]
  ];
  for (const [argument, values] of malformed) {
    expect(() => computePolygonTriangulationOnCPU({...input, [argument]: values}, options)).toThrow(
      argument
    );
  }
  const nonfinite = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [Infinity, 4],
        [0, 4]
      ]
    ]
  ]);
  const result = computePolygonTriangulationOnCPU(nonfinite, {indexCapacity: 6});
  expect(Array.from(result.valid)).toEqual([0]);
  expect(Array.from(result.triangleCount)).toEqual([0]);
  expect(Array.from(result.indices)).toEqual([0, 0, 0, 0, 0, 0]);
});

it('clamps a ring end for reads but preserves its original slice capacity', () => {
  const input = packOperationPolygons([
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ]
  ]);
  input.ringOffsets[1] = 7;
  const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 15});
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleCount)).toEqual([2]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 5]);
  expect(Array.from(result.indices.subarray(6))).toEqual(new Array(9).fill(0));
});

for (const scene of CLAMPED_RING_END_CASES) {
  it(`preserves failure padding at the signed slot boundary for ring end ${scene.ringEnd}`, () => {
    const result = computePolygonTriangulationOnCPU(createClampedRingEndInput(scene.ringEnd), {
      indexCapacity: 6
    });
    expect(Array.from(result.valid)).toEqual([0]);
    expect(Array.from(result.triangleCount)).toEqual([0]);
    expect(Array.from(result.triangleOffsets)).toEqual([0, scene.triangleEnd]);
    expect(Array.from(result.indices)).toEqual(scene.indices);
  });
}

it('validates both operation shapes at construction and contributor props when views are bound', () => {
  const input = packOperationPolygons(OPERATION_POLYGONS);
  const props = getTriangulationOperationProps(input);
  for (const Operation of [CPUPolygonTriangulationOperation, GPUPolygonTriangulationOperation]) {
    expect(() => new Operation({...props, indexCapacity: 0})).toThrow(
      /^polygon-twin.*indices.length/
    );
    expect(() => new Operation({...props, polygonCount: -1})).toThrow(
      /^polygon-twin.*valid.length/
    );
    expect(
      () => new Operation({...props, output: {...props.output, indices: 'positions'}})
    ).toThrow(/^polygon-twin/);
    expect(() => new Operation({...props, monotoneMinimumVertices: 0})).toThrow(
      /^polygon-twin.*monotoneMinimumVertices/
    );
  }
  // Routing limits are host-known and rejected above. Bound views still need one validity row
  // per polygon: the fixture has three polygons, so a two-row validity view must be rejected.
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const views = {
    positions: createTransientView(graph, 'positions', 'float32x2', input.positions.length / 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', input.featureOffsets.length),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', input.polygonOffsets.length),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', input.ringOffsets.length),
    indices: createTransientView(graph, 'indices', 'uint32', props.indexCapacity),
    valid: createTransientView(graph, 'valid', 'uint32', props.polygonCount - 1),
    triangleCount: createTransientView(graph, 'counts', 'uint32', props.polygonCount),
    triangleOffsets: createTransientView(graph, 'offsets', 'uint32', props.polygonCount + 1)
  };
  const operation = new GPUPolygonTriangulationOperation(props);
  expect(() =>
    operation.createContributor(port => views[port as keyof typeof views] as never)
  ).toThrow(/^polygon-twin.*output\.valid length must equal the polygon count/);
  expect(
    () =>
      new CPUPolygonTriangulationOperation({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/polygon-twin/);
  expect(() => new CPUPolygonTriangulationOperation({...props, maximumWork: 0})).toThrow(
    /maximumWork/
  );
});

it('CPU triangulation ignores garbage before the first ring and after the last', () => {
  const reference = computePolygonTriangulationOnCPU(createPaddingTriangulationInput(), {
    indexCapacity: 128
  });
  expect(Array.from(reference.valid)).toEqual([1, 1, 1]);
  expect(Array.from(reference.triangleCount)).toEqual([0, 8, 2]);
  for (const value of [Number.NaN, 3e30, -3e30]) {
    const input = createPaddingTriangulationInput(2, 3, value);
    const result = computePolygonTriangulationOnCPU(input, {indexCapacity: 128});
    const usedWords = 3 * result.triangleOffsets[3];
    expect(Array.from(result.indices.slice(0, usedWords), vertex => vertex - 2)).toEqual(
      Array.from(reference.indices.slice(0, usedWords))
    );
    expect(result.valid).toEqual(reference.valid);
    expect(result.triangleCount).toEqual(reference.triangleCount);
    expect(result.triangleOffsets).toEqual(reference.triangleOffsets);
  }
});
