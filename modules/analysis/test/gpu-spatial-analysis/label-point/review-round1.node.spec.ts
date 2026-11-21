// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point';
import {
  computeLabelPointOnCPU,
  publishCPUPolygonPole
} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {getCPUPolygonRings} from '../../../src/gpu-spatial-analysis/shared/cpu-polygon-rows';
import {computeMaximumInscribedCircleOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {GPUHilbertKeys} from '../../../src/gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys';
import {GPULabelPointOperation} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation';
import {GPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {GPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {computeLineSimplificationOnCPU} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function polygon(positions: number[]) {
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array([0, positions.length / 2]),
    polygonOffsets: new Uint32Array([0, 1]),
    featureOffsets: new Uint32Array([0, 1])
  };
}
function diamond(center: number, radius: number) {
  return polygon([
    center + radius,
    center,
    center,
    center + radius,
    center - radius,
    center,
    center,
    center - radius
  ]);
}
function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length = 1) =>
    createTransientView(graph, `review-${sequence++}`, format, length);
  return {device, graph, view};
}

it('B1 constructs all documented label columns within eight storage bindings', () => {
  const {device, graph, view} = fixture();
  try {
    expect(() =>
      new GPULabelPoint({
        positions: view('float32x2', 4),
        ringOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        featureOffsets: view('uint32', 2),
        output: {
          points: view('float32x2'),
          publishedCenterClearance: view('float32'),
          precisionLimited: view('uint32'),
          degenerate: view('uint32')
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B2 evaluates translated diamond clearance in published-center coordinates', () => {
  const {device, graph, view} = fixture();
  try {
    const input = diamond(1048576.125, 0.125);
    const positions = view('float32x2', 4),
      ringOffsets = view('uint32', 2),
      polygonOffsets = view('uint32', 2),
      featureOffsets = view('uint32', 2);
    const points = view('float32x2'),
      clearance = view('float32');
    const nodes = new GPULabelPoint({
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      output: {points, publishedCenterClearance: clearance}
    }).getCommandNodes(graph);
    const publish = nodes.find(node => node.id.endsWith('-published-clearance'))!;
    const memory = createCPUGraphMemory();
    memory.write(positions, input.positions);
    memory.write(ringOffsets, input.ringOffsets);
    memory.write(polygonOffsets, input.polygonOffsets);
    memory.write(featureOffsets, input.featureOffsets);
    memory.write(points, new Float32Array([1048576.125, 1048576.125]));
    executeCommandNodesOnCPU([publish], memory);
    // The perpendicular distance from the center to x+y=.125 is .125/sqrt(2).
    expect(memory.readFloat32(clearance)[0]).toBe(Math.fround(0.125 / Math.SQRT2));
  } finally {
    device.destroy();
  }
});

it('B3 identifies lost absolute witness precision and supplies a usable local witness', () => {
  const result = computeMaximumInscribedCircleOnCPU(diamond(1048576.25, 0.125), 0.00001);
  expect(result.circles[2]).toBe(Math.fround(0.125 / Math.SQRT2));
  expect(Array.from(result.witnesses)).toEqual(Array.from(result.circles.slice(0, 2)));
  expect(result.precisionLimited[0]).toBe(1);
  expect(Math.hypot(result.witnessOffsets[0], result.witnessOffsets[1])).toBeCloseTo(
    0.125 / Math.SQRT2,
    7
  );
});

it('B4 ignores ordinary radius rounding when the published center is exact', () => {
  const input = diamond(2, 2);
  const label = computeLabelPointOnCPU(input),
    circle = computeMaximumInscribedCircleOnCPU(input, 0.00001);
  expect(Array.from(label.points)).toEqual([2, 2]);
  expect(label.publishedCenterClearance[0]).toBe(Math.fround(Math.SQRT2));
  expect(label.precisionLimited[0]).toBe(0);
  expect(circle.precisionLimited[0]).toBe(0);
});

it('B5 falls back to a representable boundary when center rounding crosses outside', () => {
  const origin = 1048576;
  const input = polygon([0, 0, 3, 1, 4, 1, 1, 0].map(value => origin + value * 0.125));
  const circle = computeMaximumInscribedCircleOnCPU(input, 0.00001, {maximumIterations: 0});
  const center = Array.from(circle.circles.slice(0, 2));
  const vertices = Array.from({length: 4}, (_, row) =>
    Array.from(input.positions.slice(row * 2, row * 2 + 2))
  );
  expect(vertices).toContainEqual(center);
  expect(circle.circles[2]).toBe(0);
  expect(circle.precisionLimited[0]).toBe(1);
  const label = publishCPUPolygonPole(input.positions, getCPUPolygonRings(input, 0), {
    centerX: origin + 0.25,
    centerY: origin + 0.0625,
    radius: 0.01,
    witnessX: origin,
    witnessY: origin,
    error: 0,
    status: 0
  });
  expect(vertices).toContainEqual([label.centerX, label.centerY]);
  expect(label.clearance).toBe(0);
  expect(label.precisionLimited).toBe(true);
});

it('B6 keeps remaining search error separate from coordinate publication loss', () => {
  const origin = 1048576;
  const result = computeMaximumInscribedCircleOnCPU(
    polygon([
      origin,
      origin,
      origin + 0.125,
      origin,
      origin + 0.125,
      origin + 0.125,
      origin,
      origin + 0.125
    ]),
    0.00001
  );
  expect(result.circles[2]).toBe(0);
  expect(result.circles[3]).toBe(0);
  expect(result.precisionLimited[0]).toBe(1);
});

it('B7 preserves uncovered line rows and rejects a window in explicit compact storage', () => {
  const input = {
    positions: new Float32Array([99, 99, 0, 0, 1, 0, 2, 0, 99, 99]),
    lineOffsets: new Uint32Array([1, 1, 4]),
    importance: new Float32Array([9, 1, 2, 3, 9]),
    parameters: new Float32Array([0, 0, 0, 0])
  };
  expect(() =>
    computeLineSimplificationOnCPU({...input, storage: 'compact'}, {computeImportance: false})
  ).toThrow(/initial offset/);
  for (const storage of [undefined, 'resident-capacity'] as const) {
    const result = computeLineSimplificationOnCPU({...input, storage}, {computeImportance: false});
    expect(Array.from(result.ids.slice(0, result.count[0]))).toEqual([1, 2, 3]);
  }
});

it('B8 lowers empty geometry for zero features and one empty feature on all owned GPU forms', () => {
  for (const featureCount of [0, 1]) {
    const {device, graph, view} = fixture();
    const inputs: Record<string, GraphDataView> = {
      positions: view('float32x2', 0),
      ringOffsets: view('uint32', 1),
      polygonOffsets: view('uint32', 1),
      featureOffsets: view('uint32', featureCount + 1),
      parameters: view('float32', 4)
    };
    const shared = {
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount
    };
    const operations = [
      new GPULabelPointOperation({
        ...shared,
        output: {points: 'points', publishedCenterClearance: 'clearance', degenerate: 'degenerate'}
      }),
      new GPUMinimumBoundsOperation({
        ...shared,
        rowCount: 0,
        output: {sourceBounds: 'bounds', hullSizes: 'hullSizes', status: {overflow: 'overflow'}}
      }),
      new GPUShapeDescriptorsOperation({
        ...shared,
        parameters: 'parameters',
        output: {areas: 'areas', orientation: 'orientation'}
      })
    ];
    try {
      for (const operation of operations) {
        const outputs = Object.fromEntries(
          Object.entries(operation.outputs).map(([port, column]) => [
            port,
            view(column.format, column.length)
          ])
        );
        const producer = operation.createContributor(
          <Format extends GPUGraphVectorFormat>(port: string) =>
            (outputs[port] ?? inputs[port]) as GraphDataView<Format>
        );
        const memory = createCPUGraphMemory();
        executeCommandNodesOnCPU(producer.getCommandNodes(graph), memory);
        for (const [port, column] of Object.entries(outputs)) {
          const expected = [
            'points',
            'publishedCenterClearance',
            'sourceBounds',
            'orientation'
          ].includes(port)
            ? NaN
            : port === 'degenerate'
              ? 1
              : 0;
          const actual = column.format.startsWith('float')
            ? memory.readFloat32(column)
            : memory.readUint32(column);
          expect(actual.length).toBe((column.length * column.rowByteLength) / 4);
          for (const value of actual)
            if (Number.isNaN(expected)) expect(Number.isNaN(value)).toBe(true);
            else expect(value).toBe(expected);
        }
      }
    } finally {
      device.destroy();
    }
  }
});

it('B9 captures getter-backed and inherited Hilbert sources before validation', () => {
  const {device, view} = fixture();
  try {
    let reads = 0;
    const positions = view('float32x2');
    const props = Object.assign(
      Object.create({
        get positions() {
          reads++;
          return positions;
        }
      }),
      {output: {keys: view('uint32')}}
    );
    expect(() => new GPUHilbertKeys(props)).not.toThrow();
    expect(reads).toBe(1);
    const conflict = Object.assign(
      Object.create({minima: view('float32x2'), maxima: view('float32x2')}),
      {positions, output: {keys: view('uint32')}}
    );
    expect(() => new GPUHilbertKeys(conflict)).toThrow(/not both/);
  } finally {
    device.destroy();
  }
});

it('S2 rejects empty GPU shape operation selections at construction', () => {
  expect(
    () =>
      new GPUShapeDescriptorsOperation({
        positions: 'positions',
        ringOffsets: 'rings',
        polygonOffsets: 'polygons',
        featureOffsets: 'features',
        parameters: 'parameters',
        featureCount: 1,
        // @ts-expect-error Deliberately probe the JavaScript guard.
        output: {}
      })
  ).toThrow(/at least one/);
});

it('B7 forwards resident storage through owned CPU forms', async () => {
  const {CPULabelPointOperation} = await import(
    '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation'
  );
  const {CPUMinimumBoundsOperation} = await import(
    '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation'
  );
  const {CPUShapeDescriptorsOperation} = await import(
    '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation'
  );
  const {CPUSegmentSnapOperation} = await import(
    '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation'
  );
  const {CPULineSimplificationOperation} = await import(
    '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation'
  );
  const rows = (
    format: 'float32' | 'float32x2' | 'uint32',
    values: Float32Array | Uint32Array
  ) => ({
    format,
    length: values.length / (format === 'float32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'float32x2' ? 2 : 1)}],
    pack: () => values
  });
  const shared = {
    positions: 'positions',
    ringOffsets: 'rings',
    polygonOffsets: 'polygons',
    featureOffsets: 'features',
    featureCount: 1,
    storage: 'resident-capacity' as const
  };
  const positions = new Float32Array([99, 99, 0, 0, 1, 0, 1, 1, 0, 1]);
  const inputs = {
    positions: rows('float32x2', positions),
    ringOffsets: rows('uint32', new Uint32Array([1, 5])),
    polygonOffsets: rows('uint32', new Uint32Array([0, 1])),
    featureOffsets: rows('uint32', new Uint32Array([0, 1])),
    parameters: rows('float32', new Float32Array([0.05, 0, 0, 0]))
  };
  const label = await new CPULabelPointOperation({...shared, output: {points: 'points'}}).execute({
    inputs
  });
  expect(Array.from(label['points'] as Float32Array)).toEqual([0.5, 0.5]);
  const bounds = await new CPUMinimumBoundsOperation({
    ...shared,
    rowCount: 5,
    output: {sourceBounds: 'bounds', status: {overflow: 'overflow'}}
  }).execute({inputs});
  expect(Array.from(bounds['bounds'] as Float32Array)).toEqual([0, 0, 1, 1]);
  const shape = await new CPUShapeDescriptorsOperation({
    ...shared,
    parameters: 'parameters',
    output: {areas: 'areas'}
  }).execute({inputs});
  expect(Array.from(shape['areas'] as Float32Array)).toEqual([1]);
  const snap = await new CPUSegmentSnapOperation({
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 5,
    lineCount: 1,
    referenceCount: 0,
    tolerance: 0,
    maximumVerticesPerLine: 4,
    storage: 'resident-capacity',
    output: {positions: 'snapped', sourceEdges: 'edges', counts: 'counts'}
  }).execute({
    inputs: {
      positions: inputs.positions,
      lineOffsets: rows('uint32', new Uint32Array([1, 5])),
      referencePositions: rows('float32x2', new Float32Array())
    }
  });
  expect(Array.from(snap['counts'] as Uint32Array)).toEqual([4, 4, 0, 0]);
  const line = await new CPULineSimplificationOperation({
    positions: 'positions',
    lineOffsets: 'offsets',
    importance: 'importance',
    parameters: 'parameters',
    rowCount: 5,
    lineCount: 2,
    computeImportance: false,
    storage: 'resident-capacity',
    output: {ids: 'ids', status: {count: 'count', overflow: 'overflow'}}
  }).execute({
    inputs: {
      positions: inputs.positions,
      lineOffsets: rows('uint32', new Uint32Array([1, 1, 4])),
      importance: rows('float32', new Float32Array([9, 1, 2, 3, 9])),
      parameters: rows('float32', new Float32Array([0, 0, 0, 0]))
    }
  });
  expect(Array.from((line['ids'] as Uint32Array).slice(0, 3))).toEqual([1, 2, 3]);
});

it('B9 captures inherited raw geometry fields and nested destinations', async () => {
  const {GPUGroupGeometry} = await import(
    '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-geometry'
  );
  const {GPUGroupConvexHull} = await import(
    '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull'
  );
  const {GPUMinimumBounds} = await import(
    '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds'
  );
  const {device, graph, view} = fixture();
  try {
    const positions = view('float32x2', 4),
      labels = view('uint32', 4);
    const makeProps = (fields: object) => Object.create(fields);
    const output = makeProps({
      sourceBounds: view('float32x4'),
      status: makeProps({overflow: view('uint32')})
    });
    expect(() =>
      new GPUMinimumBounds(makeProps({positions, labels, groupCount: 1, output})).getCommandNodes(
        graph
      )
    ).not.toThrow();
    expect(() =>
      new GPUGroupGeometry(
        makeProps({
          positions,
          labels,
          groupCount: 1,
          parameters: view('float32', 8),
          output: makeProps({counts: view('uint32')})
        })
      ).getCommandNodes(graph)
    ).not.toThrow();
    expect(() =>
      new GPUGroupConvexHull(
        makeProps({
          positions,
          labels,
          groupCount: 1,
          vertexCapacity: 4,
          maximumVerticesPerGroup: 4,
          output: makeProps({
            vertexIndices: view('uint32', 4),
            offsets: view('uint32', 2),
            counts: view('uint32'),
            status: makeProps({overflow: view('uint32')})
          })
        })
      ).getCommandNodes(graph)
    ).not.toThrow();
    expect(() =>
      new GPULabelPoint(
        makeProps({
          positions,
          ringOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          featureOffsets: view('uint32', 2),
          output: makeProps({points: view('float32x2')})
        })
      ).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('U1 exposes CPU named circle publication and relative-witness columns', async () => {
  const {CPUMaximumInscribedCircleOperation} = await import(
    '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation'
  );
  const input = diamond(1048576.25, 0.125);
  const rows = (format: 'float32x2' | 'uint32', values: Float32Array | Uint32Array) => ({
    format,
    length: values.length / (format === 'float32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'float32x2' ? 2 : 1)}],
    pack: () => values
  });
  const operation = new CPUMaximumInscribedCircleOperation({
    positions: 'positions',
    ringOffsets: 'rings',
    polygonOffsets: 'polygons',
    featureOffsets: 'features',
    featureCount: 1,
    tolerance: 0.00001,
    output: {
      circles: 'circles',
      witnesses: 'witnesses',
      status: 'status',
      publishedCenterClearance: 'clearance',
      precisionLimited: 'precision',
      witnessOffsets: 'offsets'
    }
  });
  const result = await operation.execute({
    inputs: {
      positions: rows('float32x2', input.positions),
      ringOffsets: rows('uint32', input.ringOffsets),
      polygonOffsets: rows('uint32', input.polygonOffsets),
      featureOffsets: rows('uint32', input.featureOffsets)
    }
  });
  expect(Array.from(result['precision'] as Uint32Array)).toEqual([1]);
  expect(Array.from(result['clearance'] as Float32Array)).toEqual([
    Math.fround(0.125 / Math.SQRT2)
  ]);
  expect((result['offsets'] as Float32Array).length).toBe(2);
});
