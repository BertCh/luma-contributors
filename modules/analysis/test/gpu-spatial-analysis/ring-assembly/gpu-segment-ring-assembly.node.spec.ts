// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSegmentRingAssembly,
  type GPUSegmentRingAssemblyProps
} from '../../../src/analysis-core/ring-assembly/index';
import {quantizeRingVertexCoordinate} from '../../../src/analysis-core/ring-assembly/ring-vertex-quantizer';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSegmentRingAssemblyProps> = {}
): GPUSegmentRingAssemblyProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `ring-view-${serial++}`, format, length);
  return {
    endpoints: view('float32x4', 16),
    output: {
      ringOffsets: view('uint32', 5),
      positions: view('float32x2', 32),
      ringShells: view('uint32', 4),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    },
    ...overrides
  };
}

it('GPUSegmentRingAssembly declares hash, sort, tracing, scan, placement and ring nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node'});
  const nodes = new GPUSegmentRingAssembly(createProps(graph)).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(20);
  const single = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node-single'});
  const fewer = new GPUSegmentRingAssembly(
    createProps(single, {splitTouchingRings: false})
  ).getCommandNodes(single);
  expect(fewer.length).toBeLessThan(nodes.length);
});

it('snapshots ring options, context and nested output maps before graph declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph, {
      splitTouchingRings: false,
      spatialContext: {coordinateSpace: 'planar', metric: 'native'}
    });
    const producer = new GPUSegmentRingAssembly(props);
    const outputCount = props.output.status.count;
    const positions = props.output.positions;
    props.splitTouchingRings = true;
    props.spatialContext!.coordinateSpace = 'longitude-latitude';
    props.output.positions = createTransientView(graph, 'replacement-positions', 'float32x2', 0);
    props.output.status.count = createTransientView(graph, 'replacement-count', 'uint32', 0);
    expect(producer.props.splitTouchingRings).toBe(false);
    expect(producer.spatialContext.coordinateSpace).toBe('planar');
    expect(producer.props.output.positions).toBe(positions);
    expect(producer.props.output.status.count).toBe(outputCount);
    expect(producer.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('GPUSegmentRingAssembly validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-validate'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `validate-${serial++}`, format, length);
  const props = createProps(graph);
  expect(() => new GPUSegmentRingAssembly({...props, vertexTolerance: 0})).toThrow(
    /vertexTolerance/
  );
  expect(() => new GPUSegmentRingAssembly({...props, groups: view('uint32', 3)})).toThrow(
    /groups length/
  );
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringShells: view('uint32', 7)}
      })
  ).toThrow(/ring capacity/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringGroups: view('uint32', 4)}
      })
  ).toThrow(/groups input/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        interiorSide: 'up' as 'left'
      })
  ).toThrow(/interiorSide/);
});

it('GPUSegmentRingAssembly selects cos(latitude) turn comparison with a longitude-latitude context', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-context'});
  const planar = new GPUSegmentRingAssembly(createProps(graph));
  expect(planar.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  const geographic = new GPUSegmentRingAssembly(
    createProps(graph, {
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
    })
  );
  expect(geographic.spatialContext.coordinateSpace).toBe('longitude-latitude');
  for (const spatialContext of [
    {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
  ] as const) {
    expect(() => new GPUSegmentRingAssembly(createProps(graph, {spatialContext}))).toThrow(
      /does not support/
    );
  }
});

it('ring vertex quantizer separates UTM-scale vertices at several tolerances', () => {
  const originX = 500000.25;
  const originY = 4500000.5;
  for (const tolerance of [5e-5, 1e-4, 0.5]) {
    const cells = new Set<string>();
    for (let i = 0; i < 20; i++) {
      for (let j = 0; j < 20; j++) {
        const x = quantizeRingVertexCoordinate(originX + i * 2, tolerance);
        const y = quantizeRingVertexCoordinate(originY + j * 2, tolerance);
        expect(x.overflow || y.overflow).toBe(false);
        cells.add(`${x.cell},${y.cell}`);
      }
    }
    expect(cells.size, `tolerance ${tolerance}`).toBe(400);
  }
});

it('ring vertex keys are ordered across signed zero and the dense-cell boundary', () => {
  // At tolerance 1/64 the width is 1/32. At 2^18 float32 spacing first reaches 1/32.
  const tolerance = 1 / 64;
  const boundary = 2 ** 18;
  const cell = (coordinate: number) => quantizeRingVertexCoordinate(coordinate, tolerance).cell;
  const maximumFloat32 = 3.4028234663852886e38;
  expect(cell(-maximumFloat32)).toBeGreaterThan(0);
  expect(cell(maximumFloat32)).toBeLessThan(0x7fffffff);
  expect(cell(-0)).toBe(cell(0));
  expect(cell(-boundary + tolerance)).toBe(cell(-boundary));
  expect(cell(-boundary - 1 / 32)).toBe(cell(-boundary) - 1);
  for (const [first, second] of [
    [-tolerance / 2, tolerance / 2],
    [1 - tolerance / 2, 1 + tolerance / 2],
    [boundary - tolerance, boundary]
  ]) {
    expect(cell(second) - cell(first)).toBe(1);
  }
  expect(cell(boundary + 4 / 32) - cell(boundary)).toBe(2);
});

it('ring vertex quantizer reports non-finite coordinates as overflow', () => {
  expect(quantizeRingVertexCoordinate(Number.NaN, 1e-4).overflow).toBe(true);
  expect(quantizeRingVertexCoordinate(Number.POSITIVE_INFINITY, 1e-4).overflow).toBe(true);
  expect(quantizeRingVertexCoordinate(5, 1e-4).overflow).toBe(false);
});

it('GPUSegmentRingAssembly measures the frame extent before hashing', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-quantize-extent'});
  const ids = new GPUSegmentRingAssembly(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  const extent = ids.findIndex(id => id.endsWith('-quantize-extent'));
  expect(ids.findIndex(id => id.endsWith('-quantize-init'))).toBeLessThan(extent);
  expect(extent).toBeGreaterThanOrEqual(0);
  expect(extent).toBeLessThan(ids.findIndex(id => id.endsWith('-hash')));
});

it('CPL-12 captures counted segment provenance and rejects conflicting live counts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const segmentStatus = {
    count: scalar('source-count'),
    overflow: scalar('source-loss'),
    candidateOverflow: scalar('source-candidate')
  };
  const contributor = new GPUSegmentRingAssembly({...props, segmentStatus});
  expect(contributor.props.count).toBe(segmentStatus.count);
  expect(contributor.props.segmentStatus).not.toBe(segmentStatus);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain(
    'segment-ring-assembly-suppress-incomplete-0'
  );
  expect(
    () => new GPUSegmentRingAssembly({...props, segmentStatus, count: scalar('other-count')})
  ).toThrow(/count.*segmentStatus.count/);
  device.destroy();
});

for (const dispatch of ['complete', 'conflict', 'polygon-count'] as const) {
  it(`S1 rejects disjoint slices sharing a read/write buffer in ${dispatch}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph);
    const backing = createTransientView(graph, `shared-${dispatch}`, 'uint32', 192);
    const slice = (byteOffset: number, length = 1) =>
      graph.createDataView(backing.buffer, {format: 'uint32', byteOffset, length});
    const output = {...props.output, status: {...props.output.status}};
    if (dispatch === 'complete')
      Object.assign(output.status, {
        overflow: slice(0),
        candidateOverflow: slice(256),
        complete: slice(512)
      });
    if (dispatch === 'conflict')
      Object.assign(output, {
        segmentFlags: slice(0, 16),
        status: {...output.status, candidateOverflow: slice(256)}
      });
    if (dispatch === 'polygon-count')
      Object.assign(output, {
        polygons: {
          kind: 'polygons',
          positions: createTransientView(graph, 'polygon-position', 'float32x2', 32),
          ringOffsets: createTransientView(graph, 'ring-offset', 'uint32', 5),
          polygonOffsets: createTransientView(graph, 'polygon-offset', 'uint32', 5),
          featureOffsets: slice(0, 5)
        },
        status: {...output.status, polygonCount: slice(256)}
      });
    expect(() => new GPUSegmentRingAssembly({...props, output})).toThrow(/buffer/);
    device.destroy();
  });
}
