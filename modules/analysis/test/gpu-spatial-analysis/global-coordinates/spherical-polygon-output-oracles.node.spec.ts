// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUSphericalPolygonOverlay,
  type GPUSphericalPolygonOverlayProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x3' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `spherical-status-${serial++}`, format, length);
  const operand = () => ({
    kind: 'polygons' as const,
    positions: view('float32x3', 3),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  });
  const props: GPUSphericalPolygonOverlayProps = {
    id: 'spherical-status',
    left: operand(),
    right: operand(),
    chartCenter: [0, 0, 1],
    operation: 'intersection',
    vertexTolerance: 2 ** -16,
    capacity: {intersections: 8, nodedSegments: 16},
    output: {
      geometry: {
        kind: 'polygons',
        positions: view('float32x3', 8),
        ringOffsets: view('uint32', 3),
        polygonOffsets: view('uint32', 3),
        featureOffsets: view('uint32', 3)
      },
      complement: view('uint32', 1),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      },
      boundary: {
        endpoints: view('float32x4', 8),
        operandIds: view('uint32', 8),
        sourceFeatureIds: view('uint32', 8),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        }
      }
    }
  };
  return {device, graph, view, props};
}

const REMOVED_FIELDS = [
  ['result', 'iterationCount'],
  ['boundary', 'iterationCount'],
  ['result', 'invalidCount'],
  ['result', 'converged'],
  ['result', 'iterationLimitReached'],
  ['result', 'approximation'],
  ['boundary', 'uncertainCount'],
  ['boundary', 'invalidCount'],
  ['boundary', 'converged'],
  ['boundary', 'iterationLimitReached'],
  ['boundary', 'approximation']
] as const;

it.each(
  REMOVED_FIELDS
)('spherical overlay refuses the removed %s %s port at construction', (location, field) => {
  const {device, view, props} = createFixture();
  try {
    // A supplied field with no writer retains arbitrary caller bytes. A fixed-pass arrangement
    // has no solver convergence/iteration result: the independent contract expectation is refusal,
    // with completeness available through overflow/candidateOverflow instead.
    const status = location === 'result' ? props.output.status : props.output.boundary.status;
    Object.assign(status, {[field]: view('uint32', 1)});
    const label = location === 'result' ? 'output.status' : 'output.boundary.status';
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(
      `spherical-status ${label}.${field}`
    );
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(/because.*use /);
    const alternative =
      field === 'uncertainCount'
        ? 'output.status.uncertainCount'
        : field === 'invalidCount'
          ? 'output.status.candidateOverflow'
          : field === 'approximation'
            ? 'vertexTolerance'
            : 'output.status.overflow';
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(`use ${alternative}`);
  } finally {
    device.destroy();
  }
});

it('spherical overlay types expose only the implemented result and boundary status words', () => {
  expectTypeOf<keyof GPUSphericalPolygonOverlayProps['output']['status']>().toEqualTypeOf<
    | 'count'
    | 'overflow'
    | 'candidateOverflow'
    | 'uncertainCount'
    | 'ringCount'
    | 'requiredRingCount'
    | 'nodedSegmentRequiredCount'
  >();
  expectTypeOf<
    keyof GPUSphericalPolygonOverlayProps['output']['boundary']['status']
  >().toEqualTypeOf<'count' | 'requiredCount' | 'overflow' | 'candidateOverflow'>();
});

it('spherical overlay snapshots validated output descriptors before deferred construction', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const count = props.output.status.count;
    const contributor = new GPUSphericalPolygonOverlay(props);
    props.output.status.count = view('uint32', 1);
    Object.assign(props.output.status, {iterationCount: view('uint32', 1)});
    Object.assign(props.output.boundary.status, {invalidCount: view('uint32', 1)});
    expect(contributor.props.output.status.count).toBe(count);
    expect(contributor.props.output.status).not.toHaveProperty('iterationCount');
    expect(contributor.props.output.boundary.status).not.toHaveProperty('invalidCount');
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('spherical overlay preserves a declared non-enumerable output getter and reads it once', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const uncertainCount = view('uint32', 1);
    let reads = 0;
    Object.defineProperty(props.output.status, 'uncertainCount', {
      enumerable: false,
      get: () => {
        reads++;
        return uncertainCount;
      }
    });
    const contributor = new GPUSphericalPolygonOverlay(props);
    expect(contributor.props.output.status.uncertainCount).toBe(uncertainCount);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('spherical overlay refuses an inherited retired status port', () => {
  const {device, view, props} = createFixture();
  try {
    Object.setPrototypeOf(props.output.status, {iterationCount: view('uint32', 1)});
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow('output.status.iterationCount');
  } finally {
    device.destroy();
  }
});

it('result requiredCount inherited from a prototype refuses while boundary requiredCount remains supported', () => {
  const {device, view, props} = createFixture();
  try {
    expect(() => new GPUSphericalPolygonOverlay(props)).not.toThrow();
    Object.setPrototypeOf(props.output.status, {requiredCount: view('uint32', 1)});
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(
      /output.status.requiredCount.*requiredRingCount/
    );
  } finally {
    device.destroy();
  }
});

it('disjoint packed complement and feature-source words remain valid across separate dispatches', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const packed = view('uint32', 128);
    props.output.geometry.ringOffsets = view('uint32', 2);
    props.output.geometry.polygonOffsets = view('uint32', 2);
    props.output.geometry.featureOffsets = view('uint32', 2);
    props.output.featureSources = {
      sourceRows: graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 0, length: 1}),
      operandIds: view('uint32', 1),
      mixed: view('uint32', 1)
    };
    props.output.complement = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    const contributor = new GPUSphericalPolygonOverlay(props);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('lifted XYZ output cannot overlap any feature-source word', () => {
  const {device, graph, view, props} = createFixture();
  try {
    props.output.geometry.ringOffsets = view('uint32', 2);
    props.output.geometry.polygonOffsets = view('uint32', 2);
    props.output.geometry.featureOffsets = view('uint32', 2);
    props.output.featureSources = {
      sourceRows: graph.createDataView(props.output.geometry.positions.buffer, {
        format: 'uint32',
        byteOffset: 0,
        length: 1
      }),
      operandIds: view('uint32', 1),
      mixed: view('uint32', 1)
    };
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(
      /output.geometry.positions and output.featureSources.sourceRows logical byte ranges must not overlap/
    );
  } finally {
    device.destroy();
  }
});

it('refuses physical publish read/write conflicts before mutation and allows same-ID retry', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const packed = view('uint32', 128);
    props.output.status.count = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    props.output.boundary.status.overflow = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    // The publish dispatch reads boundary loss and writes result counts. Physical
    // storage usage conflicts even for these independently addressable words.
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(
      /spherical-status.*separate physical buffers/
    );
    props.output.status.count = view('uint32', 1);
    expect(() => new GPUSphericalPolygonOverlay(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('spherical contributor retains its validated spatial context after caller mutation', () => {
  const {device, props} = createFixture();
  try {
    const context = {
      coordinateSpace: 'longitude-latitude' as 'longitude-latitude' | 'planar',
      metric: 'none' as const,
      edgeModel: 'great-circle' as const
    };
    props.spatialContext = context;
    const contributor = new GPUSphericalPolygonOverlay(props);
    context.coordinateSpace = 'planar';
    expect(contributor.spatialContext.coordinateSpace).toBe('longitude-latitude');
    expect(contributor.props.spatialContext?.coordinateSpace).toBe('longitude-latitude');
    expect(contributor.props.spatialContext).toBe(contributor.spatialContext);
  } finally {
    device.destroy();
  }
});

it('refuses assembly-completeness physical read/write conflicts and permits same-ID retry', () => {
  const {device, graph, view, props} = createFixture();
  try {
    const packed = view('uint32', 128);
    props.output.status.candidateOverflow = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    props.output.boundary.status.count = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    // The atomic assembly-loss publisher reads the boundary count while writing
    // candidateOverflow. Their logical ranges are disjoint but storage usage conflicts.
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(/separate physical buffers/);
    props.output.status.candidateOverflow = view('uint32', 1);
    expect(() => new GPUSphericalPolygonOverlay(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'result',
  'boundary',
  'ring'
] as const)('refuses aligned co-written %s status overlap and permits retry', group => {
  const {device, graph, view, props} = createFixture();
  try {
    const packed = view('uint32', 128);
    const first = graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 0, length: 1});
    const second = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    if (group === 'result') {
      props.output.status.overflow = first;
      props.output.status.count = second;
    } else if (group === 'boundary') {
      props.output.boundary.status.overflow = first;
      props.output.boundary.status.count = second;
    } else {
      props.output.status.ringCount = first;
      props.output.status.requiredRingCount = second;
    }
    // A binding at byte 4 starts at aligned byte 0, so [0,4) and [0,8)
    // overlap even though each uint32 logical word has a distinct address.
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow(/aligned binding.*overlap/);
    if (group === 'result') props.output.status.count = view('uint32', 1);
    else if (group === 'boundary') props.output.boundary.status.count = view('uint32', 1);
    else props.output.status.requiredRingCount = view('uint32', 1);
    expect(() => new GPUSphericalPolygonOverlay(props).getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('spherical overlay ignores unrelated descriptor getters and capacity labels', () => {
  const {device, graph, props} = createFixture();
  try {
    Object.defineProperty(props, 'unrelated', {
      get: () => {
        throw new Error('unrelated getter evaluated');
      }
    });
    Object.defineProperty(props.capacity, 'units', {value: 'chart coordinates'});
    const contributor = new GPUSphericalPolygonOverlay(props);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'sourceRows',
  'operandIds',
  'mixed'
] as const)('spherical complement refuses aliasing featureSources.%s', field => {
  const {device, view, props} = createFixture();
  try {
    props.output.geometry.ringOffsets = view('uint32', 2);
    props.output.geometry.polygonOffsets = view('uint32', 2);
    props.output.geometry.featureOffsets = view('uint32', 2);
    props.output.featureSources = {
      sourceRows: view('uint32', 1),
      operandIds: view('uint32', 1),
      mixed: view('uint32', 1)
    };
    props.output.featureSources[field] = props.output.complement;
    expect(() => new GPUSphericalPolygonOverlay(props)).toThrow();
  } finally {
    device.destroy();
  }
});
