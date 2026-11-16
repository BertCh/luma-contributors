// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
function createProps(graph: GPUCommandGraph, kind: GPUSpatialJoinGeometry['kind'] = 'polygons') {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `predicate-distances-${serial++}`, format, length);
  const geometry: GPUSpatialJoinGeometry =
    kind === 'points'
      ? {kind, positions: view('float32x2', 2)}
      : kind === 'lines'
        ? {
            kind,
            positions: view('float32x2', 8),
            lineOffsets: view('uint32', 3)
          }
        : {
            kind,
            positions: view('float32x2', 8),
            featureOffsets: view('uint32', 3),
            polygonOffsets: view('uint32', 3),
            ringOffsets: view('uint32', 3)
          };
  return {
    id: 'predicate-distances',
    left: geometry,
    right: geometry,
    predicate: 'touches' as const,
    candidateCapacity: 4,
    output: {
      pairs: {
        leftIds: view('uint32', 4),
        rightIds: view('uint32', 4),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      relate: view('uint32', 4),
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 4),
        weights: view('float32', 4),
        distances: view('float32', 4)
      }
    }
  };
}

it('distance kernels parse for all geometry pairs with at most eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  for (const leftKind of ['points', 'lines', 'polygons'] as const) {
    for (const rightKind of ['points', 'lines', 'polygons'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(graph, leftKind);
      props.right = createProps(graph, rightKind).right;
      const join = new GPUSpatialPredicateJoin(props);
      const kernels = getGPUCommandNodeKernelSources(join.getCommandNodes(graph));
      const distanceKernel = kernels.find(kernel => kernel.id.endsWith('-distances'))!;
      const scatterKernel = kernels.find(kernel => kernel.id.endsWith('-scatter'))!;
      for (const kernel of [distanceKernel, scatterKernel]) {
        const reflected = new WgslReflect(kernel.source);
        expect(reflected.storage.length).toBeLessThanOrEqual(8);
      }
      expect(distanceKernel.source).toContain('pairDistance(left, right)');
      const clearKernel = kernels.find(kernel => kernel.id.endsWith('-clear-distance-pairs'))!;
      expect(clearKernel.constants['INVOCATION_COUNT']).toBe(8);
      const withDistances = join.getCapacityPlan();
      const withoutDistances = new GPUSpatialPredicateJoin({
        ...props,
        output: {...props.output, weights: {...props.output.weights, distances: undefined}}
      }).getCapacityPlan();
      expect(
        withDistances.estimatedPeakTransientBytes - withoutDistances.estimatedPeakTransientBytes
      ).toBe(withDistances.pairs.capacity * 8);
    }
  }
  device.destroy();
});

it('distance outputs validate lengths, metric, graph ownership and disjointness', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const weights = props.output.weights;
  expect(
    () =>
      new GPUSpatialPredicateJoin({
        ...props,
        output: {
          ...props.output,
          weights: {
            ...weights,
            distances: createTransientView(graph, 'short-distances', 'float32', 3)
          }
        }
      })
  ).toThrow(/distances length must equal neighbors length/);
  expect(
    () =>
      new GPUSpatialPredicateJoin({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/output.weights.distances requires/);
  expect(
    () =>
      new GPUSpatialPredicateJoin({
        ...props,
        output: {...props.output, weights: {...weights, distances: weights.weights}}
      })
  ).toThrow(/overlap|disjoint|alias/);
  const otherGraph = new GPUCommandGraph(device);
  const foreign = createTransientView(otherGraph, 'foreign-distances', 'float32', 4);
  const join = new GPUSpatialPredicateJoin({
    ...props,
    output: {...props.output, weights: {...weights, distances: foreign}}
  });
  expect(() => join.getCommandNodes(graph)).toThrow(/graph|belong/);
  device.destroy();
});

it('distance scratch checks the device binding limit before adding nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 'points');
  const view = <Format extends 'uint32' | 'float32'>(name: string, format: Format) =>
    createTransientView(graph, `distance-limit-${name}`, format, 8);
  // Every caller binding fits 32 bytes; only the private uint32x2 distance-pair table needs 64.
  const join = new GPUSpatialPredicateJoin({
    ...props,
    output: {
      ...props.output,
      pairs: {
        ...props.output.pairs,
        leftIds: view('left', 'uint32'),
        rightIds: view('right', 'uint32')
      },
      relate: view('relate', 'uint32'),
      weights: {
        ...props.output.weights,
        neighbors: view('neighbors', 'uint32'),
        weights: view('weights', 'float32'),
        distances: view('distances', 'float32')
      }
    }
  });
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 32}
  });
  const scratch = vi.spyOn(graph, 'createTransientBuffer');
  const views = vi.spyOn(graph, 'createDataView');
  const nodes = vi.spyOn(graph, 'add');
  const resources = vi.spyOn(device, 'createBuffer');
  expect(() => join.getCommandNodes(graph)).toThrow(/predicate-distances distance pairs.*limit 32/);
  for (const declaration of [scratch, views, nodes, resources])
    expect(declaration).not.toHaveBeenCalled();
  device.destroy();
});

it('weights-only distance output requires the aggregate overflow port and stays within the binding limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const view = (name: string) => createTransientView(graph, `weights-only-${name}`, 'uint32', 1);
  const output = {
    weights: {
      ...props.output.weights,
      status: {
        count: view('count'),
        requiredCount: view('required-count'),
        overflow: view('overflow'),
        candidateOverflow: view('candidate-overflow')
      }
    }
  };
  expect(() => new GPUSpatialPredicateJoin({...props, output})).toThrow(
    /needs output.status.overflow without pairs/
  );
  const join = new GPUSpatialPredicateJoin({
    ...props,
    output: {...output, status: {overflow: view('overall-overflow')}}
  });
  const kernels = getGPUCommandNodeKernelSources(join.getCommandNodes(graph));
  expect(kernels.some(kernel => kernel.id.endsWith('-distances'))).toBe(true);
  const finalizers = kernels.filter(kernel => /-finalize(?:-\d+)?$/.test(kernel.id));
  expect(finalizers).toHaveLength(1);
  for (const kernel of finalizers) {
    expect(new WgslReflect(kernel.source).storage.length).toBeLessThanOrEqual(8);
  }
  device.destroy();
});
