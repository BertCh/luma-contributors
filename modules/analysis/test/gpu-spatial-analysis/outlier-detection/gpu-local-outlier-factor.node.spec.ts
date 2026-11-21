// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPULocalOutlierFactorParameterValues,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';

it('W5-09 checks distance units of densityFloor against neighbor metric results', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const floating = (length: number) =>
    createTransientView(graph, `units-float-${serial++}`, 'float32', length);
  const unsigned = (length: number) =>
    createTransientView(graph, `units-uint-${serial++}`, 'uint32', length);
  const context = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const props = {
    id: 'distance-floor',
    neighbors: {
      offsets: unsigned(3),
      neighbors: unsigned(2),
      weights: floating(2),
      distances: floating(2),
      spatialContext: context
    },
    parameters: floating(4),
    output: {kDistance: floating(2), localReachabilityDensity: floating(2), lof: floating(2)}
  };
  try {
    // Omitted great-circle units resolve to meters, so an explicitly kilometer-valued
    // floor cannot be added to these distances without converting the numeric input.
    expect(
      () => new GPULocalOutlierFactor({...props, spatialContext: {...context, units: 'kilometers'}})
    ).toThrow(/distance-floor.*effective length units/);
    expect(
      () => new GPULocalOutlierFactor({...props, spatialContext: {...context, units: 'meters'}})
    ).not.toThrow();
    expect(
      () =>
        new GPULocalOutlierFactor({
          ...props,
          neighbors: {...props.neighbors, spatialContext: {...context, units: 'kilometers'}},
          spatialContext: {...context, units: 'kilometers'}
        })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const packedPort of ['density', 'score'] as const) {
  it(`keeps packed ${packedPort} and classification outputs out of conflicting buffer usages`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const packed = createTransientView(graph, 'packed', 'uint32', 128);
      const packedView = graph.createDataView(packed.buffer, {format: 'float32', length: 2});
      const density =
        packedPort === 'density' ? packedView : createTransientView(graph, 'density', 'float32', 2);
      const lof =
        packedPort === 'score' ? packedView : createTransientView(graph, 'lof', 'float32', 2);
      const outlier = graph.createDataView(packed.buffer, {
        format: 'uint32',
        length: 2,
        byteOffset: 256
      });
      const nodes = new GPULocalOutlierFactor({
        neighbors: {
          offsets: createTransientView(graph, 'offsets', 'uint32', 3),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
          distances: createTransientView(graph, 'distances', 'float32', 2),
          weights: createTransientView(graph, 'weights', 'float32', 2)
        },
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        output: {
          kDistance: createTransientView(graph, 'kd', 'float32', 2),
          localReachabilityDensity: density,
          lof,
          outlier
        }
      }).getCommandNodes(graph);
      for (const node of nodes) {
        const usages = new Map<object, Set<string>>();
        for (const resource of node.resources ?? []) {
          if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
          const buffer = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
          const current = usages.get(buffer) ?? new Set<string>();
          current.add(resource.usage);
          usages.set(buffer, current);
        }
        for (const current of usages.values())
          expect(current.has('storage-read') && current.has('storage-read-write'), node.id).toBe(
            false
          );
      }
    } finally {
      device.destroy();
    }
  });
}

describe('getGPULocalOutlierFactorParameterValues', () => {
  it('packs threshold and density floor; an omitted threshold is NaN, not a hidden 1.5', () => {
    const values = getGPULocalOutlierFactorParameterValues();
    expect(values.length).toBe(GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH);
    expect(Number.isNaN(values[0])).toBe(true);
    expect(values[1]).toBe(Math.fround(1e-10));
    expect(
      Array.from(getGPULocalOutlierFactorParameterValues({threshold: 2, densityFloor: 0.5}))
    ).toEqual([2, 0.5, 0, 0]);
  });

  it('rejects a non-positive density floor and a NaN threshold', () => {
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 0})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 1e-50})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({threshold: NaN})).toThrow();
  });

  it('rejects density floors that pack to infinity or give an infinite reciprocal', () => {
    for (const densityFloor of [1e300, 1e-40])
      expect(() => getGPULocalOutlierFactorParameterValues({densityFloor})).toThrow();
  });
});

it('does not flag an island even when its fallback LOF of one exceeds the threshold', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floating = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const unsigned = (length: number) =>
      createTransientView(graph, `uint-${serial++}`, 'uint32', length);
    const nodes = new GPULocalOutlierFactor({
      neighbors: {
        offsets: unsigned(9),
        neighbors: unsigned(16),
        weights: floating(16),
        distances: floating(16)
      },
      parameters: floating(4),
      output: {
        kDistance: floating(8),
        localReachabilityDensity: floating(8),
        lof: floating(8),
        outlier: unsigned(8)
      }
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.id.endsWith('-outlier')
    )!.source;
    // Execute the actual scalar classification expression, as in the Rose arithmetic probe.
    // Independent definition: an island's fallback score 1 is never an outlier.
    const expression = source.match(/let flagged = ([\s\S]*?);/)![1];
    const evaluate = new Function(
      'hasThreshold',
      'lof',
      'lofOffset',
      'index',
      'threshold',
      'density',
      'densityOffset',
      `const isNanBits = Number.isNaN; return ${expression};`
    );
    expect(evaluate(true, [1], 0, 0, 0.5, [0], 0)).toBe(false);
    expect(evaluate(true, [2], 0, 0, 1.25, [0.5], 0)).toBe(true);
    expect(evaluate(true, [1.25], 0, 0, 1.25, [0.5], 0)).toBe(false);
    expect(evaluate(false, [2], 0, 0, NaN, [0.5], 0)).toBe(false);
  } finally {
    device.destroy();
  }
});

it('LOF checks the density floor context against the neighbor distances when both are declared', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `context-${serial++}`, format, length);
  const context = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  } as const;
  const props = {
    neighbors: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 6),
      weights: view('float32', 6),
      distances: view('float32', 6),
      spatialContext: context
    },
    parameters: view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH),
    output: {
      kDistance: view('float32', 3),
      localReachabilityDensity: view('float32', 3),
      lof: view('float32', 3)
    }
  };
  expect(
    () =>
      new GPULocalOutlierFactor({
        ...props,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      })
  ).toThrow(/incompatible spatial contexts/);
  expect(
    () => new GPULocalOutlierFactor({...props, spatialContext: {...context, units: 'kilometers'}})
  ).toThrow(/incompatible spatial contexts/);
  expect(() => new GPULocalOutlierFactor({...props, spatialContext: {...context}})).not.toThrow();
  expect(() => new GPULocalOutlierFactor(props)).not.toThrow();
  expect(
    () =>
      new GPULocalOutlierFactor({
        ...props,
        neighbors: {...props.neighbors, spatialContext: undefined},
        spatialContext: context
      })
  ).not.toThrow();
  // Only a k-nearest-neighbor table has the k-distance as the largest row distance.
  const withCriterion = (criterion: string) => ({
    ...props,
    neighbors: {
      ...props.neighbors,
      information: {provenance: {source: 'test', criterion}}
    }
  });
  expect(() => new GPULocalOutlierFactor(withCriterion('distance-band') as never)).toThrow(
    /k-nearest-neighbor/
  );
  device.destroy();
});

it('WGS-13 instruments all three resident CSR walks within the one allocated slot', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `bounded-lof-${serial++}`, format, length);
  try {
    const nodes = new GPULocalOutlierFactor({
      neighbors: {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 1),
        weights: view('float32', 1),
        distances: view('float32', 1)
      },
      parameters: view('float32', 4),
      output: {
        kDistance: view('float32', 1),
        localReachabilityDensity: view('float32', 1),
        lof: view('float32', 1)
      }
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
      /-(k-distance|density|score)$/.test(kernel.id)
    );
    expect(kernels.map(kernel => kernel.id).sort()).toEqual([
      'local-outlier-factor-density',
      'local-outlier-factor-k-distance',
      'local-outlier-factor-score'
    ]);
    for (const kernel of kernels) {
      expect(Number(kernel.source.match(/const SLOTS: u32 = (\d+)u;/)![1])).toBe(1);
      const countVisits = (source: string, offsets: number[]) => {
        const header = source.match(/for \(var slot = [^\n]+\)/)![0];
        const slots = Number(source.match(/const SLOTS: u32 = (\d+)u;/)![1]);
        const javascript = header
          .replaceAll(/(\d+)u\b/g, '$1')
          .replaceAll('min(', 'Math.min(')
          .replaceAll('SLOTS', String(slots));
        return new Function(
          'offsets',
          'offsetsOffset',
          'index',
          `let visits = 0; ${javascript} {visits++;} return visits;`
        )(offsets, 0, 0);
      };
      expect(countVisits(kernel.source, [0, 3]), kernel.id).toBe(1);
      expect(countVisits(kernel.source, [3, 4]), kernel.id).toBe(0);
      expect(countVisits(kernel.source, [3, 0]), kernel.id).toBe(0);
      // A mutation of the emitted capacity must affect the instrumentation itself.
      expect(
        countVisits(
          kernel.source.replace('const SLOTS: u32 = 1u;', 'const SLOTS: u32 = 3u;'),
          [0, 3]
        )
      ).toBe(3);
    }
  } finally {
    device.destroy();
  }
});
