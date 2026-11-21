// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {
  getGPUDBSCANParameterValues,
  GPUDBSCAN,
  GPU_DBSCAN_PARAMETER_LENGTH,
  type GPUDBSCANProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 32;
let viewCounter = 0;
const nextName = (name: string) => `${name}-${viewCounter++}`;
const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

function createClusteringProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUDBSCANProps> = {}
): GPUDBSCANProps {
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number,
    name: string
  ): GraphDataView<Format> => createTransientView(graph, nextName(name), format, length);
  return {
    positions: view('float32x2', ROWS, 'positions'),
    parameters: view('float32', GPU_DBSCAN_PARAMETER_LENGTH, 'parameters'),
    gridSize: [8, 8],
    output: {labels: view('uint32', ROWS, 'labels')},
    ...overrides
  };
}

describe('ST-DBSCAN parameters', () => {
  const settings = {bounds: [0, 0, 10, 10] as const, epsilon: 2, minimumSamples: 4};

  it('stores timeEpsilon at index 6 and leaves it 0 without one', () => {
    expect(Array.from(getGPUDBSCANParameterValues({...settings, timeEpsilon: 5}))).toEqual([
      0, 0, 10, 10, 2, 4, 5, 0
    ]);
    expect(Array.from(getGPUDBSCANParameterValues(settings))).toEqual([0, 0, 10, 10, 2, 4, 0, 0]);
  });

  it('rejects a non-positive or non-finite timeEpsilon', () => {
    for (const timeEpsilon of [0, -1, Number.NaN, Infinity]) {
      expect(() => getGPUDBSCANParameterValues({...settings, timeEpsilon})).toThrow(
        'timeEpsilon must be positive and finite'
      );
    }
  });
});

describe('ST-DBSCAN construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const times = createTransientView(graph, 'times', 'float32', ROWS);

  it('accepts float32-relative times and exposes the resolved context', () => {
    const contributor = new GPUDBSCAN(
      createClusteringProps(graph, {times, temporalContext: CONTEXT})
    );
    expect(contributor.temporalContext).toEqual(CONTEXT);
  });

  it('has no temporal context without times', () => {
    expect(new GPUDBSCAN(createClusteringProps(graph)).temporalContext).toBeUndefined();
  });

  it('rejects times without a context, a context without times, and other encodings', () => {
    expect(() => new GPUDBSCAN(createClusteringProps(graph, {times}))).toThrow(
      'times need a temporalContext'
    );
    expect(() => new GPUDBSCAN(createClusteringProps(graph, {temporalContext: CONTEXT}))).toThrow(
      'temporalContext needs times'
    );
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {
            times,
            temporalContext: {...CONTEXT, encoding: 'int64-words', epoch: 'unix'}
          })
        )
    ).toThrow('does not support int64-words times');
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {
            times,
            temporalContext: {...CONTEXT, epoch: 'unix'}
          })
        )
    ).toThrow('float32-relative times need a bigint origin');
  });

  it('rejects wrong time formats and lengths and the dense-box shortcut', () => {
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {
            times: createTransientView(graph, 'short-times', 'float32', ROWS - 1),
            temporalContext: CONTEXT
          })
        )
    ).toThrow('times length must equal positions length');
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {
            times: createTransientView(graph, 'uint-times', 'uint32', ROWS) as never,
            temporalContext: CONTEXT
          })
        )
    ).toThrow();
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {times, temporalContext: CONTEXT, denseBoxShortcut: true})
        )
    ).toThrow('denseBoxShortcut cannot be combined with times');
  });

  it('rejects an output that aliases times', () => {
    expect(
      () =>
        new GPUDBSCAN(
          createClusteringProps(graph, {
            times,
            temporalContext: CONTEXT,
            output: {labels: times as never}
          })
        )
    ).toThrow();
  });
});

describe('ST-DBSCAN kernels', () => {
  function getKernels(withTimes: boolean) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const nodes = new GPUDBSCAN(
      createClusteringProps(
        graph,
        withTimes
          ? {times: createTransientView(graph, 'times', 'float32', ROWS), temporalContext: CONTEXT}
          : {}
      )
    ).getCommandNodes(graph);
    getParsedKernelSources(nodes);
    return getGPUCommandNodeKernelSources(nodes);
  }

  it('adds the time test to the core, union and border scans only', () => {
    const kernels = getKernels(true);
    const temporal = kernels.filter(kernel => kernel.source.includes('isWithinTime(lattice'));
    expect(
      temporal.map(kernel => kernel.id.replace(/^.*?(core|union|border).*$/, '$1')).sort()
    ).toEqual(['border', 'core', 'union']);
    for (const kernel of temporal) {
      expect(kernel.source, kernel.id).toContain('abs(tb - ta) <= lattice.timeEpsilon');
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
    expect(kernels.some(kernel => /cell-times/.test(kernel.id))).toBe(true);
  });

  it('leaves every kernel free of time code when times are absent', () => {
    for (const kernel of getKernels(false)) {
      expect(kernel.source, kernel.id).not.toMatch(/[Tt]ime/);
    }
  });
});

describe('ST-OPTICS construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, nextName(name), format, length);
  const createProps = (overrides: Record<string, unknown> = {}) => ({
    positions: view('float32x2', ROWS, 'optics-positions'),
    minimumSamples: 4,
    maximumDistance: 3,
    extractionDistance: 2,
    output: {
      ordering: view('uint32', ROWS, 'ordering'),
      coreDistances: view('float32', ROWS, 'core'),
      reachability: view('float32', ROWS, 'reachability'),
      predecessors: view('uint32', ROWS, 'predecessors'),
      labels: view('uint32', ROWS, 'optics-labels')
    },
    ...overrides
  });
  it('reports missing OPTICS ports and structured received row counts', () => {
    const props = createProps({id: 'survey'});
    expect(
      () => new GPUOPTICS({...props, output: {...props.output, ordering: undefined as never}})
    ).toThrow(/survey output.ordering/);
    // A missing positions port is diagnosed by the structured validator, not a TypeError from
    // reading its row count.
    try {
      new GPUOPTICS({...props, positions: undefined as never});
      throw new Error('expected missing positions to reject');
    } catch (error) {
      expect(error).not.toBeInstanceOf(TypeError);
      expect((error as Error).message).toMatch(/survey positions/);
    }
    try {
      new GPUOPTICS({
        ...props,
        output: {...props.output, ordering: view('uint32', 3, 'short-ordering')}
      });
      throw new Error('expected short ordering to reject');
    } catch (error) {
      expect((error as {code: unknown}).code).toBe('VIEW_LENGTH');
      expect((error as Error).message).toBe(
        'survey output.ordering must have exactly 32 rows. Received 3'
      );
      expect((error as {received: unknown}).received).toBe(3);
      expect((error as {path: unknown}).path).toBe('output.ordering');
      expect((error as {expected: unknown}).expected).toEqual({rows: ROWS});
    }
  });

  const times = view('float32', ROWS, 'optics-times');

  it('accepts times with a context and a positive window', () => {
    const optics = new GPUOPTICS(
      createProps({times, temporalContext: CONTEXT, maximumTimeDistance: 5})
    );
    expect(optics.temporalContext).toEqual(CONTEXT);
  });

  it('rejects incomplete or stray time options', () => {
    expect(() => new GPUOPTICS(createProps({times, maximumTimeDistance: 5}))).toThrow();
    expect(() => new GPUOPTICS(createProps({times, temporalContext: CONTEXT}))).toThrow();
    for (const maximumTimeDistance of [0, -1, Number.NaN]) {
      expect(
        () => new GPUOPTICS(createProps({times, temporalContext: CONTEXT, maximumTimeDistance}))
      ).toThrow();
    }
    expect(() => new GPUOPTICS(createProps({temporalContext: CONTEXT}))).toThrow();
    expect(() => new GPUOPTICS(createProps({maximumTimeDistance: 5}))).toThrow();
    expect(
      () =>
        new GPUOPTICS(
          createProps({
            times,
            temporalContext: {...CONTEXT, encoding: 'double-single'},
            maximumTimeDistance: 5
          })
        )
    ).toThrow('does not support double-single times');
  });

  it('adds the time test to the core and frontier kernels only when times are present', () => {
    const buildKernels = (overrides: Record<string, unknown>) => {
      const kernelGraph = new GPUCommandGraph(createNullWebGPUDevice());
      const kernelView = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        length: number,
        name: string
      ) => createTransientView(kernelGraph, name, format, length);
      const nodes = new GPUOPTICS({
        positions: kernelView('float32x2', ROWS, 'positions'),
        minimumSamples: 4,
        maximumDistance: 3,
        extractionDistance: 2,
        output: {
          ordering: kernelView('uint32', ROWS, 'ordering'),
          coreDistances: kernelView('float32', ROWS, 'core'),
          reachability: kernelView('float32', ROWS, 'reachability'),
          predecessors: kernelView('uint32', ROWS, 'predecessors'),
          labels: kernelView('uint32', ROWS, 'labels')
        },
        ...(overrides['times'] ? {times: kernelView('float32', ROWS, 'times')} : {}),
        ...(overrides['times'] ? {temporalContext: CONTEXT, maximumTimeDistance: 5} : {})
      }).getCommandNodes(kernelGraph);
      getParsedKernelSources(nodes);
      return getGPUCommandNodeKernelSources(nodes);
    };
    for (const kernel of buildKernels({})) {
      expect(kernel.source, kernel.id).not.toMatch(/[Tt]ime/);
    }
    const temporal = buildKernels({times: true}).filter(kernel =>
      kernel.source.includes('MAXIMUM_TIME_DISTANCE')
    );
    expect(temporal.length).toBeGreaterThanOrEqual(3);
    for (const kernel of temporal) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
  });
});

it('rejects short bounds before writing and accepts typed-array bounds', () => {
  const settings = {epsilon: 8, minimumSamples: 3};
  const target = new Float32Array(8).fill(77);
  expect(() =>
    getGPUDBSCANParameterValues({...settings, bounds: [-1, -1, 1]} as never, target)
  ).toThrow(/bounds.*4/);
  expect(Array.from(target)).toEqual(Array(8).fill(77));
  expect(
    Array.from(
      getGPUDBSCANParameterValues({
        ...settings,
        bounds: Float32Array.of(-1, -1, 1, 1)
      } as never)
    )
  ).toEqual([-1, -1, 1, 1, 8, 3, 0, 0]);
});

it('binds the time epsilon requirement to the constructed topology', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const clustering = new GPUDBSCAN(
      createClusteringProps(graph, {
        times: createTransientView(graph, 'w3-times', 'float32', ROWS),
        temporalContext: CONTEXT
      })
    );
    const settings = {bounds: [0, 0, 1, 1] as const, epsilon: 1, minimumSamples: 1};
    expect(() => clustering.getParameterValues(settings)).toThrow(/timeEpsilon/);
    expect(clustering.getParameterValues({...settings, timeEpsilon: 1})[6]).toBe(1);
  } finally {
    device.destroy();
  }
});
