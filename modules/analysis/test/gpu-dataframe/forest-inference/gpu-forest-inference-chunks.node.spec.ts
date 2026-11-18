// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUForestInference,
  type GPUForestInferenceProps
} from '../../../src/dataframe-analysis/forest-inference/gpu-forest-inference';
import {GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT} from '../../../src/dataframe-analysis/forest-inference/forest-inference-parameters';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../calendar-buckets/chunk-shader-validation.test-utils';

function createFixture(
  lengths: number[],
  mode: 'row-major' | 'feature-major' | 'raster' = 'row-major',
  featureCount = 2
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rowCount = lengths.reduce((sum, rows) => sum + rows, 0);
  const column = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    width: number
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((rows, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, rows * width + 128);
        return graph.createDataView(backing.buffer, {
          format,
          byteOffset: (index % 64) * 4,
          length: rows * width
        });
      })
    );
  const model = {
    featureCount,
    treeCount: 1,
    nodeCount: 3,
    outputCount: 1,
    maximumDepth: 1,
    task: 'regression' as const,
    transform: 'identity' as const
  };
  const props: GPUForestInferenceProps<GraphVectorView> = {
    id: 'forest',
    model,
    rowCount,
    nodes: createTransientView(graph, 'nodes', 'uint32', 18),
    trees: createTransientView(graph, 'trees', 'uint32', 4),
    leafValues: createTransientView(graph, 'leaves', 'float32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 2),
    ...(mode === 'raster'
      ? {
          rasterStack: {
            bands: Array.from({length: featureCount}, (_, index) => ({
              id: `band-${index}`,
              format: 'float32' as const,
              storage: {
                kind: 'buffer' as const,
                values: createTransientView(graph, `band-${index}`, 'float32', rowCount)
              },
              validity: createTransientView(graph, `band-validity-${index}`, 'uint32', rowCount)
            })),
            grid: {
              width: rowCount,
              height: 1,
              affine: [1, 0, 0, 0, -1, 0],
              pixelInterpretation: 'area'
            }
          }
        }
      : {features: column('features', 'float32', featureCount), featureLayout: mode}),
    mask: column('mask', 'uint32', 1),
    output: {
      predictions: column('predictions', 'float32', 1),
      validity: column('validity', 'uint32', 1),
      scores: column('scores', 'float32', 1),
      leafIndices: column('leafIndices', 'uint32', 1)
    }
  };
  return {device, graph, props, column};
}

it('accepts shared output storage at byte 256 but rejects aligned overlap at byte 4', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    const backing = createTransientView(graph, 'shared-output-storage', 'uint32', 128);
    const predictions = graph.createDataView(backing.buffer, {format: 'float32', length: 1});
    const output = {
      ...props.output,
      predictions: createVectorView('shared-predictions', 'float32', [predictions]),
      validity: createVectorView('shared-validity', 'uint32', [
        graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 256})
      ])
    };
    const contributor = new GPUForestInference({...props, output});
    expect(contributor.getCommandNodes(graph).filter(node => node.type === 'compute')).toHaveLength(
      2
    );
    expect(() =>
      new GPUForestInference({
        ...props,
        output: {
          ...output,
          validity: createVectorView('overlapping-validity', 'uint32', [
            graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 4})
          ])
        }
      }).getCommandNodes(graph)
    ).toThrow(/aligned binding.*must not overlap/);
  } finally {
    device.destroy();
  }
});

it('validates whole-row topology, uint32 totals and every overlapping writable range', () => {
  const {device, graph, props} = createFixture([1, 3, 2]);
  try {
    expect(GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT).toBe(0xffffffff);
    expect(() => new GPUForestInference({...props, rowCount: 0x100000000})).toThrow(
      /rowCount.*4294967295/
    );
    const features = props.features as GraphVectorView<'float32'>;
    expect(
      () =>
        new GPUForestInference({
          ...props,
          rasterStack: undefined,
          features: createVectorView('reversed', 'float32', [...features.data].reverse())
        })
    ).toThrow(/features chunk 0.*topology/);
    const first = graph.createDataView(features.data[0].buffer, {format: 'float32', length: 3});
    const second = graph.createDataView(features.data[1].buffer, {format: 'float32', length: 5});
    expect(
      () =>
        new GPUForestInference({
          ...props,
          rasterStack: undefined,
          features: createVectorView('partial', 'float32', [first, second, features.data[2]])
        })
    ).toThrow(/features chunks must contain whole rows of 2 elements/);
    const predictions = props.output.predictions as GraphVectorView<'float32'>;
    const overlapping = graph.createDataView(predictions.data[0].buffer, {
      format: 'float32',
      byteOffset: 0,
      length: 3
    });
    expect(
      () =>
        new GPUForestInference({
          ...props,
          output: {
            ...props.output,
            predictions: createVectorView('overlap', 'float32', [
              predictions.data[0],
              overlapping,
              predictions.data[2]
            ])
          }
        })
    ).toThrow(/must not overlap/);
    const validity = props.output.validity as GraphVectorView<'uint32'>;
    const alias = graph.createDataView(predictions.data[1].buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 3
    });
    expect(
      () =>
        new GPUForestInference({
          ...props,
          output: {
            ...props.output,
            validity: createVectorView('alias', 'uint32', [
              validity.data[0],
              alias,
              validity.data[2]
            ])
          }
        })
    ).toThrow(/must not overlap/);
  } finally {
    device.destroy();
  }
});

it('keeps traversal and model budgets per batch while reusing one score scratch buffer', () => {
  const {device, graph, props} = createFixture([50_000, 50_000, 50_000]);
  try {
    const model = {...props.model, treeCount: 1024, maximumDepth: 64};
    const trees = createTransientView(graph, 'many-trees', 'uint32', 4096);
    const contributor = new GPUForestInference({
      ...props,
      model,
      trees,
      output: {...props.output, leafIndices: undefined}
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(contributor.getCommandNodes(graph)).toHaveLength(6);
    expect(allocate).toHaveBeenCalledTimes(1);
    expect(allocate.mock.calls[0][0].byteLength).toBe(50_000 * 3 * 4);
    expect(
      () => new GPUForestInference({...props, model: {...model, maximumDepth: 65}, trees})
    ).toThrow(/maximumDepth.*64/);
    const tooMuch = createFixture([80_000]);
    expect(
      () =>
        new GPUForestInference({
          ...tooMuch.props,
          model,
          trees: createTransientView(tooMuch.graph, 'many-trees', 'uint32', 4096),
          output: {...tooMuch.props.output, leafIndices: undefined}
        })
    ).toThrow(/work.*5000000000/);
    tooMuch.device.destroy();
  } finally {
    device.destroy();
  }
});

it('prechecks actual flattened/model bindings and all derived scratch before graph mutations', () => {
  for (const failure of [
    'features',
    'output',
    'parameters',
    'scratch',
    'raster-features'
  ] as const) {
    const raster = failure === 'raster-features';
    const {device, graph, props} = createFixture(
      failure === 'scratch' ? [1, 24, 2] : raster ? [1, 20, 2] : [1, 3, 2],
      raster ? 'raster' : 'row-major',
      raster ? 4 : 2
    );
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 256});
    const backing = createTransientView(graph, 'oversized', 'float32', 128);
    if (failure === 'features') {
      const features = props.features as GraphVectorView<'float32'>;
      props.features = createVectorView('late', 'float32', [
        features.data[0],
        graph.createDataView(backing.buffer, {format: 'float32', byteOffset: 252, length: 6}),
        features.data[2]
      ]);
    }
    if (failure === 'output') {
      const scores = props.output.scores as GraphVectorView<'float32'>;
      props.output.scores = createVectorView('late', 'float32', [
        scores.data[0],
        graph.createDataView(backing.buffer, {format: 'float32', byteOffset: 252, length: 3}),
        scores.data[2]
      ]);
    }
    if (failure === 'parameters')
      props.parameters = graph.createDataView(backing.buffer, {
        format: 'float32',
        byteOffset: 252,
        length: 2
      });
    const contributor = new GPUForestInference(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      failure === 'scratch'
        ? /scratch needs 288 bytes/
        : raster
          ? /packed raster features needs 320 bytes/
          : /needs .*bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /views must belong/
    );
    device.destroy();
  }
});

it('checks reusable scratch allocations against maxBufferSize before changing the graph', () => {
  for (const raster of [false, true]) {
    const {device, graph, props} = createFixture(
      raster ? [1, 50, 2] : [1, 100, 2],
      raster ? 'raster' : 'row-major',
      raster ? 6 : 1
    );
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2048});
    Object.defineProperty(device.limits, 'maxBufferSize', {value: 1024, configurable: true});
    const contributor = new GPUForestInference(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    try {
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        raster
          ? /packed raster features buffer byteLength.*1200.*1024/
          : /scratch buffer byteLength.*1200.*1024/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      Object.defineProperty(device.limits, 'maxBufferSize', {value: 1200});
      expect(contributor.getCommandNodes(graph)).toHaveLength(
        contributor.estimatedCommandNodeCount
      );
      const allocationSizes = allocate.mock.calls.map(([descriptor]) => descriptor.byteLength);
      expect(allocationSizes).toContain(1200);
      expect(allocationSizes.every(byteLength => byteLength <= 1200)).toBe(true);
    } finally {
      device.destroy();
    }
  }
});

for (const mode of ['row-major', 'feature-major', 'raster'] as const) {
  it(`keeps ${mode} shader/pipeline counts constant across chunks and supports 30 million rows`, () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const lengths of [
      [513],
      [513, 1, 257],
      Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
      [0, 10_000_000, 9_000_000, 11_000_000]
    ]) {
      const {device, graph, props} = createFixture(lengths, mode);
      const contributor = new GPUForestInference(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      getParsedKernelSources(nodes);
      expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(
        mode === 'raster' ? 3 : 2
      );
      if (lengths[0] === 513)
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          keys.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
      expect(
        getGPUCommandNodeKernelSources(nodes).every(
          ({source}) => (source.match(/var<storage,/g) ?? []).length <= 8
        )
      ).toBe(true);
      device.destroy();
    }
    expect(keys.size).toBe(mode === 'raster' ? 3 : 2);
    expect(sources.size).toBe(mode === 'raster' ? 3 : 2);
    expectValidShaderIdentifiers(sources);
  });
}
