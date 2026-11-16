// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments,
  findConstantNonFiniteBitcasts
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function fixture(rows: number, dense = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const keys = createVectorView(
    'keys',
    'uint32',
    [3, 0, rows - 3].map((length, index) =>
      createTransientView(graph, `keys-${index}`, 'uint32', length)
    )
  );
  const values = createVectorView(
    'values',
    'float32',
    [7, rows - 7].map((length, index) =>
      createTransientView(graph, `values-${index}`, 'float32', length)
    )
  );
  const props = {
    keys,
    scratchTileSize: 16,
    keyCount: dense ? 4 : undefined,
    output: {
      keys: createTransientView(graph, 'keys-out', 'uint32', 4),
      counts: createTransientView(graph, 'counts-out', 'uint32', 4),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1)
      }
    },
    columns: [
      {
        values,
        statistics: [
          'count',
          'sum',
          'mean',
          'minimum',
          'maximum',
          'variance',
          'standardDeviation',
          'skewness',
          'kurtosis'
        ] as const,
        output: {
          counts: createTransientView(graph, 'finite-counts', 'uint32', 4),
          sums: createTransientView(graph, 'sums', 'uint32x2', 4),
          sumValues: createTransientView(graph, 'sum-values', 'float32', 4),
          means: createTransientView(graph, 'means', 'float32', 4),
          minimums: createTransientView(graph, 'minimums', 'float32', 4),
          maximums: createTransientView(graph, 'maximums', 'float32', 4),
          variances: createTransientView(graph, 'variances', 'float32', 4),
          standardDeviations: createTransientView(graph, 'deviations', 'float32', 4),
          skewness: createTransientView(graph, 'skewness', 'float32', 4),
          kurtosis: createTransientView(graph, 'kurtosis', 'float32', 4)
        }
      }
    ]
  };
  return {device, graph, props};
}

for (const dense of [false, true])
  it(`fixed group tiles reuse sources and pipelines (${dense ? 'dense' : 'compact'})`, () => {
    // Equal total-row headroom, fixed tile width, varying tile counts and tails.
    const sets = [130, 177, 225].map(rows => {
      const {device, graph, props} = fixture(rows, dense);
      try {
        const contributor = new GPUGroupStatistics(props);
        const nodes = contributor.getCommandNodes(graph);
        expect(contributor.usesTiledScratch).toBe(true);
        expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
        getParsedKernelSources(nodes);
        for (const {source, id} of getGPUCommandNodeKernelSources(nodes)) {
          expect(findConstantNonFiniteBitcasts(stripWGSLComments(source)), id).toEqual([]);
          expect(
            [...new Set(stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(
              word => WGSL_RESERVED_WORDS.has(word)
            ),
            id
          ).toEqual([]);
        }
        return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
      } finally {
        device.destroy();
      }
    });
    expect([...sets[0]].sort()).toEqual([...sets[1]].sort());
    expect([...sets[0]].sort()).toEqual([...sets[2]].sort());
  });

it('group tile scratch failures leave the graph unchanged', () => {
  const {device, graph, props} = fixture(49);
  try {
    device.limits.maxStorageBufferBindingSize = 200;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => new GPUGroupStatistics(props).getCommandNodes(graph)).toThrow();
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('tiled signed moments restore their scale only after all tile merges', () => {
  const {device, graph, props} = fixture(130);
  try {
    const nodes = new GPUGroupStatistics(props).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const localFinalizers = sources.filter(({id}) => /-tile-\d+-moments$/.test(id));
    expect(localFinalizers).toHaveLength(9);
    for (const {source} of localFinalizers) {
      expect(source).toContain('= power1[');
      expect(source).toContain('= power3[');
      expect(source).not.toContain('ldexp(');
    }
    const restore = sources.filter(({id}) => id.endsWith('-moments-restore'));
    expect(restore).toHaveLength(1);
    expect(restore[0].source).toContain('ldexp(ldexp(ldexp(moments[third]');
    const restoreIndex = nodes.findIndex(node => node.id.endsWith('-moments-restore'));
    const merges = nodes.flatMap((node, index) =>
      /-moments-merge-\d+$/.test(node.id) ? [index] : []
    );
    expect(merges).toHaveLength(9);
    expect(Math.max(...merges)).toBeLessThan(restoreIndex);
  } finally {
    device.destroy();
  }
});

it('group tile preflight rejects missing STORAGE usage and nested scratch buffer limits atomically', () => {
  for (const failure of ['storage', 'scratch'] as const) {
    const {device, graph, props} = fixture(49);
    try {
      const output =
        failure === 'storage'
          ? {
              ...props.output,
              counts: graph.createDataView(
                graph.createTransientBuffer({id: 'non-storage', usage: 8, byteLength: 16}),
                {format: 'uint32', length: 4}
              )
            }
          : props.output;
      const contributor = new GPUGroupStatistics({...props, output});
      if (failure === 'scratch') device.limits.maxBufferSize = 255;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        failure === 'storage' ? /Buffer.STORAGE/ : /tile moment scratch byteLength/
      );
      expect(allocate).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('exact order statistics retain a clear packed limit and fail before graph mutation', () => {
  const {device, graph, props} = fixture(2049);
  try {
    device.limits.maxStorageBufferBindingSize = 8192;
    const contributor = new GPUGroupStatistics({
      ...props,
      columns: [
        {
          values: props.columns[0].values,
          statistics: ['median'],
          output: {medians: createTransientView(graph, 'medians', 'float32', 4)}
        }
      ]
    });
    expect(contributor.usesTiledScratch).toBe(false);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /exact order statistics and zScore retain the packed row-scratch limit/
    );
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('group tile widths reject non-finite, unaddressable and non-power-of-two sizes at construction', () => {
  const {device, props} = fixture(49);
  try {
    for (const scratchTileSize of [NaN, Infinity, -1, 0, 3, 2 ** 30])
      expect(() => new GPUGroupStatistics({...props, scratchTileSize})).toThrow(/scratchTileSize/);
  } finally {
    device.destroy();
  }
});
