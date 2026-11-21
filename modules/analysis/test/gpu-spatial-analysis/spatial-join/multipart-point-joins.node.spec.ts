// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  GPUVisibilityWorkflow,
  createTransientView,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {CPUGPUCompaction} from '../segment-intersection/cpu-gpu-core-emulations';
import {expect, it, vi, beforeEach, afterEach} from 'vitest';
import * as segmentSums from '../../../src/analysis-core/kernels/sorted-segment-sums';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {adaptUniforms} from './multipart-point-joins-executor';
import {
  multipartInputs,
  joinOutputs,
  bufferOutputs,
  lineOutputs,
  recipeOutputs,
  createMultipartJoin,
  createMultipartBuffer,
  createMultipartLines,
  createMultipartRecipe,
  assertMultipartJoin,
  assertMultipartBuffer,
  assertMultipartLines,
  assertMultipartRecipe
} from './multipart-point-joins-fixture';
import type {FixtureOutput, FixtureInput} from '../outline-geometry/geometry-fixture';
import {
  createReviewCases,
  reviewCPUInputs,
  resolveReviewInputViews
} from './multipart-point-review-fixture';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/gpu-group-statistics';
import {captureGraphCommandNodes} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-operation';
import {
  createCPUPointsInPolygonsChoroplethOperation,
  computePointsInPolygonsChoroplethOnCPU
} from '../../../src/gpu-spatial-analysis/recipes/cpu-points-in-polygons-choropleth-operation';
import {choroplethOperationProps} from './multipart-point-operations-fixture';
import {createLineLengthPerPolygonOperation} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon-operation';
import {
  createCPULineLengthPerPolygonOperation,
  computeLineLengthPerPolygonOnCPU
} from '../../../src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation';
import {lineOperationProps} from './multipart-point-operations-fixture';
import {createPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import {
  createCPUPointInPolygonJoinOperation,
  computePointInPolygonJoinOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {pointJoinOperationProps} from './multipart-point-operations-fixture';
import {createBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import {
  createCPUBufferSelectionOperation,
  computeBufferSelectionOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import {
  bufferOperationProps,
  getOperationCPUInputs,
  getOperationResult
} from './multipart-point-operations-fixture';

useCPUExecutorTestHooks();
const restoreLocalMocks: (() => void)[] = [];
afterEach(() => {
  for (const restore of restoreLocalMocks.splice(0).reverse()) restore();
});
beforeEach(() => {
  // Cooperative GroupStatistics kernels cannot run in the sequential interpreter. This bounded
  // fixture's float32 values have exact binary64 sums; this stand-in checks membership wiring
  // only. R4 headless cases exercise the original fixed-point accumulation and both orders.
  const groups = vi
    .spyOn(GPUGroupStatistics.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUGroupStatistics) {
      const props = this.props;
      return [
        {
          id: `${this.id}-cpu-group-contract`,
          type: 'compute',
          resources: [],
          compile: () => ({
            encode: (context: {getBuffer(view: GraphDataView): {backing: ArrayBuffer}}) => {
              const read = (view: GraphDataView) =>
                new Uint32Array(context.getBuffer(view).backing, view.byteOffset, view.length);
              const floats = (view: GraphDataView) =>
                new Float32Array(context.getBuffer(view).backing, view.byteOffset, view.length);
              const keys = read(props.keys as GraphDataView),
                count = props.keyCount!;
              read(props.output.keys).set(Array.from({length: count}, (_, row) => row));
              const counts = read(props.output.counts);
              counts.fill(0);
              for (const key of keys) if (key < count) counts[key]++;
              read(props.output.status.count)[0] = count;
              read(props.output.status.overflow)[0] = 0;
              if (props.output.status.requiredCount)
                read(props.output.status.requiredCount)[0] = count;
              for (const column of props.columns) {
                const values = floats(column.values as GraphDataView);
                for (let group = 0; group < count; group++) {
                  const selected = Array.from(values).filter(
                    (value, row) => keys[row] === group && Number.isFinite(value)
                  );
                  const sum = selected.reduce((total, value) => total + value, 0);
                  for (const [field, view] of Object.entries(column.output)) {
                    if (!view || !('buffer' in view)) continue;
                    floats(view)[group] =
                      field === 'sumValues'
                        ? sum
                        : field === 'means'
                          ? selected.length
                            ? sum / selected.length
                            : NaN
                          : field === 'minimums'
                            ? selected.length
                              ? Math.min(...selected)
                              : NaN
                            : selected.length
                              ? Math.max(...selected)
                              : NaN;
                  }
                }
              }
            }
          })
        }
      ] as unknown as ReturnType<GPUGroupStatistics['getCommandNodes']>;
    });
  restoreLocalMocks.push(() => groups.mockRestore());
  // The published visibility workflow expands compaction through its internal helpers, bypassing
  // the primitive prototype hooks. This fixture has one mask and explicit packed source IDs.
  const visibility = vi
    .spyOn(GPUVisibilityWorkflow.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUVisibilityWorkflow) {
      return new CPUGPUCompaction({
        id: this.id,
        input: this.sourceIds as GraphDataView<'uint32'>,
        flags: this.predicates[0].mask as GraphDataView<'uint32'>,
        output: this.output as GraphDataView<'uint32'>,
        count: this.count
      }).getCommandNodes() as ReturnType<GPUCommandNodeProducer<unknown>['getCommandNodes']>;
    });
  restoreLocalMocks.push(() => visibility.mockRestore());
  // Only the cooperative sum primitive needs a stand-in. Segment construction, owner lookup,
  // clipping, weights and count/status publication execute the production shaders.
  const sums = vi
    .spyOn(segmentSums, 'getSortedSegmentSumNodes')
    .mockImplementation((graph, props) => {
      const reductions = [
        ...(props.sums && props.sumContributions
          ? [{output: props.sums, contributions: props.sumContributions}]
          : []),
        ...(props.reductions ?? [])
      ];
      return reductions.map((reduction, index) =>
        createWGSLKernelNode(graph, {
          id: `${props.id}-cpu-sum-${index}`,
          operation: props.operation,
          bindings: [
            {name: 'keys', view: props.segmentKeys, type: 'u32', access: 'read'},
            {name: 'values', view: reduction.contributions, type: 'f32', access: 'read'},
            {name: 'sums', view: reduction.output, type: 'f32', access: 'read_write'}
          ],
          invocationCount: props.segmentCount,
          body: `var total = 0.0; for (var row = 0u; row < ${props.segmentKeys.length}u; row++) { if (keys[keysOffset + row] == index) { total += values[valuesOffset + row]; } } sums[sumsOffset + index] = total;`
        })
      );
    });
  restoreLocalMocks.push(() => sums.mockRestore());
});
function run(
  outputs: Record<string, FixtureOutput>,
  create: (
    input: Record<string, GraphDataView>,
    output: Record<string, GraphDataView>,
    graph: GPUCommandGraph
  ) => GPUCommandNodeProducer,
  fixtureInputs: Record<string, FixtureInput> = multipartInputs
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const input = Object.fromEntries(
    Object.entries(fixtureInputs).map(([name, definition]) => {
      const view = createTransientView(
        graph,
        name,
        definition.format,
        definition.values.length / (definition.format === 'float32x2' ? 2 : 1)
      );
      memory.write(view, definition.values);
      return [name, view];
    })
  );
  const output = Object.fromEntries(
    Object.entries(outputs).map(([name, definition]) => [
      name,
      createTransientView(graph, `out-${name}`, definition.format, definition.length)
    ])
  );
  for (const view of Object.values(output))
    memory.write(view, new Uint32Array(view.length).fill(0xdeadbeef));
  try {
    const contributor = create(input, output, graph);
    executeCommandNodesOnCPU(adaptUniforms(contributor.getCommandNodes(graph)), memory);
    return Object.fromEntries(
      Object.entries(outputs).map(([name, definition]) => [
        name,
        Array.from(
          definition.format === 'uint32'
            ? memory.readUint32(output[name] as GraphDataView<'uint32'>)
            : memory.readFloat32(output[name] as GraphDataView<'float32'>)
        )
      ])
    );
  } finally {
    device.destroy();
  }
}
it('buffer CPU function and both operation forms match the Shapely literals on every output', async () => {
  const points = {
    kind: 'points' as const,
    positions: multipartInputs['points'].values as Float32Array,
    featureOffsets: multipartInputs['pointOffsets'].values as Uint32Array,
    sourceIds: multipartInputs['sourceIds'].values as Uint32Array
  };
  const features = {
    kind: 'segments' as const,
    starts: multipartInputs['starts'].values as Float32Array,
    ends: multipartInputs['ends'].values as Float32Array
  };
  const pure = computeBufferSelectionOnCPU({points, features, radius: 1}, bufferOperationProps);
  assertMultipartBuffer(
    getOperationResult(
      Object.fromEntries(
        Object.entries(bufferOperationProps.output).map(([port, name]) => [
          name,
          pure[port as keyof typeof pure]
        ])
      )
    )
  );
  const cpu = createCPUBufferSelectionOperation(bufferOperationProps);
  assertMultipartBuffer(
    getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}))
  );
  const gpu = createBufferSelectionOperation(bufferOperationProps);
  try {
    assertMultipartBuffer(
      run(
        bufferOutputs,
        (input, output) =>
          gpu.createContributor(
            port => (output[gpu.outputs[port]?.value] ?? input[gpu.inputs[port]]) as never
          ) as GPUCommandNodeProducer
      )
    );
  } finally {
    gpu.destroy();
  }
});
it('buffer CPU twin binds leaf loss and final compact capacity independently', async () => {
  const props = {...bufferOperationProps, leafCapacity: 1, matchCapacity: 2};
  const cpu = createCPUBufferSelectionOperation(props);
  const result = getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}));
  expect(result['ids']).toEqual([90, 100]);
  expect(result['count']).toEqual([2]);
  expect(result['requiredCount']).toEqual([4]);
  for (const name of ['overflow', 'candidateOverflow', 'matchCandidateOverflow', 'matchOverflow'])
    expect(result[name]).toEqual([1]);
  expect(result['truncated']).toEqual(Array(13).fill(1));
  expect(result['draw']).toEqual([2]);
});
for (const boundaryRule of ['open', 'closed', 'half-open'] as const)
  it(`polygon CPU function and both operations match Shapely ${boundaryRule} on every port`, async () => {
    const props = {...pointJoinOperationProps, boundaryRule};
    const cpu = createCPUPointInPolygonJoinOperation(props);
    assertMultipartJoin(
      getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)})),
      boundaryRule
    );
    const gpu = createPointInPolygonJoinOperation(props);
    try {
      assertMultipartJoin(
        run(
          joinOutputs,
          (input, output) =>
            gpu.createContributor(
              port => (output[gpu.outputs[port]?.value] ?? input[gpu.inputs[port]]) as never
            ) as GPUCommandNodeProducer
        ),
        boundaryRule
      );
    } finally {
      gpu.destroy();
    }
    const pure = computePointInPolygonJoinOnCPU(
      {
        points: {
          kind: 'points',
          positions: multipartInputs['points'].values as Float32Array,
          featureOffsets: multipartInputs['pointOffsets'].values as Uint32Array,
          sourceIds: multipartInputs['sourceIds'].values as Uint32Array
        },
        polygons: {
          kind: 'polygons',
          positions: multipartInputs['polygonPositions'].values as Float32Array,
          featureOffsets: multipartInputs['polygonFeatures'].values as Uint32Array,
          polygonOffsets: multipartInputs['polygonOffsets'].values as Uint32Array,
          ringOffsets: multipartInputs['ringOffsets'].values as Uint32Array,
          sourceIds: multipartInputs['polygonIds'].values as Uint32Array
        }
      },
      props
    );
    assertMultipartJoin(
      getOperationResult(
        Object.fromEntries(
          Object.entries(props.output).map(([port, name]) => [
            name,
            pure[port as keyof typeof pure]
          ])
        )
      ),
      boundaryRule
    );
  });
it('polygon CPU twin flags the owning feature after later part candidate loss', async () => {
  const props = {...pointJoinOperationProps, candidateCapacity: 1, candidateTileCapacity: 1};
  const cpu = createCPUPointInPolygonJoinOperation(props);
  const result = getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}));
  expect(result['truncated']).toEqual([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  expect(result['counts']).toEqual([1, 0, 0, 0, 0, 0]);
  expect(result['candidateCount']).toEqual([25]);
  expect(result['ids'][0]).toBe(90);
  expect(result['count']).toEqual([1]);
  expect(result['requiredCount']).toEqual([1]);
  expect(result['candidateOverflow']).toEqual([1]);
  expect(result['overflow']).toEqual([1]);
  expect(result['matchCandidateOverflow']).toEqual([1]);
  expect(result['matchOverflow']).toEqual([0]);
});
it('polygon CPU operation keeps final compact overflow separate from complete membership', async () => {
  const cpu = createCPUPointInPolygonJoinOperation({...pointJoinOperationProps, matchCapacity: 2});
  const result = getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}));
  expect(result['ids']).toEqual([90, 80]);
  expect(result['count']).toEqual([2]);
  expect(result['requiredCount']).toEqual([11]);
  expect(result['counts']).toEqual([7, 4, 2, 2, 2, 3]);
  expect(result['candidateCount']).toEqual([25]);
  expect(result['matchOverflow']).toEqual([1]);
  expect(result['overflow']).toEqual([1]);
  expect(result['candidateOverflow']).toEqual([0]);
  expect(result['matchCandidateOverflow']).toEqual([0]);
  expect(result['truncated']).toEqual(Array(13).fill(0));
});
it('line CPU function and both operation forms sum every part against the Shapely oracle', async () => {
  const cpu = createCPULineLengthPerPolygonOperation(lineOperationProps);
  assertMultipartLines(
    getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}))
  );
  const gpu = createLineLengthPerPolygonOperation(lineOperationProps);
  try {
    assertMultipartLines(
      run(
        lineOutputs,
        (input, output) =>
          gpu.createContributor(
            port => (output[gpu.outputs[port]?.value] ?? input[gpu.inputs[port]]) as never
          ) as GPUCommandNodeProducer
      )
    );
  } finally {
    gpu.destroy();
  }
  const pure = computeLineLengthPerPolygonOnCPU(
    {
      lines: {
        kind: 'lines',
        positions: multipartInputs['lines'].values as Float32Array,
        lineOffsets: multipartInputs['lineOffsets'].values as Uint32Array,
        featureOffsets: multipartInputs['lineFeatures'].values as Uint32Array
      },
      polygons: {
        kind: 'polygons',
        positions: multipartInputs['polygonPositions'].values as Float32Array,
        featureOffsets: multipartInputs['polygonFeatures'].values as Uint32Array,
        polygonOffsets: multipartInputs['polygonOffsets'].values as Uint32Array,
        ringOffsets: multipartInputs['ringOffsets'].values as Uint32Array
      },
      lineWeights: multipartInputs['weights'].values as Float32Array
    },
    lineOperationProps
  );
  assertMultipartLines({
    lengths: Array.from(pure.clippedLengths),
    weighted: Array.from(pure.weightedLengths),
    segments: Array.from(pure.segmentCounts),
    overflow: Array.from(pure.overflow),
    candidateOverflow: Array.from(pure.candidateOverflow)
  });
});
it('line CPU budgets retain one polygon pair and distinguish crossing loss', async () => {
  const cpu = createCPULineLengthPerPolygonOperation({...lineOperationProps, pairCapacity: 1});
  const result = getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}));
  // First line crosses x=0..4 at y=2.5: length 4, feature weight 2, one segment.
  expect(result['lengths']).toEqual([4, 0, 0, 0, 0, 0]);
  expect(result['weighted']).toEqual([8, 0, 0, 0, 0, 0]);
  expect(result['segments']).toEqual([1, 0, 0, 0, 0, 0]);
  expect(result['overflow']).toEqual([1]);
  expect(result['candidateOverflow']).toEqual([1]);
  const crossing = createCPULineLengthPerPolygonOperation({
    ...lineOperationProps,
    maximumCrossingCount: 1
  });
  const truncated = getOperationResult(
    await crossing.execute({inputs: getOperationCPUInputs(crossing.inputs)})
  );
  expect(truncated['overflow']).toEqual([1]);
  expect(truncated['candidateOverflow']).toEqual([0]);
});
for (const backend of ['group', 'zonal'] as const)
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const)
    it(`choropleth CPU function and both operations match Shapely ${backend} ${statistic}`, async () => {
      const props = {
        ...choroplethOperationProps,
        backend,
        statistic,
        output: {
          counts: 'counts',
          ...(statistic === 'count' ? {} : {featureValues: 'sums'}),
          overflow: 'overflow',
          uncertainCount: 'uncertainCount'
        }
      };
      const cpu = createCPUPointsInPolygonsChoroplethOperation(props);
      assertMultipartRecipe(
        getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)})),
        statistic
      );
      const gpu = createPointsInPolygonsChoroplethOperation(props);
      try {
        assertMultipartRecipe(
          run(recipeOutputs, (input, output) => ({
            getCommandNodes: graph =>
              captureGraphCommandNodes(graph, () =>
                graph.add(
                  gpu.createContributor(
                    port => (output[gpu.outputs[port]?.value] ?? input[gpu.inputs[port]]) as never
                  )
                )
              )
          })),
          statistic
        );
      } finally {
        gpu.destroy();
      }
      const pure = computePointsInPolygonsChoroplethOnCPU(
        {
          points: {
            kind: 'points',
            positions: multipartInputs['points'].values as Float32Array,
            featureOffsets: multipartInputs['pointOffsets'].values as Uint32Array
          },
          polygons: {
            kind: 'polygons',
            positions: multipartInputs['polygonPositions'].values as Float32Array,
            featureOffsets: multipartInputs['polygonFeatures'].values as Uint32Array,
            polygonOffsets: multipartInputs['polygonOffsets'].values as Uint32Array,
            ringOffsets: multipartInputs['ringOffsets'].values as Uint32Array
          },
          values: multipartInputs['values'].values as Float32Array
        },
        props
      );
      assertMultipartRecipe(
        {
          counts: Array.from(pure.counts),
          sums: Array.from(pure.featureValues),
          overflow: Array.from(pure.overflow),
          uncertainCount: Array.from(pure.uncertainCount)
        },
        statistic
      );
    });
it('choropleth CPU operation propagates later-part candidate loss', async () => {
  const cpu = createCPUPointsInPolygonsChoroplethOperation({
    ...choroplethOperationProps,
    candidateCapacity: 1
  });
  const result = getOperationResult(await cpu.execute({inputs: getOperationCPUInputs(cpu.inputs)}));
  expect(result['counts']).toEqual([1, 0, 0, 0, 0, 0]);
  expect(result['sums']).toEqual([1, 0, 0, 0, 0, 0]);
  expect(result['overflow']).toEqual([1]);
  expect(result['uncertainCount']).toEqual([0]);
});
it('choropleth CPU empty polygons preserve zero count/sum and NaN means/extrema', () => {
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const) {
    const result = computePointsInPolygonsChoroplethOnCPU(
      {
        points: {
          kind: 'points',
          positions: new Float32Array(multipartInputs['points'].values.length).fill(20),
          featureOffsets: multipartInputs['pointOffsets'].values as Uint32Array
        },
        polygons: {
          kind: 'polygons',
          positions: multipartInputs['polygonPositions'].values as Float32Array,
          featureOffsets: multipartInputs['polygonFeatures'].values as Uint32Array,
          polygonOffsets: multipartInputs['polygonOffsets'].values as Uint32Array,
          ringOffsets: multipartInputs['ringOffsets'].values as Uint32Array
        },
        values: multipartInputs['values'].values as Float32Array
      },
      {...choroplethOperationProps, statistic}
    );
    expect(Array.from(result.counts)).toEqual([0, 0, 0, 0, 0, 0]);
    if (statistic === 'count' || statistic === 'sum')
      expect(Array.from(result.featureValues)).toEqual([0, 0, 0, 0, 0, 0]);
    else
      expect(Array.from(result.featureValues).map(Number.isNaN)).toEqual([
        true,
        true,
        true,
        true,
        true,
        true
      ]);
    expect(Array.from(result.overflow)).toEqual([0]);
    expect(Array.from(result.uncertainCount)).toEqual([0]);
  }
});
for (const entry of createReviewCases())
  for (const form of ['cpu', 'shader'] as const)
    it(`review ${entry.name} ${form}`, async () => {
      let result: Record<string, number[]>;
      if (form === 'cpu') {
        const operation = entry.cpu();
        result = getOperationResult(
          await operation.execute({inputs: reviewCPUInputs(operation, entry.inputs)})
        );
      } else {
        const operation = entry.gpu();
        const outputs = Object.fromEntries(
          Object.values(operation.outputs).map(definition => [
            definition.value,
            {format: definition.format as 'float32' | 'uint32', length: definition.length}
          ])
        );
        try {
          result = run(
            outputs,
            (inputs, output) => ({
              getCommandNodes: graph => {
                resolveReviewInputViews(entry, graph, inputs);
                return captureGraphCommandNodes(graph, () =>
                  graph.add(
                    operation.createContributor(
                      port =>
                        (output[operation.outputs[port]?.value] ??
                          inputs[operation.inputs[port]]) as never
                    )
                  )
                );
              }
            }),
            entry.gpuInputs ?? entry.inputs
          );
        } finally {
          operation.destroy();
        }
      }
      for (const [port, expected] of Object.entries(entry.expected))
        expect(result[port], port).toEqual(expected);
    });
for (const rule of ['closed', 'open', 'half-open'] as const)
  it(`multipart point join matches Shapely for ${rule}, including every output`, () => {
    assertMultipartJoin(
      run(joinOutputs, (input, output) => createMultipartJoin(input, output, rule)),
      rule
    );
  });
it('multipart buffer selects any part and publishes feature identities, distances and draw count', () => {
  assertMultipartBuffer(
    run(bufferOutputs, (input, output) => createMultipartBuffer(input, output))
  );
});
it('multipart lines sum every clipped part with feature weights', () => {
  assertMultipartLines(run(lineOutputs, (input, output) => createMultipartLines(input, output)));
});
for (const backend of ['zonal', 'group'] as const)
  it(`multipart choropleth ${backend} counts once in every polygon`, () => {
    assertMultipartRecipe(
      run(recipeOutputs, (input, output, graph) => {
        const nodes: unknown[] = [];
        const add = vi.spyOn(graph, 'add').mockImplementation(source => {
          if (Array.isArray(source)) nodes.push(...source);
          else if ('getCommandNodes' in source) nodes.push(...source.getCommandNodes(graph));
        });
        try {
          createMultipartRecipe(graph, input, output, backend);
        } finally {
          add.mockRestore();
        }
        return {
          getCommandNodes: () => nodes as ReturnType<GPUCommandNodeProducer['getCommandNodes']>
        };
      })
    );
  });
it('candidate loss on a later point part flags its feature and the whole join', () => {
  const result = run(joinOutputs, (input, output) =>
    createMultipartJoin(input, output, 'closed', 1)
  );
  expect(result['candidateOverflow']).toEqual([1]);
  expect(result['overflow']).toEqual([1]);
  expect(result['matchCandidateOverflow']).toEqual([1]);
  expect(result['truncated']).toEqual([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  expect(result['counts']).toEqual([1, 0, 0, 0, 0, 0]);
});
it('multipart line candidate loss propagates to operation status', () => {
  const result = run(lineOutputs, (input, output) => createMultipartLines(input, output, 1));
  expect(result['candidateOverflow']).toEqual([1]);
  expect(result['overflow']).toEqual([1]);
});

it('multipart buffer reports BVH loss separately from final compact capacity', () => {
  const result = run(bufferOutputs, (input, output) => createMultipartBuffer(input, output, 1));
  expect(result['overflow']).toEqual([1]);
  expect(result['candidateOverflow']).toEqual([1]);
  expect(result['matchCandidateOverflow']).toEqual([1]);
  expect(result['matchOverflow']).toEqual([0]);
  expect(result['truncated']).toEqual(Array(13).fill(1));
  expect(result['mask']).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]);
});

it('multipart join distinguishes final compact overflow from candidate loss', () => {
  const result = run({...joinOutputs, ids: {format: 'uint32', length: 2}}, (input, output) =>
    createMultipartJoin(input, output)
  );
  expect(result['ids']).toEqual([90, 80]);
  expect(result['count']).toEqual([2]);
  expect(result['requiredCount']).toEqual([11]);
  expect(result['matchOverflow']).toEqual([1]);
  expect(result['overflow']).toEqual([1]);
  expect(result['candidateOverflow']).toEqual([0]);
});

for (const statistic of ['mean', 'minimum', 'maximum'] as const)
  it(`multipart choropleth ${statistic} uses one feature value per membership`, () => {
    const result = run(recipeOutputs, (input, output, graph) => {
      const nodes: unknown[] = [];
      const add = vi.spyOn(graph, 'add').mockImplementation(source => {
        if (Array.isArray(source)) nodes.push(...source);
        else if ('getCommandNodes' in source) nodes.push(...source.getCommandNodes(graph));
      });
      try {
        createMultipartRecipe(graph, input, output, 'group', statistic);
      } finally {
        add.mockRestore();
      }
      return {
        getCommandNodes: () => nodes as ReturnType<GPUCommandNodeProducer['getCommandNodes']>
      };
    });
    // Shapely memberships from the oracle; feature values are 1..13. Means are sums/counts.
    const expected =
      statistic === 'mean'
        ? [7, 6, 7.5, 7.5, 8, 28 / 3]
        : statistic === 'minimum'
          ? [1, 1, 6, 7, 6, 7]
          : [12, 13, 9, 8, 10, 13];
    for (let row = 0; row < 6; row++) {
      expect(Number.isFinite(result['sums'][row])).toBe(true);
      expect(Math.abs(result['sums'][row] - expected[row])).toBeLessThan(1e-6);
    }
  });
