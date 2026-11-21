// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {validateStatisticsExecutionPlan} from '../../../src/gpu-spatial-analysis/geographic-distribution/statistics-execution-plan';

import {createSpatialInteractionIdentificationNodes} from '../../../src/gpu-spatial-analysis/spatial-interaction/spatial-interaction-identification';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  createVectorView
} from '../../utils/gpu-contributor-test-utils';
import {
  GPUIslandAttachment,
  getGPUIslandAttachmentExecutionPlan,
  GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS,
  GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  GPUSpatialWeightsTraces,
  getGPUSpatialWeightsTracesExecutionPlan
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  getGPURankMarkovExecutionPlan,
  GPURankMarkov
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {
  GPUGeographicDistribution,
  getGPUGeographicDistributionExecutionPlan,
  type GPUGeographicDistributionOutput
} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {
  GPUSpatialInteraction,
  getGPUSpatialInteractionExecutionPlan
} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {
  GPUClassAssignment,
  type GPUClassAssignmentProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {decodeGPUSpatialInteractionCoefficientStatus} from '../../../src/gpu-spatial-analysis/spatial-interaction/spatial-interaction-identification';

useCPUExecutorTestHooks();

function withViews(
  run: (
    view: <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => GraphDataView<Format>,
    graph: GPUCommandGraph
  ) => void
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let counter = 0;
  try {
    run(
      (format, length) => createTransientView(graph, `route-${counter++}`, format, length),
      graph
    );
  } finally {
    device.destroy();
  }
}
const TINY_DEVICE = {maxBufferSize: 8, maxStorageBufferBindingSize: 8};

it('island plans distinguish 4096-row auto crossover, 65535-row forced scan and infeasible grid products/schedules', () =>
  withViews(view => {
    const props = (rows: number) => ({
      positions: view('float32x2', rows),
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      output: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows * 3),
        weights: view('float32', rows * 3),
        attachment: view('uint32', rows),
        added: view('uint32', rows * 3),
        status: {overflow: view('uint32', 1)}
      }
    });
    expect(GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS).toBe(4096);
    expect(GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS).toBe(65535);
    for (const rows of [4096, 4097]) {
      const input = props(rows);
      const plan = getGPUIslandAttachmentExecutionPlan(input);
      expect(plan.route).toBe(rows === 4096 ? 'scan' : 'grid');
      expect(plan.feasible).toBe(true);
      expect(new GPUIslandAttachment(input).estimatedCommandNodeCount).toBe(
        plan.estimatedCommandNodeCount
      );
      expect(getGPUIslandAttachmentExecutionPlan(input, TINY_DEVICE).feasible).toBe(false);
    }
    expect(() => new GPUIslandAttachment({...props(65535), nearestSearch: 'scan'})).not.toThrow();
    expect(() => new GPUIslandAttachment({...props(65536), nearestSearch: 'scan'})).toThrow(
      /65535.*grid/
    );
    // Separate the product failure from the earlier command-schedule failure.
    // ceil(rows/65536) * ceil(rows/16777215) = 1536*6 at this boundary;
    // the next row requires a seventh query batch, taking 4096 + 3*1536*7 > 32768 nodes.
    const scheduleBoundary = getGPUIslandAttachmentExecutionPlan({
      ...props(100663290),
      nearestSearch: 'grid'
    });
    expect(scheduleBoundary.feasible).toBe(true);
    const beyondSchedule = getGPUIslandAttachmentExecutionPlan({
      ...props(100663291),
      nearestSearch: 'grid'
    });
    expect(beyondSchedule.feasible).toBe(false);
    const oversized = getGPUIslandAttachmentExecutionPlan({
      ...props(268435456),
      nearestSearch: 'grid'
    });
    expect(
      oversized.constraints.find(check => check.name === 'grid-candidate-product')?.satisfied
    ).toBe(false);
    expect(oversized.constraints.find(check => check.name === 'command-schedule')?.satisfied).toBe(
      false
    );
    expect(() => new GPUIslandAttachment({...props(268435456), nearestSearch: 'scan'})).toThrow(
      /grid candidate product or command schedule/
    );
  }));

it('trace plans share exact/stochastic selection, forced maximum, padded scratch and command counts', () =>
  withViews(view => {
    const props = (rows: number) => ({
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      maximumPower: 2,
      output: {powerTraces: view('float32', 2), powerSums: view('float32', 2)}
    });
    for (const rows of [4096, 4097]) {
      const input = props(rows);
      const plan = getGPUSpatialWeightsTracesExecutionPlan(input);
      const contributor = new GPUSpatialWeightsTraces(input);
      expect(plan.route).toBe(rows === 4096 ? 'exact' : 'stochastic');
      expect(contributor.mode).toBe(plan.route);
      expect(contributor.estimatedCommandNodeCount).toBe(plan.estimatedCommandNodeCount);
      expect(getGPUSpatialWeightsTracesExecutionPlan(input, TINY_DEVICE).feasible).toBe(false);
    }
    expect(() => new GPUSpatialWeightsTraces({...props(16384), mode: 'exact'})).not.toThrow();
    const excessive = {...props(16385), mode: 'exact' as const};
    expect(getGPUSpatialWeightsTracesExecutionPlan(excessive).feasible).toBe(false);
    expect(() => new GPUSpatialWeightsTraces(excessive)).toThrow(/16384.*stochastic/);
  }));

it('rank plans cross at 64 states and suggest sorted only without bounded diagnostics', () =>
  withViews(view => {
    const props = (rows: number) => ({
      values: view('float32', rows * 2),
      periodCount: 2,
      output: {ranks: view('uint32', rows * 2), counts: view('uint32', rows * rows)}
    });
    for (const rows of [64, 65]) {
      const input = props(rows);
      const plan = getGPURankMarkovExecutionPlan(input);
      expect(plan.route).toBe(rows === 64 ? 'pairwise' : 'sorted');
      expect(new GPURankMarkov(input).rankingMethod).toBe(plan.route);
      expect(getGPURankMarkovExecutionPlan(input, TINY_DEVICE).feasible).toBe(false);
    }
    expect(() => new GPURankMarkov({...props(65), rankingMethod: 'pairwise'})).toThrow(
      /use rankingMethod: 'sorted'/
    );
    const diagnostic = {...props(65), rankingMethod: 'pairwise' as const};
    const input = {
      ...diagnostic,
      output: {...diagnostic.output, status: {converged: view('uint32', 1)}}
    };
    expect(getGPURankMarkovExecutionPlan(input).feasible).toBe(false);
    expect(() => new GPURankMarkov(input)).toThrow(/requested diagnostics/);
    expect(() => new GPURankMarkov(input)).not.toThrow(/use rankingMethod: 'sorted'/);
  }));

it('geographic plans cross at 64 groups with separate 256-group forced chunked limit', () =>
  withViews(view => {
    const props = (groups: number) => ({
      positions: view('float32x2', 8),
      parameters: view('float32', 8),
      groupCount: groups,
      groupIds: view('uint32', 8),
      output: {counts: view('uint32', groups)}
    });
    for (const groups of [64, 65]) {
      const input = props(groups);
      const plan = getGPUGeographicDistributionExecutionPlan(input);
      expect(plan.route).toBe(groups === 64 ? 'chunked' : 'sorted');
      expect(new GPUGeographicDistribution(input).reduction).toBe(plan.route);
      expect(getGPUGeographicDistributionExecutionPlan(input, TINY_DEVICE).feasible).toBe(false);
    }
    const input = {...props(257), reduction: 'chunked' as const};
    expect(getGPUGeographicDistributionExecutionPlan(input).feasible).toBe(false);
    expect(() => new GPUGeographicDistribution(input)).toThrow(/256/);
  }));

it('gravity plans cross at 4096 pairs or 128 groups only for supported sparse requests', () =>
  withViews(view => {
    const props = (pairs: number, groups = 2) => ({
      origins: view('uint32', pairs),
      destinations: view('uint32', pairs),
      flows: view('float32', pairs),
      covariate: view('float32', pairs),
      originCount: groups,
      destinationCount: groups,
      constraint: 'doubly' as const,
      coefficients: [0],
      maximumIterations: 2,
      output: {
        predictions: view('float32', pairs),
        summary: view('float32', 4),
        status: view('uint32', 1),
        candidates: view('float32', 4)
      }
    });
    for (const [pairs, groups, route] of [
      [4096, 2, 'dense'],
      [4097, 2, 'sparse'],
      [4, 128, 'dense'],
      [4, 129, 'sparse']
    ] as const) {
      const input = props(pairs, groups);
      const plan = getGPUSpatialInteractionExecutionPlan(input);
      expect(plan.route).toBe(route);
      const contributor = new GPUSpatialInteraction(input);
      expect(contributor.method).toBe(route);
      expect(contributor.estimatedCommandNodeCount).toBe(plan.estimatedCommandNodeCount);
      expect(getGPUSpatialInteractionExecutionPlan(input, TINY_DEVICE).feasible).toBe(false);
    }
    const input = {...props(4097), constraint: 'origin' as const};
    expect(getGPUSpatialInteractionExecutionPlan(input).feasible).toBe(false);
    expect(() => new GPUSpatialInteraction(input)).toThrow(/4096/);
    expect(getGPUSpatialInteractionExecutionPlan({...input, method: 'sparse'}).feasible).toBe(
      false
    );
  }));

it('class assignment rejects widened typo keys and preserves both valid closure comparisons', () =>
  withViews((view, graph) => {
    const props = {
      values: view('float32', 1),
      breaks: view('float32', 3),
      classCount: view('uint32', 1),
      output: view('uint32', 1)
    } satisfies GPUClassAssignmentProps;
    for (const intervalClosure of ['left', 'right'] as const) {
      const kernels = getGPUCommandNodeKernelSources(
        new GPUClassAssignment({...props, intervalClosure}).getCommandNodes(graph)
      );
      expect(
        kernels.some(kernel =>
          kernel.source.includes(
            `breaks[breaksOffset + edge] ${intervalClosure === 'left' ? '<=' : '<'} value`
          )
        )
      ).toBe(true);
    }
    const typo = {...props, intervalClousure: 'right'};
    // @ts-expect-error A widened object still has an unknown key.
    expect(() => new GPUClassAssignment(typo)).toThrow(/intervalClousure/);
    // @ts-expect-error Fresh literals must also reject unknown keys.
    expect(() => new GPUClassAssignment({...props, intervalClousure: 'right'})).toThrow(
      /intervalClousure/
    );
  }));

it('coefficient decoder preserves readback byte offsets and refuses truncated or unknown status', () => {
  const bytes = new Uint8Array(12);
  new DataView(bytes.buffer).setUint32(4, 1, true);
  expect(
    decodeGPUSpatialInteractionCoefficientStatus(bytes.subarray(4, 8)).structurallyAbsorbed
  ).toBe(true);
  expect(decodeGPUSpatialInteractionCoefficientStatus(Uint32Array.of(2)).sampledFlatProfile).toBe(
    true
  );
  for (const value of [new Uint8Array(0), new Uint8Array(3), new Uint8Array(5), Uint32Array.of(4)])
    expect(() => decodeGPUSpatialInteractionCoefficientStatus(value)).toThrow();
});

// @ts-expect-error An empty selection has no terminal output.
const EMPTY_OUTPUT: GPUGeographicDistributionOutput = {};
void EMPTY_OUTPUT;

it('allows every minimal geographic terminal selection', () =>
  withViews(view => {
    const selections: GPUGeographicDistributionOutput[] = [
      {counts: view('uint32', 1)},
      {weightSums: view('float32', 1)},
      {meanCenters: view('float32x2', 1)},
      {medianCenters: view('float32x2', 1)},
      {medianConverged: view('uint32', 1)},
      {standardDistances: view('float32', 1)},
      {ellipses: view('float32', 3)},
      {directionalMeans: view('float32', 3)},
      {ellipseVertices: view('float32x2', 64)},
      {circleVertices: view('float32x2', 64)}
    ];
    for (const output of selections)
      expect(
        () =>
          new GPUGeographicDistribution({
            positions: view('float32x2', 2),
            lineEnds: view('float32x2', 2),
            parameters: view('float32', 8),
            output
          })
      ).not.toThrow();
  }));

it('rejects unknown keys after a value has been widened to the declared props type', () =>
  withViews(view => {
    const extra = {
      values: view('float32', 1),
      breaks: view('float32', 3),
      classCount: view('uint32', 1),
      output: view('uint32', 1),
      intervalClousure: 'right'
    };
    const widened: GPUClassAssignmentProps = extra;
    expect(() => new GPUClassAssignment(widened)).toThrow(/intervalClousure/);
    const contributor = new GPUClassAssignment({
      values: view('float32', 1),
      breaks: view('float32', 3),
      classCount: view('uint32', 1),
      output: view('uint32', 1)
    });
    expectTypeOf(contributor.props.output).toExtend<GraphDataView<'uint32'>>();
  }));

it('interprets the optional GPU coefficient writer independently of prediction kernels', () =>
  withViews((view, graph) => {
    const input = {
      origins: view('uint32', 4),
      destinations: view('uint32', 4),
      flows: view('float32', 4),
      covariate: view('float32', 4),
      originCount: 2,
      destinationCount: 2,
      constraint: 'origin' as const,
      coefficients: [0, 1],
      output: {
        predictions: view('float32', 4),
        summary: view('float32', 4),
        candidates: view('float32', 8),
        status: view('uint32', 1),
        coefficientStatus: view('uint32', 1)
      }
    };
    const memory = createCPUGraphMemory();
    memory.write(input.origins, Uint32Array.of(0, 0, 1, 1));
    memory.write(input.destinations, Uint32Array.of(0, 1, 0, 1));
    memory.write(input.covariate, Float32Array.of(1, 1, 7, 7));
    memory.write(input.output.status, Uint32Array.of(0));
    memory.write(
      input.output.candidates,
      Float32Array.of(3.13948894, 0, 1, 0, 3.13948894, 0, 1, 0)
    );
    const contributor = new GPUSpatialInteraction(input);
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
    const nodes = createSpatialInteractionIdentificationNodes(graph, 'identification-test', input);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(input.output.coefficientStatus))).toEqual([1]);
    // The same two sampled deviances cannot prove structural absorption for this nonconstant origin.
    memory.write(input.covariate, Float32Array.of(1, 2, 7, 8));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(input.output.coefficientStatus))).toEqual([2]);
  }));

it('preserves independent packed and vector row-view inference with exact keys', () =>
  withViews((view, graph) => {
    const values = createVectorView('values', 'float32', [view('float32', 1)]);
    const output = createVectorView('classes', 'uint32', [view('uint32', 1)]);
    const contributor = new GPUClassAssignment({
      values,
      mask: view('uint32', 1),
      output,
      breaks: view('float32', 3),
      classCount: view('uint32', 1),
      intervalClosure: 'right'
    });
    expectTypeOf(contributor.props.values).toExtend<GraphVectorView<'float32'>>();
    expectTypeOf(contributor.props.output).toExtend<GraphVectorView<'uint32'>>();
    expectTypeOf(contributor.props.mask).toExtend<GraphDataView<'uint32'>>();
    expect(contributor.getCommandNodes(graph)).toHaveLength(1);
  }));

it('route planners reject unknown string options rather than selecting a default branch', () =>
  withViews(view => {
    const gravity = {
      origins: view('uint32', 1),
      destinations: view('uint32', 1),
      flows: view('float32', 1),
      covariate: view('float32', 1),
      originCount: 1,
      destinationCount: 1,
      constraint: 'origin' as const,
      coefficients: [0],
      output: {
        predictions: view('float32', 1),
        summary: view('float32', 4),
        status: view('uint32', 1),
        candidates: view('float32', 4)
      }
    };
    expect(() =>
      GPUSpatialInteraction.getExecutionPlan({...gravity, constraint: 'unknown' as never})
    ).toThrow(/unconstrained, origin, destination or doubly.*unknown/);
    expect(() =>
      GPUSpatialInteraction.getExecutionPlan({...gravity, calibration: 'unknown' as never})
    ).toThrow(/grid or bounded.*unknown/);
    const rank = {
      values: view('float32', 2),
      periodCount: 2,
      output: {ranks: view('uint32', 2), counts: view('uint32', 1)}
    };
    expect(() => GPURankMarkov.getExecutionPlan({...rank, mode: 'unknown' as never})).toThrow(
      /full-rank or geographic-rank.*unknown/
    );
    expect(() => GPURankMarkov.getExecutionPlan({...rank, tieOrder: 'unknown' as never})).toThrow(
      /giddy or row-id.*unknown/
    );
  }));

it('review B4/S1: rank plans check aligned inputs, output capacities and supported diagnostics', () =>
  withViews((view, graph) => {
    const base = {
      values: view('float32', 6),
      periodCount: 2,
      output: {ranks: view('uint32', 6), counts: view('uint32', 9)}
    };
    const shifted = graph.createDataView(view('float32', 40).buffer, {
      format: 'float32',
      byteOffset: 128,
      length: 6
    });
    expect(
      GPURankMarkov.getExecutionPlan(
        {...base, values: shifted},
        {maxBufferSize: 256, maxStorageBufferBindingSize: 64}
      ).feasible
    ).toBe(false);
    const badStatus = {...base, output: {...base.output, status: {converged: view('uint32', 1)}}};
    expect(GPURankMarkov.getExecutionPlan(badStatus).feasible).toBe(false);
    expect(
      GPURankMarkov.getExecutionPlan({
        ...base,
        output: {...base.output, homogeneity: view('float32', 5)}
      }).feasible
    ).toBe(false);
    const short = {
      values: view('float32', 142),
      periodCount: 2,
      rankingMethod: 'pairwise' as const,
      output: {ranks: view('uint32', 142), counts: view('uint32', 1)}
    };
    expect(() => new GPURankMarkov(short)).not.toThrow(/use rankingMethod: 'sorted'/);
    const converged = graph.createDataView(view('uint32', 64).buffer, {
      format: 'uint32',
      byteOffset: 252,
      length: 1
    });
    const valid = {
      ...base,
      output: {...base.output, steadyStates: view('float32', 3), status: {converged}}
    };
    expect(() => new GPURankMarkov(valid)).not.toThrow();
    expect(
      GPURankMarkov.getExecutionPlan(valid, {maxBufferSize: 256, maxStorageBufferBindingSize: 256})
        .feasible
    ).toBe(true);
    const plan = GPURankMarkov.getExecutionPlan(valid, {
      maxBufferSize: 256,
      maxStorageBufferBindingSize: 128
    });
    expect(plan.feasible).toBe(false);
    expect(
      plan.constraints
        .filter(check => check.kind !== 'auto-crossover' && !check.satisfied)
        .map(check => check.quantity)
    ).toEqual(['output.status.converged']);
  }));

it('review B5: bounded gravity includes exact-fit words and calibration admission', () =>
  withViews((view, graph) => {
    const props = {
      origins: view('uint32', 16),
      destinations: view('uint32', 16),
      flows: view('float32', 16),
      covariate: view('float32', 16),
      originCount: 4,
      destinationCount: 4,
      constraint: 'doubly' as const,
      coefficients: [-1, 1],
      calibration: 'bounded' as const,
      output: {
        predictions: view('float32', 16),
        summary: view('float32', 4),
        status: view('uint32', 1),
        candidates: view('float32', 8)
      }
    };
    const contributor = new GPUSpatialInteraction(props);
    Object.assign(graph.device.limits, {maxBufferSize: 1024, maxStorageBufferBindingSize: 128});
    const state = graph as unknown as {buffers: Map<string, unknown>};
    const before = state.buffers.size;
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /candidate predictions needs 136 bytes/
    );
    expect(state.buffers.size).toBe(before);
    expect(
      GPUSpatialInteraction.getExecutionPlan(props, {
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 128
      }).feasible
    ).toBe(false);
    expect(
      GPUSpatialInteraction.getExecutionPlan({...props, maximumCalibrationIterations: 65}).feasible
    ).toBe(false);
    expect(GPUSpatialInteraction.getExecutionPlan({...props, coefficients: [1, -1]}).feasible).toBe(
      false
    );
    expect(
      GPUSpatialInteraction.getExecutionPlan({...props, coefficients: [-3e38, 3e38]}).feasible
    ).toBe(false);
    expect(
      GPUSpatialInteraction.getExecutionPlan({...props, coefficientTolerance: 0}).feasible
    ).toBe(false);
  }));

it('review B6: trace plans include parameter bindings and two-way Gram dependency', () =>
  withViews(view => {
    const props = {
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      maximumPower: 1,
      output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
    };
    expect(
      GPUSpatialWeightsTraces.getExecutionPlan(
        {...props, parameters: view('uint32', 600)},
        {maxBufferSize: 4096, maxStorageBufferBindingSize: 1024}
      ).feasible
    ).toBe(false);
    expect(
      GPUSpatialWeightsTraces.getExecutionPlan({...props, includeGramTraces: true}).feasible
    ).toBe(false);
    expect(
      GPUSpatialWeightsTraces.getExecutionPlan({
        ...props,
        includeGramTraces: false,
        output: {...props.output, gramTraces: view('float32', 4)}
      }).feasible
    ).toBe(false);
  }));

it('review B7: geographic plans ignore unused line ends and check groups and terminals', () =>
  withViews((view, graph) => {
    const props = {
      positions: view('float32x2', 6),
      parameters: view('float32', 8),
      output: {counts: view('uint32', 1)}
    };
    const lineEnds = graph.createDataView(view('float32x2', 24).buffer, {
      format: 'float32x2',
      byteOffset: 128,
      length: 6
    });
    expect(
      GPUGeographicDistribution.getExecutionPlan(
        {...props, lineEnds},
        {maxBufferSize: 256, maxStorageBufferBindingSize: 64}
      ).feasible
    ).toBe(true);
    expect(
      GPUGeographicDistribution.getExecutionPlan({
        ...props,
        groupCount: 2,
        output: {counts: view('uint32', 2)}
      }).feasible
    ).toBe(false);
    expect(
      GPUGeographicDistribution.getExecutionPlan({
        ...props,
        output: {directionalMeans: view('float32', 3)}
      }).feasible
    ).toBe(false);
    expect(
      GPUGeographicDistribution.getExecutionPlan({...props, output: {counts: view('uint32', 0)}})
        .feasible
    ).toBe(false);
  }));

it('review B8: island plans reject mismatched domains and edge provenance capacities', () =>
  withViews(view => {
    const props = {
      positions: view('float32x2', 9),
      weights: {
        offsets: view('uint32', 10),
        neighbors: view('uint32', 9),
        weights: view('float32', 9)
      },
      output: {
        offsets: view('uint32', 10),
        neighbors: view('uint32', 27),
        weights: view('float32', 27),
        attachment: view('uint32', 9),
        added: view('uint32', 27),
        status: {overflow: view('uint32', 1)}
      }
    };
    for (const nearestSearch of ['scan', 'grid'] as const) {
      expect(
        GPUIslandAttachment.getExecutionPlan({
          ...props,
          nearestSearch,
          positions: view('float32x2', 6)
        }).feasible
      ).toBe(false);
      expect(
        GPUIslandAttachment.getExecutionPlan({
          ...props,
          nearestSearch,
          output: {...props.output, offsets: view('uint32', 7)}
        }).feasible
      ).toBe(false);
      expect(
        GPUIslandAttachment.getExecutionPlan({
          ...props,
          nearestSearch,
          output: {...props.output, attachment: view('uint32', 6)}
        }).feasible
      ).toBe(false);
      expect(
        GPUIslandAttachment.getExecutionPlan({
          ...props,
          nearestSearch,
          output: {...props.output, added: view('uint32', 26)}
        }).feasible
      ).toBe(false);
    }
  }));

it('review B9: allocation failures preserve required bytes and never recommend their failing route', () =>
  withViews(view => {
    const props = {
      positions: view('float32x2', 9),
      weights: {
        offsets: view('uint32', 10),
        neighbors: view('uint32', 9),
        weights: view('float32', 9)
      },
      output: {
        offsets: view('uint32', 10),
        neighbors: view('uint32', 27),
        weights: view('float32', 27),
        attachment: view('uint32', 9),
        added: view('uint32', 27),
        status: {overflow: view('uint32', 1)}
      },
      nearestSearch: 'grid' as const
    };
    const plan = GPUIslandAttachment.getExecutionPlan(props, {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 128
    });
    expect(() => validateStatisticsExecutionPlan('review', plan)).toThrow(
      /needs 576 bytes in one binding.*128/
    );
    expect(() => validateStatisticsExecutionPlan('review', plan)).not.toThrow(
      /nearestSearch: 'grid'/
    );
  }));

it('review B1/B2: GPU coefficient evidence handles additive components and evaluated grid precision', () =>
  withViews((view, graph) => {
    for (const [constraint, coefficients, covariate, expected] of [
      ['doubly', [-1, 0, 1], [0, 1, 1, 2, 5, 7, 8, 10], 1],
      ['doubly', [1, 0, -1], [0, 1, 1, 2, 5, 7, 8, 10], 1],
      ['origin', [1, 1 + 1e-8], [0, 1, 1, 0, 0, 1, 1, 0], 0],
      ['origin', [-1, 1], [0, 1, 1, 0, 0, 1, 1, 0], 2]
    ] as const) {
      const input = {
        origins: view('uint32', 8),
        destinations: view('uint32', 8),
        flows: view('float32', 8),
        covariate: view('float32', 8),
        originCount: 4,
        destinationCount: 4,
        constraint,
        coefficients,
        output: {
          predictions: view('float32', 8),
          summary: view('float32', 4),
          candidates: view('float32', coefficients.length * 4),
          status: view('uint32', 1),
          coefficientStatus: view('uint32', 1)
        }
      };
      const memory = createCPUGraphMemory();
      memory.write(input.origins, Uint32Array.of(0, 0, 1, 1, 2, 2, 3, 3));
      memory.write(input.destinations, Uint32Array.of(0, 1, 0, 1, 2, 3, 2, 3));
      memory.write(input.covariate, Float32Array.from(covariate));
      memory.write(input.output.status, Uint32Array.of(0));
      memory.write(
        input.output.candidates,
        Float32Array.from(coefficients.flatMap(() => [1, 0, 1, 0]))
      );
      const nodes = createSpatialInteractionIdentificationNodes(
        graph,
        `review-${constraint}-${coefficients[0]}`,
        input
      );
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(input.output.coefficientStatus))).toEqual([expected]);
    }
  }));

it('review B3: public port types follow their own packed/vector props in both mixtures', () =>
  withViews(view => {
    const packed = view('uint32', 1);
    const vector = createVectorView('mixed-output', 'uint32', [view('uint32', 1)]);
    const common = {breaks: view('float32', 3), classCount: view('uint32', 1)};
    const first = new GPUClassAssignment({
      ...common,
      values: view('float32', 1),
      mask: packed,
      output: vector
    });
    expectTypeOf(first.props.mask).toEqualTypeOf<GraphDataView<'uint32'>>();
    expectTypeOf(first.classifiedValues.classes).toEqualTypeOf<GraphVectorView<'uint32'>>();
    expectTypeOf(first.classifiedValues.mask).toEqualTypeOf<GraphDataView<'uint32'>>();
    const second = new GPUClassAssignment({
      ...common,
      values: createVectorView('mixed-values', 'float32', [view('float32', 1)]),
      mask: vector,
      output: packed
    });
    expectTypeOf(second.props.mask).toEqualTypeOf<GraphVectorView<'uint32'>>();
    expectTypeOf(second.classifiedValues.classes).toEqualTypeOf<GraphDataView<'uint32'>>();
    expectTypeOf(second.classifiedValues.mask).toEqualTypeOf<GraphVectorView<'uint32'>>();
    if (false) {
      // @ts-expect-error The vector output has no packed buffer member.
      void first.classifiedValues.classes.buffer;
      // @ts-expect-error The packed output has no vector chunks member.
      void second.classifiedValues.classes.chunks;
      // @ts-expect-error The vector mask has no packed byteOffset member.
      void second.props.mask.byteOffset;
    }
    const noMask = new GPUClassAssignment({...common, values: view('float32', 1), output: packed});
    expectTypeOf(noMask.classifiedValues.mask).toEqualTypeOf<undefined>();
    expect(first.classifiedValues.classes).toBe(vector);
    expect(second.classifiedValues.classes).toBe(packed);
  }));

it('review S4: large sparse evidence uses grouped propagation without a pair-by-pair domain scan', () =>
  withViews((view, graph) => {
    const pairs = 100000;
    const props = {
      origins: view('uint32', pairs),
      destinations: view('uint32', pairs),
      flows: view('float32', pairs),
      covariate: view('float32', pairs),
      originCount: 2,
      destinationCount: 2,
      constraint: 'doubly' as const,
      coefficients: [-1, 1],
      output: {
        predictions: view('float32', pairs),
        summary: view('float32', 4),
        candidates: view('float32', 8),
        status: view('uint32', 1),
        coefficientStatus: view('uint32', 1)
      }
    };
    const nodes = createSpatialInteractionIdentificationNodes(graph, 'large-evidence', props);
    const structural = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
      kernel.id.endsWith('-identification-structural')
    );
    expect(structural).toHaveLength(1);
    // Reject the former full-domain scan inside each of the 100,000 pair invocations.
    expect(structural[0].source).not.toMatch(/pair\s*<\s*100000u/);
    expect(GPUSpatialInteraction.getExecutionPlan(props).feasible).toBe(true);
  }));
