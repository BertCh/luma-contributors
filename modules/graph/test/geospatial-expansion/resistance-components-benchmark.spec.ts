// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {addSparseResistanceConnectivityRecipe} from '../../src/gpu-spatial-analysis/landscape-ecology/sparse-resistance';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {RecipeTestFixture} from '../gpu-spatial-analysis/recipes/recipe-harness';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../utils/benchmark-settings';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
type Settings = ReturnType<typeof getBenchmarkSettings>;
type ResistanceCase = {
  shape: 'routes' | 'chain' | 'grid' | 'star';
  vertices: number;
  order?: 'reverse' | 'shuffle';
  disconnectedVertices?: number;
};

function getProfilePhase(identifier: string): string {
  if (identifier.startsWith('resistance-components')) return 'components';
  if (identifier.startsWith('resistance-sort')) return 'sort';
  if (identifier.includes('jacobi')) return 'jacobi';
  if (identifier.includes('-product-')) return 'spmv';
  if (identifier.startsWith('resistance-pcg')) return 'solver-other';
  if (identifier.includes('true-residual')) return 'true-residual';
  if (identifier.includes('certificate')) return 'certificate';
  if (identifier.startsWith('resistance-coo') || identifier.startsWith('resistance-csr'))
    return 'assembly';
  if (
    identifier.startsWith('capture-') ||
    identifier.includes('mask') ||
    identifier.includes('currents')
  )
    return 'outputs';
  return 'other';
}

async function measureGraph(
  device: Device,
  graph: GPUCommandGraph,
  validate: () => Promise<void>,
  settings: Settings,
  readTelemetry: () => Promise<number[]>
) {
  const compileStarted = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStarted;
  const requestedQueries = 2 * (compiled.stats.nodeOrder.length + 1);
  const profileUnavailableReason = !device.features.has('timestamp-query')
    ? 'timestamp-query-unavailable'
    : requestedQueries > 4096
      ? 'query-capacity-exceeded'
      : null;
  const querySet =
    profileUnavailableReason === null
      ? device.createQuerySet({type: 'timestamp', count: requestedQueries})
      : undefined;
  const profilePhases = [...new Set(compiled.stats.nodeOrder.map(getProfilePhase))];
  const phaseNodeCounts = profilePhases.map(
    phase =>
      compiled.stats.nodeOrder.filter(identifier => getProfilePhase(identifier) === phase).length
  );
  const execute = async () => {
    const started = performance.now();
    const encodeStarted = performance.now();
    submitGraph(device, compiled, undefined);
    const encodeSubmitMilliseconds = performance.now() - encodeStarted;
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    return {wallMilliseconds: performance.now() - started, encodeSubmitMilliseconds};
  };
  try {
    const firstExecution = await execute();
    await validate();
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await execute();
    const beforeWallTelemetry = await readTelemetry();
    const wallSamples = [];
    for (let sample = 0; sample < settings.sampleCount; sample++) wallSamples.push(await execute());
    const afterWallTelemetry = await readTelemetry();
    await validate();
    const profileSamples = [];
    if (querySet) {
      for (let sample = -settings.warmupCount; sample < settings.sampleCount; sample++) {
        const encoder = device.createCommandEncoder({timeProfilingQuerySet: querySet});
        const encoding = compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const timings = await encoding.readTimings();
        if (sample >= 0) {
          const phases = profilePhases.map(phase => {
            const nodes = timings.nodes.filter(node => getProfilePhase(node.id) === phase);
            const available = nodes.filter(
              node =>
                node.hasGPUTimestamps &&
                Number.isFinite(node.gpuTimeMilliseconds) &&
                node.gpuTimeMilliseconds! >= 0
            );
            const missing = nodes.length - available.length;
            return [
              nodes.length,
              available.length,
              missing,
              missing === 0
                ? available.reduce((total, node) => total + node.gpuTimeMilliseconds!, 0)
                : null,
              nodes.reduce((total, node) => total + node.cpuEncodeTimeMilliseconds, 0)
            ];
          });
          profileSamples.push({
            cpuEncodeTimeMilliseconds: timings.cpuEncodeTimeMilliseconds,
            phases
          });
        }
      }
      await validate();
    }
    return {
      wallTelemetry: {
        executions: afterWallTelemetry[0] - beforeWallTelemetry[0],
        fallbacks: afterWallTelemetry[1] - beforeWallTelemetry[1]
      },
      finalTelemetry: await readTelemetry(),
      compileMilliseconds,
      firstExecution,
      wallSamples,
      wallDistribution: getBenchmarkDistribution(
        wallSamples.map(sample => sample.wallMilliseconds)
      ),
      profileSamples: querySet ? profileSamples : null,
      profileUnavailableReason,
      requestedQueries,
      declaredNodeCount: compiled.stats.nodeOrder.length,
      profilePhases,
      phaseNodeCounts,
      profileFields: ['nodes', 'available', 'missing', 'gpuMilliseconds', 'cpuEncodeMilliseconds'],
      physicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
      logicalTransientBytes: compiled.stats.logicalTransientBytes,
      correctnessStatus: 'passed'
    };
  } finally {
    querySet?.destroy();
    compiled.destroy();
  }
}

function makeCase(specification: ResistanceCase) {
  const {vertices, shape} = specification;
  const vertexCount = vertices + (specification.disconnectedVertices ?? 0);
  const pairs: {first: number; second: number; weight: number}[] = [];
  const add = (first: number, second: number, weight = 1) =>
    pairs.push({first, second, weight: Math.fround(weight)});
  if (shape === 'routes') {
    for (let vertex = 1; vertex < vertices - 1; vertex++) {
      add(0, vertex, 2 / (vertices - 2));
      add(vertex, vertices - 1, 2 / (vertices - 2));
    }
  } else if (shape === 'grid') {
    const side = Math.sqrt(vertices);
    for (let row = 0; row < side; row++)
      for (let column = 0; column < side; column++) {
        const vertex = row * side + column;
        if (column + 1 < side) add(vertex, vertex + 1);
        if (row + 1 < side) add(vertex, vertex + side);
      }
  } else {
    for (let vertex = 0; vertex < vertices - 1; vertex++)
      add(vertex, shape === 'star' ? vertices - 1 : vertex + 1);
  }
  for (let vertex = vertices; vertex + 1 < vertexCount; vertex++) add(vertex, vertex + 1);
  if (specification.order === 'reverse') pairs.reverse();
  if (specification.order === 'shuffle') {
    let state = 873213;
    for (let index = pairs.length - 1; index > 0; index--) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const selected = state % (index + 1);
      [pairs[index], pairs[selected]] = [pairs[selected], pairs[index]];
    }
  }
  const edges = Uint32Array.from(pairs.flatMap(pair => [pair.first, pair.second]));
  const weights = Float32Array.from(pairs.map(pair => pair.weight));
  const potentials = new Float64Array(vertexCount);
  if (shape === 'routes') {
    potentials.fill(0.5, 0, vertices);
    potentials[0] = 1;
    potentials[vertices - 1] = 0;
  } else {
    // Independent f64 grounded dense solve is used only for small non-hub controls.
    const length = vertices - 1;
    const matrix = Array.from({length}, () => new Float64Array(length + 1));
    matrix[0][length] = 1;
    for (const {first, second, weight} of pairs) {
      if (first >= vertices || second >= vertices) continue;
      if (first < length) matrix[first][first] += weight;
      if (second < length) matrix[second][second] += weight;
      if (first < length && second < length) {
        matrix[first][second] -= weight;
        matrix[second][first] -= weight;
      }
    }
    for (let pivot = 0; pivot < length; pivot++) {
      let selected = pivot;
      for (let row = pivot + 1; row < length; row++)
        if (Math.abs(matrix[row][pivot]) > Math.abs(matrix[selected][pivot])) selected = row;
      [matrix[pivot], matrix[selected]] = [matrix[selected], matrix[pivot]];
      expect(Math.abs(matrix[pivot][pivot])).toBeGreaterThan(0);
      for (let row = pivot + 1; row < length; row++) {
        const multiplier = matrix[row][pivot] / matrix[pivot][pivot];
        for (let column = pivot; column <= length; column++)
          matrix[row][column] -= multiplier * matrix[pivot][column];
      }
    }
    for (let row = length - 1; row >= 0; row--) {
      let value = matrix[row][length];
      for (let column = row + 1; column < length; column++)
        value -= matrix[row][column] * potentials[column];
      potentials[row] = value / matrix[row][row];
    }
  }
  const currents = pairs.map(({first, second, weight}) =>
    first < vertices ? weight * (potentials[first] - potentials[second]) : NaN
  );
  const labels = Uint32Array.from({length: vertexCount}, (_, vertex) =>
    vertex < vertices ? 0 : vertices
  );
  potentials.fill(NaN, vertices);
  return {
    vertexCount,
    edges,
    weights,
    potentials,
    currents,
    labels,
    sink: vertices - 1,
    iterations: shape === 'routes' || shape === 'star' ? 8 : 2 * vertices
  };
}

async function measureResistance(
  device: Device,
  specification: ResistanceCase,
  settings: Settings
) {
  const referenceStarted = performance.now();
  const input = makeCase(specification);
  const referenceMilliseconds = performance.now() - referenceStarted;
  const setupStarted = performance.now();
  const fixture = new RecipeTestFixture(device, 'resistance-comparison');
  const captured: {control?: GraphDataView<'uint32x4'>} = {};
  const createTransientBuffer = fixture.graph.createTransientBuffer.bind(fixture.graph);
  fixture.graph.createTransientBuffer = descriptor => {
    const handle = createTransientBuffer(descriptor);
    if (descriptor.id === 'resistance-component-control')
      captured.control = fixture.graph.createDataView(handle, {format: 'uint32x4', length: 1});
    return handle;
  };
  const result = addSparseResistanceConnectivityRecipe(fixture.graph, {
    id: 'resistance',
    edges: fixture.input('edges', input.edges, 'uint32x2', input.weights.length),
    conductances: fixture.input('conductances', input.weights, 'float32', input.weights.length),
    vertexCount: input.vertexCount,
    source: 0,
    sink: input.sink,
    maximumIterations: input.iterations,
    tolerance: 1e-4
  });
  fixture.graph.createTransientBuffer = createTransientBuffer;
  const certificate = fixture.output('certificate', 'float32', 8);
  const telemetry = fixture.output('telemetry', 'uint32', 2);
  const potentials = fixture.output('potentials-output', 'float32', input.vertexCount);
  const currents = fixture.output('currents-output', 'float32', input.weights.length);
  const labels = fixture.output('labels-output', 'uint32', input.vertexCount);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: 'capture-certificate',
      operation: 'benchmark-capture',
      invocationCount: 8,
      bindings: [
        {name: 'diagnostics', view: result.outputs.diagnostics, type: 'f32', access: 'read'},
        {name: 'status', view: result.status, type: 'u32', access: 'read'},
        {name: 'certificate', view: certificate.view, type: 'f32', access: 'read_write'},
        {name: 'telemetry', view: telemetry.view, type: 'u32', access: 'read_write'},
        ...(captured.control
          ? [
              {
                name: 'control',
                view: captured.control,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      body: `if (index < 4u) { certificate[certificateOffset + index] = diagnostics[diagnosticsOffset + index]; }
else { certificate[certificateOffset + index] = f32(status[statusOffset + index - 4u]); }
if (index == 0u) { telemetry[telemetryOffset] += 1u; ${captured.control ? 'telemetry[telemetryOffset + 1u] += control[controlOffset + 3u];' : ''} }`
    })
  );
  for (const [name, source, output, type] of [
    ['potentials', result.outputs.potentials, potentials.view, 'f32'],
    ['currents', result.outputs.currents, currents.view, 'f32'],
    ['labels', result.intermediates.components, labels.view, 'u32']
  ] as const)
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'capture-' + name,
        operation: 'benchmark-output-copy',
        invocationCount: source.length,
        bindings: [
          {name: 'source', view: source, type, access: 'read'},
          {name: 'output', view: output, type, access: 'read_write'}
        ],
        body: 'output[outputOffset + index] = source[sourceOffset + index];'
      })
    );
  const validate = async () => {
    const values = await fixture.readFloat32(certificate, 8);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values.slice(4)).toEqual([1, input.iterations, 1, 0]);
    expect(values[1]).toBeLessThanOrEqual(1e-4);
    if (specification.shape === 'routes') expect(values[3]).toBeLessThanOrEqual(1e-4);
    expect(Math.abs(values[0] - input.potentials[0])).toBeLessThanOrEqual(
      1e-4 * (1 + Math.abs(input.potentials[0]))
    );
    expect(Math.abs(values[2] - input.potentials[0])).toBeLessThanOrEqual(
      1e-4 * (1 + Math.abs(input.potentials[0]))
    );
    expect(await fixture.readUint32(labels, input.vertexCount)).toEqual(Array.from(input.labels));
    for (const [output, expected] of [
      [potentials, input.potentials],
      [currents, input.currents]
    ] as const) {
      const actual = await fixture.readFloat32(output, expected.length);
      for (let index = 0; index < expected.length; index++) {
        if (Number.isNaN(expected[index])) expect(actual[index]).toBeNaN();
        else
          expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
            (output === currents ? 1e-8 : 1e-5) + 1e-4 * Math.abs(expected[index])
          );
      }
    }
  };
  try {
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    const setupAndUploadMilliseconds = performance.now() - setupStarted;
    return {
      specification,
      vertexCount: input.vertexCount,
      edgeCount: input.weights.length,
      configuredIterations: input.iterations,
      componentStrategy: captured.control ? 'bounded-union' : 'serial',
      referenceMilliseconds,
      setupAndUploadMilliseconds,
      inputBytes: input.edges.byteLength + input.weights.byteLength,
      ...(await measureGraph(device, fixture.graph, validate, settings, () =>
        fixture.readUint32(telemetry, 2)
      ))
    };
  } finally {
    fixture.destroy();
  }
}

it.skipIf(environment?.['VITE_RESISTANCE_COMPONENT_BENCHMARK'] !== 'true')(
  'complete resistance workflows and component fallback telemetry',
  {timeout: 900000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `RESISTANCE_COMPONENT_METADATA ${JSON.stringify({
        settings,
        device: device.info,
        userAgent: navigator.userAgent,
        limits: {
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        timing:
          'Warm encode/submit/completion includes components, CSR, solve, certificate, all output copies and GPU telemetry. Allocation/upload/reference/readback excluded. Timestamp profiles separate. Fallback counters cover every wall execution; active PCG iterations not observed.'
      })}`
    );
    const cases: ResistanceCase[] = [
      ...[4, 66, 1026, 8194, 65538].map(vertices => ({shape: 'routes' as const, vertices})),
      {shape: 'routes', vertices: 65538, order: 'reverse'},
      {shape: 'routes', vertices: 65538, order: 'shuffle'},
      {shape: 'routes', vertices: 1026, disconnectedVertices: 32},
      {shape: 'chain', vertices: 17},
      {shape: 'chain', vertices: 65},
      {shape: 'grid', vertices: 81},
      {shape: 'star', vertices: 65}
    ];
    let measured = 0;
    for (const specification of cases) {
      const workload = `${specification.shape}-${specification.vertices}-${specification.order ?? 'forward'}-disconnected-${specification.disconnectedVertices ?? 0}`;
      if (!workload.includes(settings.workloadFilter)) continue;
      measured++;
      console.log(
        `RESISTANCE_COMPONENT_MEASUREMENT ${JSON.stringify({workload, ...(await measureResistance(device, specification, settings))})}`
      );
    }
    if (!measured) throw new Error('No resistance benchmark workloads matched');
  }
);
