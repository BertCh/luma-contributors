// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, expectTypeOf, it, vi} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPURasterEmbeddingQuantizeOutput,
  GPURasterEmbeddingDequantizeOutput
} from '@luma-contributors/analysis/raster-analysis';
import {
  GPURasterEmbeddingChange,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingQuantize,
  GPURasterEmbeddingSimilarity
} from '@luma-contributors/analysis/raster-analysis';
import {
  createKernelAuditContext,
  type KernelAuditContext
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const devices: KernelAuditContext['device'][] = [];
afterEach(() => {
  vi.restoreAllMocks();
  devices.splice(0).forEach(device => device.destroy());
});

function getFixture(lengths: number[], dimensions = 192, padded = true) {
  const context = createKernelAuditContext('embedding-scale');
  devices.push(context.device);
  const rows = <Format extends 'float32' | 'uint32'>(format: Format, width = 1, lanes = 1) =>
    createVectorView(
      `rows-${format}`,
      format,
      lengths.map((length, chunkIndex) => {
        const offset = padded ? ((chunkIndex % 61) + 1) * 4 : 0;
        const count = Math.ceil((length * width) / lanes);
        const backing = context.buffer(Math.max(count, 1) * 4 + offset);
        return context.graph.createDataView(backing, {format, length: count, byteOffset: offset});
      })
    );
  const pixelCount = lengths.reduce((total, length) => total + length, 0);
  const base = {pixelCount, dimensions, vectors: rows('float32', dimensions), mask: rows('uint32')};
  const createContributors = (trainingCount = 513) => ({
    quantize: new GPURasterEmbeddingQuantize({
      ...base,
      output: {quantized: rows('uint32', dimensions, 4), validity: rows('uint32')}
    }),
    dequantize: new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions,
      quantized: rows('uint32', dimensions, 4),
      mask: base.mask,
      normalize: true,
      output: {vectors: rows('float32', dimensions), validity: rows('uint32')}
    }),
    change: new GPURasterEmbeddingChange({
      pixelCount,
      dimensions,
      before: base.vectors,
      after: rows('float32', dimensions),
      beforeMask: base.mask,
      afterMask: rows('uint32'),
      parameters: context.view('float32', 1),
      output: {angle: rows('float32'), magnitude: rows('float32'), changed: rows('uint32')}
    }),
    probe: new GPURasterEmbeddingLinearProbe({
      ...base,
      classCount: 3,
      weights: context.view('float32', 3 * dimensions),
      bias: context.view('float32', 3),
      output: {
        labels: rows('uint32'),
        confidence: rows('float32'),
        probabilities: rows('float32', 3)
      }
    }),
    projection: new GPURasterEmbeddingProjection({
      ...base,
      componentCount: 3,
      mean: context.view('float32', dimensions),
      components: context.view('float32', 3 * dimensions),
      parameters: context.view('float32', 6),
      output: {projected: rows('float32', 3), rgb: rows('float32', 3)}
    }),
    classifier: new GPURasterEmbeddingKNearestClassifier({
      ...base,
      trainingCount,
      trainingVectors: context.view('float32', trainingCount * dimensions),
      trainingLabels: context.view('uint32', trainingCount),
      classCount: 3,
      neighborCount: 3,
      output: {
        labels: rows('uint32'),
        confidence: rows('float32'),
        probabilities: rows('float32', 3)
      }
    }),
    similarity: new GPURasterEmbeddingSimilarity({
      ...base,
      queryCount: 3,
      queries: context.view('float32', 3 * dimensions),
      parameters: context.view('float32', 1),
      output: {
        similarity: rows('float32', 3),
        maximum: rows('float32'),
        bestQuery: rows('uint32'),
        matches: rows('uint32')
      }
    })
  });
  return {...context, rows, base, createContributors};
}

it('keeps shader and pipeline counts constant with uneven chunks, offsets and training tiles', () => {
  const keysByContributor = new Map<string, Set<string>>();
  for (const chunkCount of [1, 3, 11]) {
    const lengths = Array.from({length: chunkCount}, (_, index) => [1, 129, 65][index % 3]);
    const fixture = getFixture(lengths);
    const trainingCount = chunkCount === 1 ? 257 : 1025;
    for (const [name, contributor] of Object.entries(fixture.createContributors(trainingCount))) {
      const nodes = contributor.getCommandNodes(fixture.graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const keys = keysByContributor.get(name) ?? new Set<string>();
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
      keysByContributor.set(name, keys);
      expect(() => getParsedKernelSources(nodes)).not.toThrow();
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(kernel.constants).toEqual({});
        expect(kernel.source).not.toContain('${');
        expect(auditWGSLModule(kernel.source)).toEqual([]);
        const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
      const expectedInvocations = lengths.flatMap(length => {
        if (name === 'quantize') return [length, Math.ceil((length * 192) / 4)];
        const stages =
          name === 'classifier' ? Math.ceil(trainingCount / 256) + 1 : name === 'change' ? 2 : 1;
        return new Array<number>(stages).fill(length);
      });
      const compute = nodes.filter(node => node.type === 'compute');
      expect(compute.map(node => node.workload?.maximumInvocationCount)).toEqual(
        expectedInvocations
      );
      expect(compute.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        expectedInvocations.map(count => Math.ceil(count / 64))
      );
    }
  }
  expect(
    Object.fromEntries([...keysByContributor].map(([name, keys]) => [name, keys.size]))
  ).toEqual({
    quantize: 2,
    dequantize: 1,
    change: 2,
    probe: 1,
    projection: 1,
    classifier: 2,
    similarity: 1
  });
});

for (const dimensions of [192, 256, 384, 1024]) {
  it(`audits all seven contributors at ${dimensions} dimensions without vector-sized private arrays`, () => {
    const fixture = getFixture([0, 4, 1, 0], dimensions);
    for (const contributor of Object.values(fixture.createContributors())) {
      for (const kernel of getGPUCommandNodeKernelSources(
        contributor.getCommandNodes(fixture.graph)
      )) {
        expect(auditWGSLModule(kernel.source)).toEqual([]);
        expect(kernel.source).not.toMatch(
          new RegExp(`var (?:vector|values|centered): array<f32, ${dimensions}>`)
        );
      }
    }
  });
}

it('supports 30 million complete pixel records with per-chunk uint32 indexing', () => {
  const fixture = getFixture(
    Array.from({length: 200}, () => 150_000),
    192,
    false
  );
  // Total components exceed 2^32 while every chunk binding stays below 128 MiB.
  const contributor = fixture.createContributors().similarity;
  expect(contributor.estimatedCommandNodeCount).toBe(201);
  expect(contributor.getCommandNodes(fixture.graph)).toHaveLength(201);
});

it('refuses incomplete vectors, native packed-word splits and mismatched pixel topologies', () => {
  const fixture = getFixture([1, 1], 3);
  expect(
    () =>
      new GPURasterEmbeddingQuantize({
        ...fixture.base,
        output: {quantized: fixture.rows('uint32', 3, 4)}
      })
  ).toThrow(/native packed-word alignment/);
  const good = getFixture([4, 1], 3);
  expect(
    () =>
      new GPURasterEmbeddingQuantize({
        ...good.base,
        vectors: createVectorView('split', 'float32', [
          good.view('float32', 11),
          good.view('float32', 4)
        ]),
        output: {quantized: good.rows('uint32', 3, 4)}
      })
  ).toThrow(/complete pixel vectors/);
  expect(
    () =>
      new GPURasterEmbeddingLinearProbe({
        ...good.base,
        classCount: 1,
        weights: good.view('float32', 3),
        bias: good.view('float32', 1),
        output: {
          labels: createVectorView('bad-topology', 'uint32', [
            good.view('uint32', 3),
            good.view('uint32', 2)
          ])
        }
      })
  ).toThrow(/complete pixel records/);
});

it('rejects same-dispatch aligned overlap and repeated output chunks, allowing consecutive chunks', () => {
  const fixture = getFixture([1, 2], 4);
  const backing = fixture.buffer(32);
  const chunks = [
    fixture.graph.createDataView(backing, {format: 'uint32', length: 1}),
    fixture.graph.createDataView(backing, {format: 'uint32', length: 2, byteOffset: 4})
  ];
  const labels = createVectorView('consecutive', 'uint32', chunks);
  const props = {
    ...fixture.base,
    classCount: 1,
    weights: fixture.view('float32', 4),
    bias: fixture.view('float32', 1),
    output: {labels}
  };
  expect(() => new GPURasterEmbeddingLinearProbe(props)).not.toThrow();
  const confidence = createVectorView('overlap', 'float32', [
    fixture.graph.createDataView(backing, {format: 'float32', length: 1, byteOffset: 16}),
    fixture.view('float32', 2)
  ]);
  expect(() => new GPURasterEmbeddingLinearProbe({...props, output: {labels, confidence}})).toThrow(
    /aligned binding byte ranges must not overlap/
  );
  expect(
    () =>
      new GPURasterEmbeddingLinearProbe({
        ...props,
        output: {
          labels: createVectorView('repeat', 'uint32', [
            chunks[0],
            fixture.graph.createDataView(backing, {format: 'uint32', length: 2})
          ])
        }
      })
  ).toThrow(/logical writable byte ranges must not overlap/);
});

it('checks all aligned binding prefixes and derived top-k scratch before graph mutation', () => {
  const fixture = getFixture([1, 262144], 128);
  const contributor = fixture.createContributors().classifier;
  const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
  const createView = vi.spyOn(fixture.graph, 'createDataView');
  expect(() => contributor.getCommandNodes(fixture.graph)).toThrow(/vectors chunk 1.*limit/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  const scratch = getFixture([4, 3_000_000], 1, false);
  const props = scratch.createContributors().classifier.props;
  const large = new GPURasterEmbeddingKNearestClassifier({
    ...props,
    neighborCount: 16,
    trainingCount: 16,
    trainingVectors: scratch.view('float32', 16),
    trainingLabels: scratch.view('uint32', 16)
  });
  const scratchAllocate = vi.spyOn(scratch.graph, 'createTransientBuffer');
  expect(() => large.getCommandNodes(scratch.graph)).toThrow(/nearest-neighbor scratch.*limit/);
  expect(scratchAllocate).not.toHaveBeenCalled();
});

it('retains the explicit 4096-row reference cap and budgets tiled command growth', () => {
  const fixture = getFixture([1], 1);
  const props = fixture.createContributors(4097).classifier.props;
  expect(
    () => new GPURasterEmbeddingKNearestClassifier({...props, searchStructure: 'brute-force'})
  ).toThrow(/trainingCount.*4096/);
  expect(
    () =>
      new GPURasterEmbeddingKNearestClassifier({
        ...props,
        trainingCount: 8_388_608,
        trainingVectors: fixture.view('float32', 8_388_608),
        trainingLabels: fixture.view('uint32', 8_388_608)
      })
  ).toThrow(/estimated command node count/);
  const reference = new GPURasterEmbeddingKNearestClassifier({
    ...props,
    trainingCount: 4,
    trainingVectors: fixture.view('float32', 4),
    trainingLabels: fixture.view('uint32', 4),
    searchStructure: 'brute-force'
  });
  for (const kernel of getGPUCommandNodeKernelSources(reference.getCommandNodes(fixture.graph)))
    expect(auditWGSLModule(kernel.source)).toEqual([]);
});

it('preserves packed public output types for existing callers', () => {
  expectTypeOf<GPURasterEmbeddingQuantizeOutput['quantized']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPURasterEmbeddingDequantizeOutput['vectors']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
});

it('enforces row formats and shared packed tables at construction', () => {
  const fixture = getFixture([4, 1], 192);
  const props = fixture.createContributors().similarity.props;
  expect(
    () =>
      new GPURasterEmbeddingSimilarity({
        ...props,
        queries: createVectorView('queries', 'float32', [
          fixture.view('float32', 192),
          fixture.view('float32', 384)
        ]) as never
      })
  ).toThrow(/queries must be one packed view/);
  expect(
    () =>
      new GPURasterEmbeddingSimilarity({
        ...props,
        output: {matches: fixture.rows('float32') as never}
      })
  ).toThrow(/output.matches.*uint32/);
});

it('names omitted-validity scratch per contributor and chunk on a shared graph', () => {
  const fixture = getFixture([4, 1], 3);
  const createContributor = (id: string) =>
    new GPURasterEmbeddingQuantize({
      ...fixture.base,
      id,
      output: {quantized: fixture.rows('uint32', 3, 4)}
    });
  const first = createContributor('first');
  const second = createContributor('second');
  const allocate = vi.spyOn(fixture.graph, 'createTransientBuffer');
  const nodes = [...first.getCommandNodes(fixture.graph), ...second.getCommandNodes(fixture.graph)];
  expect(nodes).toHaveLength(10);
  expect(new Set(nodes.map(node => node.id)).size).toBe(10);
  const identifiers = allocate.mock.calls.map(([props]) => props.id);
  expect(new Set(identifiers).size).toBe(identifiers.length);
  expect(identifiers).toContain('first-chunk-0-validity');
  expect(identifiers).toContain('second-chunk-1-validity');
});

it('uses each boundary chunk size in both quantize stages', () => {
  const fixture = getFixture([63, 64, 65, 127, 128, 129]);
  const nodes = fixture.createContributors().quantize.getCommandNodes(fixture.graph);
  expect(
    nodes.filter(node => node.type === 'compute').map(node => node.workload?.maximumWorkgroupCount)
  ).toEqual([1, 48, 1, 48, 2, 49, 2, 96, 2, 96, 3, 97]);
});

it('checks full WGSL reserved words and numeric literals across codec and classifier variants', () => {
  const fixture = getFixture([4, 1], 192);
  const defaults = fixture.createContributors(257);
  const validateSource = (nodes: readonly unknown[]) => {
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(kernel.source)).toEqual([]);
      const identifiers = stripWGSLComments(kernel.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      expect(kernel.constants).toEqual({});
    }
  };
  for (const format of ['alphaearth-int8', 'tessera-int8', 'float16'] as const) {
    for (const normalize of [false, true]) {
      const contributor = new GPURasterEmbeddingDequantize({
        ...defaults.dequantize.props,
        id: `codec-${format}-${normalize}`,
        format,
        normalize,
        quantized: fixture.rows('uint32', 192, format === 'float16' ? 2 : 4),
        ...(format === 'tessera-int8' ? {scales: fixture.rows('float32')} : {})
      });
      validateSource(contributor.getCommandNodes(fixture.graph));
    }
  }
  for (const metric of ['euclidean', 'cosine'] as const) {
    for (const searchStructure of ['tiled', 'brute-force'] as const) {
      const contributor = new GPURasterEmbeddingKNearestClassifier({
        ...defaults.classifier.props,
        id: `classifier-${metric}-${searchStructure}`,
        metric,
        searchStructure
      });
      validateSource(contributor.getCommandNodes(fixture.graph));
    }
  }
});
