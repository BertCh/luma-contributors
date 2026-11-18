// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUCommandNodeProducer, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPURasterEmbeddingChange,
  GPURasterEmbeddingCovariance,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingPyramidLevel,
  GPURasterEmbeddingQuantize,
  GPURasterEmbeddingSimilarity,
  GPURasterEmbeddingZoneMean
} from '@luma-contributors/analysis/raster-analysis';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  type KernelAuditContext
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

type FieldFactory = <Format extends 'float32' | 'uint32'>(
  name: string,
  format: Format,
  length: number
) => GraphDataView<Format>;
type Contributor = GPUCommandNodeProducer & {readonly id: string};
type BindingCase = {name: string; create: (field: FieldFactory) => Contributor};
const cases: BindingCase[] = [
  {
    name: 'quantize',
    create: field =>
      new GPURasterEmbeddingQuantize({
        pixelCount: 2,
        dimensions: 2,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        output: {
          quantized: field('output.quantized', 'uint32', 1),
          validity: field('output.validity', 'uint32', 2)
        }
      })
  },
  {
    name: 'dequantize',
    create: field =>
      new GPURasterEmbeddingDequantize({
        pixelCount: 2,
        dimensions: 2,
        format: 'tessera-int8',
        quantized: field('quantized', 'uint32', 1),
        scales: field('scales', 'float32', 2),
        mask: field('mask', 'uint32', 2),
        output: {
          vectors: field('output.vectors', 'float32', 4),
          validity: field('output.validity', 'uint32', 2)
        }
      })
  },
  {
    name: 'change',
    create: field =>
      new GPURasterEmbeddingChange({
        pixelCount: 2,
        dimensions: 2,
        before: field('before', 'float32', 4),
        after: field('after', 'float32', 4),
        beforeMask: field('beforeMask', 'uint32', 2),
        afterMask: field('afterMask', 'uint32', 2),
        parameters: field('parameters', 'float32', 1),
        output: {
          similarity: field('output.similarity', 'float32', 2),
          angle: field('output.angle', 'float32', 2),
          magnitude: field('output.magnitude', 'float32', 2),
          changed: field('output.changed', 'uint32', 2)
        }
      })
  },
  {
    name: 'linear-probe',
    create: field =>
      new GPURasterEmbeddingLinearProbe({
        pixelCount: 2,
        dimensions: 2,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        classCount: 2,
        weights: field('weights', 'float32', 4),
        bias: field('bias', 'float32', 2),
        output: {
          labels: field('output.labels', 'uint32', 2),
          confidence: field('output.confidence', 'float32', 2),
          probabilities: field('output.probabilities', 'float32', 4)
        }
      })
  },
  {
    name: 'projection',
    create: field =>
      new GPURasterEmbeddingProjection({
        pixelCount: 2,
        dimensions: 2,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        componentCount: 3,
        mean: field('mean', 'float32', 2),
        components: field('components', 'float32', 6),
        parameters: field('parameters', 'float32', 6),
        output: {
          projected: field('output.projected', 'float32', 6),
          rgb: field('output.rgb', 'float32', 6)
        }
      })
  },
  {
    name: 'classifier',
    create: field =>
      new GPURasterEmbeddingKNearestClassifier({
        pixelCount: 2,
        dimensions: 2,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        trainingCount: 4,
        trainingVectors: field('trainingVectors', 'float32', 8),
        trainingLabels: field('trainingLabels', 'uint32', 4),
        classCount: 2,
        neighborCount: 2,
        output: {
          labels: field('output.labels', 'uint32', 2),
          confidence: field('output.confidence', 'float32', 2),
          probabilities: field('output.probabilities', 'float32', 4)
        }
      })
  },
  {
    name: 'similarity',
    create: field =>
      new GPURasterEmbeddingSimilarity({
        pixelCount: 2,
        dimensions: 2,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        queryCount: 2,
        queries: field('queries', 'float32', 4),
        parameters: field('parameters', 'float32', 1),
        output: {
          similarity: field('output.similarity', 'float32', 4),
          maximum: field('output.maximum', 'float32', 2),
          bestQuery: field('output.bestQuery', 'uint32', 2),
          matches: field('output.matches', 'uint32', 2)
        }
      })
  },
  {
    name: 'covariance',
    create: field =>
      new GPURasterEmbeddingCovariance({
        pixelCount: 2,
        dimensions: 2,
        rangeCount: 1,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        output: {
          mean: field('output.mean', 'float32', 2),
          covariance: field('output.covariance', 'float32', 4),
          usableCount: field('output.usableCount', 'uint32', 1)
        }
      })
  },
  ...(['float', 'quantized', 'aggregation'] as const).map(source => ({
    name: `pyramid-${source}`,
    create: (field: FieldFactory) =>
      new GPURasterEmbeddingPyramidLevel({
        sourceWidth: 2,
        sourceHeight: 1,
        dimensions: 2,
        scale: 1,
        ...(source === 'quantized'
          ? {quantized: field('quantized', 'uint32', 1)}
          : source === 'aggregation'
            ? {
                aggregation: {
                  vectorSums: field('aggregation.vectorSums', 'float32', 4),
                  validCount: field('aggregation.validCount', 'uint32', 2)
                }
              }
            : {vectors: field('vectors', 'float32', 4)}),
        mask: field('mask', 'uint32', 2),
        output: {
          vectors: field('output.vectors', 'float32', 4),
          vectorSums: field('output.vectorSums', 'float32', 4),
          validity: field('output.validity', 'uint32', 2),
          validCount: field('output.validCount', 'uint32', 2),
          meanLength: field('output.meanLength', 'float32', 2)
        }
      })
  })),
  ...([false, true] as const).map(tuples => ({
    name: `zone-mean-${tuples ? 'tuples' : 'raster'}`,
    create: (field: FieldFactory) =>
      new GPURasterEmbeddingZoneMean({
        pixelCount: 2,
        dimensions: 2,
        zoneCount: 1,
        vectors: field('vectors', 'float32', 4),
        mask: field('mask', 'uint32', 2),
        ...(tuples
          ? {
              tuples: {
                zoneIds: field('tuples.zoneIds', 'uint32', 2),
                cellIds: field('tuples.cellIds', 'uint32', 2),
                fractions: field('tuples.fractions', 'float32', 2),
                status: {count: field('tuples.status.count', 'uint32', 1)}
              }
            }
          : {zoneRaster: field('zoneRaster', 'uint32', 2)}),
        output: {
          means: field('output.means', 'float32', 2),
          weights: field('output.weights', 'float32', 1),
          meanLengths: field('output.meanLengths', 'float32', 1)
        }
      })
  }))
];

function setBindingLimit(context: KernelAuditContext, byteLength: number) {
  Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {value: byteLength});
}

function createFields(
  context: KernelAuditContext,
  badField?: string,
  failure?: 'prefix' | 'usage'
) {
  const fields = new Map<string, GraphDataView>();
  const field: FieldFactory = (name, format, length) => {
    const byteOffset = name === badField && failure === 'prefix' ? 252 : 0;
    const usage = name === badField && failure === 'usage' ? Buffer.COPY_DST : Buffer.STORAGE;
    const buffer = context.graph.createTransientBuffer({
      id: `caller-${name}`,
      byteLength: Math.max(length, 1) * 4 + byteOffset,
      usage
    });
    const view = context.graph.createDataView(buffer, {format, length, byteOffset});
    fields.set(name, view);
    return view;
  };
  return {fields, field};
}

function expectUnchangedFailure(
  context: KernelAuditContext,
  contributor: Contributor,
  message: RegExp
) {
  const buffers = vi.spyOn(context.graph, 'createTransientBuffer');
  const views = vi.spyOn(context.graph, 'createDataView');
  const add = vi.spyOn(context.graph, 'add');
  const originalNodes = [...getDeclaredNodes(context)];
  try {
    expect(() => contributor.getCommandNodes(context.graph)).toThrow(message);
    expect(buffers).not.toHaveBeenCalled();
    expect(views).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(getDeclaredNodes(context)).toEqual(originalNodes);
  } finally {
    buffers.mockRestore();
    views.mockRestore();
    add.mockRestore();
  }
}

for (const bindingCase of cases) {
  for (const failure of ['prefix', 'usage'] as const) {
    it(`${bindingCase.name} checks every caller ${failure} before declaring scratch, views or nodes`, () => {
      const inventory = createKernelAuditContext(`embedding-inventory-${bindingCase.name}`);
      const inventoryFields = createFields(inventory);
      bindingCase.create(inventoryFields.field);
      const names = [...inventoryFields.fields.keys()];
      inventory.device.destroy();
      for (const name of names) {
        const context = createKernelAuditContext(
          `embedding-${bindingCase.name}-${failure}-${name}`
        );
        try {
          setBindingLimit(context, 64);
          const {field} = createFields(context, name, failure);
          const contributor = bindingCase.create(field);
          const fieldName = name.replaceAll('.', '\\.');
          expectUnchangedFailure(
            context,
            contributor,
            new RegExp(
              `^${contributor.id} ${fieldName}.*${failure === 'prefix' ? 'limit 64' : 'Buffer\\.STORAGE'}`
            )
          );
        } finally {
          context.device.destroy();
        }
      }
    });
  }

  it(`${bindingCase.name} validates physical read/write aliases in the actual dispatch scope`, () => {
    const context = createKernelAuditContext(`embedding-alias-${bindingCase.name}`);
    const physical = context.device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
    try {
      const shared = context.graph.importBuffer(
        {id: 'shared-physical', byteLength: physical.byteLength, usage: physical.usage},
        physical
      );
      let outputAssigned = false;
      const fields = createFields(context);
      const field: FieldFactory = (name, format, length) => {
        const inputName =
          bindingCase.name === 'classifier'
            ? 'trainingLabels'
            : bindingCase.name === 'dequantize' || bindingCase.name === 'pyramid-quantized'
              ? 'quantized'
              : bindingCase.name === 'change'
                ? 'before'
                : bindingCase.name === 'pyramid-aggregation'
                  ? 'aggregation.vectorSums'
                  : 'vectors';
        if (name === inputName) {
          return context.graph.createDataView(shared, {format, length});
        }
        if (name.startsWith('output.') && !outputAssigned) {
          outputAssigned = true;
          return context.graph.createDataView(shared, {format, length, byteOffset: 256});
        }
        return fields.field(name, format, length);
      };
      if (/covariance|zone-mean/.test(bindingCase.name)) {
        // Caller inputs are consumed before the output stages. Disjoint ranges are valid.
        const contributor = bindingCase.create(field);
        expectNoMixedStorageScopes(contributor.getCommandNodes(context.graph));
      } else {
        expect(() => bindingCase.create(field)).toThrow(/same physical buffer/);
      }
    } finally {
      physical.destroy();
      context.device.destroy();
    }
  });

  it(`${bindingCase.name} accepts disjoint writable ranges imported once`, () => {
    const context = createKernelAuditContext(`embedding-packed-${bindingCase.name}`);
    const physical = context.device.createBuffer({byteLength: 2048, usage: Buffer.STORAGE});
    try {
      const shared = context.graph.importBuffer(
        {id: 'packed-outputs', byteLength: physical.byteLength, usage: physical.usage},
        physical
      );
      const fields = createFields(context);
      let outputIndex = 0;
      const contributor = bindingCase.create((name, format, length) =>
        name.startsWith('output.')
          ? context.graph.createDataView(shared, {format, length, byteOffset: outputIndex++ * 256})
          : fields.field(name, format, length)
      );
      const nodes = contributor.getCommandNodes(context.graph);
      expectNoMixedStorageScopes(nodes);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(auditWGSLModule(kernel.source)).toEqual([]);
      }
    } finally {
      physical.destroy();
      context.device.destroy();
    }
  });
}

function expectNoMixedStorageScopes(nodes: ReturnType<Contributor['getCommandNodes']>) {
  for (const node of nodes) {
    const usages = new Map<object, Set<string>>();
    for (const resource of node.resources ?? []) {
      if (!('buffer' in resource) || !resource.usage.startsWith('storage')) continue;
      const buffer = 'rowByteLength' in resource.buffer ? resource.buffer.buffer : resource.buffer;
      const modes = usages.get(buffer) ?? new Set<string>();
      modes.add(resource.usage);
      usages.set(buffer, modes);
    }
    for (const modes of usages.values())
      expect([...modes]).not.toEqual(
        expect.arrayContaining(['storage-read', 'storage-read-write'])
      );
  }
}

it('separate output stages can share an alignment window without sharing logical bytes', () => {
  for (const name of ['change', 'covariance', 'zone-mean-raster']) {
    const bindingCase = cases.find(entry => entry.name === name)!;
    const context = createKernelAuditContext(`embedding-output-stages-${name}`);
    try {
      const shared = context.buffer(1024);
      const offsets: Record<string, number> =
        name === 'change'
          ? {
              'output.similarity': 0,
              'output.angle': 256,
              'output.magnitude': 512,
              'output.changed': 264
            }
          : name === 'covariance'
            ? {'output.mean': 0, 'output.covariance': 8, 'output.usableCount': 256}
            : {'output.means': 0, 'output.weights': 256, 'output.meanLengths': 260};
      const fields = createFields(context);
      const contributor = bindingCase.create((fieldName, format, length) =>
        fieldName.startsWith('output.')
          ? context.graph.createDataView(shared, {format, length, byteOffset: offsets[fieldName]})
          : fields.field(fieldName, format, length)
      );
      expectNoMixedStorageScopes(contributor.getCommandNodes(context.graph));
    } finally {
      context.device.destroy();
    }
  }
});

it('allows caller inputs and outputs to share a buffer when consumed in separate dispatches', () => {
  for (const [name, inputName, outputName] of [
    ['quantize', 'mask', 'output.quantized'],
    ['change', 'before', 'output.changed'],
    ['classifier', 'vectors', 'output.labels']
  ]) {
    const bindingCase = cases.find(entry => entry.name === name)!;
    const context = createKernelAuditContext(`embedding-input-stages-${name}`);
    try {
      const shared = context.buffer(1024);
      const fields = createFields(context);
      const contributor = bindingCase.create((fieldName, format, length) =>
        fieldName === inputName || fieldName === outputName
          ? context.graph.createDataView(shared, {
              format,
              length,
              byteOffset: fieldName === inputName ? 0 : 256
            })
          : fields.field(fieldName, format, length)
      );
      expectNoMixedStorageScopes(contributor.getCommandNodes(context.graph));
    } finally {
      context.device.destroy();
    }
  }
});

it('unused threshold and stretch parameters do not impose binding or alias restrictions', () => {
  const context = createKernelAuditContext('embedding-unused-parameters');
  try {
    for (const name of ['change', 'similarity', 'projection'] as const) {
      const output = context.view('float32', name === 'projection' ? 3 : 1);
      const common = {pixelCount: 1, dimensions: 1, parameters: output};
      const vectors = context.view('float32', 1);
      const contributor =
        name === 'change'
          ? new GPURasterEmbeddingChange({
              ...common,
              before: vectors,
              after: vectors,
              output: {similarity: output}
            })
          : name === 'similarity'
            ? new GPURasterEmbeddingSimilarity({
                ...common,
                vectors,
                queryCount: 1,
                queries: vectors,
                output: {maximum: output}
              })
            : new GPURasterEmbeddingProjection({
                ...common,
                vectors,
                componentCount: 3,
                mean: vectors,
                components: context.view('float32', 3),
                output: {projected: output}
              });
      expectNoMixedStorageScopes(contributor.getCommandNodes(context.graph));
    }
  } finally {
    context.device.destroy();
  }
});

it('covariance preflights partial sums and partial covariance independently before its usable mask', () => {
  for (const [dimensions, bindingLimit, scratch] of [
    [2, 1024, 'partialSums'],
    [4, 4096, 'partialCovariance']
  ] as const) {
    const context = createKernelAuditContext(`embedding-covariance-${scratch}`);
    try {
      setBindingLimit(context, bindingLimit);
      const contributor = new GPURasterEmbeddingCovariance({
        pixelCount: 1,
        dimensions,
        rangeCount: 256,
        vectors: context.view('float32', dimensions),
        output: {
          mean: context.view('float32', dimensions),
          covariance: context.view('float32', dimensions * dimensions)
        }
      });
      expectUnchangedFailure(
        context,
        contributor,
        new RegExp(`^raster-embedding-covariance ${scratch} scratch.*limit ${bindingLimit}`)
      );
    } finally {
      context.device.destroy();
    }
  }
});

it('zone mean preflights its three-word accumulator before allocating any scratch', () => {
  const context = createKernelAuditContext('embedding-zone-scratch');
  try {
    setBindingLimit(context, 768);
    const contributor = new GPURasterEmbeddingZoneMean({
      pixelCount: 1,
      dimensions: 1,
      zoneCount: 128,
      vectors: context.view('float32', 1),
      zoneRaster: context.view('uint32', 1),
      output: {means: context.view('float32', 128)}
    });
    expectUnchangedFailure(
      context,
      contributor,
      /^raster-embedding-zone-mean accumulator scratch needs 3072 bytes.*limit 768/
    );
  } finally {
    context.device.destroy();
  }
});

it('new preflights accept fitting layouts and preserve audited kernels', () => {
  for (const bindingCase of cases.filter(entry =>
    /covariance|pyramid|zone-mean/.test(entry.name)
  )) {
    const context = createKernelAuditContext(`embedding-fitting-${bindingCase.name}`);
    try {
      setBindingLimit(context, 64);
      const contributor = bindingCase.create(createFields(context).field);
      for (const kernel of getGPUCommandNodeKernelSources(
        contributor.getCommandNodes(context.graph)
      )) {
        expect(auditWGSLModule(kernel.source)).toEqual([]);
      }
    } finally {
      context.device.destroy();
    }
  }
});
