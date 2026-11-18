// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Device, Resource} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GraphDataView,
  GraphVectorView,
  GraphTextureHandle,
  GraphTextureView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, vi} from 'vitest';
import {RecipeBuilder} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';

/** Inspects public result ports and tracks each contributor's first declaration graph. */
export function auditRecipeResult<Result extends object>(
  graph: GPUCommandGraph,
  declare: () => Result
) {
  const origins = new WeakMap<object, object>();
  const expanded = new WeakSet<object>();
  const emitted = new Map<object, readonly string[]>();
  const nodeIds: string[] = [];
  const ports: Record<string, unknown> = {};
  function traceProducer(producer: object, target: object) {
    if (origins.has(producer)) expect(origins.get(producer)).toBe(target);
    else origins.set(producer, target);
    if (expanded.has(producer)) return;
    expanded.add(producer);
    const getNodes = Reflect.get(producer, 'getCommandNodes') as (
      target: GPUCommandGraph
    ) => readonly GPUCommandNode[];
    expect(
      Reflect.set(producer, 'getCommandNodes', (target: GPUCommandGraph) => {
        const nodes = getNodes.call(producer, target);
        if (target === graph)
          emitted.set(
            producer,
            nodes.map(node => node.id)
          );
        return nodes;
      })
    ).toBe(true);
  }
  const add = RecipeBuilder.prototype.add;
  vi.spyOn(RecipeBuilder.prototype, 'add').mockImplementation(function (
    this: RecipeBuilder<unknown>,
    producer
  ) {
    traceProducer(producer, this.graph);
    return add.call(this, producer);
  });
  const glue = RecipeBuilder.prototype.addGlue;
  vi.spyOn(RecipeBuilder.prototype, 'addGlue').mockImplementation(function (
    this: RecipeBuilder<unknown>,
    ...args
  ) {
    const result = glue.apply(this, args);
    traceProducer(this.contributors[this.contributors.length - 1], this.graph);
    return result;
  });
  const graphAdd = GPUCommandGraph.prototype.add;
  vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(function (
    this: GPUCommandGraph,
    producer
  ) {
    if (producer && typeof producer === 'object' && 'getCommandNodes' in producer)
      traceProducer(producer, this);
    return graphAdd.call(this, producer);
  });
  const compute = graph.addComputePass.bind(graph);
  vi.spyOn(graph, 'addComputePass').mockImplementation(node => {
    nodeIds.push(node.id);
    return compute(node);
  });
  const render = graph.addRenderPass.bind(graph);
  vi.spyOn(graph, 'addRenderPass').mockImplementation(node => {
    nodeIds.push(node.id);
    return render(node);
  });
  const copy = graph.addCopyPass.bind(graph);
  vi.spyOn(graph, 'addCopyPass').mockImplementation(node => {
    nodeIds.push(node.id);
    return copy(node);
  });
  const visited = new Set<object>();
  function visit(value: unknown, path: string) {
    if (!value || typeof value !== 'object') return;
    if (value instanceof GraphDataView) {
      expect(value.buffer.graph, path).toBe(graph);
      ports[path] = {
        id: value.buffer.id,
        format: value.format,
        length: value.length,
        byteOffset: value.byteOffset,
        byteStride: value.byteStride,
        rowByteLength: value.rowByteLength
      };
      return;
    }
    if (value instanceof GraphVectorView) {
      value.data.forEach((chunk, index) => visit(chunk, `${path}.chunks[${index}]`));
      return;
    }
    if (value instanceof GraphTextureView) {
      expect(value.texture.graph, path).toBe(graph);
      ports[path] = {id: value.texture.id};
      return;
    }
    if (value instanceof GraphTextureHandle) {
      expect(value.graph, path).toBe(graph);
      ports[path] = {id: value.id};
      return;
    }
    if (value instanceof GPUCommandGraph) {
      expect(value, path).toBe(graph);
      return;
    }
    if (value instanceof Resource || value instanceof Device || ArrayBuffer.isView(value)) return;
    if (visited.has(value)) return;
    visited.add(value);
    if (origins.has(value)) expect(origins.get(value), path).toBe(graph);
    for (const [name, child] of Object.entries(value)) visit(child, `${path}.${name}`);
  }
  try {
    const result = declare();
    const contributors = Reflect.get(result, 'contributors') as readonly object[] | undefined;
    const entries = (contributors ?? []).map(producer => {
      expect(origins.get(producer)).toBe(graph);
      // addGlue producers return already constructed nodes without declaring more resources.
      if (!emitted.has(producer)) {
        const nodes = (
          Reflect.get(producer, 'getCommandNodes') as (
            target: GPUCommandGraph
          ) => readonly GPUCommandNode[]
        ).call(producer, graph);
        emitted.set(
          producer,
          nodes.map(node => node.id)
        );
      }
      const ids = emitted.get(producer)!;
      return {
        id: Reflect.get(producer, 'id') ?? ids[0],
        class: producer.constructor.name,
        nodes: ids
      };
    });
    expect(new Set(contributors).size).toBe(contributors?.length ?? 0);
    expect(entries.every(entry => typeof entry.id === 'string')).toBe(true);
    expect(new Set(entries.map(entry => entry.id)).size).toBe(entries.length);
    visit(Reflect.get(result, 'outputs'), 'outputs');
    visited.clear();
    visit(Reflect.get(result, 'intermediates'), 'intermediates');
    visited.clear();
    visit(result, 'result');
    return {result, entries, nodeIds, ports, exposesContributors: contributors !== undefined};
  } finally {
    vi.restoreAllMocks();
  }
}
