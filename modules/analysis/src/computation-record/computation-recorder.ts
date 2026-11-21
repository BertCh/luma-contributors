// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CompiledGPUCommandGraph,
  GPUCommandGraph,
  GPUCommandNode,
  GraphBufferHandle,
  GraphExternalTextureHandle,
  GraphTextureHandle
} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../analysis-core/kernels/gpu-contributor-utils';
import {recordWGSLKernelNodes} from '../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUComputationNodeAnnotation,
  GPUComputationRecordScope
} from './computation-record-types';
import {
  buildGPUCommandGraphDescription,
  type DescribeGPUCommandGraphOptions,
  type GPUCommandGraphDescription
} from './describe-command-graph';

/** Construction options of a recorder. */
export type GPUComputationRecorderProps = {
  /** Keep full source text on recorded sources. Defaults to false (hash and metadata only). */
  sourceText?: boolean;
};

/** Public resource handle fields the recorder keeps; no GPU objects are retained. */
export type GPUComputationRecordedResource =
  | GraphBufferHandle
  | GraphTextureHandle
  | GraphExternalTextureHandle;

/** Annotation after scope merging, ready for `describeGPUCommandGraph`. */
export type GPUComputationRecordedNode = GPUComputationNodeAnnotation & {
  /** Node ids passed as `dependsOn`, copied from the node when it was recorded. */
  dependsOn?: readonly string[];
};

// Weak links avoid retaining the compiled GPU resources or wrapping a builder twice.
const RECORDED_COMPILATIONS = new WeakMap<object, WeakSet<object>>();

/**
 * Analysis-side record of one command graph under construction. Published `GPUCommandGraph` has
 * no recording hooks, so analysis code passes each node through the recorder as it builds the graph.
 * The recorder only reads public node properties and never touches the graph's internals.
 */
export class GPUComputationRecorder {
  /** Whether recorded sources keep their full text. */
  readonly sourceText: boolean;

  private state = {
    kernelGraph: undefined as GPUCommandGraph<any> | undefined,
    nodes: new Map<string, GPUComputationRecordedNode>(),
    resources: new Map<string, GPUComputationRecordedResource>()
  };
  private readonly scopes: GPUComputationRecordScope[] = [];

  constructor(props: GPUComputationRecorderProps = {}) {
    this.sourceText = props.sourceText ?? false;
  }

  /**
   * Keeps the kernel facts of every node `createWGSLKernelNode` builds on `graph` from now on (source,
   * workgroup size, dispatch, bindings, operation), so `describeGPUCommandGraph(compiled, {recorder})`
   * fills them in for nodes that were never passed to `recordNode`. Call it before contributors add
   * nodes. Annotations passed to `recordNode` win field by field. Returns the graph.
   */
  recordKernels<Graph extends GPUCommandGraph<any>>(graph: Graph): Graph {
    recordWGSLKernelNodes(graph);
    if (!RECORDED_COMPILATIONS.has(graph)) {
      const compilations = new WeakSet<object>();
      const compile = graph.compile.bind(graph);
      const compileAsync = graph.compileAsync.bind(graph);
      graph.compile = options => {
        const compiled = compile(options);
        compilations.add(compiled);
        return compiled;
      };
      graph.compileAsync = async options => {
        const compiled = await compileAsync(options);
        compilations.add(compiled);
        return compiled;
      };
      RECORDED_COMPILATIONS.set(graph, compilations);
    }
    this.state.kernelGraph = graph;
    return graph;
  }

  /**
   * Describes the exact object returned by compiling the recorded graph. Returns undefined when
   * `recordKernels` has not been called and rejects a compilation from another builder.
   */
  getRecordedGraphDescription(
    compiled: Pick<CompiledGPUCommandGraph, 'id' | 'stats' | 'preflight'>,
    options: Omit<DescribeGPUCommandGraphOptions, 'graph' | 'recorder'> = {}
  ): GPUCommandGraphDescription | undefined {
    const graph = this.state.kernelGraph;
    if (!graph) return undefined;
    validateRequirement(
      'GPUComputationRecorder',
      RECORDED_COMPILATIONS.get(graph)?.has(compiled) === true,
      'compiled must come from the graph passed to recordKernels'
    );
    return buildGPUCommandGraphDescription(compiled, {...options, graph, recorder: this});
  }

  /**
   * Runs a synchronous callback in a copied scope. Scopes nest, innermost wins, and unwind on throw.
   * PromiseLike results are rejected; use `withAsyncScope` and its callback's scoped recorder for
   * asynchronous authoring. A native async callback is rejected before it starts. Other callbacks
   * returning thenables are rejected on return; their rejection is observed, but their already-started
   * work cannot be cancelled.
   */
  withSynchronousScope<Result>(
    scope: GPUComputationRecordScope,
    callback: (() => Result) &
      (Extract<Result, PromiseLike<unknown>> extends never ? unknown : never)
  ): Result {
    validateRequirement(
      'GPUComputationRecorder',
      Object.prototype.toString.call(callback) !== '[object AsyncFunction]',
      'callback must be synchronous; use withAsyncScope(scope, scopedRecorder => ...)'
    );
    this.scopes.push(this.snapshotScope(scope));
    try {
      const result = callback();
      const isPromiseLike =
        result !== null &&
        (typeof result === 'object' || typeof result === 'function') &&
        'then' in result &&
        typeof result.then === 'function';
      // The boundary owns the returned promise once it rejects async use: the caller never receives
      // it. Observe an already-started rejection (including an assimilated thenable) before throwing.
      if (isPromiseLike) void Promise.resolve(result).catch(() => {});
      validateRequirement(
        'GPUComputationRecorder',
        !isPromiseLike,
        'callback must not return PromiseLike; use withAsyncScope(scope, scopedRecorder => ...)'
      );
      return result;
    } finally {
      this.scopes.pop();
    }
  }

  /**
   * Authors asynchronously through a separate recorder with a captured scope stack. Use the recorder
   * passed to `callback` for all nested authoring. It shares records, resources and kernel graph with
   * this recorder, while concurrent callbacks keep independent namespaces across awaits and errors.
   */
  async withAsyncScope<Result>(
    scope: GPUComputationRecordScope,
    callback: (scopedRecorder: GPUComputationRecorder) => Result | PromiseLike<Result>
  ): Promise<Result> {
    const scopedRecorder = new GPUComputationRecorder({sourceText: this.sourceText});
    scopedRecorder.state = this.state;
    scopedRecorder.scopes.push(
      ...this.scopes.map(value => this.snapshotScope(value)),
      this.snapshotScope(scope)
    );
    return await callback(scopedRecorder);
  }

  /**
   * Records a node's annotation, merged with the enclosing scopes, plus its resource handles and
   * explicit dependencies. Returns the same node so it can be passed straight to `graph.add`.
   */
  recordNode<Node extends GPUCommandNode<any>>(
    node: Node,
    annotation: GPUComputationNodeAnnotation = {}
  ): Node {
    this.state.nodes.set(node.id, this.mergeScopes(node, annotation));
    for (const use of node.resources ?? []) {
      if ('buffer' in use) {
        const handle = 'buffer' in use.buffer ? use.buffer.buffer : use.buffer;
        this.state.resources.set(handle.id, handle);
      } else if ('texture' in use) {
        const handle = 'texture' in use.texture ? use.texture.texture : use.texture;
        this.state.resources.set(handle.id, handle);
      } else {
        this.state.resources.set(use.externalTexture.id, use.externalTexture);
      }
    }
    return node;
  }

  /** Records a node and adds it to the graph through the public `add` method. */
  addNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    node: GPUCommandNode<Parameters>,
    annotation?: GPUComputationNodeAnnotation
  ): void {
    graph.add(this.recordNode(node, annotation));
  }

  /** Returns the merged annotation for a node id, when it was recorded. */
  getNodeRecord(nodeId: string): GPUComputationRecordedNode | undefined {
    return this.state.nodes.get(nodeId);
  }

  /** Returns the handle of a resource used by a recorded node. */
  getResource(resourceId: string): GPUComputationRecordedResource | undefined {
    return this.state.resources.get(resourceId);
  }

  private snapshotScope(scope: GPUComputationRecordScope): GPUComputationRecordScope {
    const snapshot = snapshotConstructionFields(scope, [
      'namespace',
      'derivedFrom',
      'loop',
      'op',
      'datum'
    ]);
    return Object.freeze({
      ...snapshot,
      ...(snapshot.derivedFrom && {derivedFrom: Object.freeze([...snapshot.derivedFrom])}),
      ...(snapshot.loop && {
        loop: snapshotConstructionFields(snapshot.loop, [
          'id',
          'iteration',
          'phase',
          'maximumIterations'
        ])
      }),
      ...(snapshot.datum && {
        datum: snapshotConstructionFields(snapshot.datum, ['keySpace', 'relation'])
      })
    });
  }

  private mergeScopes(
    node: GPUCommandNode<any>,
    annotation: GPUComputationNodeAnnotation
  ): GPUComputationRecordedNode {
    const namespaces = this.scopes.flatMap(scope => (scope.namespace ? [scope.namespace] : []));
    if (annotation.namespace) namespaces.push(annotation.namespace);
    const innermost = <Key extends 'derivedFrom' | 'loop' | 'op' | 'datum'>(
      key: Key
    ): GPUComputationRecordScope[Key] => {
      for (let index = this.scopes.length - 1; index >= 0; index--) {
        const value = this.scopes[index][key];
        if (value !== undefined) return value;
      }
      return undefined;
    };
    const derivedFrom = annotation.derivedFrom ?? innermost('derivedFrom');
    const loop = annotation.loop ?? innermost('loop');
    const op = annotation.op ?? innermost('op');
    const datum = annotation.datum ?? innermost('datum');
    let source = annotation.source;
    if (source && !this.sourceText) {
      const {text: omittedText, ...sourceWithoutText} = source;
      source = sourceWithoutText;
    }
    return {
      ...annotation,
      namespace: namespaces.join('/'),
      ...(derivedFrom !== undefined && {derivedFrom}),
      ...(loop !== undefined && {loop}),
      ...(op !== undefined && {op}),
      ...(datum !== undefined && {datum}),
      ...(source !== undefined && {source}),
      ...(node.dependsOn?.length && {dependsOn: [...node.dependsOn]})
    };
  }
}
