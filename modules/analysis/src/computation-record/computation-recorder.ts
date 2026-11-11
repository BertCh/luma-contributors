// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GraphBufferHandle,
  GraphExternalTextureHandle,
  GraphTextureHandle
} from '@luma.gl/graph/gpu-core';
import {recordWGSLKernelNodes} from '../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUComputationNodeAnnotation,
  GPUComputationRecordScope
} from './computation-record-types';

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

/**
 * Analysis-side record of one command graph under construction. Published `GPUCommandGraph` has
 * no recording hooks, so analysis code passes each node through the recorder as it builds the graph.
 * The recorder only reads public node properties and never touches the graph's internals.
 */
export class GPUComputationRecorder {
  /** Whether recorded sources keep their full text. */
  readonly sourceText: boolean;

  /** Graph whose `createWGSLKernelNode` kernels this recorder describes, set by `recordKernels`. */
  get graph(): GPUCommandGraph<any> | undefined {
    return this.kernelGraph;
  }

  private kernelGraph?: GPUCommandGraph<any>;
  private readonly scopes: GPUComputationRecordScope[] = [];
  private readonly nodes = new Map<string, GPUComputationRecordedNode>();
  private readonly resources = new Map<string, GPUComputationRecordedResource>();

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
    this.kernelGraph = graph;
    return graph;
  }

  /** Runs `callback`; nodes recorded inside inherit the scope. Scopes nest, innermost wins. */
  withScope<Result>(scope: GPUComputationRecordScope, callback: () => Result): Result {
    this.scopes.push(scope);
    try {
      return callback();
    } finally {
      this.scopes.pop();
    }
  }

  /**
   * Records a node's annotation, merged with the enclosing scopes, plus its resource handles and
   * explicit dependencies. Returns the same node so it can be passed straight to `graph.add`.
   */
  recordNode<Node extends GPUCommandNode<any>>(
    node: Node,
    annotation: GPUComputationNodeAnnotation = {}
  ): Node {
    this.nodes.set(node.id, this.mergeScopes(node, annotation));
    for (const use of node.resources ?? []) {
      if ('buffer' in use) {
        const handle = 'buffer' in use.buffer ? use.buffer.buffer : use.buffer;
        this.resources.set(handle.id, handle);
      } else if ('texture' in use) {
        const handle = 'texture' in use.texture ? use.texture.texture : use.texture;
        this.resources.set(handle.id, handle);
      } else {
        this.resources.set(use.externalTexture.id, use.externalTexture);
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
    return this.nodes.get(nodeId);
  }

  /** Returns the handle of a resource used by a recorded node. */
  getResource(resourceId: string): GPUComputationRecordedResource | undefined {
    return this.resources.get(resourceId);
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
