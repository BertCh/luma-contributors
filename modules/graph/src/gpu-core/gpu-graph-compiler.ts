// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {lowerGPUUpload, lowerGPUCopy, lowerGPUTextureCopy} from './gpu-transfer-lowering';
import {GPUUploadOperation, GPUCopyOperation} from './gpu-transfer-operation';
import {GPUCompactionOperation} from './gpu-compaction-operation';
import {GPUCompaction} from './gpu-compaction';

import type {TextureFormat} from '@luma.gl/core';
import {getTextureBindingInfo, getBoundTextureViewProps} from './gpu-texture-plan';
import type {
  GraphBufferHandle,
  GraphTextureHandle,
  GraphImportedTexture
} from './gpu-command-graph';
import {GPUGraphTextureView, type GPUGraphTexture} from './gpu-graph-value';

import {assert, Buffer, TextureView, type Device} from '@luma.gl/core';
import type {
  GPUCommandGraphRecordingOptions,
  GPUCommandGraphComputeNode,
  GraphBufferDescriptor
} from './gpu-command-graph-types';
import {getGPUGraphIntentGraphId, getGPUGraphIntentNodeId} from './gpu-graph-describe';
import {getGPUVectorFormatInfo, type GPUVectorInput} from '@luma-contributors/graph/gpu-data';
import type {GPUGraphVectorFormat} from './gpu-graph-value';
import {type GPUCommandGraph, GraphVectorView} from './gpu-command-graph';
import {
  addGPUCommandNodes,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from './gpu-command-node';
import {GPUCommandLoweringGraph} from './gpu-command-graph-lowering-scope';
import {createTransientView} from './graph-data-view-utils';
import {GPUScalarDispatchGate} from './gpu-scalar-dispatch-gate';
import {createGPUScalar, type GPUScalar} from './gpu-scalar';
import {GPUScalarCompute} from './gpu-scalar-operation';
import {GPUScalarLiteral} from './gpu-scalar-literal';
import {GPUVectorScalarMADD} from './gpu-elementwise-scalar';
import {GPUDotProductScalar} from './gpu-dot-product-scalar';
import {GPUAdaptiveSpMV} from './gpu-adaptive-spmv';
import {GPUPredicateConjunction} from './gpu-predicate-conjunction';
import {GPUGraphScalarOperation} from './gpu-semantic-scalar-operation';
import {GPUGraphScalarLiteral} from './gpu-semantic-state-operation';
import {GPUGraphVectorMADD, GPUGraphDotProduct} from './gpu-semantic-vector-operation';
import {GPUGraphSpMV} from './gpu-semantic-spmv';
import {
  setGPUComputeDispatchWorkgroups,
  getGPUComputeDispatchWorkgroups
} from './gpu-command-dispatch-metadata';
import {
  GPUCompositeOperation,
  getGPUOperationTree,
  isGPUCommandNodeProducer,
  isGPUOperation,
  type GPUGraphOperation,
  type GPUOperationTree
} from './gpu-operation';
import {
  GPUConditionalOperation,
  GPULoopOperation,
  type GPUOperationPredicate
} from './gpu-control-flow-operation';
import {
  type GPUOperationLowering,
  type GPUOperationLoweringDecision,
  type GPUOperationLoweringContext,
  type GPUGraphBackendCapabilities
} from './gpu-graph-lowering';
import {resolveGPUGraphDeclarations} from './gpu-graph-declarations';
import {validateGPUGraph, type GPUGraphValidationReport} from './gpu-graph-validation';
import type {GPUGraphScalar, GPUGraphVector} from './gpu-graph-value';
import type {GPUGraph} from './gpu-graph';
export type GPUGraphBindings = {
  scalars?: Readonly<Record<string, number>>;
  vectors?: Readonly<Record<string, GPUVectorInput>>;
  textures?: Readonly<Record<string, GraphImportedTexture | TextureView>>;
};
export type GPUGraphLoweredNode = {
  nodeId: string;
  nodeType: 'compute' | 'render' | 'copy';
  operationPath: readonly string[];
};
export type GPUGraphLoweringReport = {
  operations: readonly GPUOperationTree[];
  nodes: readonly GPUGraphLoweredNode[];
  decisions: readonly GPUOperationLoweringDecision[];
};
export type GPUGraphCompilation<Parameters = void> = {
  program: GPUGraph;
  graph: GPUCommandGraph<Parameters>;
  scalars: ReadonlyMap<string, GPUScalar>;
  vectors: ReadonlyMap<string, GraphVectorView>;
  textures: ReadonlyMap<string, GraphTextureHandle>;
  validation: GPUGraphValidationReport;
  lowering: GPUGraphLoweringReport;
};
type LoweringState = {
  path: string[];
  nodes: GPUGraphLoweredNode[];
  decisions: GPUOperationLoweringDecision[];
  predicates: GPUOperationPredicate[];
  scalars: Map<string, GPUScalar>;
  vectors: Map<string, GraphVectorView>;
  textures: Map<string, GraphTextureHandle>;
  textureValues: readonly GPUGraphTexture[];
  textureViews: ReadonlyMap<string, TextureView>;
  predicateConjunctions: Map<string, GPUScalar<'uint32'>>;
  commandNodeIds: Set<string>;
  recording?: {graphId: string; counts: Map<string, number>};
  transientResultNames: ReadonlySet<string>;
  retainedBuffers: readonly GraphBufferHandle[];
};
export class GPUGraphCompiler<Parameters = void> {
  readonly device: Device;
  readonly capabilities: GPUGraphBackendCapabilities;
  readonly lowerings: readonly GPUOperationLowering<Parameters>[];
  private compactionIndex = 0;
  constructor(device: Device, lowerings: readonly GPUOperationLowering<Parameters>[] = []) {
    this.device = device;
    this.capabilities = Object.freeze({
      backend: 'webgpu',
      gpuConditionals: true,
      nativeLoops: false,
      childGraphs: false
    });
    const combined = [...this.createCoreLowerings(), ...lowerings];
    const types = new Set<string>();
    for (const lowering of combined) {
      if (!lowering.type || types.has(lowering.type))
        throw new Error(`Duplicate or empty lowering type: ${lowering.type}`);
      types.add(lowering.type);
    }
    this.lowerings = Object.freeze(
      combined.map(lowering =>
        Object.freeze({type: lowering.type, lower: lowering.lower.bind(lowering)})
      )
    );
  }
  /** Lowers a semantic program, optionally retaining computation provenance. */
  compile(
    program: GPUGraph,
    bindings: GPUGraphBindings = {},
    options: {recording?: GPUCommandGraphRecordingOptions} = {}
  ): GPUGraphCompilation<Parameters> {
    this.compactionIndex = 0;
    // Bound external partitions must be available when resolving shape-inheriting results.
    program = program.clone();
    for (const vector of program.vectors) {
      const binding = bindings.vectors?.[vector.id];
      if (!vector.external || !binding || vector.chunkLengths) continue;
      const data = 'data' in binding ? binding.data : 'buffer' in binding ? [binding] : binding;
      program.constrainResult(vector, {
        type: 'vector',
        format: vector.format,
        chunkLengths: data.map(chunk => chunk.length)
      });
    }
    program = resolveGPUGraphDeclarations(program);
    const transferUsages = new Map<string, number>();
    const addTransferUsage = (name: string, usage: number): void => {
      transferUsages.set(name, (transferUsages.get(name) ?? 0) | usage);
    };
    const collectTransferUsages = (operation: GPUGraphOperation): void => {
      if (operation instanceof GPUCompositeOperation)
        operation.operations.forEach(collectTransferUsages);
      else if (operation instanceof GPUUploadOperation)
        addTransferUsage(operation.output, Buffer.COPY_DST);
      else if (operation instanceof GPUCopyOperation) {
        addTransferUsage(operation.input, Buffer.COPY_SRC);
        addTransferUsage(operation.output, Buffer.COPY_DST);
      }
    };
    program.operations.forEach(collectTransferUsages);
    const validation = validateGPUGraph(program, bindings);
    if (!validation.valid)
      throw new Error(
        `GPUGraph "${program.id}" validation failed: ${validation.issues
          .filter(i => i.level === 'error')
          .map(i => i.message)
          .join('; ')}`
      );
    const graph = new GPUGraphCommandGraph<Parameters>(this.device, {
      id: `${program.id}-commands`,
      recording: options.recording
    });
    const scalars = new Map<string, GPUScalar>();
    for (const scalar of program.scalars)
      scalars.set(scalar.id, createGPUScalar(graph, scalar.id, scalar.format));
    for (const [name, value] of Object.entries(bindings.scalars ?? {})) {
      const scalar = scalars.get(name);
      if (!scalar) throw new Error(`Undeclared scalar binding ${name}`);
      const view = scalar.view;
      graph.addCopyPass({
        id: `${name}-binding`,
        resources: [{buffer: view, usage: 'copy-destination'}],
        compile: ({device}) => {
          const data =
            scalar.format === 'float32'
              ? new Float32Array([value])
              : scalar.format === 'uint32'
                ? new Uint32Array([value])
                : new Int32Array([value]);
          const buffer = device.createBuffer({data, usage: Buffer.COPY_SRC});
          return {
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: buffer,
                destinationBuffer: getBuffer(view),
                destinationOffset: view.byteOffset,
                size: 4
              }),
            destroy: () => buffer.destroy()
          };
        }
      });
    }
    const vectors = new Map<string, GraphVectorView>();
    for (const vector of program.vectors) {
      if (vector.length === undefined)
        throw new Error(
          `${vector.id}: bind dynamic dimensions through program.compile(device).run(bindings)`
        );
      let resolved: GraphVectorView;
      if (vector.external) {
        const binding = bindings.vectors![vector.id];
        const source =
          'data' in binding
            ? binding
            : {
                format: vector.format,
                length: vector.length,
                data: 'buffer' in binding ? [binding] : binding
              };
        resolved = graph.importGPUVector(vector.id, source);
        if (resolved.format !== vector.format || resolved.length !== vector.length) {
          throw new Error(`${vector.id} binding format and length must match its logical vector`);
        }
        if (
          vector.chunkLengths &&
          (vector.chunkLengths.length !== resolved.data.length ||
            vector.chunkLengths.some((length, index) => length !== resolved.data[index].length))
        ) {
          throw new Error(`${vector.id} binding must preserve its declared chunk topology`);
        }
      } else {
        const chunkLengths =
          vector.chunkLengths ??
          getTransientChunkLengths(
            vector.length,
            this.device,
            getGPUVectorFormatInfo(vector.format).byteLength
          );
        const data = chunkLengths.map((length, index) =>
          createTransientView(
            graph,
            chunkLengths.length === 1 ? vector.id : `${vector.id}-chunk-${index}`,
            vector.format,
            length,
            vector.role === 'value'
              ? Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
              : Buffer.STORAGE | (transferUsages.get(vector.id) ?? 0)
          )
        );
        const formatInfo = getGPUVectorFormatInfo(vector.format);
        resolved = new GraphVectorView({
          id: vector.id,
          name: vector.id,
          format: vector.format,
          length: vector.length,
          valueLength: vector.length,
          stride: formatInfo.components,
          byteStride: formatInfo.byteLength,
          rowByteLength: formatInfo.byteLength,
          data
        });
      }
      vectors.set(vector.id, resolved);
    }
    const textures = new Map<string, GraphTextureHandle>();
    const textureViews = new Map<string, TextureView>();
    for (const texture of program.textures) {
      // Reusable declarations must have execution-resolved dimensions before low-level lowering.
      assert(texture.width !== undefined && texture.height !== undefined);
      let binding = bindings.textures?.[texture.id];
      let descriptor = {...texture, width: texture.width, height: texture.height};
      if (binding instanceof TextureView) {
        const info = getTextureBindingInfo(binding);
        assert(
          info.format === texture.format &&
            info.width === texture.width &&
            info.height === texture.height
        );
        textureViews.set(texture.id, binding);
        binding = info.texture;
        descriptor = {
          ...descriptor,
          dimension: binding.dimension,
          width: binding.width,
          height: binding.height,
          depth: binding.depth,
          mipLevels: binding.mipLevels,
          samples: binding.samples
        };
      }
      const handle = texture.external
        ? graph.importTexture(descriptor, binding)
        : graph.createTransientTexture(descriptor);
      textures.set(texture.id, handle);
    }
    const state: LoweringState = {
      path: [],
      nodes: [],
      decisions: [],
      predicates: [],
      scalars,
      vectors,
      textures,
      textureValues: program.textures,
      textureViews,
      predicateConjunctions: new Map(),
      commandNodeIds: new Set(),
      transientResultNames: new Set(
        program.vectors
          .filter(vector => vector.role === 'value' && !vector.external)
          .map(vector => vector.id)
      ),
      retainedBuffers: []
    };
    if (graph.recording) {
      state.recording = {graphId: getGPUGraphIntentGraphId(program), counts: new Map()};
    }
    for (const operation of program.operations) this.lower(graph, operation, state);
    return Object.freeze({
      program,
      graph,
      scalars,
      vectors,
      textures,
      validation,
      lowering: Object.freeze({
        operations: program.operationTree,
        nodes: Object.freeze(
          state.nodes.map(node =>
            Object.freeze({...node, operationPath: Object.freeze([...node.operationPath])})
          )
        ),
        decisions: Object.freeze(state.decisions.map(decision => Object.freeze({...decision})))
      })
    });
  }
  protected createCoreLowerings(): GPUOperationLowering<Parameters>[] {
    return [
      {type: 'vector-upload', lower: lowerGPUUpload},
      {type: 'vector-copy', lower: lowerGPUCopy},
      {
        type: 'compaction',
        lower: (operation: GPUCompactionOperation, context) => {
          const {input, flags, output, count} = operation.props;
          context.graph.add(
            new GPUCompaction({
              // Unrolled iterations need distinct scratch resources as well as distinct nodes.
              id: `${operation.id}-${this.compactionIndex++}`,
              input: context.resolveVector<'uint32'>(input),
              flags: context.resolveVector<'uint32'>(flags),
              output: context.resolveVector<'uint32'>(output),
              count: context.resolveScalar<'uint32'>(count).view
            })
          );
        }
      },
      {
        type: 'scalar-literal',
        lower: (operation: GPUGraphScalarLiteral, context) => {
          const e = new GPUScalarLiteral({
            id: operation.id,
            output: context.resolveScalar(operation.props.output),
            value: operation.props.value
          });
          this.emitProducer(context.graph, e, [[1, 1, 1]]);
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'webgpu-scalar-literal',
            reason: 'host-known literal lowers to one compute command node'
          });
        }
      },
      {
        type: 'scalar-operation',
        lower: (operation: GPUGraphScalarOperation, context) => {
          const e = new GPUScalarCompute({
            id: operation.id,
            operation: operation.operation,
            left: context.resolveScalar(operation.left),
            right: operation.right ? context.resolveScalar(operation.right) : undefined,
            output: context.resolveScalar(operation.output)
          });
          this.emitProducer(context.graph, e, [[1, 1, 1]]);
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'webgpu-scalar-kernel',
            reason: 'scalar operation lowers to one compute command node'
          });
        }
      },
      {
        type: 'vector-madd',
        lower: (operation: GPUGraphVectorMADD, context) => {
          const p = operation.props,
            e = new GPUVectorScalarMADD({
              id: operation.id,
              input: context.resolveVector(p.input),
              scale: context.resolveScalar(p.scale),
              addend: context.resolveVector(p.addend),
              output: context.resolveVector(p.output)
            });
          this.emitProducer(context.graph, e);
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'webgpu-vector-madd',
            reason: 'dense float32 MADD lowers through explicit command nodes'
          });
        }
      },
      {
        type: 'dot-product',
        lower: (operation: GPUGraphDotProduct, context) => {
          const p = operation.props,
            e = new GPUDotProductScalar({
              id: operation.id,
              left: context.resolveVector(p.left),
              right: context.resolveVector(p.right),
              output: context.resolveScalar(p.output)
            });
          this.emitProducer(context.graph, e);
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'webgpu-hierarchical-dot',
            reason: 'chunk dot products use bounded hierarchical reductions into the scalar arena'
          });
        }
      },
      {
        type: 'spmv',
        lower: (operation: GPUGraphSpMV, context) => {
          const p = operation.props,
            m = p.matrix,
            e = new GPUAdaptiveSpMV({
              id: operation.id,
              rowOffsets: context.resolveVector(m.rowOffsets),
              columnIndices: context.resolveVector(m.columnIndices),
              values: context.resolveVector(m.values),
              vector: context.resolveVector(p.vector),
              output: context.resolveVector(p.output),
              columns: m.columns,
              statistics: m.statistics,
              strategy: p.strategy
            });
          const d = e.getStrategy(context.graph);
          this.emitProducer(context.graph, e);
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: `webgpu-spmv:${d.id}`,
            reason: d.reason
          });
        }
      },
      {
        type: 'conditional',
        lower: (operation: GPUConditionalOperation, context) => {
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'gpu-indirect-gate',
            reason: 'WebGPU realizes runtime predicates with zero/nonzero indirect dispatch'
          });
          for (const child of operation.operations)
            context.lowerConditional(operation.predicate, child);
        }
      },
      {
        type: 'loop',
        lower: (operation: GPULoopOperation, context) => {
          if (!operation.predicate || operation.lowering === 'unroll') {
            context.recordDecision({
              operationId: operation.id,
              operationType: operation.type,
              lowering: 'bounded-unroll',
              reason: operation.predicate
                ? 'explicit unroll preference'
                : 'loop has no runtime predicate'
            });
            for (let iteration = 0; iteration < operation.maximumIterations; iteration++) {
              if (context.withLoopIteration) {
                context.withLoopIteration(
                  iteration,
                  'unconditional',
                  operation.maximumIterations,
                  () => {
                    for (const child of operation.operations) context.lower(child);
                  }
                );
              } else {
                for (const child of operation.operations) context.lower(child);
              }
            }
            return;
          }
          context.recordDecision({
            operationId: operation.id,
            operationType: operation.type,
            lowering: 'gpu-gated-bounded-sequence',
            reason: 'WebGPU has indirect dispatch but no native graph loop'
          });
          for (let iteration = 0; iteration < operation.maximumIterations; iteration++) {
            const gated = iteration >= operation.minimumIterations;
            const lowerIteration = () => {
              for (const child of operation.operations)
                gated
                  ? context.lowerConditional(operation.predicate!, child)
                  : context.lower(child);
            };
            if (context.withLoopIteration) {
              context.withLoopIteration(
                iteration,
                gated ? 'gated' : 'unconditional',
                operation.maximumIterations,
                lowerIteration
              );
            } else {
              lowerIteration();
            }
          }
        }
      }
    ];
  }
  protected lower(
    graph: GPUCommandLoweringGraph<Parameters>,
    operation: GPUGraphOperation,
    state: LoweringState
  ): void {
    if (state.recording) {
      const tree = getGPUOperationTree(operation);
      const intentNodeId = getGPUGraphIntentNodeId(tree.id, state.recording.counts);
      graph.withRecordScope(
        {namespace: intentNodeId, derivedFrom: [`${state.recording.graphId}:${intentNodeId}`]},
        () => {
          this.lowerOperation(graph, operation, state, intentNodeId);
        }
      );
      // Opaque producers may describe children without lowering them independently.
      if (tree.children && ('getNodes' in operation || isGPUCommandNodeProducer(operation))) {
        const visit = (child: GPUOperationTree): void => {
          getGPUGraphIntentNodeId(child.id, state.recording!.counts);
          child.children?.forEach(visit);
        };
        tree.children.forEach(visit);
      }
      return;
    }
    this.lowerOperation(graph, operation, state);
  }
  private lowerOperation(
    graph: GPUCommandLoweringGraph<Parameters>,
    operation: GPUGraphOperation,
    state: LoweringState,
    intentNodeId?: string
  ): void {
    const id =
      'id' in operation && typeof operation.id === 'string'
        ? operation.id
        : (operation.constructor?.name ?? 'command-node-producer');
    state.path.push(id);
    const previousRetainedBuffers = state.retainedBuffers;
    const retainedBuffers = [...previousRetainedBuffers];
    if (isGPUOperation(operation)) {
      for (const use of operation.contract?.values ?? []) {
        const name = typeof use.value === 'string' ? use.value : use.value.id;
        if (state.transientResultNames.has(name))
          for (const chunk of state.vectors.get(name)!.data) retainedBuffers.push(chunk.buffer);
      }
    }
    state.retainedBuffers = retainedBuffers;
    try {
      if (operation instanceof GPUCompositeOperation && operation.type === 'composite') {
        state.decisions.push({
          operationId: intentNodeId ?? operation.id,
          operationType: operation.type,
          lowering: 'flatten',
          reason: 'WebGPU has no executable child-graph primitive'
        });
        for (const child of operation.operations) this.lower(graph, child, state);
        return;
      }
      if (isGPUOperation(operation)) {
        for (const feature of operation.contract?.requiredFeatures ?? [])
          if (!this.device.features.has(feature)) throw new Error(`${id} requires ${feature}`);
        const lowerer = this.lowerings.find(lowering => lowering.type === operation.type)?.lower;
        if (lowerer) {
          const iterationCounts =
            state.recording && operation.type === 'loop'
              ? new Map(state.recording.counts)
              : undefined;
          const context: GPUOperationLoweringContext<Parameters> = {
            graph,
            capabilities: this.capabilities,
            lower: child => this.lower(graph, child, state),
            lowerConditional: (predicate, child) => {
              state.predicates.push(predicate);
              try {
                this.lower(graph, child, state);
              } finally {
                state.predicates.pop();
              }
            },
            resolveScalar: scalar => this.resolveScalar(scalar, state),
            resolveVector: vector => this.resolveVector(vector, state),
            resolveTexture: texture => this.resolveTexture(texture, state),
            resolveTextureView: value => {
              const texture = value instanceof GPUGraphTextureView ? value.texture : value;
              const props = value instanceof GPUGraphTextureView ? value.props : {};
              const binding = state.textureViews.get(
                typeof texture === 'string' ? texture : texture.id
              );
              return graph.createTextureView(
                this.resolveTexture(texture, state),
                binding ? getBoundTextureViewProps(binding, props) : props
              );
            },
            recordDecision: decision =>
              state.decisions.push(
                intentNodeId !== undefined ? {...decision, operationId: intentNodeId} : decision
              )
          };
          if (iterationCounts) {
            context.withLoopIteration = (iteration, phase, maximumIterations, callback) => {
              state.recording!.counts = new Map(iterationCounts);
              graph.withRecordScope(
                {
                  namespace: `iteration-${iteration}`,
                  loop: {
                    id: intentNodeId!,
                    iteration,
                    phase,
                    maximumIterations
                  }
                },
                callback
              );
            };
          }
          const invoke = () =>
            (operation instanceof GPUCopyOperation &&
              state.textureValues.some(value => value.id === operation.input)
              ? (lowerGPUTextureCopy as typeof lowerer)
              : lowerer)(operation, context);
          if (operation instanceof GPUConditionalOperation || operation instanceof GPULoopOperation)
            invoke();
          else this.withNodeDecoration(graph, state, invoke);
          return;
        }
      }
      if ('getNodes' in operation || isGPUCommandNodeProducer(operation)) {
        state.decisions.push({
          operationId: intentNodeId ?? id,
          operationType: 'command-node-producer',
          lowering: 'explicit-command-nodes',
          reason: 'execution primitive constructs explicit command nodes'
        });
        this.withNodeDecoration(graph, state, () => graph.add(operation));
        return;
      }
      throw new Error(`GPUGraph operation "${id}" has no WebGPU lowering`);
    } finally {
      state.retainedBuffers = previousRetainedBuffers;
      state.path.pop();
    }
  }
  private emitProducer(
    graph: GPUCommandGraph<Parameters>,
    producer: GPUCommandNodeProducer<Parameters>,
    dispatches?: [number, number, number][]
  ): void {
    const nodes = producer.getCommandNodes(graph);
    if (!dispatches) {
      addGPUCommandNodes(graph, nodes);
      return;
    }
    if (nodes.length !== dispatches.length)
      throw new Error(
        `lowering produced ${nodes.length} command nodes but declared ${dispatches.length} dispatches`
      );
    const decorated = nodes.map((node, index) => this.attachDispatch(node, dispatches[index]));
    addGPUCommandNodes(graph, decorated);
  }
  private attachDispatch(
    node: GPUCommandNode<Parameters>,
    dispatch: [number, number, number]
  ): GPUCommandNode<Parameters> {
    if (node.type !== 'compute')
      throw new Error(
        `dispatch geometry can only be attached to compute command nodes; got ${node.type} node "${node.id}"`
      );
    return setGPUComputeDispatchWorkgroups(node, dispatch) as GPUCommandNode<Parameters>;
  }
  private withNodeDecoration<Result>(
    graph: GPUCommandLoweringGraph<Parameters>,
    state: LoweringState,
    callback: () => Result
  ): Result {
    return graph.withComputeLowering(
      node => this.decorateComputeNode(graph, node, state),
      () =>
        graph.withOtherLowering(node => {
          if (state.predicates.length)
            throw new Error(`GPU predicates do not support ${node.type} node "${node.id}"`);
          const decorated = this.makeCommandNodeIdUnique(node, state, Boolean(graph.recording));
          state.nodes.push({
            nodeId: decorated.id,
            nodeType: decorated.type,
            operationPath: [...state.path]
          });
          return {...decorated, retainedBuffers: state.retainedBuffers};
        }, callback)
    );
  }
  private decorateComputeNode(
    graph: GPUCommandLoweringGraph<Parameters>,
    node: Omit<GPUCommandGraphComputeNode<Parameters>, 'type'>,
    state: LoweringState
  ): Omit<GPUCommandGraphComputeNode<Parameters>, 'type'> {
    node = {
      ...this.makeCommandNodeIdUnique(node, state, Boolean(graph.recording)),
      retainedBuffers: state.retainedBuffers
    };
    if (state.path.length)
      state.nodes.push({nodeId: node.id, nodeType: 'compute', operationPath: [...state.path]});
    if (!state.predicates.length) return node;
    if (node.condition) throw new Error(`compute node "${node.id}" already has condition`);
    const workgroups = getGPUComputeDispatchWorkgroups(node);
    if (!workgroups)
      throw new Error(
        `compute node "${node.id}" under runtime predicate must declare exact dispatch geometry`
      );
    let active: GPUScalar<'uint32'>;
    if (state.predicates.length === 1) {
      active = state.scalars.get(state.predicates[0].value.id) as GPUScalar<'uint32'>;
    } else {
      const key = state.predicates.map(p => p.value.id).join('&&');
      let conjunction = state.predicateConjunctions.get(key);
      if (!conjunction) {
        conjunction = createGPUScalar(
          graph,
          `predicate-and-${state.predicateConjunctions.size}`,
          'uint32'
        );
        state.predicateConjunctions.set(key, conjunction);
      }
      const inputs = state.predicates.map(
        p => state.scalars.get(p.value.id) as GPUScalar<'uint32'>
      );
      const conjunctionNodes = new GPUPredicateConjunction({
        id: `${node.id}-predicate-and`,
        inputs,
        output: conjunction!
      })
        .getCommandNodes(graph)
        .map(node =>
          this.attachDispatch(
            graph.recording
              ? {
                  ...node,
                  annotation: {
                    ...node.annotation,
                    attrs: {...node.annotation?.attrs, role: 'predicate-conjunction'}
                  }
                }
              : node,
            [1, 1, 1]
          )
        );
      graph.withoutComputeLowering(() => addGPUCommandNodes(graph, conjunctionNodes));
      active = conjunction;
    }
    if (!active) throw new Error(`runtime predicate for "${node.id}" could not be resolved`);
    const gate = new GPUScalarDispatchGate(graph, {id: `${node.id}-predicate`, active, workgroups});
    const gateNodes = gate.getUpdateCommandNodes(graph, `${node.id}-predicate-update`).map(node =>
      this.attachDispatch(
        graph.recording
          ? {
              ...node,
              annotation: {
                ...node.annotation,
                attrs: {...node.annotation?.attrs, role: 'predicate-gate'}
              }
            }
          : node,
        [1, 1, 1]
      )
    );
    graph.withoutComputeLowering(() => addGPUCommandNodes(graph, gateNodes));
    if (graph.recording && node.annotation?.dispatch) {
      node = {
        ...node,
        annotation: {
          ...node.annotation,
          dispatch: {...node.annotation.dispatch, indirect: true}
        }
      };
    }
    return {...node, condition: gate.condition};
  }
  private makeCommandNodeIdUnique<
    Node extends {id: string; annotation?: GPUCommandGraphComputeNode<Parameters>['annotation']}
  >(node: Node, state: LoweringState, recording: boolean): Node {
    if (!state.commandNodeIds.has(node.id)) {
      state.commandNodeIds.add(node.id);
      return node;
    }
    let iteration = 1;
    let id = `${node.id}-iteration-${iteration}`;
    while (state.commandNodeIds.has(id)) {
      iteration++;
      id = `${node.id}-iteration-${iteration}`;
    }
    state.commandNodeIds.add(id);
    return recording
      ? {...node, id, annotation: {...node.annotation, dedupName: node.id}}
      : {...node, id};
  }
  private resolveTexture<Format extends TextureFormat>(
    texture: GPUGraphTexture<Format> | string,
    state: LoweringState
  ): GraphTextureHandle<Format> {
    const name = typeof texture === 'string' ? texture : texture.id;
    if (
      typeof texture === 'string'
        ? !state.textures.has(name)
        : !state.textureValues.includes(texture)
    )
      throw new Error(`Undeclared program texture "${name}"`);
    // The map erases formats; identity validation above preserves the declared texture's format.
    return state.textures.get(name)! as GraphTextureHandle<Format>;
  }

  private resolveScalar<T extends 'float32' | 'uint32' | 'sint32'>(
    scalar: GPUGraphScalar<T> | string,
    state: LoweringState
  ): GPUScalar<T> {
    const name = typeof scalar === 'string' ? scalar : scalar.id;
    const resolved = state.scalars.get(name);
    if (!resolved || (typeof scalar !== 'string' && resolved.format !== scalar.format))
      throw new Error(`GPUGraph scalar "${name}" is not part of compilation`);
    return resolved as GPUScalar<T>;
  }
  private resolveVector<T extends GPUGraphVectorFormat>(
    vector: GPUGraphVector<T> | string,
    state: LoweringState
  ): GraphVectorView<T> {
    const name = typeof vector === 'string' ? vector : vector.id;
    const resolved = state.vectors.get(name);
    if (
      !resolved ||
      (typeof vector !== 'string' &&
        (resolved.format !== vector.format ||
          (vector.length !== undefined && resolved.length !== vector.length)))
    )
      throw new Error(`GPUGraph vector "${name}" is not part of compilation`);
    return resolved as GraphVectorView<T>;
  }
}

/** Assigns distinct scratch storage to each realization of a program operation. */
class GPUGraphCommandGraph<Parameters> extends GPUCommandLoweringGraph<Parameters> {
  private readonly transientBufferIds = new Set<string>();

  override createTransientBuffer(descriptor: GraphBufferDescriptor) {
    let id = descriptor.id;
    let iteration = 0;
    while (this.transientBufferIds.has(id)) {
      id = `${descriptor.id}-iteration-${++iteration}`;
    }
    // Producers allocate scratch before the compiler decorates their command nodes. Repeated
    // loop bodies need distinct logical buffers even when their local scratch names repeat.
    const buffer = super.createTransientBuffer(
      id === descriptor.id ? descriptor : {...descriptor, id}
    );
    this.transientBufferIds.add(id);
    return buffer;
  }
}

function getTransientChunkLengths(
  length: number,
  device: Device,
  rowByteLength: number
): readonly number[] {
  const maximumLength = Math.floor(device.limits.maxStorageBufferBindingSize / rowByteLength);
  if (maximumLength < 1 || length <= maximumLength) return [length];
  const lengths: number[] = [];
  for (let offset = 0; offset < length; offset += maximumLength) {
    lengths.push(Math.min(maximumLength, length - offset));
  }
  return lengths;
}
