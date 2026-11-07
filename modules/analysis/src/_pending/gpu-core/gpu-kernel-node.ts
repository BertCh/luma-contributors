// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUComputationLoopCoordinate} from '../../computation-record/computation-record-types';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GraphBufferUse} from '@luma.gl/graph/gpu-core';
import {setGPUComputeDispatchWorkgroups} from '@luma.gl/graph/gpu-core';
import {createGPUComputeCommandNode, type GPUComputeCommandNode} from '@luma.gl/graph/gpu-core';
import {getGPUComputeKernelAnnotation} from './gpu-command-node-record';
import type {GPUBoundedDispatchLayout} from '@luma.gl/graph/gpu-core';
import {getViewBinding, getViewBindingRange} from '@luma.gl/graph/gpu-core';

/** One storage binding of a {@link createGPUKernelNode} kernel; array order is binding order. */
export type GPUKernelBinding = {
  /** WGSL variable name and luma.gl binding name. */
  name: string;
  /** Packed or strided graph view bound as a storage buffer. */
  view: GraphDataView;
  /** How the shader accesses the view. Drives graph hazards, the bind type, and the annotation. */
  access: 'read' | 'write' | 'read_write';
  /** Declares the storage array as `array<atomic<u32>>` for kernels that synchronize through it. */
  atomic?: boolean;
  /** WGSL element type. Defaults to the type of the view format; set `u32` to copy raw words. */
  elementType?: 'u32' | 'i32' | 'f32';
};

/** Binding list accepted by {@link createGPUKernelNode}; falsy entries are skipped. */
export type GPUKernelBindingList = readonly (GPUKernelBinding | false | undefined | null)[];

/** Properties for {@link createGPUKernelNode}. */
export type GPUKernelNodeProps = {
  /** Graph node and pipeline identifier. */
  id: string;
  /** Exported operator name recorded in the workload estimate and annotation. */
  operation: string;
  /** Workload variant used by autotuning calibration, for example a strategy name. */
  variant?: string;
  /** Stable module identifier for recording graphs. Defaults to `${operation}/${variant}`. */
  moduleId?: string;
  /** Formula reference id for recording graphs. */
  formula?: string;
  /** Shared name of unrolled copies for recording graphs. */
  dedupName?: string;
  /** Unrolled pass coordinate, recorded on recording graphs only. */
  loop?: GPUComputationLoopCoordinate;
  /** Complete WGSL, including binding declarations (see {@link getGPUKernelBindingDeclarations}). */
  source: string;
  /** Lanes per workgroup. Defaults to 256. */
  workgroupSize?: number;
  /** Bounded dispatch geometry. */
  dispatchLayout: GPUBoundedDispatchLayout;
  /** Storage bindings in declaration order. Falsy entries are skipped so callers can inline `&&`. */
  bindings: GPUKernelBindingList;
  /** Pipeline override constants; declare them with {@link getGPUKernelOverrideDeclarations}. */
  constants?: Readonly<Record<string, number>>;
  /** Logical rows handled by the pass, recorded on recording graphs only. */
  invocationCount?: number;
  /** Rejects bindings larger than the device's maximum storage binding when the node is created. */
  checkBindingSize?: boolean;
  /** Replaces the byte estimates derived from the bindings. */
  workload?: {readByteLength?: number; writeByteLength?: number};
  /**
   * Attach the workload estimate to every graph (default), to recording graphs only, or never.
   * Kernels that historically carried no estimate keep it that way.
   */
  workloadScope?: 'always' | 'recording' | 'never';
};

/** Resolved binding list without skipped entries. */
function getKernelBindings(props: GPUKernelNodeProps): GPUKernelBinding[] {
  return props.bindings.filter((binding): binding is GPUKernelBinding => Boolean(binding));
}

const WGSL_ELEMENT_TYPES: Record<string, string> = {uint32: 'u32', sint32: 'i32', float32: 'f32'};

/**
 * Generates `@group(0) @binding(n)` storage declarations from a binding list, so the WGSL binding
 * order cannot drift from the order passed to {@link createGPUKernelNode}.
 */
export function getGPUKernelBindingDeclarations(bindings: GPUKernelBindingList): string {
  return bindings
    .filter((binding): binding is GPUKernelBinding => Boolean(binding))
    .map(({name, view, access, atomic, elementType: requestedType}, location) => {
      const elementType = requestedType ?? WGSL_ELEMENT_TYPES[view.format] ?? 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access === 'read' ? 'read' : 'read_write'}> ${name}: array<${atomic ? `atomic<${elementType}>` : elementType}>;`;
    })
    .join('\n');
}

/**
 * Generates `override NAME: type;` declarations for pipeline constants. Constants are `u32` unless
 * `types` names another scalar type, which lets float and signed values travel as constants too.
 */
export function getGPUKernelOverrideDeclarations(
  constants: Readonly<Record<string, number>>,
  types: Readonly<Record<string, 'u32' | 'i32' | 'f32'>> = {}
): string {
  return Object.keys(constants)
    .map(name => `override ${name}: ${types[name] ?? 'u32'};`)
    .join('\n');
}

/**
 * Creates one compute node: kernel compilation, view-to-buffer binding with a reusable bind group,
 * workload estimate, recording annotation, and dispatch metadata. Every gpu-core primitive that
 * dispatches one WGSL kernel over graph views goes through this helper so cross-cutting changes
 * (bind-group caching, dispatch validation, conditions) happen in one place.
 *
 * Problem-size values belong in `constants` rather than the WGSL text, so one pipeline serves every
 * row count, view offset, and dispatch shape.
 */
export function createGPUKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUKernelNodeProps
): GPUComputeCommandNode<Parameters> {
  const bindings = getKernelBindings(props);
  if (props.checkBindingSize) {
    for (const {view} of bindings) {
      // Every active binding must fit one storage binding of the device.
      if (getViewBindingRange(view).size > graph.device.limits.maxStorageBufferBindingSize) {
        throw new Error(`${props.id} active chunks must fit a storage binding`);
      }
    }
  }
  const workgroupSize = props.workgroupSize ?? 256;
  const {dispatchLayout} = props;
  const workgroupCount = dispatchLayout.x * dispatchLayout.y * dispatchLayout.z;
  const resources: GraphBufferUse[] = bindings.map(({view, access}) => ({
    buffer: view,
    usage:
      access === 'read'
        ? 'storage-read'
        : access === 'write'
          ? 'storage-write'
          : 'storage-read-write'
  }));
  const getByteLength = (matches: (usage: GraphBufferUse['usage']) => boolean) =>
    resources.reduce(
      (total, resource) =>
        total +
        (matches(resource.usage)
          ? 'length' in resource.buffer
            ? resource.buffer.length * resource.buffer.rowByteLength
            : resource.buffer.byteLength
          : 0),
      0
    );

  let node = createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources,
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source: props.source,
        ...(props.constants && {constants: {...props.constants}}),
        shaderLayout: {
          bindings: bindings.map(({name, access}, location) => ({
            name,
            type: access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      // Bind groups are keyed by the resolved physical buffers, so a recurring graph reuses them.
      let resolvedBuffers: unknown[] = [];
      let resolvedBindings: Record<string, Binding> = {};
      let bindGroupCacheKey = {};
      return {
        encode: ({computePass, getBuffer}) => {
          const nextBuffers = bindings.map(({view}) => getBuffer(view));
          if (
            nextBuffers.length !== resolvedBuffers.length ||
            nextBuffers.some((buffer, index) => buffer !== resolvedBuffers[index])
          ) {
            resolvedBuffers = nextBuffers;
            resolvedBindings = {};
            for (const {name, view} of bindings) {
              resolvedBindings[name] = getViewBinding(view, getBuffer);
            }
            bindGroupCacheKey = {};
          }
          kernel.dispatch(computePass, {
            bindings: resolvedBindings,
            _bindGroupCacheKeys: {0: bindGroupCacheKey},
            x: dispatchLayout.x,
            y: dispatchLayout.y,
            z: dispatchLayout.z
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });

  if (props.workloadScope !== 'never' && (props.workloadScope !== 'recording' || graph.recording)) {
    node.workload = {
      operation: props.operation,
      ...(props.variant && {variant: props.variant}),
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * workgroupSize,
      readByteLength:
        props.workload?.readByteLength ??
        getByteLength(usage => usage === 'storage-read' || usage === 'storage-read-write'),
      writeByteLength:
        props.workload?.writeByteLength ??
        getByteLength(usage => usage === 'storage-write' || usage === 'storage-read-write')
    };
  }

  const annotation = getGPUComputeKernelAnnotation(graph, () => ({
    op: props.operation,
    moduleId: props.moduleId ?? `${props.operation}/${props.variant ?? 'kernel'}`,
    source: props.source,
    ...(props.constants && {constants: props.constants}),
    workgroupSize: [workgroupSize, 1, 1],
    workgroups: [dispatchLayout.x, dispatchLayout.y, dispatchLayout.z],
    invocationCount: props.invocationCount,
    loop: props.loop,
    dedupName: props.dedupName,
    formula: props.formula,
    bindings: bindings.map(({name, view, access}) => ({name, view, access}))
  }));
  if (annotation) node.annotation = annotation;

  node = setGPUComputeDispatchWorkgroups(node, [
    dispatchLayout.x,
    dispatchLayout.y,
    dispatchLayout.z
  ]);
  return node;
}
