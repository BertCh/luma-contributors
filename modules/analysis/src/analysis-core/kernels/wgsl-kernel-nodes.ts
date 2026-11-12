// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding, type BindingDeclaration, type TextureView} from '@luma.gl/core';
import {getPlatformInfo, Kernel} from '@luma.gl/graph';
import {
  fp64,
  fp64arithmetic,
  getShaderModuleDependencies,
  WGSLShaderAssembler,
  type ShaderModule
} from '@luma.gl/shaders';
import {maskWGSLComments} from '@luma.gl/shaders/wgsl';
import {
  createGPUComputeCommandNode,
  createTransientView,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewBindingRange,
  getViewElementOffset,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandGraphGPUIndirectCondition,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView,
  type GraphResourceUse,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import type {GPUCompactOutput} from './gpu-contributor-types';
import {validateMaximumCount, validateRequirement} from './gpu-contributor-utils';

const DEFAULT_WORKGROUP_SIZE = 256;

/** GPU-written dispatch and its update node for a runtime active-item count. @internal */
export type WGSLActiveCountDispatch<Parameters> = {
  updateNode: GPUCommandNode<Parameters>;
  condition: GPUCommandGraphGPUIndirectCondition;
};

/**
 * Writes a bounded three-dimensional indirect dispatch from a GPU-resident item count.
 *
 * `invocationsPerItem` is one for ordinary linear kernels and equals the workgroup size for
 * cooperative one-workgroup-per-item kernels. The emitted dimensions preserve the static bounded
 * layout used by {@link createWGSLKernelNode}, so its linear invocation mapping remains valid.
 *
 * @internal
 */
export function createWGSLActiveCountDispatch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    count: GraphDataView<'uint32'>;
    countRow?: number;
    maximumItemCount: number;
    workgroupSize?: number;
    invocationsPerItem?: number;
    additionalInvocationCount?: number;
  }
): WGSLActiveCountDispatch<Parameters> {
  const workgroupSize = props.workgroupSize ?? DEFAULT_WORKGROUP_SIZE;
  const invocationsPerItem = props.invocationsPerItem ?? 1;
  const additionalInvocationCount = props.additionalInvocationCount ?? 0;
  const countRow = props.countRow ?? 0;
  if (
    !Number.isSafeInteger(countRow) ||
    countRow < 0 ||
    countRow >= props.count.length ||
    !Number.isSafeInteger(props.maximumItemCount) ||
    props.maximumItemCount < 0 ||
    !Number.isSafeInteger(invocationsPerItem) ||
    invocationsPerItem < 1 ||
    !Number.isSafeInteger(additionalInvocationCount) ||
    additionalInvocationCount < 0 ||
    props.maximumItemCount * invocationsPerItem + additionalInvocationCount > 0xffffffff
  ) {
    throw new Error(`${props.operation}: invalid active-count dispatch topology`);
  }
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.maximumItemCount * invocationsPerItem + additionalInvocationCount, 1),
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const dispatch = createTransientView(
    graph,
    `${props.id}-buffer`,
    'uint32',
    3,
    Buffer.STORAGE | Buffer.INDIRECT
  );
  const updateNode = createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'active-count-dispatch',
    bindings: [
      {name: 'count', view: props.count, type: 'u32', access: 'read'},
      {name: 'dispatch', view: dispatch, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    workgroupSize: 2,
    // Topology values are pipeline-overridable so every active-count updater shares one WGSL
    // module; only the specialized pipeline constants differ between call sites.
    declarations: `override COUNT_ROW: u32;
override MAXIMUM_ITEM_COUNT: u32;
override INVOCATIONS_PER_ITEM: u32;
override ADDITIONAL_INVOCATION_COUNT: u32;
override TARGET_WORKGROUP_SIZE: u32;
override MAXIMUM_X: u32;
override MAXIMUM_Y: u32;`,
    constants: {
      COUNT_ROW: countRow,
      MAXIMUM_ITEM_COUNT: props.maximumItemCount,
      INVOCATIONS_PER_ITEM: invocationsPerItem,
      ADDITIONAL_INVOCATION_COUNT: additionalInvocationCount,
      TARGET_WORKGROUP_SIZE: workgroupSize,
      MAXIMUM_X: layout.x,
      MAXIMUM_Y: layout.y
    },
    body: `let itemCount = min(count[countOffset + COUNT_ROW], MAXIMUM_ITEM_COUNT);
  let invocationCount = itemCount * INVOCATIONS_PER_ITEM + ADDITIONAL_INVOCATION_COUNT;
  let workgroupCount = (invocationCount + TARGET_WORKGROUP_SIZE - 1u) / TARGET_WORKGROUP_SIZE;
  var x = min(workgroupCount, MAXIMUM_X);
  var y = 1u;
  var z = 1u;
  if (workgroupCount > MAXIMUM_X) {
    x = MAXIMUM_X;
    y = min((workgroupCount + MAXIMUM_X - 1u) / MAXIMUM_X, MAXIMUM_Y);
  }
  if (workgroupCount > MAXIMUM_X * MAXIMUM_Y) {
    y = MAXIMUM_Y;
    z = (workgroupCount + MAXIMUM_X * MAXIMUM_Y - 1u) / (MAXIMUM_X * MAXIMUM_Y);
  }
  dispatch[dispatchOffset] = x;
  dispatch[dispatchOffset + 1u] = y;
  dispatch[dispatchOffset + 2u] = z;`
  });
  return {
    updateNode,
    condition: {
      id: `${props.id}-condition`,
      source: 'gpu',
      mode: 'indirect',
      buffer: dispatch.buffer,
      byteOffset: dispatch.byteOffset
    }
  };
}
/** GPU-written indirect dispatch gate of a loop with a data-dependent exit. */
export type WGSLDispatchGate<Parameters> = {
  /** Transient buffer view holding one indirect dispatch triple per shape. */
  view: GraphDataView<'uint32'>;
  /** Read-write binding named `dispatch` for the one-thread node that arms or disarms the gate. */
  binding: WGSLKernelBinding;
  /** Returns the `condition` and `extraResources` that gate a node on triple `slot`. */
  getGate: (slot: number) => {
    condition: GPUCommandGraphNodeCondition<Parameters>;
    extraResources: GraphBufferUse[];
  };
  /** WGSL statements that write every triple with its bounded dispatch layout. */
  armWGSL: string;
  /** WGSL statements that set the x dimension of every triple to zero. */
  disarmWGSL: string;
};

/**
 * Creates GPU-written indirect dispatch triples that gate unrolled loop kernels on a GPU state
 * word (library rule 13): once the one-thread node that owns the state word runs `disarmWGSL`,
 * every later round launches nothing. Slot `i` holds the bounded layout of `shapes[i]`, the same
 * layout {@link createWGSLKernelNode} computes, so the gated kernel's linear index stays valid.
 * An indirect condition also supplies the gated node's dispatch dimensions, so each slot must
 * match the invocation count and workgroup size of the nodes that use it.
 *
 * The node that writes `binding` must itself be ungated: one buffer cannot be both an indirect
 * argument and storage-written in one dispatch.
 *
 * @param shapes One `[invocationCount, workgroupSize]` per slot; the workgroup size defaults to
 *   the automatic size for the invocation count.
 */
export function createWGSLDispatchGate<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    shapes: readonly (readonly [invocationCount: number, workgroupSize?: number])[];
  }
): WGSLDispatchGate<Parameters> {
  const {id} = props;
  const maximumWorkgroupsPerDimension = graph.device.limits.maxComputeWorkgroupsPerDimension;
  const layouts = props.shapes.map(([invocationCount, workgroupSize]) =>
    getBoundedDispatchLayout(
      props.operation,
      Math.max(invocationCount, 1),
      workgroupSize ?? getAutomaticWGSLKernelWorkgroupSize(invocationCount),
      maximumWorkgroupsPerDimension
    )
  );
  const view = createTransientView(
    graph,
    id,
    'uint32',
    3 * layouts.length,
    Buffer.STORAGE | Buffer.INDIRECT
  );
  return {
    view,
    binding: {name: 'dispatch', view, type: 'u32', access: 'read_write'},
    getGate: slot => ({
      condition: {
        id: `${id}-${slot}`,
        source: 'gpu',
        mode: 'indirect',
        buffer: view.buffer,
        byteOffset: view.byteOffset + slot * 12
      },
      extraResources: [{buffer: view, usage: 'indirect'}]
    }),
    armWGSL: layouts
      .map(
        (layout, slot) => `dispatch[dispatchOffset + ${3 * slot}u] = ${layout.x}u;
  dispatch[dispatchOffset + ${3 * slot + 1}u] = ${layout.y}u;
  dispatch[dispatchOffset + ${3 * slot + 2}u] = ${layout.z}u;`
      )
      .join('\n  '),
    disarmWGSL: layouts.map((_, slot) => `dispatch[dispatchOffset + ${3 * slot}u] = 0u;`).join(' ')
  };
}

const MINIMUM_AUTOMATIC_WORKGROUP_SIZE = 32;

/** WGSL element type of one storage binding generated by {@link createWGSLKernelNode}. */
export type WGSLKernelElementType = 'u32' | 'i32' | 'f32' | 'atomic<u32>' | 'atomic<i32>';

/** One storage binding of a generated WGSL kernel. */
export type WGSLKernelBinding = {
  /**
   * WGSL variable name. The generated `${name}Offset` constant holds the view's element offset
   * from its 256-byte-aligned binding, so it is part of the WGSL source: per-iteration slices of
   * one buffer must start at 256-byte strides to share a module (rule 12).
   */
  name: string;
  /** Bound packed view. */
  view: GraphDataView;
  /** WGSL array element type. */
  type: WGSLKernelElementType;
  /** Storage access mode. */
  access: 'read' | 'read_write';
};

/** Texel sample type of one sampled-texture binding of a generated WGSL kernel. */
export type WGSLKernelTextureSampleType = 'float' | 'unfilterable-float' | 'uint' | 'sint';

/** One sampled-texture binding of a generated WGSL kernel. */
export type WGSLKernelTextureBinding = {
  /** WGSL variable name of the `texture_*<T>` declaration, read with `textureLoad`. */
  name: string;
  /** Bound graph texture view. Its dimension (`1d`, `2d`, `2d-array` or `3d`) selects the WGSL type. */
  texture: GraphTextureView;
  /**
   * Texel sample type. `'float'` and `'unfilterable-float'` declare `texture_*<f32>`, `'uint'`
   * declares `<u32>` and `'sint'` declares `<i32>`. Use `'unfilterable-float'` for formats that
   * cannot be filtered, such as `r32float` without the filtering feature.
   */
  sampleType: WGSLKernelTextureSampleType;
};

/** Properties for one linear WGSL compute kernel. */
export type WGSLKernelProps<Parameters> = {
  /** Graph-wide node ID. */
  id: string;
  /** Operation name reported in the workload estimate, such as `GPUPointDensity`. */
  operation: string;
  /** Optional implementation variant reported in the workload estimate. */
  variant?: string;
  /**
   * Storage bindings in `@binding` order. At most the device's `maxStorageBuffersPerShaderStage`
   * (8 by WebGPU default).
   */
  bindings: readonly WGSLKernelBinding[];
  /**
   * Sampled-texture bindings, declared after the storage bindings at the following `@binding`
   * locations (`bindings.length + i`). They are resolved with the encode context's
   * `getTextureView`, declared as `'sampled'` node resources, and supplied with the storage
   * bindings on every dispatch. At most `maxSampledTexturesPerShaderStage`.
   */
  textures?: readonly WGSLKernelTextureBinding[];
  /** Number of linear invocations. Each invocation sees `index` in `[0, invocationCount)`. */
  invocationCount: number;
  /**
   * Power-of-two workgroup size. Guarded kernels choose 32, 64, 128, or 256 from the invocation
   * count. Cooperative kernels (`guardIndex: false`) retain 256 unless this is specified.
   */
  workgroupSize?: number;
  /** Module-scope WGSL declarations such as constants and helper functions. */
  declarations?: string;
  /**
   * Values for `override` declarations in {@link declarations}. Every named override must be
   * statically used by the kernel body. Moving size-like values here instead of baking them into
   * `const` declarations lets kernels that differ only in those values share one shader module.
   */
  constants?: Readonly<Record<string, number>>;
  /** WGSL statements executed once per in-range `index`. */
  body: string;
  /**
   * When false, `body` runs for every dispatched invocation and must check
   * `index < INVOCATION_COUNT` itself. Use it for kernels with workgroup barriers. Default true.
   */
  guardIndex?: boolean;
  /** Optional node condition, for example a GPU-written indirect dispatch gate. */
  condition?: GPUCommandGraphNodeCondition<Parameters>;
  /** Additional declared resource uses, such as the `'indirect'` use of a condition buffer. */
  extraResources?: readonly GraphBufferUse[];
  /**
   * Published shader modules from `@luma.gl/shaders`, such as `fp64`. When the list is non-empty
   * the generated WGSL is assembled with the published `WGSLShaderAssembler` for
   * `getPlatformInfo(graph.device)`: the modules (with their dependencies) are emitted before the
   * kernel, and the application bindings keep their locations. Omitted or empty, nothing is
   * assembled and the kernel is exactly the generated WGSL. See {@link createWGSLKernelNode} for
   * what is accepted.
   */
  modules?: readonly ShaderModule<any, any, any>[];
  /**
   * Preprocessor defines for the module assembly; they win over the platform and module defines.
   * Needs `modules`.
   */
  defines?: Readonly<Record<string, boolean | number>>;
};

/**
 * Builds one compute node from bindings and a WGSL body, following the `GPUMask` pattern.
 *
 * Bindings use `getViewBinding` and expose `${name}Offset` WGSL constants. Every declared binding
 * is kept in the pipeline layout even when `body` never reads it, so callers may declare bindings
 * conditionally used by generated WGSL. Dispatch uses a bounded 3D layout so large inputs never
 * exceed `maxComputeWorkgroupsPerDimension`.
 *
 * With `modules`, the WGSL is assembled before the kernel is built, and the node record holds the
 * assembled text. The node binds only its own storage and texture bindings, so it accepts only
 * modules that need nothing bound, and it checks that at construction:
 * - Application bindings must stay below `@binding(100)`; the assembler gives group-0 locations
 *   from 100 up to module bindings.
 * - With `fp64arithmetic` in the module set (directly or through `fp64`), the node always
 *   defines `LUMA_FP64_INTEGER_ARITHMETIC`, on every platform, and rejects `defines` that turn it
 *   off. The integer arithmetic functions do not read the module's `fp64arithmetic` uniform,
 *   so WebGPU's `layout: 'auto'` leaves it out when no caller code references it. The
 *   published assembler itself chooses the integer variant only for Apple GPUs; elsewhere it
 *   chooses the classic variant without the code-elimination barrier, which measured at float32
 *   accuracy (`dev-docs/research/extended-precision-2026-11-11`), and the per-vendor helpers
 *   that choose classic safely are internal to `@luma.gl/graph`. A classic path also has to bind
 *   the uniform; until one exists, the node masks WGSL comments in `body`, `declarations` and
 *   every module source except the published `fp64` and `fp64arithmetic` objects, then rejects
 *   any whole identifier naming the uniform or its reader helpers (`fp64_runtime_zero_bits`,
 *   `fp64_runtime_zero`, and `prevent_fp64_optimization`, which reads it when the code-elimination
 *   workaround is enabled).
 * - Any other module binding (a uniform block, a texture, a buffer) is rejected.
 * Section 2 of `dev-docs/research/extended-precision-2026-11-11/README.md` measured the loop probe
 * on the test GPU (Apple Metal, headless Chromium), including upload and readback: classic with
 * the workaround 145 ms, integer 380 ms, and local double-float 132 ms. Separately, luma.gl's
 * `node_modules/@luma.gl/graph/dist/geospatial/geospatial-utils.js` says D3D12 inlining makes
 * integer shaders too large to compile in practical time; its platform selector uses classic
 * on D3D11, D3D12 and Vulkan.
 */
export function createWGSLKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  const workgroupSize =
    props.workgroupSize ??
    (props.guardIndex === false
      ? DEFAULT_WORKGROUP_SIZE
      : getAutomaticWGSLKernelWorkgroupSize(props.invocationCount));
  // Fail at declaration, naming the kernel, instead of at pipeline creation. WebGPU's default is
  // 8 storage buffers per stage; devices created with higher limits accept more.
  const storageBufferLimit = graph.device.limits.maxStorageBuffersPerShaderStage;
  if (props.bindings.length > storageBufferLimit) {
    throw new Error(
      `${props.operation}: kernel ${props.id} binds ${props.bindings.length} storage buffers, ` +
        `more than the device limit of ${storageBufferLimit}`
    );
  }
  // Same for one binding above maxStorageBufferBindingSize (128 MiB by default): WebGPU would
  // otherwise reject the bind group at encode time, naming neither the kernel nor the view.
  const bindingByteLimit = graph.device.limits.maxStorageBufferBindingSize;
  for (const binding of props.bindings) {
    const bindingByteLength = getViewBindingRange(binding.view).size;
    if (bindingByteLength > bindingByteLimit) {
      throw new Error(
        `${props.operation}: kernel ${props.id} binding ${binding.name} needs ` +
          `${bindingByteLength} bytes, above the device limit of ${bindingByteLimit}`
      );
    }
  }
  const textures = props.textures ?? [];
  const sampledTextureLimit = graph.device.limits.maxSampledTexturesPerShaderStage;
  if (textures.length > sampledTextureLimit) {
    throw new Error(
      `${props.operation}: kernel ${props.id} binds ${textures.length} sampled textures, ` +
        `more than the device limit of ${sampledTextureLimit}`
    );
  }
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const storageDeclarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const textureDeclarations = textures.map(
    (binding, textureIndex) =>
      `@group(0) @binding(${props.bindings.length + textureIndex}) var ${binding.name}: ${getWGSLTextureType(binding)};`
  );
  const declarations = [storageDeclarations, ...textureDeclarations].join('\n');
  // WebGPU `layout: 'auto'` only lists bindings the entry point statically uses, and a bind group
  // with an extra entry fails validation. A phony assignment of each binding's address is a
  // static use with no memory access, so every declared binding stays in the layout.
  // Handle-space texture variables have no address, so they are phony-assigned by value.
  const bindingReferences = [
    ...props.bindings.map(binding => `_ = &${binding.name};`),
    ...textures.map(binding => `_ = ${binding.name};`)
  ].join('\n  ');
  // The invocation count and the bounded dispatch shape are pipeline-overridable constants, so the
  // WGSL source does not depend on the problem size and same-shaped kernels share one shader
  // module. The phony use keeps INVOCATION_COUNT statically used by unguarded kernels.
  const maximumLinearWorkgroupCount = Math.floor(0xffffffff / workgroupSize) + 1;
  const source = /* wgsl */ `
override INVOCATION_COUNT: u32;
override WGSL_KERNEL_DISPATCH_X: u32;
override WGSL_KERNEL_DISPATCH_Y: u32;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${bindingReferences}
  _ = INVOCATION_COUNT;
  let workgroupIndex =
    (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) * WGSL_KERNEL_DISPATCH_X +
    workgroupId.x;
  if (workgroupIndex >= ${maximumLinearWorkgroupCount}u) { return; }
  let index = workgroupIndex * ${workgroupSize}u + localInvocationIndex;
  ${props.guardIndex === false ? '' : 'if (index >= INVOCATION_COUNT) {\n    return;\n  }'}
  ${props.body}
}`;
  const kernelSource = props.modules?.length
    ? assembleWGSLKernelSource(graph, props, source)
    : source;
  if (props.defines && !props.modules?.length) {
    throw new Error(`${props.id} defines need modules: without modules nothing is assembled`);
  }
  const resources: GraphResourceUse[] = props.bindings.map(binding => ({
    buffer: binding.view,
    usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
  }));
  resources.push(
    ...textures.map(binding => ({texture: binding.texture, usage: 'sampled' as const}))
  );
  resources.push(...(props.extraResources ?? []));
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') {
      readByteLength += byteLength;
    } else {
      writeByteLength += byteLength;
    }
  }
  const node = createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources,
    condition: props.condition,
    workload: {
      operation: props.operation,
      // GPUCommandGraph rejects a present-but-undefined variant.
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: props.invocationCount,
      readByteLength,
      writeByteLength
    },
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source: kernelSource,
        constants: {
          ...props.constants,
          INVOCATION_COUNT: props.invocationCount,
          WGSL_KERNEL_DISPATCH_X: layout.x,
          WGSL_KERNEL_DISPATCH_Y: layout.y
        },
        shaderLayout: {
          bindings: [
            ...props.bindings.map(
              (binding, location): BindingDeclaration => ({
                name: binding.name,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
                group: 0,
                location
              })
            ),
            ...textures.map(
              (binding, textureIndex): BindingDeclaration => ({
                name: binding.name,
                type: 'texture' as const,
                group: 0,
                location: props.bindings.length + textureIndex,
                viewDimension: binding.texture.dimension as '1d' | '2d' | '2d-array' | '3d',
                sampleType: binding.sampleType
              })
            )
          ]
        }
      });
      // The binding record is rebuilt only when a resolved buffer or texture view changes. That
      // saves the JavaScript allocation of the `getViewBinding` records on every encode, nothing
      // more: `Kernel.dispatch` passes the bindings to `ComputePass.setBindings`, which asks the
      // bind-group factory without cache keys, so WebGPU creates a new bind group on every
      // dispatch whatever the identity of these objects (luma.gl 10.0.0-alpha.13).
      let resolvedBuffers: Buffer[] = [];
      let resolvedTextureViews: TextureView[] = [];
      let resolvedBindings: Record<string, Binding> = {};
      return {
        encode: ({computePass, getBuffer, getTextureView}) => {
          const nextBuffers = props.bindings.map(binding => getBuffer(binding.view));
          const nextTextureViews = textures.map(binding => getTextureView(binding.texture));
          const bindingsChanged =
            nextBuffers.some((buffer, index) => buffer !== resolvedBuffers[index]) ||
            nextTextureViews.some((view, index) => view !== resolvedTextureViews[index]);
          if (
            bindingsChanged ||
            nextBuffers.length !== resolvedBuffers.length ||
            nextTextureViews.length !== resolvedTextureViews.length
          ) {
            resolvedBuffers = nextBuffers;
            resolvedTextureViews = nextTextureViews;
            resolvedBindings = {};
            for (const [index, binding] of props.bindings.entries()) {
              const buffer = nextBuffers[index]!;
              resolvedBindings[binding.name] = getViewBinding(binding.view, () => buffer);
            }
            for (const [index, binding] of textures.entries()) {
              resolvedBindings[binding.name] = nextTextureViews[index]!;
            }
          }
          kernel.dispatch(computePass, {
            bindings: resolvedBindings,
            x: layout.x,
            y: layout.y,
            z: layout.z
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });
  // One WeakMap lookup when the graph records nothing. When it does, the entry references only
  // values the node's compile and encode closures already hold.
  WGSL_KERNEL_NODE_RECORDS.get(graph)?.set(props.id, {
    props,
    source: kernelSource,
    workgroupSize,
    layout
  });
  return node;
}

/** Name of the published double-single arithmetic module and of its uniform block. */
const FP64_ARITHMETIC_MODULE_NAME = 'fp64arithmetic';
/** Define that selects the integer-controlled double-single variant of `fp64arithmetic`. */
const FP64_INTEGER_ARITHMETIC_DEFINE = 'LUMA_FP64_INTEGER_ARITHMETIC';
/** First group-0 binding location the published assembler gives to module bindings. */
const FIRST_MODULE_BINDING_LOCATION = 100;
/** Whole identifiers for the uniform and helpers that can read it under integer arithmetic. */
const FP64_UNIFORM_READER_PATTERN =
  /\b(?:fp64arithmetic|fp64_runtime_zero(?:_bits)?|prevent_fp64_optimization)\b/;

/** One assembler for every kernel, so module binding locations stay stable between kernels. */
let wgslKernelAssembler: WGSLShaderAssembler | undefined;

/**
 * Assembles the generated WGSL of {@link createWGSLKernelNode} with its shader modules, and
 * rejects module sets whose bindings the node would have to bind.
 */
function assembleWGSLKernelSource<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>,
  source: string
): string {
  const modules = [...(props.modules ?? [])];
  validateMaximumCount(
    props.id,
    'storage and sampled-texture bindings with shader modules',
    props.bindings.length + (props.textures?.length ?? 0),
    FIRST_MODULE_BINDING_LOCATION
  );
  const resolvedModules = getShaderModuleDependencies(modules);
  const usesFP64Arithmetic = resolvedModules.some(
    module => module.name === FP64_ARITHMETIC_MODULE_NAME
  );
  const defines: Record<string, boolean | number> = {...props.defines};
  if (usesFP64Arithmetic) {
    const integerDefine = defines[FP64_INTEGER_ARITHMETIC_DEFINE];
    validateRequirement(
      props.id,
      integerDefine === undefined || Boolean(integerDefine),
      `defines must leave ${FP64_INTEGER_ARITHMETIC_DEFINE} on with module ` +
        `${FP64_ARITHMETIC_MODULE_NAME}: the classic variant needs its uniform bound, which ` +
        'createWGSLKernelNode does not bind'
    );
    defines[FP64_INTEGER_ARITHMETIC_DEFINE] = true;
    const readerSources = [
      `${props.declarations ?? ''}\n${props.body}`,
      ...resolvedModules
        .filter(module => module !== fp64 && module !== fp64arithmetic)
        .map(module => module.source ?? '')
    ];
    validateRequirement(
      props.id,
      !readerSources.some(readerSource =>
        FP64_UNIFORM_READER_PATTERN.test(maskWGSLComments(readerSource))
      ),
      `body, declarations and modules must not read the ${FP64_ARITHMETIC_MODULE_NAME} uniform ` +
        '(fp64arithmetic, fp64_runtime_zero_bits, fp64_runtime_zero, ' +
        'prevent_fp64_optimization): createWGSLKernelNode ' +
        'does not bind it'
    );
  }
  wgslKernelAssembler ??= new WGSLShaderAssembler();
  const assembly = wgslKernelAssembler.assembleWGSLShader({
    platformInfo: getPlatformInfo(graph.device),
    source,
    modules,
    defines,
    shaderStage: 'compute',
    computeEntryPoint: 'main',
    scanVertexAttributes: false
  });
  for (const assignment of assembly.bindingAssignments) {
    validateRequirement(
      props.id,
      usesFP64Arithmetic &&
        assignment.moduleName === FP64_ARITHMETIC_MODULE_NAME &&
        assignment.name === FP64_ARITHMETIC_MODULE_NAME,
      `module ${assignment.moduleName} binding ${assignment.name} needs a bound resource, which ` +
        'createWGSLKernelNode does not bind'
    );
  }
  return assembly.source;
}

/** Kernel facts of one node built by {@link createWGSLKernelNode}, for computation records. */
export type WGSLKernelNodeRecord = {
  /** Graph-wide node ID. */
  id: string;
  /** Operation (contributor or operator) name from the workload estimate. */
  operation: string;
  /** Implementation variant from the workload estimate, when set. */
  variant?: string;
  /**
   * Full WGSL module text exactly as passed to the kernel: the generated text, assembled with its
   * shader modules when the kernel has any.
   */
  source: string;
  /** Author-written body (`WGSLKernelProps.body`), shared by differently bound instances. */
  body: string;
  /** Author-written module-scope declarations, when set. */
  declarations?: string;
  /** Compute entry point. */
  entryPoint: 'main';
  /** Lanes per workgroup. */
  workgroupSize: readonly [number, number, number];
  /** Bounded static dispatch; the upper bound when `indirect` is true. */
  workgroups: readonly [number, number, number];
  /** Linear invocations the kernel guards to. */
  invocationCount: number;
  /** Pipeline override constants, including the generated invocation and dispatch overrides. */
  constants: Readonly<Record<string, number>>;
  /** Storage bindings in `@binding` order. */
  bindings: readonly WGSLKernelBinding[];
  /** Sampled-texture bindings, after the storage bindings. */
  textures: readonly WGSLKernelTextureBinding[];
  /** Whether a GPU-written indirect condition supplies the dispatch. */
  indirect: boolean;
};

type WGSLKernelNodeRecordEntry = {
  props: WGSLKernelProps<any>;
  source: string;
  workgroupSize: number;
  layout: {x: number; y: number; z: number};
};

/**
 * Graphs whose kernel facts are kept, keyed weakly by graph. The compiled graph publishes node ids,
 * not node objects, and `graph.add` copies nodes, so entries are keyed by node id within a graph.
 */
const WGSL_KERNEL_NODE_RECORDS = new WeakMap<object, Map<string, WGSLKernelNodeRecordEntry>>();

/**
 * Keeps the kernel facts (WGSL source, entry point, workgroup size, dispatch, bindings with their
 * access, operation name) of every node that {@link createWGSLKernelNode} creates on `graph` from
 * now on, for {@link getWGSLKernelNodeRecord} and `describeGPUCommandGraph`. Call it right after
 * creating a graph that will be described, before any contributor adds nodes. Graphs that never
 * call it keep nothing. The table is held weakly by the graph and retains no string or binding that
 * the graph's nodes do not already hold.
 */
export function recordWGSLKernelNodes(graph: GPUCommandGraph<any>): void {
  if (!WGSL_KERNEL_NODE_RECORDS.has(graph)) WGSL_KERNEL_NODE_RECORDS.set(graph, new Map());
}

/** Whether {@link recordWGSLKernelNodes} was called for `graph`. */
export function isRecordingWGSLKernelNodes(graph: GPUCommandGraph<any>): boolean {
  return WGSL_KERNEL_NODE_RECORDS.has(graph);
}

/**
 * Returns the kernel facts of node `nodeId` on a graph passed to {@link recordWGSLKernelNodes}, or
 * `undefined` when the graph records nothing or the node was not built by
 * {@link createWGSLKernelNode} (copy nodes, published `@luma.gl/graph` primitives, hand-built
 * nodes). When two kernels were created with the same id, the last one is returned.
 */
export function getWGSLKernelNodeRecord(
  graph: GPUCommandGraph<any>,
  nodeId: string
): WGSLKernelNodeRecord | undefined {
  const entry = WGSL_KERNEL_NODE_RECORDS.get(graph)?.get(nodeId);
  if (!entry) return undefined;
  const {props, source, workgroupSize, layout} = entry;
  return {
    id: props.id,
    operation: props.operation,
    ...(props.variant ? {variant: props.variant} : {}),
    source,
    body: props.body,
    ...(props.declarations ? {declarations: props.declarations} : {}),
    entryPoint: 'main',
    workgroupSize: [workgroupSize, 1, 1],
    workgroups: [layout.x, layout.y, layout.z],
    invocationCount: props.invocationCount,
    constants: {
      ...props.constants,
      INVOCATION_COUNT: props.invocationCount,
      WGSL_KERNEL_DISPATCH_X: layout.x,
      WGSL_KERNEL_DISPATCH_Y: layout.y
    },
    bindings: props.bindings,
    textures: props.textures ?? [],
    indirect: props.condition?.source === 'gpu' && props.condition.mode === 'indirect'
  };
}

/** Returns the WGSL sampled-texture type of one texture binding. */
function getWGSLTextureType(binding: WGSLKernelTextureBinding): string {
  const texelType = {float: 'f32', 'unfilterable-float': 'f32', uint: 'u32', sint: 'i32'}[
    binding.sampleType
  ];
  const dimension = {'1d': '1d', '2d': '2d', '2d-array': '2d_array', '3d': '3d'}[
    binding.texture.dimension as string
  ];
  if (!dimension) {
    throw new Error(`WGSL kernel texture ${binding.name} must be a 1d, 2d, 2d-array or 3d view`);
  }
  return `texture_${dimension}<${texelType}>`;
}

/** Selects a warp-friendly workgroup without reserving 256 lanes for small guarded kernels. */
export function getAutomaticWGSLKernelWorkgroupSize(invocationCount: number): number {
  if (!Number.isSafeInteger(invocationCount) || invocationCount < 0) {
    throw new Error('WGSL kernel invocation count must be a non-negative safe integer');
  }
  const boundedCount = Math.max(invocationCount, MINIMUM_AUTOMATIC_WORKGROUP_SIZE);
  return Math.min(DEFAULT_WORKGROUP_SIZE, 2 ** Math.ceil(Math.log2(boundedCount)));
}

/**
 * Builds one storage binding record with the WGSL element type taken from the view's format
 * (`uint32*` gives `u32`, `sint32*` gives `i32`, everything else `f32`).
 *
 * @param name WGSL variable name.
 * @param view Bound packed view.
 * @param access Storage access mode; `'atomic'` declares a read-write `atomic<u32>` or
 * `atomic<i32>` array and needs an integer view.
 */
export function getWGSLKernelBinding(
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write' | 'atomic' = 'read'
): WGSLKernelBinding {
  const scalarType = view.format.startsWith('uint32')
    ? 'u32'
    : view.format.startsWith('sint32')
      ? 'i32'
      : 'f32';
  if (access !== 'atomic') {
    return {name, view, type: scalarType, access};
  }
  if (scalarType === 'f32') {
    throw new Error(`WGSL kernel binding ${name} must be an integer view to be atomic`);
  }
  return {name, view, type: `atomic<${scalarType}>`, access: 'read_write'};
}

/** Formats a JS number as a WGSL `f32` literal. */
export function getWGSLFloatLiteral(value: number): string {
  const float32Value = Math.fround(value);
  if (!Number.isFinite(float32Value)) {
    throw new Error('WGSL float literals must be finite');
  }
  if (Object.is(float32Value, -0)) {
    return '-0.0';
  }
  const text = String(float32Value);
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * Creates graph-owned `uint32` scratch rows with the same chunk topology as `template`.
 *
 * Unlike `createTransientVectorView`, the template may have any fixed-width format.
 *
 * @internal
 */
export function createTransientUint32Rows<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: GraphDataView | GraphVectorView
): GraphDataView<'uint32'> | GraphVectorView<'uint32'> {
  if (!(template instanceof GraphVectorView)) {
    return createTransientView(graph, id, 'uint32', template.length);
  }
  let emptyChunk: GraphDataView<'uint32'> | undefined;
  const data = template.data.map((chunk, chunkIndex) => {
    if (chunk.length === 0) {
      emptyChunk ??= createTransientView(graph, `${id}-empty`, 'uint32', 0);
      return emptyChunk;
    }
    return createTransientView(graph, `${id}-chunk-${chunkIndex}`, 'uint32', chunk.length);
  });
  return new GraphVectorView({
    id,
    name: id,
    format: 'uint32',
    length: template.length,
    valueLength: template.length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
}

/** Properties for {@link createPublishNode}. */
export type PublishNodeProps = {
  /** Graph-wide node ID. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** One-row unclamped accepted count written by an earlier compaction. */
  requiredCount: GraphDataView<'uint32'>;
  /**
   * Full-size compacted IDs to copy into `output.ids`. Omit when the compaction already wrote
   * `output.ids` directly.
   */
  compactIds?: GraphDataView<'uint32'>;
  /** Caller-owned bounded output. */
  output: GPUCompactOutput;
  /** Optional one-row flags ORed into `output.status.overflow`, for example index overflow. */
  overflowSources?: readonly GraphDataView<'uint32'>[];
  /** Additional packed one-row destinations that receive the clamped count. */
  extraCounts?: readonly GraphDataView<'uint32'>[];
  /**
   * Optional 32-bit column aligned with `compactIds` whose bounded prefix is copied verbatim into
   * `destination`, for example per-result priorities.
   */
  extraColumn?: {source: GraphDataView; destination: GraphDataView};
};

/**
 * Clamps an unclamped compaction count to the capacity of a {@link GPUCompactOutput}.
 *
 * Writes `count = min(total, capacity)`, `overflow = total > capacity || any overflow source`,
 * optional `requiredCount` and extra count words, and copies the bounded prefix when compaction used
 * a full-size scratch buffer. Every word is rewritten on every encoding.
 */
export function createPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PublishNodeProps
): GPUCommandNode<Parameters> {
  const {output, compactIds} = props;
  const capacity = output.ids.length;
  const copyRowCount = compactIds ? Math.min(capacity, compactIds.length) : 0;
  const overflowSources = props.overflowSources ?? [];
  const extraCounts = props.extraCounts ?? [];
  const bindings: WGSLKernelBinding[] = [
    {name: 'totalIn', view: props.requiredCount, type: 'u32', access: 'read'}
  ];
  if (compactIds && copyRowCount > 0) {
    bindings.push({name: 'compactIn', view: compactIds, type: 'u32', access: 'read'});
    bindings.push({name: 'idsOut', view: output.ids, type: 'u32', access: 'read_write'});
  }
  for (const [sourceIndex, source] of overflowSources.entries()) {
    bindings.push({name: `overflowIn${sourceIndex}`, view: source, type: 'u32', access: 'read'});
  }
  bindings.push({name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'});
  bindings.push({
    name: 'overflowOut',
    view: output.status.overflow,
    type: 'u32',
    access: 'read_write'
  });
  if (output.status.requiredCount) {
    bindings.push({
      name: 'totalOut',
      view: output.status.requiredCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  const extraColumn = copyRowCount > 0 ? props.extraColumn : undefined;
  if (extraColumn) {
    bindings.push(
      {name: 'columnIn', view: extraColumn.source, type: 'u32', access: 'read'},
      {name: 'columnOut', view: extraColumn.destination, type: 'u32', access: 'read_write'}
    );
  }
  for (const [extraIndex, extra] of extraCounts.entries()) {
    bindings.push({name: `extraOut${extraIndex}`, view: extra, type: 'u32', access: 'read_write'});
  }
  // createWGSLKernelNode rejects more bindings than the device's maxStorageBuffersPerShaderStage,
  // naming the operation, the node and the limit.
  const overflowExpression = [
    'total > CAPACITY',
    ...overflowSources.map(
      (_, sourceIndex) => `overflowIn${sourceIndex}[overflowIn${sourceIndex}Offset] != 0u`
    )
  ].join(' || ');
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'publish',
    bindings,
    invocationCount: Math.max(copyRowCount, 1),
    // Capacities are overrides so publish nodes with the same binding shape share one module.
    declarations:
      copyRowCount > 0
        ? 'override CAPACITY: u32;\noverride COPY_ROW_COUNT: u32;'
        : 'override CAPACITY: u32;',
    constants:
      copyRowCount > 0 ? {CAPACITY: capacity, COPY_ROW_COUNT: copyRowCount} : {CAPACITY: capacity},
    body: `let total = totalIn[totalInOffset];
  let count = min(total, CAPACITY);
  ${
    copyRowCount > 0
      ? `if (index < COPY_ROW_COUNT && index < count) {
    idsOut[idsOutOffset + index] = compactIn[compactInOffset + index];
    ${extraColumn ? 'columnOut[columnOutOffset + index] = columnIn[columnInOffset + index];' : ''}
  }`
      : ''
  }
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, ${overflowExpression});
    ${output.status.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${extraCounts.map((_, extraIndex) => `extraOut${extraIndex}[extraOut${extraIndex}Offset] = count;`).join('\n    ')}
  }`
  });
}

/**
 * Builds one node that fills `view` with a constant `u32`, `i32`, or `f32` value.
 */
export function createFillNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    view: GraphDataView;
    type: 'u32' | 'i32' | 'f32';
    value: string;
    componentCount?: number;
  }
): GPUCommandNode<Parameters> {
  const componentCount = props.componentCount ?? props.view.length;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'fill',
    bindings: [{name: 'fillTarget', view: props.view, type: props.type, access: 'read_write'}],
    invocationCount: componentCount,
    body: `fillTarget[fillTargetOffset + index] = ${props.value};`
  });
}
