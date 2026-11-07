// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding, ComputePipelineProps, Shader} from '@luma.gl/core';
import {setComputePassBindings} from '../../utils/luma-core-compat';
import {
  Buffer,
  ComputePass,
  ComputePipeline,
  Device,
  PipelineFactory,
  ShaderFactory
} from '@luma.gl/core';

/** Properties for a lightweight immutable WebGPU compute kernel. */
export type KernelProps = Omit<ComputePipelineProps, 'shader'> & {
  /** WGSL compute shader source. Shader assembly belongs to higher-level APIs. */
  source: string;
  /** Factory used to create/reuse the compute pipeline. */
  pipelineFactory?: PipelineFactory;
  /** Factory used to create/reuse the compute shader. */
  shaderFactory?: ShaderFactory;
  /** Show shader source in browser debugging UI. */
  debugShaders?: 'never' | 'errors' | 'warnings' | 'always';
};

/** Bindings and workgroup dimensions for one direct dispatch. */
export type KernelDispatchOptions = {
  bindings?: Record<string, Binding>;
  /** @internal Stable keys for backend bind-group reuse. */
  _bindGroupCacheKeys?: Partial<Record<number, object>>;
  x: number;
  y?: number;
  z?: number;
};

/** Bindings and GPU-written dispatch record for one indirect dispatch. */
export type KernelDispatchIndirectOptions = {
  bindings?: Record<string, Binding>;
  /** @internal Stable keys for backend bind-group reuse. */
  _bindGroupCacheKeys?: Partial<Record<number, object>>;
  indirectBuffer: Buffer;
  indirectOffset?: number;
};

/**
 * Lightweight executable WebGPU compute program.
 *
 * `Kernel` deliberately does not assemble shaders modules or own ShaderInputs/uniform state.
 * Higher-level APIs may assemble WGSL before constructing a kernel. Bindings are supplied per
 * dispatch so a kernel remains reusable and does not retain mutable resource binding state.
 */
export class Kernel {
  /** Creates a kernel while allowing the backend to compile its pipeline asynchronously. */
  static async createAsync(device: Device, props: KernelProps): Promise<Kernel> {
    const ownsCompilation = !PipelineFactory.getAsyncCompilation(device);
    const asyncCompilation = ownsCompilation
      ? PipelineFactory.beginAsyncCompilation(device)
      : PipelineFactory.getAsyncCompilation(device)!;
    let kernel: Kernel;
    try {
      kernel = new Kernel(device, props);
    } finally {
      if (ownsCompilation) PipelineFactory.endAsyncCompilation(device, asyncCompilation);
    }
    try {
      if (ownsCompilation) {
        await Promise.all(asyncCompilation);
      } else {
        await kernel._pipelineInitialization;
      }
      return kernel;
    } catch (error) {
      kernel.destroy();
      throw error;
    }
  }

  readonly device: Device;
  readonly id: string;
  readonly source: string;
  readonly pipelineFactory: PipelineFactory;
  readonly shaderFactory: ShaderFactory;

  shader: Shader;
  pipeline!: ComputePipeline;

  private _pipelineInitialization?: Promise<ComputePipeline>;
  private _destroyed = false;
  private _bindingCacheEntries: BufferBindingCacheEntry[] = [];
  private _bindGroupIndices?: number[];

  constructor(device: Device, props: KernelProps) {
    if (device.type !== 'webgpu') {
      throw new Error('Kernel is only supported in WebGPU');
    }
    const {source, pipelineFactory, shaderFactory, debugShaders, ...pipelineProps} = props;

    this.device = device;
    this.id = props.id ?? 'kernel';
    this.source = source;
    this.pipelineFactory = pipelineFactory ?? PipelineFactory.getDefaultPipelineFactory(device);
    this.shaderFactory = shaderFactory ?? ShaderFactory.getDefaultShaderFactory(device);
    this.shader = this.shaderFactory.createShader({
      id: `${this.id}-compute`,
      stage: 'compute',
      source: this.source,
      debugShaders
    });

    const computePipelineProps: ComputePipelineProps = {...pipelineProps, shader: this.shader};

    const asyncCompilation = PipelineFactory.getAsyncCompilation(device);
    if (asyncCompilation) {
      const pipelineInitialization =
        this.pipelineFactory.createComputePipelineAsync(computePipelineProps);
      this._pipelineInitialization = pipelineInitialization;
      asyncCompilation.push(
        pipelineInitialization.then(pipeline => {
          if (this._destroyed) {
            // Destroyed while compiling: release the refcounted pipeline instead of leaking it.
            this.pipelineFactory.release(pipeline);
            return pipeline;
          }
          this.pipeline = pipeline;
          return pipeline;
        })
      );
    } else {
      this.pipeline = this.pipelineFactory.createComputePipeline(computePipelineProps);
    }
  }

  /** Applies per-dispatch bindings and records a direct compute dispatch. */
  dispatch(computePass: ComputePass, options: KernelDispatchOptions): void {
    const maximumWorkgroupCount = this.device.limits?.maxComputeWorkgroupsPerDimension;
    if (
      maximumWorkgroupCount &&
      (options.x > maximumWorkgroupCount ||
        (options.y ?? 1) > maximumWorkgroupCount ||
        (options.z ?? 1) > maximumWorkgroupCount)
    ) {
      // Dispatch exceeds maxComputeWorkgroupsPerDimension: plan it with getBoundedDispatchLayout()
      // from gpu-core/gpu-dispatch-utils and linearize workgroup ids in the shader.
      throw new Error('Kernel dispatch exceeds maxComputeWorkgroupsPerDimension');
    }
    this._setPipeline(computePass, options.bindings, options._bindGroupCacheKeys);
    computePass.dispatch(options.x, options.y, options.z);
  }

  /** Applies per-dispatch bindings and records a GPU-driven indirect compute dispatch. */
  dispatchIndirect(computePass: ComputePass, options: KernelDispatchIndirectOptions): void {
    this._setPipeline(computePass, options.bindings, options._bindGroupCacheKeys);
    computePass.dispatchIndirect(options.indirectBuffer, options.indirectOffset ?? 0);
  }

  destroy(): void {
    if (this._destroyed) return;
    if (this.pipeline) this.pipelineFactory.release(this.pipeline);
    this.shaderFactory.release(this.shader);
    this._destroyed = true;
    this._bindingCacheEntries = [];
  }

  private _setPipeline(
    computePass: ComputePass,
    bindings: Record<string, Binding> = {},
    bindGroupCacheKeys?: Partial<Record<number, object>>
  ): void {
    computePass.setPipeline(this.pipeline);
    bindGroupCacheKeys ||= this._getAutomaticBindGroupCacheKeys(bindings);
    setComputePassBindings(computePass, bindings, bindGroupCacheKeys);
  }

  /**
   * Derives stable bind-group cache keys when every binding is a buffer (or buffer range).
   *
   * A cache entry matches only when each named binding refers to the same live Buffer object with
   * the same offset and size. Buffers never change their GPU handle after creation, so a reallocated
   * buffer is a new object and misses the cache. Entries that reference destroyed buffers are
   * dropped. Texture, sampler and external-texture bindings are never cached automatically.
   */
  private _getAutomaticBindGroupCacheKeys(
    bindings: Record<string, Binding>
  ): Partial<Record<number, object>> | undefined {
    const bindingNames = Object.keys(bindings);
    if (bindingNames.length === 0) return undefined;
    for (const name of bindingNames) {
      const binding = bindings[name];
      const buffer = binding instanceof Buffer ? binding : (binding as {buffer?: Buffer}).buffer;
      // Only plain buffers and buffer ranges are cached; the binding may be a texture or sampler.
      if (!(buffer instanceof Buffer) || buffer.destroyed) return undefined;
      if (!(binding instanceof Buffer) && !('buffer' in binding)) return undefined;
    }

    const entries = this._bindingCacheEntries;
    for (let entryIndex = 0; entryIndex < entries.length; entryIndex++) {
      const entry = entries[entryIndex];
      if (isBufferBindingCacheEntryMatch(entry, bindingNames, bindings)) {
        if (entryIndex > 0) {
          entries.splice(entryIndex, 1);
          entries.unshift(entry);
        }
        return entry.bindGroupCacheKeys;
      }
    }

    // Drop entries that reference destroyed buffers, then bound the cache size.
    this._bindingCacheEntries = entries.filter(
      entry => !entry.bindings.some(binding => binding.buffer.destroyed)
    );
    this._bindingCacheEntries.length = Math.min(
      this._bindingCacheEntries.length,
      MAXIMUM_BINDING_CACHE_ENTRIES - 1
    );

    const cacheKey = {};
    const bindGroupCacheKeys: Partial<Record<number, object>> = {};
    this._bindGroupIndices ||= Array.from(
      new Set(this.pipeline.shaderLayout.bindings.map(binding => binding.group))
    );
    for (const group of this._bindGroupIndices) {
      bindGroupCacheKeys[group] = cacheKey;
    }
    const entry: BufferBindingCacheEntry = {
      bindGroupCacheKeys,
      bindings: bindingNames.map(name => {
        const binding = bindings[name];
        return binding instanceof Buffer
          ? {name, buffer: binding, offset: 0, size: undefined}
          : {
              name,
              buffer: (binding as {buffer: Buffer}).buffer,
              offset: (binding as {offset?: number}).offset ?? 0,
              size: (binding as {size?: number}).size
            };
      })
    };
    this._bindingCacheEntries.unshift(entry);
    return bindGroupCacheKeys;
  }
}

/** Maximum distinct buffer-binding sets remembered per kernel. */
const MAXIMUM_BINDING_CACHE_ENTRIES = 16;

type BufferBindingCacheEntry = {
  bindGroupCacheKeys: Partial<Record<number, object>>;
  bindings: {name: string; buffer: Buffer; offset: number; size: number | undefined}[];
};

function isBufferBindingCacheEntryMatch(
  entry: BufferBindingCacheEntry,
  bindingNames: string[],
  bindings: Record<string, Binding>
): boolean {
  if (entry.bindings.length !== bindingNames.length) return false;
  for (let index = 0; index < bindingNames.length; index++) {
    const cached = entry.bindings[index];
    if (cached.name !== bindingNames[index]) return false;
    const binding = bindings[cached.name];
    if (binding instanceof Buffer) {
      if (cached.buffer !== binding || cached.offset !== 0 || cached.size !== undefined) {
        return false;
      }
    } else {
      const range = binding as {buffer: Buffer; offset?: number; size?: number};
      if (
        cached.buffer !== range.buffer ||
        cached.offset !== (range.offset ?? 0) ||
        cached.size !== range.size
      ) {
        return false;
      }
    }
  }
  return true;
}
