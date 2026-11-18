// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GraphVectorView,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphEncoding,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {NullDevice} from '@luma.gl/core/test-utils';
import {WgslReflect} from 'wgsl_reflect';
import {getWGSLModuleUsage, getWGSLStorageBindingCount} from './wgsl-memory-usage';

/**
 * Returns a CPU-only device that GPUCommandGraph accepts, for `.node.spec.ts` wiring tests.
 *
 * It supports graph declaration and `getCommandNodes(graph)` but not compilation or encoding.
 */
export function createNullWebGPUDevice(): Device {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64
    }
  });
  return device;
}

/** WGSL of one compute kernel, as recorded by {@link getGPUCommandNodeKernelSources}. */
export type RecordedGPUKernelSource = {
  /** Kernel ID (the pipeline ID). */
  id: string;
  /** `workload.operation` of the command node that compiled the kernel. */
  operation: string;
  /** The complete WGSL module passed to the pipeline. */
  source: string;
  /** Pipeline-overridable constants passed with the module (a copy; empty when none). */
  constants: Readonly<Record<string, number>>;
  /** Shader layout passed with the module, if any. */
  shaderLayout?: unknown;
  /** Compute entry point, defaulting to `main`. */
  entryPoint: string;
  /** Source plus the exact serialized configuration used by PipelineFactory. */
  configurationKey: string;
};

/**
 * Compiles command nodes on a CPU-only WebGPU device that records the WGSL of every compute
 * pipeline instead of creating it, for `.node.spec.ts` checks of generated kernels (WGSL parsing,
 * binding counts, private and workgroup memory) at the core limits of {@link createNullWebGPUDevice}.
 * Each compiled node is destroyed after recording, including any node-owned parameter buffers.
 *
 * Use it instead of `vi.mock`: the node project runs with `isolate: false`, so a module mock of
 * `@luma.gl/engine` is not applied reliably when another spec file loaded the module first.
 */
export function getGPUCommandNodeKernelSources(
  nodes: readonly unknown[]
): RecordedGPUKernelSource[] {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const sources: RecordedGPUKernelSource[] = [];
  let operation = '';
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {
      id?: string;
      shader: {source: string};
      constants?: Record<string, number>;
      shaderLayout?: unknown;
      entryPoint?: string;
    }) => {
      sources.push({
        id: props.id ?? '',
        operation,
        source: props.shader.source,
        constants: {...props.constants},
        shaderLayout: props.shaderLayout,
        entryPoint: props.entryPoint ?? 'main',
        configurationKey: JSON.stringify([
          props.shader.source,
          props.shaderLayout,
          props.entryPoint ?? 'main',
          props.constants
        ])
      });
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  try {
    for (const node of nodes as {
      workload?: {operation?: string};
      compile?: (context: {device: Device}) => {destroy?: () => void} | undefined;
    }[]) {
      operation = node.workload?.operation ?? '';
      const compiled = node.compile?.({device});
      compiled?.destroy?.();
    }
  } finally {
    device.destroy();
  }
  return sources;
}

/** Options of {@link getGPUCommandNodePipelineKeys}. */
export type PipelineKeyOptions = {
  /** Only count kernels whose ID matches, for checks on one family (for example every round). */
  idPattern?: RegExp;
  /** @deprecated Layout is always included. Use shader-module keys to ignore configuration. */
  includeShaderLayout?: boolean;
};

/**
 * Compiles command nodes (see {@link getGPUCommandNodeKernelSources}) and returns one key per
 * distinct compute pipeline: WGSL source, shader layout, entry point and serialized constants,
 * exactly the inputs `PipelineFactory` hashes. Library rule 12: building a graph at N and N + 1 rounds (batches,
 * periods, powers, classes) must give the same `size`. Use `getGPUCommandNodeKernelSources(nodes).length`
 * for the total number of pipeline requests.
 */
export function getGPUCommandNodePipelineKeys(
  nodes: readonly unknown[],
  options: PipelineKeyOptions = {}
): Set<string> {
  const keys = new Set<string>();
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    if (options.idPattern && !options.idPattern.test(kernel.id)) {
      continue;
    }
    keys.add(kernel.configurationKey);
  }
  return keys;
}

/** Unique WGSL modules, deliberately ignoring layout, entry point and override constants. */
export function getGPUCommandNodeShaderModuleKeys(nodes: readonly unknown[]): Set<string> {
  return new Set(getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source));
}

/**
 * Returns the WGSL of every compute kernel the nodes compile, by kernel ID, after checking that
 * each module parses with `wgsl_reflect` (a parse failure names the kernel).
 * The ID lookup retains the last request. Resource checks must iterate `configurations`, which
 * preserves each distinct (ID, configuration) pair, including earlier requests with the same ID.
 */
export function getParsedKernelSources(nodes: readonly unknown[]): Map<string, string> & {
  constantsById: Map<string, Readonly<Record<string, number>>>;
  configurations: readonly RecordedGPUKernelSource[];
  getConfigurations: (id: string) => readonly RecordedGPUKernelSource[];
} {
  const configurations: RecordedGPUKernelSource[] = [];
  const sources = Object.assign(new Map<string, string>(), {
    constantsById: new Map<string, Readonly<Record<string, number>>>(),
    configurations,
    getConfigurations: (id: string) => {
      const matching = configurations.filter(kernel => kernel.id === id);
      if (!matching.length) throw new Error(`no recorded kernel configurations for ${id}`);
      return matching;
    }
  });
  const recordedConfigurations = new Set<string>();
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    try {
      new WgslReflect(kernel.source);
    } catch (error) {
      throw new Error(`kernel ${kernel.id} does not parse: ${String(error)}`);
    }
    sources.set(kernel.id, kernel.source);
    sources.constantsById.set(kernel.id, kernel.constants);
    const key = JSON.stringify([kernel.id, kernel.configurationKey]);
    if (!recordedConfigurations.has(key)) {
      recordedConfigurations.add(key);
      sources.configurations.push(kernel);
    }
  }
  return sources;
}

/** Core WebGPU limit on private (function-scope) memory per invocation, in bytes. */
export const CORE_PRIVATE_MEMORY_LIMIT_BYTES = 8192;
/** Core WebGPU `maxComputeWorkgroupStorageSize`, in bytes. */
export const CORE_WORKGROUP_STORAGE_LIMIT_BYTES = 16384;

function requireResolvedMemoryUsage(source: string, constants: Readonly<Record<string, number>>) {
  const usage = getWGSLModuleUsage(source, constants);
  const unresolved = [...usage.unresolvedLengths, ...usage.unresolvedTypes];
  if (unresolved.length) throw new Error(`unresolved shader memory: ${unresolved.join(', ')}`);
  return usage;
}

/**
 * Declared private storage plus local array/struct storage, summed across functions; scalar locals are
 * treated as registers. This estimates declared storage, not total compiler/device allocation.
 * Resolves named constants, pipeline overrides, aliases and struct padding. Throws when a memory
 * layout cannot be resolved rather than returning an unsafe bound. This is not device allocation.
 */
export function getPrivateMemoryBytes(
  source: string,
  constants: Readonly<Record<string, number>> = {}
): number {
  const usage = requireResolvedMemoryUsage(source, constants);
  return usage.privateBytes + usage.localAggregateBytes;
}

/** WebGPU workgroup storage usage, rounding each variable up to 16 bytes; refuses unresolved layouts. */
export function getWorkgroupMemoryBytes(
  source: string,
  constants: Readonly<Record<string, number>> = {}
): number {
  return requireResolvedMemoryUsage(source, constants).workgroupStorageBytes;
}

/** Number of storage bindings, including WGSL's default read access. */
export function getStorageBindingCount(source: string): number {
  return getWGSLStorageBindingCount(source);
}

/** Creates a storage input buffer initialized with `values`. */
export function createInputBuffer(
  device: Device,
  values: Float32Array | Uint32Array | Int32Array
): Buffer {
  return device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST});
}

/** Creates a zero-filled storage output buffer with `length` 32-bit elements. */
export function createOutputBuffer(device: Device, length: number, usage: number = 0): Buffer {
  return device.createBuffer({
    byteLength: Math.max(length, 1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | usage
  });
}

/** Reads the first `length` (default all) uint32 values of a submitted buffer. */
export async function readUint32(buffer: Buffer, length?: number): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(
    new Uint32Array(bytes.buffer, bytes.byteOffset, length ?? bytes.byteLength / 4)
  );
}

/** Reads the first `length` (default all) float32 values of a submitted buffer. */
export async function readFloat32(buffer: Buffer, length?: number): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(
    new Float32Array(bytes.buffer, bytes.byteOffset, length ?? bytes.byteLength / 4)
  );
}

/** Reads a compact result prefix: `count` first, then that many IDs. */
export async function readCompactIds(ids: Buffer, count: Buffer): Promise<number[]> {
  const [visibleCount] = await readUint32(count, 1);
  return readUint32(ids, visibleCount);
}

/** Numeric comparator for stable assertions on unordered GPU output. */
export function sortNumbers(left: number, right: number): number {
  return left - right;
}

/** Wraps ordered packed chunks of one format in a graph vector view. */
export function createVectorView<Format extends GPUVectorFormat>(
  id: string,
  format: Format,
  data: GraphDataView<Format>[]
): GraphVectorView<Format> {
  const formatInfo = getGPUVectorFormatInfo(format);
  const length = data.reduce((sum, chunk) => sum + chunk.length, 0);
  const components = formatInfo.byteLength / 4;
  return new GraphVectorView({
    id,
    name: id,
    format,
    length,
    valueLength: length * components,
    stride: components,
    byteStride: formatInfo.byteLength,
    rowByteLength: formatInfo.byteLength,
    data
  });
}

/** Returns whether a device is a software or fallback adapter with relaxed float behavior. */
export function isSoftwareDevice(device: Device): boolean {
  return (
    device.info.gpu === 'software' || device.info.gpuType === 'cpu' || Boolean(device.info.fallback)
  );
}

/**
 * Encodes one compiled graph into a new command encoder and submits it.
 *
 * Contributors never submit; this is a test convenience for headless specs.
 */
export function submitGraph<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  ...[parameters]: undefined extends Parameters ? [Parameters?] : [Parameters]
): GPUCommandGraphEncoding {
  const commandEncoder = device.createCommandEncoder({id: `${compiled.id}-encoder`});
  const encoding = compiled.encode(commandEncoder, {parameters: parameters as Parameters});
  device.submit(commandEncoder.finish());
  return encoding;
}

/** One compute pipeline a command-node list would create. */
export type RecordedPipeline = {
  /** Pipeline ID. */
  id: string;
  /** Complete WGSL module. */
  source: string;
  /** Override constants passed to the pipeline (empty when none). */
  constants: Record<string, number>;
  /** Shader layout passed to the pipeline. */
  shaderLayout?: unknown;
  /** Compute entry point, defaulting to `main`. */
  entryPoint: string;
  /** Source plus serialized layout, entry point and constants, as the pipeline factory hashes. */
  key: string;
};

/** Compiles `nodes` on a recording device and returns each pipeline's source and constants. */
export function recordComputePipelines(nodes: readonly unknown[]): RecordedPipeline[] {
  return getGPUCommandNodeKernelSources(nodes).map(kernel => ({
    id: kernel.id,
    source: kernel.source,
    constants: {...kernel.constants},
    shaderLayout: kernel.shaderLayout,
    entryPoint: kernel.entryPoint,
    key: kernel.configurationKey
  }));
}

/** Number of distinct source/layout/entry-point/constants configurations requested. */
export function countDistinctPipelines(nodes: readonly unknown[]): number {
  return new Set(recordComputePipelines(nodes).map(pipeline => pipeline.key)).size;
}
