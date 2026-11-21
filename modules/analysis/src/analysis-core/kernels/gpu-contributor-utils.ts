// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputeCommandNode,
  createGPUCopyCommandNode,
  createGPURenderCommandNode,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUComputeCommandNode,
  type GPUCopyCommandNode,
  type GPURenderCommandNode,
  type GraphDataView,
  validatePackedView as validatePackedGraphView
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {GPUCompactOutput} from './gpu-contributor-types';
import {getCoreDefaultBuffer} from '../contracts/network-topology';
import type {GPUSpatialParameterArray} from '../contracts/parameter-schema';

import {
  AnalysisValidationError,
  type AnalysisValidationIdentity,
  validateOption,
  validateRequirement
} from './analysis-validation-error';

export {getPackedUint32Values, getPackedFloat32Values, readPackedBuffer} from './packed-readback';
export type {GPUPackedReadbackValues, GPUPackedReadbackFormat} from './packed-readback';

export {
  AnalysisValidationError,
  validateOption,
  validateRequirement
} from './analysis-validation-error';
export type {
  AnalysisValidationCode,
  AnalysisValidationIdentity,
  AnalysisValidationDetails
} from './analysis-validation-error';

/** Published graph-data shape, independent of the module condition that constructed it. */
export function isGraphDataView(view: unknown): view is GraphDataView {
  if (!view || typeof view !== 'object') return false;
  const candidate = view as GraphDataView;
  return (
    typeof candidate.format === 'string' &&
    typeof candidate.length === 'number' &&
    typeof candidate.byteOffset === 'number' &&
    typeof candidate.byteStride === 'number' &&
    typeof candidate.rowByteLength === 'number' &&
    Boolean(
      candidate.buffer &&
        typeof candidate.buffer === 'object' &&
        typeof candidate.buffer.id === 'string' &&
        typeof candidate.buffer.transient === 'boolean' &&
        candidate.buffer.graph &&
        typeof candidate.buffer.graph === 'object'
    )
  );
}

/** Published ordered-vector shape; accepts empty vectors and preserves data-view identities. */
export function isGraphVectorView<Format extends GPUVectorFormat>(
  view: GraphDataView<Format> | GraphVectorView<Format>
): view is GraphVectorView<Format>;
export function isGraphVectorView(view: unknown): view is GraphVectorView;
export function isGraphVectorView(view: unknown): view is GraphVectorView {
  if (!view || typeof view !== 'object') return false;
  const candidate = view as GraphVectorView;
  // Published graph vectors own their id and chunk arrays. GPUVector instead exposes a computed
  // chunks getter and physical GPUData, even when empty. Never walk chunks to discriminate shape.
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.name === 'string' &&
    typeof candidate.format === 'string' &&
    typeof candidate.length === 'number' &&
    Array.isArray(Object.getOwnPropertyDescriptor(view, 'data')?.value) &&
    Array.isArray(Object.getOwnPropertyDescriptor(view, 'chunks')?.value)
  );
}

/** A named constructor port: presence, accepted published view shape, and explicit formats. */
export type GPUGraphPortDescriptor = {
  readonly required: boolean;
  readonly viewType: 'data' | 'vector' | 'either';
  readonly formats: readonly GPUVectorFormat[];
};

/**
 * Checks all named ports before a caller dereferences, snapshots, or validates packed layouts.
 * Descriptor keys may be dotted paths into nested port maps. No GPU resources are changed.
 */
export function validateGraphPorts(
  id: AnalysisValidationIdentity,
  ports: Readonly<Record<string, unknown>>,
  descriptors: Readonly<Record<string, GPUGraphPortDescriptor>>
): void {
  const capturedPorts = Object.entries(descriptors).map(([path, descriptor]) => {
    let value: unknown;
    if (Object.prototype.hasOwnProperty.call(ports, path)) {
      value = ports[path];
    } else {
      value = ports;
      for (const name of path.split('.')) {
        value =
          value && typeof value === 'object' ? (value as Record<string, unknown>)[name] : undefined;
      }
    }
    return {path, descriptor, value};
  });
  // Presence has precedence over any shape/format metadata, including accessor-backed formats.
  for (const {path, descriptor, value} of capturedPorts) {
    if (descriptor.required && (value === undefined || value === null)) {
      throw new AnalysisValidationError(
        id,
        `${path} must be a ${descriptor.viewType} graph view of ${descriptor.formats.join(' or ')}; received ${value}`,
        {
          code: 'MISSING_PROP',
          path,
          received: value,
          expected: {viewType: descriptor.viewType, formats: descriptor.formats}
        }
      );
    }
  }
  for (const {path, descriptor, value} of capturedPorts) {
    validateOption(id, `${path} descriptor.viewType`, descriptor.viewType, [
      'data',
      'vector',
      'either'
    ]);
    if (value === undefined || value === null) continue;
    const expected = {viewType: descriptor.viewType, formats: descriptor.formats};
    const matchesType =
      descriptor.viewType === 'data'
        ? isGraphDataView(value)
        : descriptor.viewType === 'vector'
          ? isGraphVectorView(value)
          : isGraphDataView(value) || isGraphVectorView(value);
    if (!matchesType)
      throw new AnalysisValidationError(id, `${path} must be a ${descriptor.viewType} graph view`, {
        code: 'VIEW_FORMAT',
        path,
        received: value,
        expected
      });
    const view = value as GraphDataView | GraphVectorView;
    if (!descriptor.formats.includes(view.format))
      throw new AnalysisValidationError(
        id,
        `${path} must have format ${descriptor.formats.join(' or ')}; received ${view.format}`,
        {code: 'VIEW_FORMAT', path, received: view.format, expected}
      );
    for (const chunk of getGraphViewChunks(view)) {
      if (!isGraphDataView(chunk)) {
        throw new AnalysisValidationError(id, `${path} chunks must be graph data views`, {
          code: 'VIEW_FORMAT',
          path,
          received: chunk,
          expected
        });
      }
      if (!descriptor.formats.includes(chunk.format))
        throw new AnalysisValidationError(
          id,
          `${path} chunks must have format ${descriptor.formats.join(' or ')}; received ${chunk.format}`,
          {code: 'VIEW_FORMAT', path, received: chunk.format, expected}
        );
    }
  }
}

/** Scalar formats supported by {@link GPUParameterBuffer}. */
export type GPUParameterFormat = 'float32' | 'uint32' | 'sint32';

/** Typed values whose scalar representation matches a parameter buffer's declared format. */
export type GPUParameterValues<Format extends GPUParameterFormat> =
  GPUSpatialParameterArray<Format>;

// The intrinsic getter reads the typed-array kind across realms and ignores an own toStringTag.
const GET_PARAMETER_TYPED_ARRAY_NAME = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float32Array.prototype),
  Symbol.toStringTag
)!.get!;

/** Reads the intrinsic typed-array kind across realms, ignoring caller tags. @internal */
export function getTypedArrayName(values: unknown): string | undefined {
  return GET_PARAMETER_TYPED_ARRAY_NAME.call(values) as string | undefined;
}

/** Properties for one application-owned, per-frame parameter buffer. */
export type GPUParameterBufferProps<Format extends GPUParameterFormat> = {
  /** Buffer ID, also used as the graph resource ID by {@link GPUParameterBuffer.importToGraph}. */
  id: string;
  /** Packed scalar format of every element. */
  format: Format;
  /** Element count. Fixed for the lifetime of any compiled graph that imports the buffer. */
  length: number;
  /** Optional initial values matching `format`, with at most `length` elements. Unwritten elements are zero. */
  values?: GPUParameterValues<NoInfer<Format>>;
};

/**
 * A small application-owned storage buffer for values that change between encodings.
 *
 * Contributors read viewports, time windows, thresholds, and similar per-frame state from a packed
 * storage view instead of baking them into WGSL, so the application can call {@link write} and
 * encode the same compiled graph again without recompiling. Like `DrawCommandBuffer`, the buffer is
 * created on a device and imported into one or more graphs, which never destroy it.
 */
export class GPUParameterBuffer<Format extends GPUParameterFormat = GPUParameterFormat> {
  /** Buffer and default graph resource ID. */
  readonly id: string;
  /** Packed scalar format of every element. */
  readonly format: Format;
  /** Element count. */
  readonly length: number;
  /** Owned backing buffer with storage, copy-source, and copy-destination usage. */
  readonly buffer: Buffer;

  constructor(device: Device, props: GPUParameterBufferProps<Format>) {
    this.id = props.id;
    this.format = props.format;
    this.length = props.length;
    // Each prop is read once: the checked values are the ones allocated and written.
    const values = props.values;
    const identity = {
      id: this.id,
      contributor: 'GPUParameterBuffer',
      messagePrefix: `GPUParameterBuffer ${this.id}`
    };
    validateOption(identity, 'format', this.format, ['float32', 'uint32', 'sint32']);
    validateCapacity(identity, 'length', this.length, 0);
    if (values !== undefined) {
      validateGPUParameterValues(this, values);
    }
    this.buffer = device.createBuffer({
      id: this.id,
      byteLength: Math.max(this.length, 1) * Uint32Array.BYTES_PER_ELEMENT,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    try {
      if (values !== undefined) {
        this.write(values);
      }
    } catch (error) {
      // Preserve the upload error even if device-loss cleanup also fails.
      try {
        this.buffer.destroy();
      } catch {
        /* The original upload error is authoritative. */
      }
      throw error;
    }
  }

  /**
   * Imports the buffer into a graph and returns a packed view to pass into contributor props.
   *
   * @param graph Graph that reads the parameters.
   * @param id Graph resource ID. Defaults to the buffer ID.
   */
  importToGraph<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    id: string = this.id
  ): GraphDataView<Format> {
    return importGraphBuffer(graph, id, this.buffer, this.format, this.length);
  }

  /** Overwrites a matching-format range at an integer `firstElement` within the logical length. */
  write(values: GPUParameterValues<Format>, firstElement: number = 0): void {
    validateGPUParameterValues(this, values, firstElement);
    this.buffer.write(values, firstElement * Uint32Array.BYTES_PER_ELEMENT);
  }

  /** Destroys the owned backing buffer. Destroy compiled graphs that use it first. */
  destroy(): void {
    this.buffer.destroy();
  }
}

/** Checks a parameter upload without accessing or changing its backing buffer. @internal */
export function validateGPUParameterValues<Format extends GPUParameterFormat>(
  parameters: Pick<GPUParameterBuffer<Format>, 'id' | 'format' | 'length'>,
  values: GPUParameterValues<Format>,
  firstElement = 0
): void {
  const id = {
    id: parameters.id,
    contributor: 'GPUParameterBuffer',
    messagePrefix: `GPUParameterBuffer ${parameters.id}`
  };
  const parameterArrayName =
    parameters.format === 'float32'
      ? 'Float32Array'
      : parameters.format === 'uint32'
        ? 'Uint32Array'
        : 'Int32Array';
  validateRequirement(
    id,
    ArrayBuffer.isView(values) &&
      GET_PARAMETER_TYPED_ARRAY_NAME.call(values) === parameterArrayName,
    `values must match format ${parameters.format}`,
    {
      code: 'VIEW_FORMAT',
      path: 'values',
      received: values,
      expected: {format: parameters.format, array: parameterArrayName}
    }
  );
  validateCapacity(id, 'firstElement', firstElement, 0, parameters.length);
  validateMaximumCount(id, 'values length', values.length, parameters.length - firstElement);
}

/**
 * Imports one caller-owned buffer into a graph and returns a packed typed view over it.
 *
 * @param graph Graph that receives the imported buffer handle.
 * @param id Graph resource ID. Must be unique within the graph.
 * @param buffer Caller-owned buffer. The graph never destroys it.
 * @param format Fixed-width element format.
 * @param length Row count. Defaults to the number of whole rows the buffer can hold.
 */
export function importGraphBuffer<Format extends GPUVectorFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  buffer: Buffer,
  format: Format,
  length?: number
): GraphDataView<Format> {
  const rowByteLength = getGPUVectorFormatInfo(format).byteLength;
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {
    format,
    length: length ?? Math.floor(buffer.byteLength / rowByteLength)
  });
}

/** Returns the ordered source chunks of a data or vector view without repacking them. */
export function getGraphViewChunks<Format extends GPUVectorFormat>(
  view: GraphDataView<Format> | GraphVectorView<Format>
): readonly GraphDataView<Format>[] {
  return isGraphVectorView<Format>(view) ? view.data : [view];
}

/** Throws when any view or vector chunk belongs to a different graph than `graph`. */
export function validateGraphViewsBelongToGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  for (const view of views) {
    if (!view) {
      continue;
    }
    for (const chunk of getGraphViewChunks(view)) {
      if (chunk.buffer.graph !== graph) {
        // Contributor inputs and outputs must be created or imported on the graph passed to graph.add().
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
  }
}

/**
 * Throws when any writable view shares a graph or imported physical buffer with an input.
 *
 * The graph compiler rejects overlapping writable bindings within one node. Contributors whose
 * outputs are written by separate nodes must also validate output-versus-output separation.
 * Pass records keyed by port name for actionable errors. Array callers receive indexed port names.
 */
export function validateGraphOutputsDisjointFromInputs(
  id: string,
  outputs:
    | readonly (GraphDataView | GraphVectorView | undefined)[]
    | Readonly<Record<string, GraphDataView | GraphVectorView | undefined>>,
  inputs:
    | readonly (GraphDataView | GraphVectorView | undefined)[]
    | Readonly<Record<string, GraphDataView | GraphVectorView | undefined>>
): void {
  const getBufferIdentity = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const inputBuffers = new Map<unknown, string>();
  for (const [name, view] of Object.entries(inputs)) {
    if (!view) continue;
    const port = Array.isArray(inputs) ? `inputs[${name}]` : name;
    for (const chunk of getGraphViewChunks(view)) {
      if (!inputBuffers.has(getBufferIdentity(chunk)))
        inputBuffers.set(getBufferIdentity(chunk), port);
    }
  }
  for (const [name, view] of Object.entries(outputs)) {
    if (!view) continue;
    const port = Array.isArray(outputs) ? `outputs[${name}]` : name;
    for (const chunk of getGraphViewChunks(view)) {
      const inputPort = inputBuffers.get(getBufferIdentity(chunk));
      if (inputPort !== undefined) {
        throw new Error(
          `${id} outputs must not share buffers with inputs: ${port} conflicts with ${inputPort}`
        );
      }
    }
  }
}

/**
 * Throws unless `view` has the same row count and chunk boundaries as `template`.
 *
 * A packed template requires a packed view of equal length; a vector template requires a vector
 * with identical ordered chunk lengths.
 */
export function validateGraphViewTopology(
  id: string,
  name: string,
  template: GraphDataView | GraphVectorView,
  view: GraphDataView | GraphVectorView | undefined
): void {
  if (!view) {
    return;
  }
  const templateChunks = getGraphViewChunks(template);
  const viewChunks = getGraphViewChunks(view);
  if (
    isGraphVectorView(template) !== isGraphVectorView(view) ||
    templateChunks.length !== viewChunks.length ||
    templateChunks.some((chunk, chunkIndex) => chunk.length !== viewChunks[chunkIndex].length)
  ) {
    throw new Error(`${id} ${name} must match the source topology`);
  }
}

/**
 * Throws unless `view` is one packed, uint32-aligned view of one of `formats`. Like the published
 * `validatePackedView` from `@luma.gl/graph/gpu-core` (same arguments), but it also accepts a
 * `GraphVectorView` argument and rejects a chunked vector with a short `must be one packed view`
 * message before the published check runs.
 *
 * @param view View to check.
 * @param formats Accepted formats.
 * Pass `{id, contributor}` and a separate path for structured diagnostics. Legacy combined
 * labels stay opaque unless an explicit identity lets the helper strip its exact ID prefix.
 * @param name Prop name prefixed with the contributor ID, for the legacy message.
 */
export function validateSinglePackedView<Format extends GPUVectorFormat>(
  view: GraphDataView | GraphVectorView,
  formats: readonly Format[],
  identity: Exclude<AnalysisValidationIdentity, string>,
  path: string
): asserts view is GraphDataView<Format>;
export function validateSinglePackedView<Format extends GPUVectorFormat>(
  view: GraphDataView | GraphVectorView,
  formats: readonly Format[],
  name: string,
  identity?: AnalysisValidationIdentity
): asserts view is GraphDataView<Format>;
export function validateSinglePackedView<Format extends GPUVectorFormat>(
  view: GraphDataView | GraphVectorView,
  formats: readonly Format[],
  nameOrIdentity: AnalysisValidationIdentity,
  identityOrPath?: AnalysisValidationIdentity
): asserts view is GraphDataView<Format> {
  const separatePath = typeof nameOrIdentity !== 'string' && typeof identityOrPath === 'string';
  const identity =
    separatePath || typeof nameOrIdentity !== 'string'
      ? nameOrIdentity
      : (identityOrPath ?? nameOrIdentity);
  const identityId = typeof identity === 'string' ? identity : identity.id;
  // Legacy combined labels are opaque: guessing at spaces corrupts IDs and chunk paths.
  const path = separatePath
    ? identityOrPath
    : typeof nameOrIdentity === 'string' &&
        identityOrPath &&
        nameOrIdentity.startsWith(`${identityId} `)
      ? nameOrIdentity.slice(identityId.length + 1)
      : 'view';
  const messagePath = separatePath || identityOrPath ? `${path} ` : '';
  const fail = (
    code: 'MISSING_PROP' | 'VIEW_FORMAT',
    message: string,
    received: unknown,
    expected: unknown
  ): never => {
    throw new AnalysisValidationError(identity, `${messagePath}${message}`, {
      code,
      path,
      received,
      expected
    });
  };
  if (view === undefined || view === null)
    return fail('MISSING_PROP', `must be one packed view; received ${view}`, view, {formats});
  if (isGraphVectorView(view))
    return fail('VIEW_FORMAT', 'must be one packed view', 'vector', {viewType: 'data', formats});
  if (!isGraphDataView(view))
    return fail('VIEW_FORMAT', 'must be a graph data view', view, {viewType: 'data', formats});
  const label = separatePath || identityOrPath ? `${identityId} ${path}` : String(nameOrIdentity);
  try {
    validatePackedGraphView(view, formats, label);
  } catch {
    fail(
      'VIEW_FORMAT',
      `must be packed, uint32-aligned ${formats.join(' or ')} GPU data; received ${view.format}`,
      view.format,
      {formats, packed: true, byteAlignment: 4}
    );
  }
}

/**
 * Throws unless a capacity or count prop is an integer in `[minimum, maximum]`.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 * @param minimum Smallest accepted value. Defaults to 1.
 * @param maximum Largest accepted value. Defaults to `2^32 - 1`, the largest uint32 row count.
 */
export function validateCapacity(
  id: AnalysisValidationIdentity,
  name: string,
  value: number,
  minimum: number = 1,
  maximum: number = 0xffffffff,
  source?: string
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new AnalysisValidationError(
      id,
      `${name} must be an integer in [${minimum}, ${maximum}]`,
      {
        code: 'INTEGER_RANGE',
        path: name,
        received: value,
        expected: {minimum, maximum, ...(source ? {source} : {})}
      }
    );
  }
}

/**
 * Throws unless a count derived from props (rows of a view, cells of a grid, a product of
 * capacities) is at most `maximum`. Use it for documented maxima, such as the core-limit maxima of
 * {@link getCoreBindingElementLimit}; the message names the prop and the limit.
 *
 * @param id Contributor ID, for the message.
 * @param name What is counted, such as `'positions rows'` or `'indexGridSize cells'`.
 * @param value Count to check.
 * @param maximum Largest accepted count.
 */
export function validateMaximumCount(
  id: AnalysisValidationIdentity,
  name: string,
  value: number,
  maximum: number,
  source?: string
): void {
  if (!(value <= maximum)) {
    throw new AnalysisValidationError(id, `${name} must be at most ${maximum}`, {
      code: 'INTEGER_RANGE',
      path: name,
      received: value,
      expected: {maximum, ...(source ? {source} : {})}
    });
  }
}

/**
 * Default WebGPU `maxStorageBufferBindingSize`, 128 MiB. The documented maxima of the analysis
 * contributors are sized so that every binding fits at this limit (library rule 16), so they hold
 * on every device.
 *
 * @internal
 */
export const GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE = 134217728;

/**
 * Largest element count `n` whose binding of `(n + extraElementCount) * bytesPerElement` bytes fits
 * the device's `maxStorageBufferBindingSize`. Contributors derive their documented maxima from it at
 * declaration (in `getCommandNodes`, with `graph.device`), so a device created with larger
 * binding limits accepts larger inputs; a default device (128 MiB) gives the core values.
 *
 * @param device Device whose limit applies.
 * @param bytesPerElement Bytes of one element of the largest binding.
 * @param extraElementCount Elements the binding holds beyond `n`, such as a trailing offset.
 */
export function getStorageBindingElementLimit(
  device: Device,
  bytesPerElement: number,
  extraElementCount: number = 0
): number {
  // Element sizes and extra counts are constants of the calling contributor's layout.
  if (!(bytesPerElement > 0 && extraElementCount >= 0)) {
    throw new Error(
      `getStorageBindingElementLimit bytesPerElement must be positive and extraElementCount nonnegative, received ${bytesPerElement} and ${extraElementCount}`
    );
  }
  return (
    Math.floor(device.limits.maxStorageBufferBindingSize / bytesPerElement) - extraElementCount
  );
}

/**
 * Largest cell count (`gridSize[0] * gridSize[1]`) of a uniform grid index over points on
 * `device`: its `cellCount + 1` uint32 cell offsets must fit one storage binding, so `2^25 - 1` on
 * a default device. Kernel density, inverse distance weighting, kriging, spatial clustering,
 * neighbor search and trajectory encounters validate their index grid against it.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUGridIndexMaximumCellCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4, 1);
}

/**
 * Largest element count `n` whose binding of `(n + extraElementCount) * bytesPerElement` bytes fits
 * {@link GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE}. For example an offsets array of `n + 1` uint32
 * rows gives `getCoreBindingElementLimit(4, 1)`, `2^25 - 1`.
 *
 * @param bytesPerElement Bytes of one element of the largest binding.
 * @param extraElementCount Elements the binding holds beyond `n`, such as a trailing offset.
 * @internal
 */
export function getCoreBindingElementLimit(
  bytesPerElement: number,
  extraElementCount: number = 0
): number {
  // Element sizes and extra counts are constants of the calling contributor's layout.
  if (!(bytesPerElement > 0 && extraElementCount >= 0)) {
    throw new Error(
      `getCoreBindingElementLimit bytesPerElement must be positive and extraElementCount nonnegative, received ${bytesPerElement} and ${extraElementCount}`
    );
  }
  return Math.floor(GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE / bytesPerElement) - extraElementCount;
}

/**
 * Throws unless `extent` is `[minX, minY, maxX, maxY]` with finite entries and `max > min` on both
 * axes. A reversed or empty extent would mirror or collapse the grid.
 *
 * @param id Contributor ID, for the message.
 * @param extent Extent to check.
 * @param name Prop name, for the message. Defaults to `'extent'`.
 */
export function validateExtent(
  id: AnalysisValidationIdentity,
  extent: readonly number[],
  name: string = 'extent'
): void {
  const [minX, minY, maxX, maxY] = extent;
  if (extent.length !== 4 || !extent.every(Number.isFinite) || !(maxX > minX) || !(maxY > minY)) {
    throw new AnalysisValidationError(id, `${name} must be finite with max > min`, {
      code: 'NUMBER_RANGE',
      path: name,
      received: extent,
      expected: {length: 4, finite: true, relationship: 'max > min on both axes'}
    });
  }
}

/**
 * Throws unless `value` is a power of two in `[minimum, maximum]`.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 * @param minimum Smallest accepted value. Defaults to 1.
 * @param maximum Largest accepted value. Defaults to `2^31`, the largest uint32 power of two.
 */
export function validatePowerOfTwo(
  id: AnalysisValidationIdentity,
  name: string,
  value: number,
  minimum: number = 1,
  maximum: number = 2 ** 31
): void {
  if (
    !Number.isInteger(value) ||
    value < minimum ||
    value > maximum ||
    getNextPowerOfTwo(value) !== value
  ) {
    throw new AnalysisValidationError(
      id,
      `${name} must be a power of two in [${minimum}, ${maximum}]`,
      {
        code: 'INTEGER_RANGE',
        path: name,
        received: value,
        expected: {minimum, maximum, powerOfTwo: true}
      }
    );
  }
}

/** Validates the packed layout and minimum row counts of a {@link GPUCompactOutput}. */
export function validateCompactOutput(
  id: AnalysisValidationIdentity,
  output: GPUCompactOutput,
  path = 'output'
): void {
  const status = output?.status;
  if (!status || typeof status !== 'object') {
    throw new AnalysisValidationError(
      id,
      `${path}.status must contain count and overflow uint32 views`,
      {
        code: 'MISSING_PROP',
        path: `${path}.status`,
        received: status,
        expected: {required: ['count', 'overflow']}
      }
    );
  }
  const ids = output.ids;
  // Canonical properties may be prototype getters. Capture each exactly once before validating.
  const scalars = [
    ['count', status.count],
    ['overflow', status.overflow],
    ['candidateOverflow', status.candidateOverflow],
    ['requiredCount', status.requiredCount]
  ] as const;
  const ports = Object.fromEntries([
    [`${path}.ids`, ids],
    ...scalars.map(([name, value]) => [`${path}.status.${name}`, value])
  ]);
  validateGraphPorts(id, ports, {
    [`${path}.ids`]: {required: true, viewType: 'data', formats: ['uint32']},
    [`${path}.status.count`]: {required: true, viewType: 'data', formats: ['uint32']},
    [`${path}.status.overflow`]: {required: true, viewType: 'data', formats: ['uint32']},
    [`${path}.status.candidateOverflow`]: {required: false, viewType: 'data', formats: ['uint32']},
    [`${path}.status.requiredCount`]: {required: false, viewType: 'data', formats: ['uint32']}
  });
  const identity = typeof id === 'string' ? {id, contributor: id} : id;
  validateSinglePackedView(ids, ['uint32'], identity, `${path}.ids`);
  for (const [name, scalar] of scalars) {
    if (scalar === undefined || scalar === null) continue;
    validateSinglePackedView(scalar, ['uint32'], identity, `${path}.status.${name}`);
    if (scalar.length < 1) {
      throw new AnalysisValidationError(id, `${path}.status.${name} must contain one uint32 row`, {
        code: 'VIEW_LENGTH',
        path: `${path}.status.${name}`,
        received: scalar.length,
        expected: {minimumRows: 1}
      });
    }
  }
}

/**
 * Collects the command nodes that an `addToGraph(graph)`-style contributor schedules.
 *
 * Geospatial, raster, and projection operations in `@luma.gl/graph` schedule nodes directly
 * through `addToGraph(graph)`, while GPU Core contributors return them from
 * `getCommandNodes`. A contributor's `getCommandNodes` wraps such calls with this helper
 * so every contributor composes through `graph.add(contributor)` in declaration order. Transient resources
 * created by `addNodes` are still declared on `graph`.
 *
 * @param graph Graph passed to the contributor.
 * @param addNodes Callback that calls one or more `addToGraph(graph)` methods.
 * @returns The scheduled nodes, in order, without adding them to `graph`.
 */
export function captureGraphCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  addNodes: () => void
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const recorders = {
    addComputePass: (node: Omit<GPUComputeCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPUComputeCommandNode(node)),
    addRenderPass: (node: Omit<GPURenderCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPURenderCommandNode(node)),
    addCopyPass: (node: Omit<GPUCopyCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPUCopyCommandNode(node))
  };
  const previousDescriptors = Object.keys(recorders).map(
    name => [name, Object.getOwnPropertyDescriptor(graph, name)] as const
  );
  for (const [name, recorder] of Object.entries(recorders)) {
    Object.defineProperty(graph, name, {configurable: true, writable: true, value: recorder});
  }
  try {
    addNodes();
  } finally {
    for (const [name, descriptor] of previousDescriptors) {
      if (descriptor) {
        Object.defineProperty(graph, name, descriptor);
      } else {
        Reflect.deleteProperty(graph, name);
      }
    }
  }
  return nodes;
}

/** Smallest power of two at or above `value`, and at least one. */
export function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

/**
 * Throws when one storage binding of `byteLength` bytes exceeds the device's
 * `maxStorageBufferBindingSize` (128 MiB on default WebGPU devices). Without this check WebGPU
 * rejects the bind group at dispatch with a validation error; call it where row-proportional
 * scratch is sized, with `hint` naming the prop to change. Byte lengths must be nonnegative
 * safe integers, so NaN, infinity and malformed arithmetic cannot bypass the size guard.
 *
 * @param device Device whose limit applies.
 * @param id Contributor ID, for the message.
 * @param name Binding or scratch name, for the message.
 * @param byteLength Bytes the binding needs.
 * @param hint What to reduce, for the message.
 */
export function validateGPUStorageBindingByteLength(
  device: Device,
  id: AnalysisValidationIdentity,
  name: string,
  byteLength: number,
  hint: string
): void {
  validateRequirement(
    id,
    Number.isSafeInteger(byteLength) && byteLength >= 0,
    `${name} byteLength must be a nonnegative safe integer, received ${byteLength}`,
    {
      code: 'INTEGER_RANGE',
      path: name,
      received: byteLength,
      expected: {minimum: 0, safeInteger: true}
    }
  );
  const limit = device.limits.maxStorageBufferBindingSize;
  if (byteLength > limit) {
    throw new AnalysisValidationError(
      id,
      `${name} needs ${byteLength} bytes in one binding (limit ${limit}); ${hint}`,
      {
        code: 'INTEGER_RANGE',
        path: name,
        received: byteLength,
        expected: {maximum: limit, source: 'device.limits.maxStorageBufferBindingSize', hint}
      }
    );
  }
}

/**
 * Largest command node count one contributor may emit. Iterative contributors unroll their rounds
 * into command nodes because the published compiler has no GPU-gated continuation, so the count
 * grows with a round or batch prop and sets the compile and encode cost. The largest measured
 * configuration (sampled-pivot centrality on a 224 x 224 grid) emits about 23,000 nodes.
 */
export const GPU_MAXIMUM_COMMAND_NODE_COUNT = 32768;

/**
 * Throws unless the command node count a contributor is about to emit fits the budget. Every
 * contributor whose node count depends on a prop (rounds, batches, lanes, levels) computes the
 * count at construction, exposes it as `estimatedCommandNodeCount`, and calls this.
 *
 * @param id Contributor ID, for the message.
 * @param estimatedCommandNodeCount Number of command nodes `getCommandNodes` will return.
 * @param maximum Largest accepted count.
 */
export function validateCommandNodeBudget(
  id: string,
  estimatedCommandNodeCount: number,
  maximum: number = GPU_MAXIMUM_COMMAND_NODE_COUNT
): void {
  validateMaximumCount(id, 'estimated command node count', estimatedCommandNodeCount, maximum);
}

/** Formats diagnostic values without losing signed zero or masking the validation error. */
function formatValidationValue(value: unknown): string {
  if (Object.is(value, -0)) return '-0';
  try {
    return String(value);
  } catch {
    return '<unprintable>';
  }
}

/**
 * Throws unless `value` is a finite number inside `[minimum, maximum]` after rounding to float32,
 * which is the precision the kernel receives.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 * @param minimum Smallest accepted value.
 * @param maximum Largest accepted value.
 */
export function validateFiniteNumber(
  id: AnalysisValidationIdentity,
  name: string,
  value: number,
  minimum: number = -Infinity,
  maximum: number = Infinity
): void {
  const float32Value = typeof value === 'number' ? Math.fround(value) : NaN;
  if (!Number.isFinite(float32Value) || float32Value < minimum || float32Value > maximum) {
    throw new AnalysisValidationError(
      id,
      `${name} must be a finite number in [${minimum}, ${maximum}]; received ${formatValidationValue(value)}${
        typeof value !== 'number' || Object.is(value, float32Value)
          ? ''
          : `; encoded ${formatValidationValue(float32Value)}`
      }`,
      {
        code: 'NUMBER_RANGE',
        path: name,
        received: value,
        expected: {minimum, maximum, precision: 'float32'}
      }
    );
  }
}

/**
 * Throws unless `value` is finite and not zero after rounding to float32, which is the precision
 * the kernel receives. For a signed step such as a raster cell size that a kernel divides by.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 */
export function validateNonZeroFiniteNumber(
  id: AnalysisValidationIdentity,
  name: string,
  value: number
): void {
  const float32Value = typeof value === 'number' ? Math.fround(value) : NaN;
  if (!Number.isFinite(float32Value) || float32Value === 0) {
    throw new AnalysisValidationError(id, `${name} must be a finite number other than 0`, {
      code: 'NUMBER_RANGE',
      path: name,
      received: value,
      expected: {finite: true, nonzero: true, precision: 'float32'}
    });
  }
}

/**
 * Throws unless a view holds exactly `expectedLength` rows.
 *
 * @param id Contributor ID, for the message.
 * @param name View name, for the message.
 * @param view View to check.
 * @param expectedLength Required row count.
 */
export function validateViewLength(
  id: AnalysisValidationIdentity,
  name: string,
  view: {readonly length: number},
  expectedLength: number,
  source?: string
): void {
  if (!view) {
    throw new AnalysisValidationError(id, `${name} must be a view`, {
      code: 'MISSING_PROP',
      path: name,
      received: view,
      expected: {rows: expectedLength, ...(source ? {source} : {})}
    });
  }
  if (view.length !== expectedLength) {
    throw new AnalysisValidationError(id, `${name} must have ${expectedLength} rows`, {
      code: 'VIEW_LENGTH',
      path: name,
      received: view.length,
      expected: {rows: expectedLength, ...(source ? {source} : {})}
    });
  }
}

/** Lists scalar/vector outputs, including canonical nested status words. @internal */
export function getGraphOutputViews(output: object): GraphDataView[] {
  return Object.entries(output)
    .flatMap(([name, value]) => (name === 'status' ? Object.values(value ?? {}) : [value]))
    .filter((value): value is GraphDataView => isGraphDataView(value));
}
