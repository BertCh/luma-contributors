// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Device, Resource} from '@luma.gl/core';
import {GPUSpatialJoinPrepared} from '../spatial-join/gpu-spatial-join-prepared';
import {
  GPUCommandGraph,
  type GraphDataView,
  GraphVectorView,
  GraphTextureHandle,
  GraphTextureView,
  type GraphBufferHandle,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  isGraphDataView,
  isGraphVectorView,
  GPUParameterBuffer,
  validateGPUParameterValues,
  type GPUParameterFormat,
  type GPUParameterValues,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type RecipeContext = {
  preparation: boolean;
  writes: (() => void)[];
  contributors: Set<{destroy(): void}>;
  finalizers: (() => void)[];
};
const recipeContexts = new WeakMap<object, RecipeContext>();

/**
 * Prepares an analysis recipe before declaring anything on the caller's graph.
 *
 * The preparation graph uses the published graph API to validate the complete declaration,
 * including nested primitive expansion and device binding limits. It is never compiled and
 * never executes GPU commands. Each original handle maps to one preparation handle;
 * offset views and imported physical aliases retain their relationships.
 *
 * Owned preparation contributors are destroyed even when a later stage throws. Parameter writes
 * are validated without uploading and committed only after the full declaration succeeds.
 * Luma resources/devices in auxiliary metadata are opaque references, never reflected over.
 * Consumed configuration fields (including inherited getters and Proxy reads) are captured once.
 * Views, parameter buffers and prepared handles retain their original committed identities;
 * configuration records are read-only captured facades. Nested recipes share preparation.
 * This checks recipe configuration failures; it cannot inspect existing caller node IDs, frozen
 * graph state or recover from an allocation failure during the final declaration.
 * @internal
 */
export function addPreparedRecipe<Parameters, Props extends object, Result>(
  graph: GPUCommandGraph<Parameters>,
  props: Props,
  declare: (target: GPUCommandGraph<Parameters>, captured: Props) => Result
): Result {
  if (recipeContexts.has(graph)) return declare(graph, props);
  const preparation = new GPUCommandGraph<Parameters>(graph.device, {
    id: graph.id,
    autotuner: graph.autotuner
  });
  const buffers = new Map<GraphBufferHandle, GraphBufferHandle>();
  const textures = new Map<GraphTextureHandle, GraphTextureHandle>();
  const resources = new Map<RecipeResource, RecipeResource>();
  const mapper = createDetachableRecipeMapper(prepareResource);
  const snapshot = captureRecipeRecords(props, mapper.copy);
  function prepareResource(view: RecipeResource): RecipeResource {
    validateRecipeResource(graph, view);
    if (resources.has(view)) return resources.get(view)!;
    if (view instanceof GPUSpatialJoinPrepared) {
      const mirror = new GPUSpatialJoinPrepared({
        id: `${view.id}-recipe-preflight`,
        geometry: captureRecipeRecords(view.geometry, prepareResource),
        storage: captureRecipeRecords(view.storage, prepareResource),
        partitioning: view.partitioning && captureRecipeRecords(view.partitioning, prepareResource),
        leafCapacity: view.leafCapacity,
        spatialSort: view.spatialSort,
        edgeSlabs: view.edgeSlabs
      });
      resources.set(view, mirror);
      registerRecipeContributor(preparation, mirror);
      preparation.add(mirror);
      return mirror;
    }
    const prepared =
      view instanceof GPUParameterBuffer
        ? view
        : isGraphVectorView(view)
          ? new GraphVectorView({...view, data: view.data.map(copyBufferView)})
          : view instanceof GraphTextureView
            ? preparation.createTextureView(copyTexture(view.texture), view)
            : view instanceof GraphTextureHandle
              ? copyTexture(view)
              : copyBufferView(view);
    resources.set(view, prepared);
    return prepared;
  }
  function copyTexture(texture: GraphTextureHandle): GraphTextureHandle {
    let copied = textures.get(texture);
    if (!copied) {
      const descriptor = {...texture};
      copied = texture.transient
        ? preparation.createTransientTexture(descriptor)
        : texture.frameScoped
          ? preparation.importFrameTexture(descriptor)
          : preparation.importTexture(descriptor, texture.defaultTexture);
      textures.set(texture, copied);
    }
    return copied;
  }
  function copyBufferView(view: GraphDataView): GraphDataView {
    let buffer = buffers.get(view.buffer);
    if (!buffer) {
      const descriptor = {
        id: view.buffer.id,
        byteLength: view.buffer.byteLength,
        usage: view.buffer.usage
      };
      buffer = view.buffer.transient
        ? preparation.createTransientBuffer(descriptor)
        : preparation.importBuffer(descriptor, view.buffer.defaultBuffer);
      buffers.set(view.buffer, buffer);
    }
    return preparation.createDataView(buffer, {
      format: view.format,
      length: view.length,
      byteOffset: view.byteOffset,
      byteStride: view.byteStride,
      rowByteLength: view.rowByteLength
    });
  }
  const context: RecipeContext = {
    preparation: true,
    writes: [],
    contributors: new Set(),
    finalizers: []
  };
  recipeContexts.set(preparation, context);
  try {
    declare(preparation, snapshot);
    for (const finalize of context.finalizers) finalize();
  } finally {
    recipeContexts.delete(preparation);
    mapper.detach();
    for (const contributor of [...context.contributors].reverse()) contributor.destroy();
  }
  const committed: RecipeContext = {
    preparation: false,
    writes: [],
    contributors: new Set(),
    finalizers: []
  };
  recipeContexts.set(graph, committed);
  try {
    const result = declare(graph, snapshot);
    for (const finalize of committed.finalizers) finalize();
    for (const write of committed.writes) write();
    return result;
  } finally {
    recipeContexts.delete(graph);
  }
}

/** Builders defer expansion until the entire enclosing recipe has constructed. @internal */
export function isRecipeDeclaration(graph: object): boolean {
  return recipeContexts.has(graph);
}

/** Queues a contributor, glue node or published primitive in declaration order. @internal */
export function scheduleRecipeStage(graph: object, declare: () => void): void {
  const context = recipeContexts.get(graph);
  if (context) context.finalizers.push(declare);
  else declare();
}

type RecipeResource =
  | GraphDataView
  | GraphVectorView
  | GraphTextureHandle
  | GraphTextureView
  | GPUParameterBuffer
  | GPUSpatialJoinPrepared;

// Keep this closure outside addPreparedRecipe: retained metadata must not capture its shadow
// graph, resource maps or contributor ledger after preflight. Only the current mapper is held.
function createDetachableRecipeMapper(copy: (view: RecipeResource) => RecipeResource) {
  let current = copy;
  return {
    copy: (view: RecipeResource) => current(view),
    detach: () => {
      current = preserveRecipeResource;
    }
  };
}

function preserveRecipeResource(view: RecipeResource): RecipeResource {
  return view;
}

function validateRecipeResource<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  resource: RecipeResource
): void {
  if (resource instanceof GPUParameterBuffer) return;
  if (resource instanceof GPUSpatialJoinPrepared) {
    validateRequirement(
      'GPURecipePreparation',
      resource.isDeclaredIn(graph),
      'prepared must be declared in the caller graph before adding a recipe'
    );
    return;
  }
  if (isGraphDataView(resource) || isGraphVectorView(resource)) {
    validateGraphViewsBelongToGraph('GPURecipePreparation', graph, [resource]);
  } else {
    const texture = resource instanceof GraphTextureView ? resource.texture : resource;
    validateRequirement(
      'GPURecipePreparation',
      texture.graph === graph,
      'textures must belong to the target graph'
    );
  }
}

/** Tracks only newly constructed preparation contributors, never caller resources. @internal */
export function registerRecipeContributor<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUCommandNodeProducer<Parameters>
): void {
  const context = recipeContexts.get(graph);
  if (
    context?.preparation &&
    'destroy' in contributor &&
    typeof contributor.destroy === 'function'
  ) {
    context.contributors.add(contributor as GPUCommandNodeProducer<Parameters> & {destroy(): void});
  }
}

/** Validate through the public parameter writer; upload only after complete declaration. @internal */
export function writeRecipeParameters<Parameters, Format extends GPUParameterFormat>(
  graph: GPUCommandGraph<Parameters>,
  parameters: GPUParameterBuffer<Format>,
  values: GPUParameterValues<Format>
): void {
  const context = recipeContexts.get(graph);
  if (!context) {
    parameters.write(values);
    return;
  }
  validateGPUParameterValues(parameters, values);
  if (!context.preparation) context.writes.push(() => parameters.write(values));
}

/** Captures consumed properties once; unrelated fields are never evaluated. */
function captureRecipeRecords<Value>(
  value: Value,
  copyView: (view: RecipeResource) => RecipeResource,
  copies = new Map<object, unknown>()
): Value {
  if (!value || typeof value !== 'object') return value;
  if (
    // Presence checks do not evaluate unrelated configuration accessors (for example a
    // raster band's format). Only a full candidate shape reaches the public discriminator.
    (['buffer', 'format', 'length', 'byteOffset', 'byteStride', 'rowByteLength'].every(
      field => field in value
    ) &&
      isGraphDataView(value)) ||
    (['format', 'length', 'data', 'chunks'].every(field => field in value) &&
      isGraphVectorView(value)) ||
    value instanceof GraphTextureHandle ||
    value instanceof GraphTextureView ||
    value instanceof GPUParameterBuffer ||
    value instanceof GPUSpatialJoinPrepared
  ) {
    return copyView(value) as Value;
  }
  if (copies.has(value)) return copies.get(value) as Value;
  // A caller may spread a geometry fixture/owner into a port, including its backing buffers.
  // Their implementation state can contain native WebGPU accessors with receiver branding.
  // Preserve ownership and identity; only graph views above are mirrored into preparation.
  if (value instanceof Resource || value instanceof Device) return value;
  if (ArrayBuffer.isView(value)) return value;
  const target = Array.isArray(value) ? [] : Object.create(Object.getPrototypeOf(value));
  const captured = new Map<PropertyKey, unknown>();
  const descriptors = new Map<PropertyKey, PropertyDescriptor | undefined>();
  const membership = new Map<PropertyKey, boolean>();
  let keys: (string | symbol)[] | undefined;
  // The facade is independent of the source: frozen objects and JavaScript Proxies cannot
  // force later declarations to reread their properties or violate frozen-target invariants.
  const facade = new Proxy(target, {
    get(_target, key) {
      if (!captured.has(key)) captured.set(key, Reflect.get(value, key));
      return captureRecipeRecords(captured.get(key), copyView, copies);
    },
    has(_target, key) {
      if (!membership.has(key)) membership.set(key, Reflect.has(value, key));
      return membership.get(key)!;
    },
    ownKeys() {
      keys ??= Reflect.ownKeys(value);
      return keys;
    },
    getOwnPropertyDescriptor(_target, key) {
      if (Array.isArray(target) && key === 'length')
        return Reflect.getOwnPropertyDescriptor(target, key);
      if (!descriptors.has(key)) descriptors.set(key, Reflect.getOwnPropertyDescriptor(value, key));
      const descriptor = descriptors.get(key);
      return descriptor
        ? {configurable: true, enumerable: descriptor.enumerable, writable: true, value: undefined}
        : undefined;
    },
    set: () => false
  });
  copies.set(value, facade);
  return facade as Value;
}
