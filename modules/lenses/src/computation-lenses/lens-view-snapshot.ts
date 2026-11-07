// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GraphDataView,
  type CompiledGPUCommandGraph,
  type GraphBufferDescriptor,
  type GPUCommandGraphEncodeOptions,
  type GPUCommandGraphEncoding,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {
  GPUSnapshotArena,
  GPUSnapshotTap,
  type GPUComputationCapture,
  type GPUComputationCaptureElementType
} from '@luma-contributors/analysis/computation-record';

/**
 * Observer effect: every lens graph differs from its production graph. Each transient gains
 * `COPY_SRC` and the graph ends in one snapshot copy node, which ends compute-pass coalescing, so
 * node counts, pass grouping and buffer reuse in lens records describe the observation twin, not
 * the production graph. Records state this in `metrics[].encoding` (`coalesced: false`, `taps`,
 * `lensTwin: true`). Tests must not assert node counts or pass indices as production facts.
 */

/**
 * Lens observation graph: every transient also declares `COPY_SRC`, so a snapshot tap may copy
 * any intermediate. Observation-only allocation choice; production graphs keep their declarations.
 */
export class LensCommandGraph<Parameters = void> extends GPUCommandGraph<Parameters> {
  override createTransientBuffer(descriptor: GraphBufferDescriptor): GraphBufferHandle {
    return super.createTransientBuffer({...descriptor, usage: descriptor.usage | Buffer.COPY_SRC});
  }
}

/** Named packed views or whole buffers a lens snapshots at the end of its graph. */
export type LensSnapshotViews = Readonly<Record<string, GraphDataView | GraphBufferHandle>>;

/** Pending end-of-run snapshot: an arena plus the views it copies. */
export type LensViewSnapshot = {
  /** Arena the tap node copies into; read once with {@link readLensViewSnapshot}. */
  arena: GPUSnapshotArena;
  /** Resource identifier behind each captured name. */
  resources: Readonly<Record<string, string>>;
  /** Id of the tap node added to the graph. */
  tapNodeId: string;
};

function getSnapshotElementType(
  view: GraphDataView | GraphBufferHandle
): GPUComputationCaptureElementType {
  const format = view instanceof GraphDataView ? view.format : '';
  return format.startsWith('float32') ? 'f32' : format.startsWith('sint32') ? 'i32' : 'u32';
}

/**
 * Adds one observation tap as the last node of a lens graph. The tap is an ordinary copy node, so
 * it ends compute-pass coalescing (an observer effect) and requires `COPY_SRC` on the views.
 */
export function addLensViewSnapshot<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  device: Device,
  props: {
    id: string;
    views: LensSnapshotViews;
    elementTypes?: Readonly<Record<string, GPUComputationCaptureElementType>>;
    afterNode?: string;
  }
): LensViewSnapshot {
  const entries = Object.entries(props.views);
  const slots = entries.map(([name, view]) => {
    if (view instanceof GraphDataView && view.byteStride !== view.rowByteLength) {
      // Capture views must be packed
      throw new Error(`Capture view "${name}" must be packed`);
    }
    const format = view instanceof GraphDataView ? view.format : '';
    const components = /^(?:float32|sint32|uint32)x([234])$/.exec(format);
    return {
      name,
      byteLength:
        view instanceof GraphDataView ? view.length * view.rowByteLength : view.byteLength,
      elementType: props.elementTypes?.[name] ?? getSnapshotElementType(view),
      ...(components && {components: Number(components[1])})
    };
  });
  const arena = new GPUSnapshotArena(device, {id: `${props.id}-snapshots`, slots, capacity: 1});
  graph.add(
    new GPUSnapshotTap({
      id: `${props.id}-snapshot-tap`,
      arena,
      views: props.views,
      ...(props.afterNode !== undefined && {afterNode: props.afterNode})
    })
  );
  return {
    arena,
    tapNodeId: `${props.id}-snapshot-tap`,
    resources: Object.fromEntries(
      entries.map(([name, view]) => [
        name,
        view instanceof GraphDataView ? view.buffer.id : view.id
      ])
    )
  };
}

/** Reads the single snapshot frame after the compiled graph was submitted, then frees the arena. */
export async function readLensViewSnapshot(
  snapshot: LensViewSnapshot,
  props: {graphId: string; runId?: string}
): Promise<GPUComputationCapture[]> {
  try {
    return await snapshot.arena.read({
      graph: props.graphId,
      resources: snapshot.resources,
      ...(props.runId !== undefined && {runId: props.runId})
    });
  } finally {
    snapshot.arena.destroy();
  }
}

/** Encodes and submits a compiled lens graph once, including its snapshot tap. */
export function submitLensGraph<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  encodeOptions: GPUCommandGraphEncodeOptions<Parameters>
): GPUCommandGraphEncoding {
  const commandEncoder = device.createCommandEncoder();
  try {
    const encoding = compiled.encode(commandEncoder, encodeOptions);
    device.submit(commandEncoder.finish());
    return encoding;
  } finally {
    commandEncoder.destroy();
  }
}
