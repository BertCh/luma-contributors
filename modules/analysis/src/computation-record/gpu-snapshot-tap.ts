// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUCopyCommandNode,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUSnapshotArena, GPUSnapshotLabel} from './gpu-snapshot-arena';
import type {GPUComputationRecorder} from './computation-recorder';

/** One snapshot tap: copies named views into the next frame of an arena. */
export type GPUSnapshotTapProps = {
  /** Node id; also the default frame label. */
  id: string;
  /** Arena receiving one frame per encoding of the tap. Its slot names must equal the `views` keys. */
  arena: GPUSnapshotArena;
  /**
   * Views to copy. The underlying buffers must have `Buffer.COPY_SRC` usage; the tap declares the
   * `copy-source` use, so the graph orders it after earlier writers and keeps transients alive.
   * Whole-buffer handles are copied from byte offset 0; data views from their `byteOffset`.
   */
  views: Readonly<Record<string, GraphDataView | GraphBufferHandle>>;
  /** Node the observation follows; stored as the capture's `afterNode` and as an explicit dependency. */
  afterNode?: string;
  /** Frame description stored as the capture's `meaning`. */
  label?: string;
  /** Frame iteration coordinate; defaults to the frame index. */
  iteration?: number;
  /** Records the tap as an observation-tap node in a computation record. */
  recorder?: GPUComputationRecorder;
};

/**
 * A copy node that snapshots views of a graph into a {@link GPUSnapshotArena} at a chosen point of
 * the authored sequence. Taps are ordinary published `GPUCommandGraph` copy nodes, so they need no
 * graph hooks. Add each tap right after the node it observes, because resource hazards follow
 * authoring order. Taps end compute-pass coalescing (an observer effect), so list the tap count in
 * the record's `GPUComputationEncodingFlags.taps`. Call `arena.read()` after the encoding is submitted.
 */
export class GPUSnapshotTap implements GPUCommandNodeProducer<any> {
  constructor(readonly props: GPUSnapshotTapProps) {}

  getCommandNodes<Parameters>(
    _graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, arena, views, afterNode, label, iteration, recorder} = this.props;
    const entries = Object.entries(views);
    const node = createGPUCopyCommandNode<Parameters>({
      id,
      ...(afterNode !== undefined && {dependsOn: [afterNode]}),
      resources: entries.map(([, view]) => ({buffer: view, usage: 'copy-source' as const})),
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) => {
          const sources = Object.fromEntries(
            entries.map(([name, view]) => [
              name,
              {buffer: getBuffer(view), byteOffset: 'buffer' in view ? view.byteOffset : 0}
            ])
          );
          const frameLabel: GPUSnapshotLabel = {
            ...(afterNode !== undefined && {afterNode}),
            ...(iteration !== undefined && {iteration}),
            label: label ?? id
          };
          arena.encodeSnapshot(commandEncoder, sources, frameLabel);
        }
      })
    });
    return [
      recorder
        ? recorder.recordNode(node, {op: 'GPUSnapshotTap', attrs: {observationTap: true}})
        : node
    ];
  }
}
