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
import {validateRequirement} from '../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';

/** One snapshot tap: copies named views into the next frame of an arena. */
export type GPUSnapshotTapProps = {
  /** Node id; also the default frame label. */
  id: string;
  /** Arena receiving one frame per encoding of the tap. Its slot names must equal the `views` keys. */
  arena: GPUSnapshotArena;
  /**
   * Views to copy. The underlying buffers must have `Buffer.COPY_SRC` usage; the tap declares the
   * `copy-source` use, so the graph orders it after earlier writers and keeps transients alive.
   * Whole-buffer handles are copied from byte offset 0; packed data views from their `byteOffset`.
   * Each slot's byteLength must fit within its logical view; strided views are rejected.
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
  readonly props: GPUSnapshotTapProps;
  private readonly sources: readonly {
    name: string;
    buffer: GraphBufferHandle;
    byteOffset: number;
  }[];

  constructor(props: GPUSnapshotTapProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'arena',
      'views',
      'afterNode',
      'label',
      'iteration',
      'recorder'
    ]);
    this.props = Object.freeze({...snapshot, views: Object.freeze({...snapshot.views})});
    const {id, arena, views} = this.props;
    validateRequirement(
      id,
      Object.keys(views).length === arena.slots.length &&
        arena.slots.every(slot => Object.hasOwn(views, slot.name)),
      'views keys must equal the arena slot names'
    );
    for (const slot of arena.slots) {
      const view = views[slot.name];
      if ('buffer' in view) {
        validateRequirement(
          id,
          view.byteStride === view.rowByteLength,
          `views.${slot.name} byteStride must equal rowByteLength (packed rows)`
        );
      }
      const byteLength = 'buffer' in view ? view.length * view.rowByteLength : view.byteLength;
      const byteOffset = 'buffer' in view ? view.byteOffset : 0;
      validateRequirement(
        id,
        slot.byteLength <= byteLength,
        `views.${slot.name} byteLength must be at least slot byteLength ${slot.byteLength}`
      );
      validateRequirement(
        id,
        Number.isSafeInteger(byteOffset) && byteOffset >= 0 && byteOffset % 4 === 0,
        `views.${slot.name} byteOffset must be a non-negative multiple of 4`
      );
    }
    this.sources = arena.slots.map(slot => {
      const view = views[slot.name];
      return Object.freeze({
        name: slot.name,
        buffer: 'buffer' in view ? view.buffer : view,
        byteOffset: 'buffer' in view ? view.byteOffset : 0
      });
    });
  }

  getCommandNodes<Parameters>(
    _graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, arena, afterNode, label, iteration, recorder} = this.props;
    const node = createGPUCopyCommandNode<Parameters>({
      id,
      ...(afterNode !== undefined && {dependsOn: [afterNode]}),
      resources: this.sources.map(source => ({
        buffer: source.buffer,
        usage: 'copy-source' as const
      })),
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) => {
          const sources = Object.fromEntries(
            this.sources.map(source => [
              source.name,
              {buffer: getBuffer(source.buffer), byteOffset: source.byteOffset}
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
