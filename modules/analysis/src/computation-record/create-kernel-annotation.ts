// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  getViewElementOffset,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUComputationDatumBinding,
  GPUComputationLoopCoordinate,
  GPUComputationNodeAnnotation
} from './computation-record-types';
import {hashGPUComputationText} from './hash-computation-text';

/** Kernel facts an analysis operator records for one compute node. */
export type GPUKernelAnnotationProps = {
  /** Exported operator name, also the pack's formula key. */
  op: string;
  /** Stable kernel module identifier. */
  moduleId: string;
  /** Full WGSL text exactly as passed to the kernel. */
  source: string;
  /** Entry point; defaults to main. */
  entryPoint?: string;
  /** Pipeline override constants. */
  constants?: Readonly<Record<string, number>>;
  /** Lanes per workgroup. */
  workgroupSize: readonly [number, number, number];
  /** Dispatched workgroups along each dimension. */
  workgroups: readonly [number, number, number];
  /** Logical invocations, excluding idle lanes. */
  invocationCount?: number;
  /** Named storage or uniform bindings. */
  bindings?: readonly {
    name: string;
    view: GraphDataView | GraphBufferHandle;
    access: 'read' | 'read_write' | 'write' | 'uniform';
    /** WGSL element type when it differs from the view format, such as `atomic<u32>`. */
    elementType?: string;
  }[];
  /** Key space indexed by invocations. */
  datum?: GPUComputationDatumBinding;
  /** Coordinate of an unrolled pass. */
  loop?: GPUComputationLoopCoordinate;
  /** Shared name of unrolled copies. */
  dedupName?: string;
  /** Formula reference id. */
  formula?: string;
  /** Display label. */
  label?: string;
  /** Free-form attributes. */
  attrs?: GPUComputationNodeAnnotation['attrs'];
};

/**
 * Builds the node annotation for one kernel. The recorder drops `source.text` unless it was created
 * with `sourceText: true`. Hashing costs a pass over the source, so call this only when a recorder exists.
 */
export function createGPUKernelAnnotation(
  props: GPUKernelAnnotationProps
): GPUComputationNodeAnnotation {
  const {
    source,
    moduleId,
    entryPoint,
    constants,
    workgroupSize,
    workgroups,
    invocationCount,
    bindings,
    ...rest
  } = props;
  const hash = hashGPUComputationText(source);
  return {
    ...Object.fromEntries(Object.entries(rest).filter(([, value]) => value !== undefined)),
    source: {
      id: `wgsl:${hash}`,
      language: 'wgsl',
      moduleId,
      hash,
      entryPoint: entryPoint ?? 'main',
      text: source,
      ...(constants && {constants})
    },
    constants: constants ?? {},
    entryPoint: entryPoint ?? 'main',
    workgroupSize,
    dispatch: {
      workgroups,
      ...(invocationCount !== undefined && {invocationCount}),
      indirect: false
    },
    ...(bindings && {
      bindings: bindings.map(({name, view, access, elementType}) => ({
        name,
        resource: 'buffer' in view ? view.buffer.id : view.id,
        access,
        ...('buffer' in view && {
          elementType: elementType ?? getGPUElementType(view.format),
          elementOffset: getViewElementOffset(view),
          bindingByteOffset: getViewBindingRange(view).offset,
          bindingByteLength: getViewBindingRange(view).size,
          length: view.length
        })
      }))
    })
  };
}

/** Maps a packed GPU vector format to the scalar WGSL element type used in bindings. */
export function getGPUElementType(format: string): string {
  return format.startsWith('float32')
    ? 'f32'
    : format.startsWith('sint32')
      ? 'i32'
      : format.startsWith('uint32')
        ? 'u32'
        : format;
}
