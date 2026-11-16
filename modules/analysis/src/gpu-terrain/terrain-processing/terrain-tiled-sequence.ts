// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUComputeCommandNode,
  type GraphBufferHandle,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/** Fixed terrain dispatch schedule; repetitions reuse compiled kernels and immutable parameters. @internal */
export type TerrainKernelSchedule<Parameters> = {
  nodes: readonly GPUCommandNode<Parameters>[];
  repetitions: number;
}[];

/** A tile is one scheduling unit; its bounded passes share caller-independent tile scratch. @internal */
export function createTerrainTileSequence<Parameters>(
  id: string,
  schedule: TerrainKernelSchedule<Parameters>
): GPUCommandNode<Parameters> {
  const resources = new Map<GraphBufferHandle, GraphBufferUse>();
  let commandCount = 0;
  let invocationCount = 0;
  let workgroupCount = 0;
  for (const stage of schedule) {
    for (const node of stage.nodes) {
      commandCount += (node.workload?.commandCount ?? 1) * stage.repetitions;
      invocationCount += (node.workload?.maximumInvocationCount ?? 0) * stage.repetitions;
      workgroupCount += (node.workload?.maximumWorkgroupCount ?? 0) * stage.repetitions;
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource))
          throw new Error(`${id} terrain tile sequences require buffer-only compute passes`);
        const view = resource.buffer as GraphDataView;
        const handle = view.buffer;
        const previous = resources.get(handle);
        const previousView = previous?.buffer as GraphDataView | undefined;
        const start = Math.min(view.byteOffset, previousView?.byteOffset ?? view.byteOffset);
        const end = Math.max(
          view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength,
          previousView
            ? previousView.byteOffset +
                (previousView.length - 1) * previousView.byteStride +
                previousView.rowByteLength
            : 0
        );
        resources.set(handle, {
          // Caller fields can share one handle at different byte offsets. Preserve the union
          // in this scheduling node's resource declaration; child kernels keep their own views.
          buffer: (handle.graph as GPUCommandGraph<Parameters>).createDataView(handle, {
            format: 'uint32',
            byteOffset: start,
            length: (end - start) / 4
          }),
          usage:
            previous?.usage === 'storage-read-write' || resource.usage === 'storage-read-write'
              ? 'storage-read-write'
              : 'storage-read'
        });
      }
    }
  }
  return createGPUComputeCommandNode<Parameters>({
    id,
    resources: [...resources.values()],
    workload: {
      operation: 'GPUTerrainTiledFlow',
      commandCount,
      maximumInvocationCount: invocationCount,
      maximumWorkgroupCount: workgroupCount
    },
    compile: context => {
      const compiled = new Map<
        GPUCommandNode<Parameters>,
        ReturnType<GPUComputeCommandNode<Parameters>['compile']>
      >();
      try {
        for (const stage of schedule)
          for (const node of stage.nodes) {
            if (node.type !== 'compute')
              throw new Error(`${id} terrain tile sequences require compute passes`);
            if (!compiled.has(node)) compiled.set(node, node.compile(context));
          }
      } catch (error) {
        for (const kernel of compiled.values()) kernel.destroy?.();
        throw error;
      }
      return {
        encode: encoding => {
          for (const stage of schedule)
            for (let repetition = 0; repetition < stage.repetitions; repetition++)
              for (const node of stage.nodes) compiled.get(node)!.encode(encoding);
        },
        destroy: () => {
          for (const kernel of compiled.values()) kernel.destroy?.();
        }
      };
    }
  });
}

/** Pointer doublings alternate two fixed binding sets; the round index never reaches WGSL. @internal */
export function appendTerrainAlternatingSchedule<Parameters>(
  schedule: TerrainKernelSchedule<Parameters>,
  nodes: readonly [GPUCommandNode<Parameters>, GPUCommandNode<Parameters>],
  rounds: number
): void {
  if (rounds >= 2) schedule.push({nodes, repetitions: Math.floor(rounds / 2)});
  if (rounds % 2) schedule.push({nodes: [nodes[0]], repetitions: 1});
}
