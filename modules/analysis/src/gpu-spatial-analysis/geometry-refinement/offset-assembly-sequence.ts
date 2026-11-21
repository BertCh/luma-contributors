// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  createGPUCopyCommandNode,
  type GPUCommandNode,
  type GPUCommandGraph,
  type GPUComputeCommandNode,
  type GraphDataView,
  type GraphBufferUse
} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Maximum declared invocation budget per composite assembly command (uint32). @internal */
export const MAXIMUM_OFFSET_ASSEMBLY_SEQUENCE_INVOCATION_COUNT = 0xffffffff;

/** Maximum assembly dispatches grouped into one compute pass. @internal */
const MAXIMUM_OFFSET_ASSEMBLY_PASS_DISPATCH_COUNT = 2048;

/** Splits complete repetition rounds by invocation and dispatch budgets. @internal */
export function createBoundedOffsetAssemblySequences<Parameters>(
  id: string,
  schedule: readonly {nodes: readonly GPUCommandNode<Parameters>[]; repetitions: number}[]
): readonly GPUCommandNode<Parameters>[] {
  const commands: GPUCommandNode<Parameters>[] = [];
  let batchCount = 0;
  for (const stage of schedule) {
    const invocationsPerRound = stage.nodes.reduce(
      (count, node) => count + (node.workload?.maximumInvocationCount ?? 0),
      0
    );
    validateCapacity(
      id,
      'assembly invocations per round',
      invocationsPerRound,
      0,
      MAXIMUM_OFFSET_ASSEMBLY_SEQUENCE_INVOCATION_COUNT
    );
    const dispatchesPerRound = stage.nodes.reduce(
      (count, node) => count + (node.workload?.commandCount ?? 1),
      0
    );
    validateCapacity(
      id,
      'assembly dispatches per round',
      dispatchesPerRound,
      0,
      MAXIMUM_OFFSET_ASSEMBLY_PASS_DISPATCH_COUNT
    );
    const roundsPerPass = Math.floor(
      MAXIMUM_OFFSET_ASSEMBLY_PASS_DISPATCH_COUNT / Math.max(dispatchesPerRound, 1)
    );
    const roundsPerCommand = Math.min(
      roundsPerPass,
      Math.floor(
        MAXIMUM_OFFSET_ASSEMBLY_SEQUENCE_INVOCATION_COUNT / Math.max(invocationsPerRound, 1)
      )
    );
    for (let firstRound = 0; firstRound < stage.repetitions; firstRound += roundsPerCommand) {
      const batch = createOffsetAssemblySequence(
        batchCount === 0 ? id : `${id}-batch-${batchCount}`,
        [
          {
            nodes: stage.nodes,
            repetitions: Math.min(roundsPerCommand, stage.repetitions - firstRound)
          }
        ]
      );
      commands.push(batch);
      batchCount++;
    }
  }
  return commands;
}

/** Budget one assembly pass across setup, continuation batches and status. @internal */
export function boundOffsetAssemblyPasses<Parameters>(
  id: string,
  metadata: GraphDataView<'uint32'>,
  nodes: readonly GPUCommandNode<Parameters>[]
): readonly GPUCommandNode<Parameters>[] {
  // Graph resource hazards survive operation lowering. Declare exclusive access to the fragment
  // table even in setup and at boundaries so the compiler keeps each pass in schedule order.
  const ordering: GraphBufferUse = {buffer: metadata, usage: 'storage-read-write'};
  const commands: GPUCommandNode<Parameters>[] = [];
  const addBoundary = () => {
    commands.push(
      createGPUCopyCommandNode<Parameters>({
        id: `${id}-pass-boundary-${commands.length}`,
        resources: [ordering],
        workload: {operation: 'GPUOffsetCurveComponents', commandCount: 0},
        // Pass-independent graph nodes end the coalesced compute pass before encoding.
        compile: () => ({encode: () => {}})
      })
    );
  };
  addBoundary();
  let dispatchesInPass = 0;
  for (const node of nodes) {
    if (node.type === 'compute') {
      const dispatches = node.workload?.commandCount ?? 1;
      validateCapacity(
        id,
        'assembly dispatches per command',
        dispatches,
        0,
        MAXIMUM_OFFSET_ASSEMBLY_PASS_DISPATCH_COUNT
      );
      if (dispatchesInPass + dispatches > MAXIMUM_OFFSET_ASSEMBLY_PASS_DISPATCH_COUNT) {
        addBoundary();
        dispatchesInPass = 0;
      }
      dispatchesInPass += dispatches;
    } else {
      dispatchesInPass = 0;
    }
    commands.push({
      ...node,
      resources: [
        ...(node.resources ?? []).filter(
          resource => !('buffer' in resource && resource.buffer === metadata)
        ),
        ordering
      ]
    });
  }
  addBoundary();
  return commands;
}

/** Reuses public command encoders for a fixed offset-assembly schedule, without expanding the graph. */
export function createOffsetAssemblySequence<Parameters>(
  id: string,
  schedule: readonly {nodes: readonly GPUCommandNode<Parameters>[]; repetitions: number}[]
): GPUCommandNode<Parameters> {
  const conditions = new Map<GPUCommandNode<Parameters>, GraphDataView>();
  const resources = new Map<GraphDataView, GraphBufferUse>();
  let commandCount = 0;
  let invocationCount = 0;
  let workgroupCount = 0;
  for (const stage of schedule)
    for (const node of stage.nodes) {
      commandCount += (node.workload?.commandCount ?? 1) * stage.repetitions;
      invocationCount += (node.workload?.maximumInvocationCount ?? 0) * stage.repetitions;
      workgroupCount += (node.workload?.maximumWorkgroupCount ?? 0) * stage.repetitions;
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource) || !('format' in resource.buffer))
          throw new Error(`${id} offset assembly requires buffer views`);
        const view = resource.buffer;
        const previous = resources.get(view);
        resources.set(view, {
          buffer: view,
          usage:
            previous?.usage === 'storage-read-write' || resource.usage === 'storage-read-write'
              ? 'storage-read-write'
              : 'storage-read'
        });
      }
      if (node.condition?.source === 'gpu') {
        const condition = node.condition;
        const view = (condition.buffer.graph as GPUCommandGraph<Parameters>).createDataView(
          condition.buffer,
          {format: 'uint32', length: 3, byteOffset: condition.byteOffset ?? 0}
        );
        conditions.set(node, view);
        resources.set(view, {buffer: view, usage: 'indirect'});
      }
    }
  return createGPUComputeCommandNode<Parameters>({
    id,
    resources: [...resources.values()],
    workload: {
      operation: 'GPUOffsetCurveComponents',
      variant: 'bounded-assembly',
      commandCount,
      ...(Number.isSafeInteger(invocationCount) ? {maximumInvocationCount: invocationCount} : {}),
      ...(Number.isSafeInteger(workgroupCount) ? {maximumWorkgroupCount: workgroupCount} : {})
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
              throw new Error(`${id} offset assembly requires compute commands`);
            if (!compiled.has(node)) compiled.set(node, node.compile(context));
          }
      } catch (error) {
        for (const executable of compiled.values()) executable.destroy?.();
        throw error;
      }
      return {
        encode: encoding => {
          const encodings = new Map<GPUCommandNode<Parameters>, typeof encoding>();
          for (const node of compiled.keys()) {
            const condition = node.condition;
            if (condition?.source === 'gpu') {
              // Preserve the public indirect condition when composing command encoders. The
              // adapter is captured once per encode, rather than allocated in each repetition.
              const pass = encoding.computePass;
              const indirect = encoding.getBuffer(conditions.get(node)!);
              const adapter = new Proxy(pass, {
                get(target, property) {
                  if (property === 'dispatch')
                    return () => target.dispatchIndirect(indirect, condition.byteOffset ?? 0);
                  const value = Reflect.get(target, property, target);
                  return typeof value === 'function' ? value.bind(target) : value;
                }
              });
              encodings.set(node, {...encoding, computePass: adapter});
            } else encodings.set(node, encoding);
          }
          for (const stage of schedule)
            for (let repetition = 0; repetition < stage.repetitions; repetition++)
              for (const node of stage.nodes) compiled.get(node)!.encode(encodings.get(node)!);
        },
        destroy: () => {
          for (const executable of compiled.values()) executable.destroy?.();
        }
      };
    }
  });
}
