// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {vi} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import * as records from '@luma-contributors/analysis/computation-record';
import type {LensScenarioDescriptor} from '../../src/computation-lenses/types';
import {getSpatialOperatorEntry} from '../../src/gpu-spatial-analysis/computation-pack/pack';
import * as dataframeRuns from '../../src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support';
import {getDataFrameOperatorEntry} from '../../src/gpu-dataframe/computation-pack/pack';
import {getRasterOperatorEntries} from '../../src/gpu-raster/computation-pack/pack';
import {getTerrainOperatorEntries} from '../../src/gpu-terrain/computation-pack/pack';
import {getLensCaptureNames} from '../../src/computation-lenses/lens-view-snapshot';
import {assertValidLensRecord} from '../../src/computation-lenses/validate-lens-record';

export type ScenarioDeclaration = {
  record: records.GPUComputationRecord;
  captureNames: string[];
  missingChoices: string[];
};

/**
 * Execute the real scenario closure through record assembly, stopping before submission/readback.
 * Pipelines are inert null-device objects, while real node compilation still registers kernel
 * facts. Encoding is forbidden. Published scheduling, validation and preflight still run.
 * Spies are scoped to one scenario and restored even on a declaration error.
 */
export async function declareLensScenario(
  scenario: LensScenarioDescriptor,
  storageBufferLimit = 8
): Promise<ScenarioDeclaration> {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  Object.defineProperty(device, 'submit', {
    value: () => {
      throw new Error('Null-device declaration must never submit');
    }
  });
  Object.defineProperty(device.limits, 'maxStorageBuffersPerShaderStage', {
    value: storageBufferLimit
  });
  const boundary = new Error('Null-device declaration reached record assembly');
  let record: records.GPUComputationRecord | undefined;
  const captureNames = new Set<string>();
  let derivedCaptureNames: readonly string[] | undefined;
  const compiledGraphs: {destroy(): void}[] = [];
  const compiledStructures: records.GPUComputationGraph[] = [];
  const buffers: {destroy(): void}[] = [];
  const originalCreateBuffer = device.createBuffer;
  const originalCompile = GPUCommandGraph.prototype.compile;
  const originalRecord = records.createGPUComputationRecord;
  const originalTap = records.GPUSnapshotTap.prototype.getCommandNodes;
  const originalDataframeRun = dataframeRuns.runDataFrameContributorLens;
  const spies = [
    vi
      .spyOn(dataframeRuns, 'runDataFrameContributorLens')
      .mockImplementation((frameDevice, props) => {
        derivedCaptureNames = props.derivedCaptureNames;
        return originalDataframeRun(frameDevice, props);
      }),
    vi.spyOn(device, 'createBuffer').mockImplementation(function (props) {
      const buffer = originalCreateBuffer.call(device, props);
      buffers.push(buffer);
      return buffer;
    }),
    vi.spyOn(device, 'createCommandEncoder').mockImplementation(() => {
      throw new Error('Null-device declaration must never execute');
    }),
    vi.spyOn(GPUCommandGraph.prototype, 'compile').mockImplementation(function (
      this: GPUCommandGraph
    ) {
      const compiled = originalCompile.call(this);
      compiledGraphs.push(compiled);
      compiledStructures.push(records.describeGPUCommandGraph(compiled, {graph: this}).graph);
      for (const name of getLensCaptureNames(this)) captureNames.add(name);
      return compiled;
    }),
    vi.spyOn(GPUCommandGraph.prototype, 'compileAsync').mockImplementation(async function (
      this: GPUCommandGraph
    ) {
      return this.compile();
    }),
    vi.spyOn(records.GPUSnapshotTap.prototype, 'getCommandNodes').mockImplementation(function (
      this: records.GPUSnapshotTap,
      graph
    ) {
      for (const slot of this.props.arena.slots) captureNames.add(slot.name);
      return originalTap.call(this, graph);
    }),
    vi.spyOn(records, 'createGPUComputationRecord').mockImplementation(props => {
      record = originalRecord(props);
      assertValidLensRecord(record);
      // Record graph IDs can override compiled IDs. Match complete ordered node/resource
      // structure, consuming each record graph once so duplicate omitted graphs also fail.
      const remainingGraphs = record.graphs.filter(graph => graph.form === 'compiled');
      for (const compiled of compiledStructures) {
        const index = remainingGraphs.findIndex(
          graph =>
            JSON.stringify(getGraphNodeSignature(graph)) ===
            JSON.stringify(getGraphNodeSignature(compiled))
        );
        if (index === -1)
          throw new Error(
            `${scenario.id}: compiled graph ${compiled.id} (nodes: ${compiled.nodes.map(node => node.id).join(', ')}) is missing or structurally different in the record`
          );
        remainingGraphs.splice(index, 1);
      }
      if (remainingGraphs.length)
        throw new Error(
          `${scenario.id}: recorded compiled graph ${remainingGraphs.map(graph => graph.id).join(', ')} was not compiled during declaration`
        );
      throw boundary;
    })
  ];
  try {
    try {
      await (await scenario.load()).run(device);
    } catch (error) {
      if (error !== boundary) throw error;
    }
    if (!record) throw new Error(`${scenario.id}: scenario never assembled its record`);
    const missingChoices = getMissingScenarioChoices(scenario, record);
    return {
      record,
      captureNames: [...(derivedCaptureNames ?? captureNames)].sort(),
      missingChoices
    };
  } finally {
    for (const spy of spies.reverse()) spy.mockRestore();
    for (const compiled of compiledGraphs) compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

export function getMissingScenarioChoices(
  scenario: LensScenarioDescriptor,
  record: records.GPUComputationRecord
): string[] {
  const entry =
    scenario.entry === 'gpu-spatial-analysis'
      ? getSpatialOperatorEntry(scenario.element)
      : scenario.entry === 'gpu-dataframe'
        ? getDataFrameOperatorEntry(scenario.element)
        : scenario.entry === 'gpu-raster'
          ? getRasterOperatorEntries()[scenario.element]
          : scenario.entry === 'gpu-terrain'
            ? getTerrainOperatorEntries()[scenario.element]
            : undefined;
  // Network and GPU-core packs expose no choice templates; their actual record choices are
  // still checked for undefined values below.
  const requiredChoices = entry?.choices ?? [];
  const requiredNames = requiredChoices.map(choice =>
    typeof choice === 'string' ? choice : choice.name
  );
  return [
    ...requiredNames.filter(name => !record.choices.some(choice => choice.name === name)),
    ...record.choices.filter(choice => choice.value === undefined).map(choice => choice.name)
  ];
}

/**
 * Legacy records omit device limits. Replay with the minimum core-or-higher storage-buffer limit
 * their retained bindings require, so device-dependent fusion is compared on the recorded route.
 * Uniform buffers and texture bindings do not consume storage-buffer binding slots.
 */
export function getRecordedStorageBufferLimit(record: records.GPUComputationRecord): number {
  let storageBufferLimit = 8;
  for (const graph of record.graphs) {
    const bufferIds = new Set(
      graph.resources?.filter(resource => resource.kind === 'buffer').map(resource => resource.id)
    );
    for (const node of graph.nodes) {
      const bindingCount =
        node.bindings?.filter(
          binding =>
            binding.access !== 'uniform' &&
            binding.access !== 'sampled' &&
            bufferIds.has(binding.resource)
        ).length ?? 0;
      storageBufferLimit = Math.max(storageBufferLimit, bindingCount);
    }
  }
  return storageBufferLimit;
}

/** Every named sidecar capture participates, including geometry aliases. */
export function getRecordedCaptureNames(
  record: records.GPUComputationRecordWithSidecars
): string[] {
  const captures = record.captures;
  if (!captures) throw new Error(`${record.id}: missing captures sidecar`);
  return [
    ...new Set(
      [...captures.snapshots, ...captures.stamps, ...captures.replays].map(capture => capture.name)
    )
  ].sort();
}

/**
 * Ignore timings, source text, allocations and values; retain the declared observation graph and
 * each kernel binding's layout (name, resource, access, element type, element offset and length),
 * so a binding that shifts by one element changes the signature (LENS-106).
 */
export function getStructureSignature(
  record: records.GPUComputationRecord,
  captureNames: readonly string[]
) {
  return {
    graphs: record.graphs.map(graph => ({
      id: graph.id,
      nodes: getGraphNodeSignature(graph)
    })),
    captureNames: [...captureNames].sort()
  };
}

/** Declared binding order is part of the layout; absent optional fields stay absent. */
function getBindingLayoutSignature(node: records.GPUComputationNode) {
  return (node.bindings ?? []).map(binding => ({
    name: binding.name,
    resource: binding.resource,
    access: binding.access,
    ...(binding.elementType !== undefined ? {elementType: binding.elementType} : {}),
    ...(binding.elementOffset !== undefined ? {elementOffset: binding.elementOffset} : {}),
    ...(binding.length !== undefined ? {length: binding.length} : {}),
    ...(binding.bindingByteOffset !== undefined
      ? {bindingByteOffset: binding.bindingByteOffset}
      : {}),
    ...(binding.bindingByteLength !== undefined
      ? {bindingByteLength: binding.bindingByteLength}
      : {})
  }));
}

function getGraphNodeSignature(graph: records.GPUComputationGraph) {
  return graph.nodes.map(node => ({
    id: node.id,
    kind: node.kind,
    reads: [...(node.reads ?? [])].sort(),
    writes: [...(node.writes ?? [])].sort(),
    bindings: getBindingLayoutSignature(node)
  }));
}
