// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {transpileModule, ModuleKind, ScriptTarget} from 'typescript';
import {entries as networkEntries} from '../../../../../website/src/catalog/entries/network-routing-access';
import {entries as geometryEntries} from '../../../../../website/src/catalog/entries/geometry-construction-overlay';
import {entries as recipeEntries} from '../../../../../website/src/catalog/entries/recipes-trajectories';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {
  GPUOffsetCurve,
  getGPUOffsetCurveParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {addPopulationSupportAlignmentRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `catalog-final3-${serial++}`, format, length);
  return {device, graph, view};
}

it.each([
  'facility',
  'rational'
] as const)('H1 declares the authored %s example with its required summary', kind => {
  const {device, graph, view} = createContext();
  const declare =
    kind === 'facility'
      ? vi.spyOn(GPUNetworkFacilityOptimization.prototype, 'getCommandNodes')
      : vi.spyOn(GPUNetworkRationalAccess.prototype, 'getCommandNodes');
  try {
    const summary = view('uint32', 4);
    const status = {
      invalidCount: view('uint32', 1),
      iterationCount: view('uint32', 1),
      converged: view('uint32', 1),
      iterationLimitReached: view('uint32', 1)
    };
    const imported = (values: ReturnType<typeof view>) => ({
      importToGraph: (target: GPUCommandGraph) => {
        expect(target).toBe(graph);
        return values;
      }
    });
    const bindings =
      kind === 'facility'
        ? {
            GPUNetworkFacilityOptimization,
            graph,
            summary,
            status,
            costs: imported(view('float32', 8)),
            demand: imported(view('float32', 1)),
            capacities: imported(view('float32', 8)),
            threshold: imported(view('float32', 1)),
            selectedFacilities: view('uint32', 8),
            assignments: view('uint32', 1),
            objective: view('float32', 1)
          }
        : {
            GPUNetworkRationalAccess,
            graph,
            summary,
            status,
            matrix: {costs: view('float32', 2)},
            demand: imported(view('float32', 1)),
            supply: imported(view('float32', 2)),
            parameters: imported(view('float32', 6)),
            assignments: view('float32', 2),
            scores: view('float32', 1),
            unservedDemand: view('float32', 1)
          };
    const entry = networkEntries.find(
      entry =>
        entry.id === `network-${kind === 'facility' ? 'facility-optimization' : 'rational-access'}`
    )!;
    const source = entry.example.replace(/^import[^\n]*\n/gm, '');
    const javascript = transpileModule(source, {
      compilerOptions: {target: ScriptTarget.ES2022, module: ModuleKind.ESNext}
    }).outputText;
    const add = vi.spyOn(graph, 'add');
    new Function(...Object.keys(bindings), javascript)(...Object.values(bindings));
    expect(add).toHaveBeenCalledTimes(1);
    const contributor = add.mock.calls[0][0] as
      | GPUNetworkFacilityOptimization
      | GPUNetworkRationalAccess;
    expect(contributor.props.output.summary).toBe(summary);
    if (contributor instanceof GPUNetworkFacilityOptimization)
      expect(contributor.props.facilityCount).toBe(8);
    expect(declare).toHaveBeenCalledTimes(1);
    expect(declare.mock.results[0].value).toHaveLength(kind === 'facility' ? 9 : 3);
    add.mockRestore();
  } finally {
    declare.mockRestore();
    device.destroy();
  }
});

it('H2 documents terminal vertices separately from the single compact output line', () => {
  const {device, graph, view} = createContext();
  try {
    const positions = view('float32x2', 3);
    const lineOffsets = view('uint32', 2);
    const parameters = view('float32', 4);
    const output = {
      layout: 'compact' as const,
      positions: view('float32x2', 3),
      lineOffsets: view('uint32', 2),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        vertexCount: view('uint32', 1),
        requiredVertexCount: view('uint32', 1)
      }
    };
    const nodes = new GPUOffsetCurve({
      positions,
      lineOffsets,
      parameters,
      joinStyle: 'bevel',
      output
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.of(0, 0, 1, 0, 2, 0));
    memory.write(lineOffsets, Uint32Array.of(0, 3));
    memory.write(parameters, getGPUOffsetCurveParameterValues({distance: 1}));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(output.lineOffsets))).toEqual([0, 3]);
    expect(Array.from(memory.readUint32(output.status.count))).toEqual([1]);
    expect(Array.from(memory.readUint32(output.status.vertexCount))).toEqual([3]);
    expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([0]);
    const entry = geometryEntries.find(entry => entry.id === 'offset-curve')!;
    const description = entry.outputs.find(output => output.name === 'lineOffsets')!.description;
    expect(description).toContain('vertices written');
    expect(description).toContain('status.count');
    expect(description).not.toContain('terminal line count');
  } finally {
    device.destroy();
  }
});

it('H3 documents actual alignment recipe view paths and empty intermediates', () => {
  const {device, graph, view} = createContext();
  try {
    const populations = view('float32x2', 2);
    const sourceIds = view('uint32', 2);
    const targetIds = view('uint32', 2);
    const result = addPopulationSupportAlignmentRecipe(graph, {populations, sourceIds, targetIds});
    expect(result.intermediates).toEqual({});
    expect(result.provenance.sourceIds).toBe(sourceIds);
    expect(result.provenance.targetIds).toBe(targetIds);
    expect(result.outputs.populations).not.toBe(populations);
    expect(result.outputs.populations.length).toBe(2);
    const entry = recipeEntries.find(entry => entry.id === 'population-support-alignment-recipe')!;
    const documented = entry.outputs.flatMap(output => output.name.split(/\s*,\s*/));
    expect(documented).toEqual(
      expect.arrayContaining([
        'provenance.sourceIds',
        'provenance.targetIds',
        'outputs.populations'
      ])
    );
    expect(entry.outputs.find(output => output.name === 'intermediates')!.description).toMatch(
      /empty/i
    );
  } finally {
    device.destroy();
  }
});
