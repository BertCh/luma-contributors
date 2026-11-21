// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  captureGraphCommandNodes
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {
  GPUHotSpotAnalysis,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createRecipeLossReadback} from './recipe-loss-readback';

function createContext() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'sint32' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `loss-review-${serial++}`, format, length);
  return {graph, view};
}

function createHotSpot(
  context: ReturnType<typeof createContext>,
  sumOverflowLength = 1,
  candidateOverflow?: GraphDataView<'uint32'>
) {
  const {graph, view} = context;
  return addHotSpotAnalysisRecipe(graph, {
    id: 'hot-loss-review',
    source: {
      kind: 'points',
      family: 'quadbin',
      resolution: 7,
      cellCapacity: 5,
      pointCellCapacity: 2,
      positions: view('float32x2', 5),
      values: view('float32', 5),
      studyArea: {
        kind: 'polygons',
        positions: view('float32x2', 4),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2),
        candidateCapacity: 64
      }
    },
    minimumNeighbors: 1,
    settings: {fixedMoments: {count: 5, mean: 2, variance: 1}},
    scratch: {
      pointTable: {status: {sumOverflow: view('uint32', sumOverflowLength)}},
      table: {status: {candidateOverflow}}
    },
    parameters: new GPUParameterBuffer(graph.device, {
      id: 'parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    })
  });
}

it('B1 and B4 gate inference independently of public occupancy, fixed moments and loss kind', () => {
  const context = createContext();
  let recipe!: ReturnType<typeof createHotSpot>;
  const nodes = captureGraphCommandNodes(context.graph, () => {
    recipe = createHotSpot(context);
  });
  const hotSpots = recipe.contributors.find(
    contributor => contributor instanceof GPUHotSpotAnalysis
  )! as GPUHotSpotAnalysis;
  expect(hotSpots.props.mask).not.toBe(recipe.mask);
  expect(recipe.table!.status.candidateOverflow).toBeDefined();
  const point = recipe.status.stages.find(stage => stage.stage === 'point-aggregation')!.status;
  const memory = createCPUGraphMemory();
  memory.write(recipe.mask!, Uint32Array.of(1, 1, 1, 1, 1));
  const gates = nodes.filter(node => /-(joined-loss|inference-mask)$/.test(node.id));
  expect(gates).toHaveLength(2);
  for (const [sumLoss, capacityLoss, destinationLoss] of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ]) {
    memory.write(point.sumOverflow!, Uint32Array.of(sumLoss));
    memory.write(point.overflow!, Uint32Array.of(capacityLoss));
    memory.write(recipe.table!.status.overflow!, Uint32Array.of(destinationLoss));
    executeCommandNodesOnCPU(gates, memory);
    expect(Array.from(memory.readUint32(recipe.mask!))).toEqual([1, 1, 1, 1, 1]);
    expect(Array.from(memory.readUint32(recipe.table!.status.candidateOverflow!))).toEqual([
      capacityLoss
    ]);
    expect(Array.from(memory.readUint32(hotSpots.props.mask!))).toEqual(
      new Array(5).fill(sumLoss || capacityLoss || destinationLoss ? 0 : 1)
    );
  }
});

function createSpaceTime(
  context: ReturnType<typeof createContext>,
  occupiedCellsOnly: boolean | null = true
) {
  const {graph, view} = context;
  const positions = view('float32x2', 5);
  const recipe = addSpaceTimeHotSpotsRecipe(graph, {
    id: 'cube-loss-review',
    timestamps: view('uint32x2', 5),
    calendarParameters: view('sint32', 2),
    slices: {step: 'month', origin: {year: 2023}, count: 3},
    cells: {kind: 'lattice', positions, width: 4, height: 1, bounds: [0, 0, 4, 1]},
    occupiedCellsOnly: occupiedCellsOnly as boolean,
    settings: {radius: 0, temporalWindow: 0},
    scratch: {groupKeys: view('uint32', 2), groupCounts: view('uint32', 2)},
    parameters: new GPUParameterBuffer(graph.device, {
      id: 'parameters',
      format: 'float32',
      length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
    })
  });
  return {recipe, positions};
}

it('B2 derives occupancy from eligible events and gates truncated sorted inference until recovery', () => {
  const context = createContext();
  let result!: ReturnType<typeof createSpaceTime>;
  const nodes = captureGraphCommandNodes(context.graph, () => {
    result = createSpaceTime(context);
  });
  const {recipe, positions} = result;
  const emerging = recipe.contributors.find(
    contributor => contributor instanceof GPUEmergingHotSpots
  )! as GPUEmergingHotSpots;
  const overflow = recipe.status.stages.find(stage => stage.stage === 'cube-counts')!.status
    .overflow!;
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5));
  memory.write(recipe.intermediates.periods, Uint32Array.of(0, 0, 1, 0, 0));
  memory.write(recipe.cube, Uint32Array.of(2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  memory.write(overflow, Uint32Array.of(1));
  const maskNodes = nodes.filter(node =>
    /-(occupied-clear|occupied-events|occupied-cell-mask|inference-mask)$/.test(node.id)
  );
  executeCommandNodesOnCPU(maskNodes, memory);
  expect(Array.from(memory.readUint32(recipe.intermediates.cellMask!))).toEqual([1, 1, 1, 0]);
  expect(Array.from(memory.readUint32(emerging.props.mask!))).toEqual([0, 0, 0, 0]);
  memory.write(positions, new Float32Array(10).fill(0.5));
  memory.write(recipe.intermediates.periods, Uint32Array.of(0, 0, 1, 1, 0));
  memory.write(overflow, Uint32Array.of(0));
  executeCommandNodesOnCPU(maskNodes, memory);
  expect(Array.from(memory.readUint32(recipe.intermediates.cellMask!))).toEqual([1, 0, 0, 0]);
  expect(Array.from(memory.readUint32(emerging.props.mask!))).toEqual([1, 0, 0, 0]);
});

it('B5 constructs the actual device readback helper with valid WGSL binding names', () => {
  const {graph, view} = createContext();
  expect(() =>
    createRecipeLossReadback(graph, 'copy', view('uint32', 4), view('uint32', 4), 'u32')
  ).not.toThrow();
});

it('S5 rejects explicit null occupancy and validates formerly ignored sum-loss overrides', () => {
  expect(() => createSpaceTime(createContext(), null)).toThrow(/occupiedCellsOnly/);
  expect(() => createHotSpot(createContext(), 0)).toThrow(/sumOverflow.*one uint32 row/);
});

it('B4 retains the caller joined candidate-loss destination without changing destination overflow', () => {
  const context = createContext();
  const candidate = context.view('uint32', 1);
  const recipe = createHotSpot(context, 1, candidate);
  expect(recipe.table!.status.candidateOverflow).toBe(candidate);
  const study = recipe.status.stages.find(stage => stage.stage === 'study-table')!.status;
  expect(recipe.table!.status.overflow).toBe(study.overflow);
});
