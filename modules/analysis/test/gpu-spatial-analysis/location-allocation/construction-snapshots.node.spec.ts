// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  type GPULocationAllocationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {
  GPUBoundedFacilityAllocation,
  type GPUBoundedFacilityAllocationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation';
import {
  GPUDorlingCartogram,
  type GPUDorlingCartogramProps
} from '../../../src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram';
import {
  GPULocationAllocationGraphOperation,
  type LocationAllocationOperationProps
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation';
import {CPULocationAllocationOperation} from '../../../src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('captures facility method, sparse topology and output bindings while preserving graph views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `view-${sequence++}`, format, length);
    const props: GPULocationAllocationProps = {
      operation: 'p-median',
      facilityCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      sparseCosts: {
        offsets: view('uint32', 3),
        facilities: view('uint32', 2),
        costs: view('float32', 2)
      },
      output: {
        selectedFacilities: view('uint32', 2),
        assignments: view('uint32', 2),
        objective: view('float32', 1),
        status: view('uint32', 7)
      }
    };
    const model = new GPULocationAllocation(props);
    const offsets = props.sparseCosts!.offsets,
      objective = props.output.objective;
    props.operation = 'set-covering';
    props.facilityCount = 99;
    props.sparseCosts!.offsets = view('uint32', 5);
    props.output.objective = view('float32', 8);
    expect(model.props.operation).toBe('p-median');
    expect(model.props.facilityCount).toBe(2);
    expect(model.props.sparseCosts!.offsets).toBe(offsets);
    expect(model.props.output.objective).toBe(objective);
    expect(model.getCommandNodes(graph).length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});

it('captures the facility operation factory and CPU execution closure before caller mutation', async () => {
  const device = createNullWebGPUDevice();
  try {
    const props: LocationAllocationOperationProps = {
      id: 'captured-operation',
      operation: 'p-median',
      facilityCount: 2,
      demandCount: 2,
      maximumFacilityCount: 1,
      maximumSwapIterations: 0,
      costs: 'costs',
      output: {
        selectedFacilities: 'selected',
        assignments: 'assignments',
        objective: 'objective',
        status: 'status'
      }
    };
    const operation = new GPULocationAllocationGraphOperation(props);
    const cpuOperation = new CPULocationAllocationOperation(props);
    props.operation = 'set-covering';
    props.facilityCount = 99;
    props.output.objective = 'changed';
    const graph = new GPUCommandGraph(device);
    const views = new Map<string, GraphDataView>();
    for (const [port, output] of Object.entries(operation.outputs))
      views.set(port, createTransientView(graph, port, output.format, output.length));
    views.set('costs', createTransientView(graph, 'costs', 'float32', 4));
    const lowered = operation.createContributor(
      <Format extends GPUGraphVectorFormat>(port: string) =>
        views.get(port)! as GraphDataView<Format>
    ) as GPULocationAllocation;
    expect(lowered.props.operation).toBe('p-median');
    expect(lowered.props.facilityCount).toBe(2);
    expect(lowered.props.output.objective).toBe(views.get('objective'));
    const actual = await runOperationGraph(device, {
      inputs: {costs: {format: 'float32', values: Float32Array.of(1, 4, 2, 5)}},
      operations: [cpuOperation],
      outputs: ['selected', 'assignments', 'objective']
    });
    expect(Array.from(actual['selected'])).toEqual([1, 0]);
    expect(Array.from(actual['assignments'])).toEqual([0, 0]);
    expect(Array.from(actual['objective'])).toEqual([3]);
  } finally {
    device.destroy();
  }
});

it('captures bounded facility operation and nested outputs', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `view-${sequence++}`, format, length);
    const props: GPUBoundedFacilityAllocationProps = {
      operation: 'p-center',
      facilityCount: 2,
      selectCount: 1,
      costs: view('float32', 4),
      capacities: view('float32', 2),
      demand: view('float32', 2),
      output: {
        selectedFacilities: view('uint32', 2),
        assignments: view('uint32', 2),
        objective: view('float32', 1),
        status: view('uint32', 3)
      }
    };
    const model = new GPUBoundedFacilityAllocation(props),
      objective = props.output.objective;
    props.operation = 'capacitated-p-median';
    props.selectCount = 2;
    props.output.objective = view('float32', 1);
    expect(model.props.operation).toBe('p-center');
    expect(model.props.selectCount).toBe(1);
    expect(model.props.output.objective).toBe(objective);
    expect(
      getGPUCommandNodeKernelSources(model.getCommandNodes(graph)).some(kernel =>
        kernel.source.includes('const CAPACITATED: bool = false')
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('captures Dorling tiling, method, output and nested spatial metadata', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `view-${sequence++}`, format, length);
    const props: GPUDorlingCartogramProps = {
      positions: view('float32x2', 2),
      values: view('float32', 2),
      maximumIterations: 1,
      neighborGrid: false,
      spatialContext: {
        coordinateSpace: 'planar',
        metric: 'native',
        frame: {id: 'local', kind: 'local-azimuthal-equidistant', origin: [0, 0]}
      },
      output: {
        positions: view('float32x2', 2),
        radii: view('float32', 2),
        sourceIds: view('uint32', 2),
        metrics: view('float32x2', 1),
        status: view('uint32', 4)
      }
    };
    const model = new GPUDorlingCartogram(props),
      positions = props.output.positions;
    props.rowTileSize = 1;
    props.neighborGrid = true;
    props.attraction = 1;
    props.output.positions = view('float32x2', 3);
    props.spatialContext!.frame!.id = 'changed';
    expect(model.props.neighborGrid).toBe(false);
    expect(model.props.rowTileSize).toBeUndefined();
    expect(model.props.attraction).toBeUndefined();
    expect(model.props.output.positions).toBe(positions);
    expect(model.props.spatialContext!.frame!.id).toBe('local');
    expect(model.getCommandNodes(graph).length).toBe(model.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
