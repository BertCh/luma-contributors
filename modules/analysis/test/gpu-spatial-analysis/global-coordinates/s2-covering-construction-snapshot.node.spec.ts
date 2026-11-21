// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  GPUS2CapCovering,
  type GPUS2CapCoveringProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {
  GPUS2CapCoveringOperation,
  type S2CapCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';
import {CPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {
  GPUS2PolygonCoveringOperation,
  type S2PolygonCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {CPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

function createContext(): GPUSpatialContext {
  return {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel: 'great-circle'};
}
function createVector(
  values: Float32Array | Uint32Array,
  format: GPUCPUVector['format'],
  width: number
): GPUCPUVector {
  const length = values.length / width;
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it('captures direct cap route, dimensions, context and resource selections (W2-04)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['uniform', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const context = createContext();
      const props: GPUS2CapCoveringProps = {
        caps: createTransientView(graph, 'caps', 'float32x4', 1),
        level: 1,
        capacity: 6,
        refinement,
        spatialContext: context,
        ...(refinement === 'level-synchronous' ? {frontierCapacity: 24} : {}),
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 6),
          counts: createTransientView(graph, 'counts', 'uint32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      };
      const originalCells = props.output.cells;
      const originalCaps = props.caps;
      const contributor = new GPUS2CapCovering(props);
      props.refinement = refinement === 'uniform' ? 'level-synchronous' : 'uniform';
      props.level = 15;
      props.capacity = 99;
      props.frontierCapacity = 999;
      props.mode = 'interior';
      props.caps = createTransientView(graph, 'other-caps', 'float32x4', 2);
      props.output.cells = createTransientView(graph, 'other-cells', 'uint32x2', 99);
      context.coordinateSpace = 'planar';
      expect(contributor.props.refinement).toBe(refinement);
      expect(contributor.props.level).toBe(1);
      expect(contributor.props.capacity).toBe(6);
      expect(contributor.props.output.cells).toBe(originalCells);
      expect(contributor.props.caps).toBe(originalCaps);
      expect(contributor.props.spatialContext?.coordinateSpace).toBe('longitude-latitude');
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      const bindings = nodes.flatMap(
        node => getWGSLKernelNodeRecord(graph, node.id)?.bindings ?? []
      );
      const capBindings = bindings.filter(binding => binding.name === 'caps');
      const cellBindings = bindings.filter(binding => binding.name === 'cells');
      expect(capBindings.length).toBeGreaterThan(0);
      expect(cellBindings.length).toBeGreaterThan(0);
      for (const binding of capBindings) expect(binding.view).toBe(originalCaps);
      for (const binding of cellBindings) expect(binding.view).toBe(originalCells);
      expect(
        bindings.some(binding => binding.view === props.caps || binding.view === props.output.cells)
      ).toBe(false);
    }
  } finally {
    device.destroy();
  }
});

it('captures named GPU cap and polygon forms before deferred lowering (W2-04)', () => {
  const cap: S2CapCoveringOperationProps = {
    caps: 'caps',
    capCount: 1,
    level: 1,
    capacity: 6,
    spatialContext: createContext(),
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  const hemisphereCenter: [number, number, number] = [1, 0, 0];
  const polygon: S2PolygonCoveringOperationProps = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    positionCount: 3,
    ringOffsetCount: 2,
    hemisphereCenter,
    maximumLevel: 1,
    capacity: 6,
    spatialContext: createContext(),
    output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visited'}
  };
  const capOperation = new GPUS2CapCoveringOperation(cap);
  const polygonOperation = new GPUS2PolygonCoveringOperation(polygon);
  cap.capCount = 2;
  cap.capacity = 99;
  cap.level = 15;
  cap.refinement = 'level-synchronous';
  cap.caps = 'other-caps';
  cap.output.cells = 'other-cells';
  cap.spatialContext!.coordinateSpace = 'planar';
  polygon.positionCount = 0;
  polygon.capacity = 99;
  polygon.maximumLevel = 15;
  polygon.refinement = 'level-synchronous';
  polygon.positions = 'other-positions';
  polygon.output.cells = 'other-cells';
  polygon.output.visitedCount = undefined;
  hemisphereCenter[0] = 0;
  polygon.spatialContext!.coordinateSpace = 'planar';
  const capContributor = constructNetworkOperationContributor(capOperation, {
    caps: {format: 'float32x4', length: 1}
  }) as GPUS2CapCovering;
  const polygonContributor = constructNetworkOperationContributor(polygonOperation, {
    positions: {format: 'float32x3', length: 3},
    ringOffsets: {format: 'uint32', length: 2}
  }) as GPUS2PolygonCovering;
  expect(capContributor.props.capacity).toBe(6);
  expect(capContributor.props.level).toBe(1);
  expect(capContributor.props.refinement).toBeUndefined();
  expect(capOperation.inputs).toEqual({caps: 'caps'});
  expect(capOperation.outputs['cells'].value).toBe('cells');
  expect(polygonContributor.props.capacity).toBe(6);
  expect(polygonContributor.props.hemisphereCenter).toEqual([1, 0, 0]);
  expect(polygonContributor.props.output.visitedCount).toBeDefined();
  expect(polygonOperation.outputs['cells'].value).toBe('cells');
});

it('captures CPU cap and polygon options while reading current input contents (W2-04)', async () => {
  const cap: S2CapCoveringOperationProps = {
    caps: 'caps',
    capCount: 1,
    level: 0,
    capacity: 6,
    spatialContext: createContext(),
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  const hemisphereCenter: [number, number, number] = [1, 0, 0];
  const polygon: S2PolygonCoveringOperationProps = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    positionCount: 3,
    ringOffsetCount: 2,
    hemisphereCenter,
    maximumLevel: 0,
    capacity: 6,
    spatialContext: createContext(),
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  const capOperation = new CPUS2CapCoveringOperation(cap);
  const polygonOperation = new CPUS2PolygonCoveringOperation(polygon);
  cap.level = 30;
  cap.capacity = 1;
  cap.capCount = 2;
  cap.caps = 'other-caps';
  cap.output.cells = 'other-cells';
  cap.spatialContext!.coordinateSpace = 'planar';
  polygon.maximumLevel = 30;
  polygon.positionCount = 4;
  polygon.capacity = 1;
  polygon.output.cells = 'other-cells';
  hemisphereCenter[0] = 0;
  polygon.spatialContext!.coordinateSpace = 'planar';
  const capValues = Float32Array.of(1, 0, 0, 0);
  const capInputs = {caps: createVector(capValues, 'float32x4', 4)};
  const first = await capOperation.execute({inputs: capInputs});
  expect(Array.from(first['cells'] as Uint32Array)).toEqual([
    0, 0x10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  capValues.set([0, 1, 0, 0]);
  const second = await capOperation.execute({inputs: capInputs});
  expect(Array.from(second['cells'] as Uint32Array)).toEqual([
    0, 0x30000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  const positions = Float32Array.from(
    [
      [1, 0.2, 0],
      [1, -0.1, 0.2],
      [1, -0.1, -0.2]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  const inputs = {
    positions: createVector(positions, 'float32x3', 3),
    ringOffsets: createVector(Uint32Array.of(0, 3), 'uint32', 1)
  };
  const result = await polygonOperation.execute({inputs});
  expect(Array.from(result['cells'] as Uint32Array)).toEqual([
    0, 0x10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  positions[0] = NaN;
  const invalid = await polygonOperation.execute({inputs});
  expect(Array.from(invalid['status'] as Uint32Array)).toEqual([1]);
  expect(Array.from(invalid['cells'] as Uint32Array)).toEqual(new Array(12).fill(0));
});

it('rejects unknown mode and refinement values at named construction (W2-04)', () => {
  const cap: S2CapCoveringOperationProps = {
    id: 'cap-options',
    caps: 'caps',
    capCount: 1,
    level: 0,
    capacity: 6,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  const polygon: S2PolygonCoveringOperationProps = {
    id: 'polygon-options',
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    positionCount: 3,
    ringOffsetCount: 2,
    maximumLevel: 0,
    capacity: 6,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  };
  for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation]) {
    expect(() => new Operation({...cap, mode: 'bad' as never})).toThrow(
      /cap-options.*mode.*exterior.*interior.*bad/
    );
    expect(() => new Operation({...cap, refinement: 'bad' as never})).toThrow(
      /cap-options.*refinement.*uniform.*level-synchronous.*bad/
    );
  }
  for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
    expect(() => new Operation({...polygon, mode: 'bad' as never})).toThrow(
      /polygon-options.*mode.*exterior.*interior.*bad/
    );
    expect(() => new Operation({...polygon, refinement: 'bad' as never})).toThrow(
      /polygon-options.*refinement.*serial.*level-synchronous.*bad/
    );
  }
});

it('captures prototype getter options and output names in every named covering form (review B4)', async () => {
  class CapOptions implements S2CapCoveringOperationProps {
    caps = 'caps';
    capCount = 1;
    level = 0;
    capacity = 6;
    selectedMode: 'interior' | 'exterior' = 'interior';
    get mode() {
      return this.selectedMode;
    }
    get output() {
      return new OutputNames();
    }
  }
  class PolygonOptions implements S2PolygonCoveringOperationProps {
    positions = 'positions';
    ringOffsets = 'ringOffsets';
    positionCount = 3;
    ringOffsetCount = 2;
    maximumLevel = 0;
    capacity = 6;
    selectedMode: 'interior' | 'exterior' = 'interior';
    get mode() {
      return this.selectedMode;
    }
    get output() {
      return new OutputNames();
    }
  }
  class OutputNames {
    get cells() {
      return 'cells';
    }
    get counts() {
      return 'counts';
    }
    get status() {
      return 'status';
    }
  }
  const caps = Float32Array.of(1, 0, 0, 0);
  const capOptions = new CapOptions();
  const cpuCap = new CPUS2CapCoveringOperation(capOptions);
  const gpuCap = new GPUS2CapCoveringOperation(capOptions);
  capOptions.selectedMode = 'exterior';
  const capResult = await cpuCap.execute({inputs: {caps: createVector(caps, 'float32x4', 4)}});
  expect(Array.from(capResult['counts'] as Uint32Array)).toEqual([0, 0]);
  const capContributor = constructNetworkOperationContributor(gpuCap, {
    caps: {format: 'float32x4', length: 1}
  }) as GPUS2CapCovering;
  expect(capContributor.props.mode).toBe('interior');
  expect(gpuCap.outputs['cells'].value).toBe('cells');
  const polygonOptions = new PolygonOptions();
  const cpuPolygon = new CPUS2PolygonCoveringOperation(polygonOptions);
  const gpuPolygon = new GPUS2PolygonCoveringOperation(polygonOptions);
  polygonOptions.selectedMode = 'exterior';
  const positions = Float32Array.from(
    [
      [1, 0.2, 0],
      [1, -0.1, 0.2],
      [1, -0.1, -0.2]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  const polygonResult = await cpuPolygon.execute({
    inputs: {
      positions: createVector(positions, 'float32x3', 3),
      ringOffsets: createVector(Uint32Array.of(0, 3), 'uint32', 1)
    }
  });
  expect(Array.from(polygonResult['counts'] as Uint32Array)).toEqual([0, 0]);
  const polygonContributor = constructNetworkOperationContributor(gpuPolygon, {
    positions: {format: 'float32x3', length: 3},
    ringOffsets: {format: 'uint32', length: 2}
  }) as GPUS2PolygonCovering;
  expect(polygonContributor.props.mode).toBe('interior');
});

it('rejects unknown prototype getter modes in all named covering constructors (review B4)', () => {
  class CapOptions {
    caps = 'caps';
    capCount = 1;
    level = 0;
    capacity = 6;
    output = {cells: 'cells', counts: 'counts', status: 'status'};
    get mode() {
      return 'bogus' as never;
    }
  }
  class PolygonOptions {
    positions = 'positions';
    ringOffsets = 'ringOffsets';
    positionCount = 3;
    ringOffsetCount = 2;
    maximumLevel = 0;
    capacity = 6;
    output = {cells: 'cells', counts: 'counts', status: 'status'};
    get mode() {
      return 'bogus' as never;
    }
  }
  for (const Operation of [CPUS2CapCoveringOperation, GPUS2CapCoveringOperation])
    expect(() => new Operation(new CapOptions())).toThrow(/mode.*exterior.*interior.*bogus/);
  for (const Operation of [CPUS2PolygonCoveringOperation, GPUS2PolygonCoveringOperation])
    expect(() => new Operation(new PolygonOptions())).toThrow(/mode.*exterior.*interior.*bogus/);
});

it('checks both captured polygon input dimensions during GPU lowering (review N1)', () => {
  const operation = new GPUS2PolygonCoveringOperation({
    id: 'dimension-test',
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    positionCount: 3,
    ringOffsetCount: 2,
    maximumLevel: 0,
    capacity: 6,
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  });
  expect(() =>
    constructNetworkOperationContributor(operation, {
      positions: {format: 'float32x3', length: 8},
      ringOffsets: {format: 'uint32', length: 2}
    })
  ).toThrow(/dimension-test.*positionCount.*3.*8/);
  expect(() =>
    constructNetworkOperationContributor(operation, {
      positions: {format: 'float32x3', length: 3},
      ringOffsets: {format: 'uint32', length: 3}
    })
  ).toThrow(/dimension-test.*ringOffsetCount.*2.*3/);
});
