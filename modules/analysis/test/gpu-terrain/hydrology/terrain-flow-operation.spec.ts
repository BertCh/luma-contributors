// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {CPUTerrainFlowOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {
  GPUTerrainFlowOperation,
  type TerrainFlowOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-operation';
import {
  getGPUTerrainFlowParameterValues,
  GPUTerrainFlow
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const GRID = {
  width: 5,
  height: 5,
  affine: [2, 0, 0, 0, -3, 15] as const,
  pixelInterpretation: 'area' as const
};

for (const scene of [
  'weighted-d8',
  'd-infinity',
  'mfd-freeman',
  'mfd-quinn',
  'fill-flats',
  'epsilon-fill',
  'high-edge-flats',
  'all-invalid',
  'singleton',
  'geographic-area',
  'mercator-area'
] as const) {
  it(`terrain-flow operations agree on ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid =
      scene === 'singleton'
        ? {...GRID, width: 1, height: 1}
        : scene === 'geographic-area'
          ? {...GRID, affine: [0.01, 0, 0, 0, -0.01, 50] as const}
          : scene === 'mercator-area'
            ? {...GRID, affine: [100, 0, 0, 0, -100, 6000000] as const}
            : GRID;
    const count = grid.width * grid.height;
    const fillDepressions =
      scene === 'fill-flats' || scene === 'epsilon-fill' || scene === 'all-invalid';
    const resolveFlats =
      scene === 'fill-flats' || scene === 'high-edge-flats' || scene === 'all-invalid';
    const flowRouting =
      scene === 'd-infinity' || scene === 'mfd-freeman' || scene === 'mfd-quinn' ? scene : 'd8';
    const props: TerrainFlowOperationProps = {
      id: `flow-${scene}`,
      grid,
      fillDepressions,
      resolveFlats,
      flowRouting,
      maximumFillIterations: 32,
      maximumFlatIterations: 32,
      maximumAccumulationIterations: 32,
      cellSizeMode:
        scene === 'geographic-area'
          ? 'geographic'
          : scene === 'mercator-area'
            ? 'web-mercator'
            : 'uniform',
      accumulationUnits:
        scene === 'weighted-d8' || scene === 'geographic-area' || scene === 'mercator-area'
          ? 'area'
          : 'cells',
      elevation: {
        id: 'dem-band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'dem'},
        validity: 'validity',
        scale: 0.5,
        offset: 2,
        noDataValue: -9999.1
      },
      parameters: 'parameters',
      runoff: scene === 'weighted-d8' || flowRouting !== 'd8' ? 'runoff' : undefined,
      output: {
        ...(fillDepressions ? {filledElevation: 'filledElevation'} : {}),
        flowDirections: 'flowDirections',
        cellClasses: 'cellClasses',
        accumulation: 'accumulation',
        streams: 'streams',
        status: {
          ...(fillDepressions
            ? {fillConverged: 'fillConverged', fillIterationCount: 'fillIterationCount'}
            : {}),
          ...(resolveFlats
            ? {flatsConverged: 'flatsConverged', flatsIterationCount: 'flatsIterationCount'}
            : {}),
          accumulationConverged: 'accumulationConverged',
          accumulationIterationCount: 'accumulationIterationCount'
        }
      }
    };
    const values = Float32Array.from({length: count}, (_, cell) => {
      const row = Math.floor(cell / grid.width);
      const column = cell % grid.width;
      if (scene === 'all-invalid') return NaN;
      if (fillDepressions)
        return row === 0 || column === 0 || row === grid.height - 1 || column === grid.width - 1
          ? 20
          : 0;
      if (scene === 'high-edge-flats') return column === 0 ? 24 : column === 4 ? 0 : 10;
      // Unequal slopes produce a multi-step DAG with actual splits and merges. The D-infinity
      // winning facet is interior, with a moderate diagonal fraction rather than 0 or 1.
      return 100 - 4 * column - 1.5 * row;
    });
    const validity = new Uint32Array(count).fill(scene === 'all-invalid' ? 0 : 1);
    if (scene === 'weighted-d8') {
      // Exercise both validity and the rounded float32 nodata sentinel, away from the main trunk.
      validity[0] = 0;
      values[4] = Math.fround(-9999.1);
    }
    const inputs: Record<string, OperationGraphInput> = {
      dem: {format: 'float32', values},
      validity: {format: 'uint32', values: validity},
      parameters: {
        format: 'float32',
        values: getGPUTerrainFlowParameterValues(
          {
            fillEpsilon: scene === 'epsilon-fill' ? 0.25 : 0,
            streamThreshold:
              scene === 'weighted-d8'
                ? 33.25
                : scene === 'geographic-area'
                  ? 2e6
                  : scene === 'mercator-area'
                    ? 7000
                    : 2.625,
            flowExponent: scene === 'mfd-freeman' ? 1.5 : scene === 'mfd-quinn' ? 2 : 0
          },
          grid
        )
      }
    };
    if (props.runoff !== undefined)
      inputs['runoff'] = {
        format: 'float32',
        values: Float32Array.from({length: count}, (_, cell) => 0.5 + (cell % 7) * 0.25)
      };
    const outputs = [
      ...Object.values(props.output).filter((value): value is string => typeof value === 'string'),
      ...Object.values(props.output.status!)
    ];
    // Each run receives a fresh operation: runOperationGraph destroys its graph and operations.
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainFlowOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainFlowOperation(props)],
      outputs
    });
    for (const name of outputs) {
      expect(gpu[name].length).toBe(cpu[name].length);
      if (name.endsWith('IterationCount')) {
        // These are deliberately algorithm-specific diagnostics, not numerical results. CPU
        // executes zero GPU rounds. Bounds cover the spill/tour confirmations and three flat
        // relaxations, and are the stated tolerance for this output alone.
        expect(cpu[name][0]).toBe(0);
        const maximumRounds = name === 'flatsIterationCount' ? 3 * 32 : 32;
        expect(gpu[name][0]).toBeGreaterThanOrEqual(0);
        expect(Math.abs(gpu[name][0] - cpu[name][0])).toBeLessThanOrEqual(maximumRounds);
      } else if (name === 'accumulation' || name === 'filledElevation') {
        for (let cell = 0; cell < count; cell++) {
          const expected = cpu[name][cell];
          if (Number.isNaN(expected)) expect(gpu[name][cell]).toBeNaN();
          else if (name === 'filledElevation' || expected === 0)
            expect(gpu[name][cell] === expected).toBe(true);
          // Float64 fractions/sums vs GPU float32: relative tolerance, so small positive
          // results cannot pass as zero. No sign-of-zero comparison.
          else
            expect(Math.abs(gpu[name][cell] - expected) / Math.abs(expected)).toBeLessThanOrEqual(
              3e-5
            );
        }
      } else expect(Array.from(gpu[name])).toEqual(Array.from(cpu[name]));
    }
    if (scene !== 'all-invalid' && scene !== 'singleton') {
      expect(cpu['accumulation'].some(value => value > 1)).toBe(true);
      expect(cpu['streams'].some(value => value === 1)).toBe(true);
      expect(cpu['streams'].some(value => value === 0)).toBe(true);
    }
    if (scene === 'fill-flats' || scene === 'high-edge-flats')
      expect(cpu['flowDirections'][12]).not.toBe(0);
  });
}

for (const [format, noDataValue] of [
  ['uint32', 0],
  ['uint32', 0xffffffff],
  ['sint32', -0x80000000],
  ['sint32', 0x7fffffff],
  ['float32', -9999.1],
  ['float32', Math.fround(3.4028234663852886e38)]
] as const) {
  it(`terrain-flow operations agree on valid ${format} no-data boundary ${noDataValue}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = {...GRID, width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1] as const};
    const props: TerrainFlowOperationProps = {
      id: 'review-flow',
      grid,
      elevation: {
        id: 'elevation',
        format,
        noDataValue,
        storage: {kind: 'buffer', values: 'elevation'}
      },
      parameters: 'parameters',
      output: {
        flowDirections: 'directions',
        cellClasses: 'classes',
        accumulation: 'accumulation',
        streams: 'streams'
      }
    };
    const outputs = ['directions', 'classes', 'accumulation', 'streams'];
    for (const missing of [false, true]) {
      const value = missing ? noDataValue : 7;
      const inputs: Record<string, OperationGraphInput> = {
        elevation: {
          format,
          values:
            format === 'uint32'
              ? Uint32Array.of(value)
              : format === 'sint32'
                ? Int32Array.of(value)
                : Float32Array.of(value)
        },
        parameters: {format: 'float32', values: getGPUTerrainFlowParameterValues({}, grid)}
      };
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUTerrainFlowOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUTerrainFlowOperation(props)],
        outputs
      });
      // One cell: every output is exact, including terminal/missing integer codes and NaN.
      for (const name of outputs) expect(Array.from(gpu[name])).toEqual(Array.from(cpu[name]));
      expect(cpu['directions'][0]).toBe(missing ? 0xffffffff : 0);
      expect(cpu['classes'][0]).toBe(missing ? 4 : 3);
      if (missing) expect(cpu['accumulation'][0]).toBeNaN();
      else expect(cpu['accumulation'][0]).toBe(1);
      expect(cpu['streams'][0]).toBe(0);
    }
  });
}

for (const noDataValue of [-1, 0.5]) {
  it(`both terrain-flow paths reject uint32 noDataValue ${noDataValue} before dispatch`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = {...GRID, width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1] as const};
    const props: TerrainFlowOperationProps = {
      id: 'review-flow',
      grid,
      elevation: {
        id: 'elevation',
        format: 'uint32',
        noDataValue,
        storage: {kind: 'buffer', values: 'elevation'}
      },
      parameters: 'parameters',
      output: {accumulation: 'accumulation'}
    };
    // This is a metadata rejection check, not a numerical comparison outside the scalar domain.
    // Invalid nonfinite metadata is node-only; its documented CPU result is an Error.
    expect(() => new CPUTerrainFlowOperation(props)).toThrow(/^review-flow.*elevation.noDataValue/);
    const graph = new GPUCommandGraph(device);
    const operation = new GPUTerrainFlowOperation(props);
    const views: Record<string, GraphDataView> = {
      values: createTransientView(graph, 'review-elevation', 'uint32', 1),
      parameters: createTransientView(graph, 'review-settings', 'float32', 8),
      accumulation: createTransientView(graph, 'review-accumulation', 'float32', 1)
    };
    const getView = <Format extends GPUGraphVectorFormat>(port: string) =>
      views[port] as GraphDataView<Format>;
    try {
      const contributor = operation.createContributor(getView);
      expect(contributor).toBeInstanceOf(GPUTerrainFlow);
      if (!(contributor instanceof GPUTerrainFlow)) throw new Error('expected GPUTerrainFlow');
      // getCommandNodes builds GPURasterNeighborhood, which owns the upstream scalar check.
      expect(() => contributor.getCommandNodes(graph)).toThrow(/nodata must fit in uint32/);
    } finally {
      // Graph views are declarations; no execution was compiled or GPU allocation created.
      operation.destroy();
    }
  });
}
