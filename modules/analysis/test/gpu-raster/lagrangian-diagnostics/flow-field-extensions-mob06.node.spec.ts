// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {evaluateProjectionProgram, invertProjectionProgram} from '@luma.gl/graph/gpu-project';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {describe, expect, it} from 'vitest';
import {
  GPUScalarFieldRidges,
  getGPUScalarFieldRidgesParameterValues,
  GPULagrangianFlowMap,
  GPUFiniteTimeLyapunovExponent,
  type GPUScalarFieldRidgesProps,
  type GPULagrangianFlowMapProps
} from '../../../src/raster-analysis/lagrangian-diagnostics/index';
import {LAGRANGIAN_GEOGRAPHIC_PROJECTION} from '../../../src/raster-analysis/lagrangian-diagnostics/lagrangian-geographic-projection';
import fixture from '../../oracles/fixtures/flow-field-extensions-mob06-numpy.json';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GRID: GPURasterMetadata = {
  width: 9,
  height: 7,
  affine: [2, 0, 0, 0, -3, 21],
  pixelInterpretation: 'area'
};
let serial = 0;
function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `flow-extensions-${serial++}`});
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const ridges: GPUScalarFieldRidgesProps = {
    grid: GRID,
    input: {
      id: 'density',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32', 63)},
      validity: view('uint32', 63)
    },
    parameters: view('float32', 8),
    output: {
      ridges: view('uint32', 63),
      validity: view('uint32', 63),
      strength: view('float32', 63)
    }
  };
  const flow: GPULagrangianFlowMapProps = {
    seeds: view('float32x2', 5),
    velocitiesStart: view('float32x2', 63),
    velocitiesEnd: view('float32x2', 63),
    fieldWidth: 9,
    fieldHeight: 7,
    stepCount: 20,
    parameters: view('float32', 8),
    output: {positions: view('float32x2', 5), validity: view('uint32', 5)}
  };
  return {device, graph, view, ridges, flow};
}

describe('MOB-06 construction and projection contract', () => {
  it('packs per-frame affine scales, row edges and thresholds', () => {
    expect(
      Array.from(
        getGPUScalarFieldRidgesParameterValues({minimumValue: 2, minimumCurvature: 3}, GRID)
      )
    ).toEqual([2, 3, 21, 0, 2, 3, 0, 0]);
    expect(() => getGPUScalarFieldRidgesParameterValues({minimumCurvature: -1}, GRID)).toThrow(
      /GPUScalarFieldRidges.*minimumCurvature/
    );
    expect(() => getGPUScalarFieldRidgesParameterValues({}, GRID, new Float32Array(7))).toThrow(
      /target.*8/
    );
  });

  it('validates ridge dimensions, missing flags, aliases and orthogonal ground-distance modes at construction', () => {
    const {device, graph, ridges, view} = createContext();
    try {
      expect(new GPUScalarFieldRidges(ridges).getCommandNodes(graph)).toHaveLength(1);
      expect(() => new GPUScalarFieldRidges({...ridges, grid: {...GRID, width: 2}})).toThrow(
        /scalar-field-ridges.*grid.width.*3/
      );
      expect(() => new GPUScalarFieldRidges({...ridges, grid: {...GRID, width: 65536}})).toThrow(
        /grid.width.*65535/
      );
      expect(
        () =>
          new GPUScalarFieldRidges({
            ...ridges,
            output: {...ridges.output, validity: view('uint32', 62)}
          })
      ).toThrow(/output.validity.*63/);
      expect(
        () =>
          new GPUScalarFieldRidges({
            ...ridges,
            output: {...ridges.output, validity: ridges.output.ridges}
          })
      ).toThrow(/scalar-field-ridges/);
      expect(
        () => new GPUScalarFieldRidges({...ridges, grid: {...GRID, affine: [2, 1, 0, 0, 3, 0]}})
      ).toThrow(/orthogonal/);
      expect(
        () =>
          new GPUScalarFieldRidges({
            ...ridges,
            input: {
              ...ridges.input,
              storage: {kind: 'buffer', values: view('uint32', 63)}
            } as unknown as GPUScalarFieldRidgesProps['input']
          })
      ).toThrow(/input/);
      expect(
        () => new GPUScalarFieldRidges({...ridges, parameters: view('float32', 12)})
      ).not.toThrow();
      const rotatedGrid: GPURasterMetadata = {...GRID, affine: [0, -3, 0, 2, 0, 0]};
      expect(() => new GPUScalarFieldRidges({...ridges, grid: rotatedGrid})).not.toThrow();
      expect(
        () => new GPUScalarFieldRidges({...ridges, grid: rotatedGrid, cellSizeMode: 'web-mercator'})
      ).toThrow(/axis-aligned/);
      expect(() => new GPUScalarFieldRidges({...ridges, cellSizeMode: 'geographic'})).toThrow(
        /spatialContext/
      );
      expect(
        () =>
          new GPUScalarFieldRidges({
            ...ridges,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
            cellSizeMode: 'geographic'
          })
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });

  it('validates flow field sizes, geographic context and validity outputs before graph construction', () => {
    const {device, graph, flow, view} = createContext();
    try {
      expect(() => new GPULagrangianFlowMap({...flow, stepCount: 8193})).toThrow(/stepCount.*8192/);
      expect(() => new GPULagrangianFlowMap({...flow, fieldWidth: 65536})).toThrow(
        /fieldWidth.*65535/
      );
      expect(
        () =>
          new GPULagrangianFlowMap({
            ...flow,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          })
      ).toThrow(/metric/);
      expect(
        () =>
          new GPULagrangianFlowMap({...flow, output: {...flow.output, validity: view('uint32', 4)}})
      ).toThrow(/output.validity.*5/);
      expect(
        new GPULagrangianFlowMap({
          ...flow,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        }).getCommandNodes(graph)
      ).toHaveLength(3);
      expect(
        () =>
          new GPUFiniteTimeLyapunovExponent({
            positions: view('float32x2', 9),
            seedWidth: 3,
            seedHeight: 3,
            parameters: view('float32', 4),
            validity: view('uint32', 8),
            output: {ftle: view('float32', 9)}
          })
      ).toThrow(/validity.*9/);
    } finally {
      device.destroy();
    }
  });

  it('matches the NumPy analytic uniform-flow oracle through the published projection program', () => {
    const inverse = invertProjectionProgram(LAGRANGIAN_GEOGRAPHIC_PROJECTION);
    for (const flow of fixture.flowCases) {
      for (let index = 0; index < flow.validity.length; index++) {
        if (!flow.validity[index]) {
          continue;
        }
        const projected = evaluateProjectionProgram(LAGRANGIAN_GEOGRAPHIC_PROJECTION, [
          flow.seeds[2 * index],
          flow.seeds[2 * index + 1]
        ]);
        expect(projected.valid).toBe(true);
        const duration = flow.stepCount * flow.timeStep;
        const position = evaluateProjectionProgram(inverse, [
          projected.position[0] + 600 * duration,
          projected.position[1] + 200 * duration
        ]);
        expect(position.valid).toBe(true);
        expect(position.position[0]).toBeCloseTo(flow.positions[2 * index]!, 10);
        expect(position.position[1]).toBeCloseTo(flow.positions[2 * index + 1]!, 10);
      }
    }
    expect(evaluateProjectionProgram(LAGRANGIAN_GEOGRAPHIC_PROJECTION, [1, 86]).valid).toBe(false);
  });
});
