// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {
  GPUSpaceTimeDiggleTest,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  type GPUSpaceTimeDiggleTestProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const Contributor of [GPUSpaceTimeKFunction, GPUSpaceTimeDiggleTest])
  for (const firstUndefined of [true, false]) {
    it(`S1 ${Contributor.name} captures spatial frame getters once (first undefined=${firstUndefined})`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      let originReads = 0;
      let areaReads = 0;
      let unitReads = 0;
      const origin: [number, number] = [12, 34];
      const area: [number, number, number, number] = [0, 0, 40, 40];
      const unit = {name: 'meters', metersPerUnit: 1};
      const spatialContext: GPUSpatialContext = {
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'meters',
        frame: {
          id: 'capture-frame',
          kind: 'crs',
          crs: 'EPSG:3857',
          get origin() {
            originReads++;
            return firstUndefined && originReads === 1 ? undefined : origin;
          },
          get areaOfUse() {
            areaReads++;
            return area;
          },
          get unit() {
            unitReads++;
            return unit;
          }
        }
      };
      const props = {
        spatialContext,
        positions: view('positions', 'float32x2', 2),
        times: view('times', 'float32', 2),
        parameters: view('parameters', 'float32', 11),
        randomParameters: view('random', 'uint32', 2),
        gridSize: [2, 2] as const,
        distanceClassCount: 1,
        timeClassCount: 1,
        permutationCount: 1
      };
      const contributor =
        Contributor === GPUSpaceTimeKFunction
          ? new GPUSpaceTimeKFunction({...props, output: {k: view('k', 'float32', 1)}})
          : new GPUSpaceTimeDiggleTest({
              ...props,
              output: {
                d: view('d', 'float32', 1),
                statistics: view('statistics', 'float32', 2),
                summary: view('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
              }
            });
      try {
        expect([originReads, areaReads, unitReads]).toEqual([1, 1, 1]);
        origin[0] = 99;
        area[0] = 99;
        unit.metersPerUnit = 99;
        contributor.getCommandNodes(graph);
        expect(contributor.spatialContext.frame?.origin).toEqual(
          firstUndefined ? undefined : [12, 34]
        );
        expect(contributor.spatialContext.frame?.areaOfUse).toEqual([0, 0, 40, 40]);
        expect(contributor.spatialContext.frame?.unit).toEqual({name: 'meters', metersPerUnit: 1});
        expect([originReads, areaReads, unitReads]).toEqual([1, 1, 1]);
      } finally {
        device.destroy();
      }
    });
  }

for (const Contributor of [GPUSpaceTimeKFunction, GPUSpaceTimeDiggleTest]) {
  it(`PTS-133 ${Contributor.name} captures declared fields once before validation`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    let interval = 1;
    let intervalReads = 0;
    const step = {
      calendarUnit: 'day' as const,
      get interval() {
        intervalReads++;
        return interval;
      }
    };
    const temporalContext: GPUTemporalContext = {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'regular', step}
    };
    const gridSize: [number, number] = [2, 3];
    const output: {
      k: GraphDataView<'float32'>;
      d: GraphDataView<'float32'>;
      statistics?: GraphDataView<'float32'>;
      summary?: GraphDataView<'float32'>;
    } = {
      k: view('k', 'float32', 2),
      d: view('d', 'float32', 2),
      statistics: view('statistics', 'float32', 2),
      summary: view('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH)
    };
    // Non-enumerable inherited getters must be captured too, and only once.
    let reads = 0;
    let currentPositions = view('positions', 'float32x2', 3);
    const positions = currentPositions;
    const prototype = Object.defineProperty({}, 'positions', {
      get: () => {
        reads++;
        return currentPositions;
      }
    });
    const props = Object.assign(Object.create(prototype) as {positions: typeof positions}, {
      times: view('times', 'float32', 3),
      parameters: view('parameters', 'float32', 11),
      randomParameters: view('random', 'uint32', 2),
      gridSize,
      distanceClassCount: 2,
      timeClassCount: 1,
      permutationCount: 1,
      temporalContext,
      output
    });
    // Give each contributor only its declared output fields.
    if (Contributor === GPUSpaceTimeKFunction) {
      delete output.statistics;
      delete output.summary;
    }
    const originalK = output.k;
    const contributor =
      Contributor === GPUSpaceTimeKFunction
        ? new GPUSpaceTimeKFunction(props)
        : new GPUSpaceTimeDiggleTest(props as GPUSpaceTimeDiggleTestProps);
    expect(reads).toBe(1);
    currentPositions = view('replacement', 'float32x2', 1);
    gridSize[0] = 99;
    props.distanceClassCount = 64;
    props.timeClassCount = 64;
    props.permutationCount = 999;
    output.k = view('replacement-k', 'float32', 1);
    temporalContext.unit = 'nanosecond';
    interval = 9;
    try {
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(sources.map(kernel => kernel.source).join('\n')).toContain(
        'const DISTANCE_CLASS_COUNT: u32 = 2u;'
      );
      expect(contributor.props.positions).toBe(positions);
      expect(contributor.props.output.k).toBe(originalK);
      expect(contributor.props.gridSize).toEqual([2, 3]);
      expect(contributor.temporalContext?.unit).toBe('second');
      expect(contributor.temporalContext?.axis).toEqual({
        kind: 'regular',
        step: {calendarUnit: 'day', interval: 1}
      });
      expect(reads).toBe(1);
      expect(intervalReads).toBe(1);
    } finally {
      device.destroy();
    }
  });
}
