// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {GPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import {
  GPURasterWalkingCost,
  getGPURasterWalkingCostParameterValues
} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';

it('R7 all walking forms keep the original two-cell GRASS model after nested edits', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  for (const form of ['direct', 'gpu-operation', 'cpu-operation'] as const) {
    const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
    const grid = {width: 2, height: 1, affine, pixelInterpretation: 'area' as const};
    const model = {
      kind: 'grass' as 'grass' | 'tobler',
      coefficients: [1, 1, -1, -1] as [number, number, number, number],
      downhillSlopeThreshold: -0.5
    };
    const parameters = getGPURasterWalkingCostParameterValues({model}, grid);
    const mutate = () => {
      model.kind = 'tobler';
      model.coefficients[0] = 9;
      model.downhillSlopeThreshold = -2;
      grid.width = 1;
      affine[0] = 9;
    };
    // The fixed, flat GRASS unit edge costs exactly 1; its original parameter block is valid.
    if (form === 'direct') {
      const rig = createParityGraph(device);
      const costs = rig.output('costs', 2, 'float32');
      const invalid = rig.output('invalid', 1, 'uint32');
      const elevation = {
        id: 'elevation',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: rig.input('elevation', [0, 0], 'float32').view}
      };
      const friction = {
        id: 'friction',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: rig.input('friction', [0, 0], 'float32').view}
      };
      const output = {
        costs: costs.view,
        validity: rig.output('validity', 2, 'uint32').view,
        status: {converged: rig.output('done', 1, 'uint32').view, invalidCount: invalid.view}
      };
      const contributor = new GPURasterWalkingCost({
        grid,
        model,
        elevation,
        friction,
        sources: rig.input('sources', [1, 0], 'uint32').view,
        parameters: rig.input('parameters', Array.from(parameters), 'float32').view,
        maximumIterations: 1,
        output
      });
      mutate();
      output.costs = rig.output('replacement-costs', 2, 'float32').view;
      elevation.storage.values = rig.input('replacement-elevation', [0, -9], 'float32').view;
      rig.graph.add(contributor);
      const compiled = rig.graph.compile();
      try {
        invalid.buffer.write(Uint32Array.of(12345));
        submitGraph(device, compiled, undefined);
        expect(await readFloat32(costs.buffer, 2)).toEqual([0, 1]);
        expect(await readUint32(invalid.buffer, 1)).toEqual([0]);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    } else {
      const Operation =
        form === 'gpu-operation' ? GPURasterWalkingCostOperation : CPURasterWalkingCostOperation;
      const operation = new Operation({
        grid,
        model,
        maximumIterations: 1,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: 'elevation'}
        },
        friction: {
          id: 'friction',
          format: 'float32',
          storage: {kind: 'buffer', values: 'friction'}
        },
        sources: 'sources',
        parameters: 'parameters',
        output: {
          costs: 'costs',
          validity: 'validity',
          status: {converged: 'done', invalidCount: 'invalid'}
        }
      });
      mutate();
      const result = await runOperationGraph(device, {
        inputs: {
          elevation: {format: 'float32', values: Float32Array.of(0, 0)},
          friction: {format: 'float32', values: Float32Array.of(0, 0)},
          sources: {format: 'uint32', values: Uint32Array.of(1, 0)},
          parameters: {format: 'float32', values: parameters}
        },
        operations: [operation],
        outputs: ['costs', 'invalid']
      });
      expect(Array.from(result['costs'])).toEqual([0, 1]);
      expect(Array.from(result['invalid'])).toEqual([0]);
    }
  }
});

it('R2 walking forms declare inherited status names and preserve them after rename', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  for (const Operation of [GPURasterWalkingCostOperation, CPURasterWalkingCostOperation]) {
    const status = Object.create({converged: 'done', iterationLimitReached: 'limit'});
    const output = Object.assign(Object.create({costs: 'costs', validity: 'validity'}), {status});
    const operation = new Operation({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: 'elevation'}
      },
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      sources: 'sources',
      parameters: 'parameters',
      output
    });
    status.converged = 'renamed';
    status.iterationLimitReached = 'renamed-limit';
    output.costs = 'renamed-costs';
    output.validity = 'renamed-validity';
    const result = await runOperationGraph(device, {
      inputs: {
        elevation: {format: 'float32', values: Float32Array.of(0)},
        friction: {format: 'float32', values: Float32Array.of(0)},
        sources: {format: 'uint32', values: Uint32Array.of(1)},
        parameters: {format: 'float32', values: getGPURasterWalkingCostParameterValues({}, grid)}
      },
      operations: [operation],
      outputs: ['costs', 'validity', 'done', 'limit']
    });
    // Usable seeded singleton: cost zero, validity one and no transition to change a cost.
    expect(Array.from(result['costs'])).toEqual([0]);
    expect(Array.from(result['validity'])).toEqual([1]);
    expect(Array.from(result['done'])).toEqual([1]);
    expect(Array.from(result['limit'])).toEqual([0]);
  }
});

for (const scene of ['partial', 'complete', 'mismatch'] as const) {
  it(`walking forms pin independent ${scene} count, validity and stopping words`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const width = scene === 'mismatch' ? 1 : 6;
    const grid = {
      width,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const model = {
      kind: 'grass' as const,
      coefficients: [1, 1, -1, -1] as const,
      downhillSlopeThreshold: -0.5
    };
    // One flat GRASS edge costs exactly 1. Two rounds reach cell 2, leaving cell 3 pending;
    // five advances and one unchanged pass finish. Other-model parameters are one mismatch.
    const expected =
      scene === 'partial' ? [2, 0, 1, 0] : scene === 'complete' ? [6, 1, 0, 0] : [1, 1, 0, 1];
    for (const Operation of [GPURasterWalkingCostOperation, CPURasterWalkingCostOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          elevation: {format: 'float32', values: new Float32Array(width)},
          friction: {format: 'float32', values: new Float32Array(width)},
          sources: {
            format: 'uint32',
            values: Uint32Array.from({length: width}, (_, cell) => Number(cell === 0))
          },
          parameters: {
            format: 'float32',
            values: getGPURasterWalkingCostParameterValues(
              {model: scene === 'mismatch' ? {kind: 'tobler'} : model},
              grid
            )
          }
        },
        operations: [
          new Operation({
            grid,
            model,
            maximumIterations: scene === 'partial' ? 1 : 8,
            elevation: {
              id: 'elevation',
              format: 'float32',
              storage: {kind: 'buffer', values: 'elevation'}
            },
            friction: {
              id: 'friction',
              format: 'float32',
              storage: {kind: 'buffer', values: 'friction'}
            },
            sources: 'sources',
            parameters: 'parameters',
            output: {
              costs: 'costs',
              validity: 'validity',
              status: {
                iterationCount: 'count',
                converged: 'done',
                iterationLimitReached: 'limit',
                invalidCount: 'invalid'
              }
            }
          })
        ],
        outputs: ['costs', 'validity', 'count', 'done', 'limit', 'invalid']
      });
      expect(['count', 'done', 'limit', 'invalid'].map(name => result[name][0])).toEqual(expected);
      expect(Array.from(result['validity'])).toEqual(new Array(width).fill(1));
      expect(Array.from(result['costs'])).toEqual(
        scene === 'partial'
          ? [0, 1, 2, Infinity, Infinity, Infinity]
          : scene === 'complete'
            ? [0, 1, 2, 3, 4, 5]
            : [0]
      );
    }
  });
}
