// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSinkhornDivergenceOperation} from '../../../src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation';
import {
  GPUSinkhornDivergenceOperation,
  type SinkhornDivergenceOperationProps
} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeSinkhornDivergenceReference} from './stm09-sinkhorn-reference';

// Log masses and dual potentials round at the unit cost/epsilon scale in this two-cell case.
// Allow four float32 unit roundoffs for components, still over 3,000 times below the old
// premature-stop self-cost error (8.8e-4). The divergence retains its tighter bound below.
const TWO_CELL_COMPONENT_TOLERANCE = 4 * 2 ** -24;

it('Sinkhorn operations retain distributions with overflowing raw float32 totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = {
    width: 2,
    height: 1,
    pixelInterpretation: 'area' as const,
    affine: [1, 0, 0, 0, -1, 0] as [number, number, number, number, number, number]
  };
  const settings = {cellWidth: 1, cellHeight: 1, epsilon: 1, tolerance: 1e-6};
  const reference = computeSinkhornDivergenceReference({
    ...grid,
    ...settings,
    source: [3, 2],
    target: [2, 3]
  });
  const props: SinkhornDivergenceOperationProps = {
    grid,
    source: 'source',
    target: 'target',
    parameters: 'parameters',
    maximumIterations: 100,
    output: {
      divergence: 'divergence',
      components: 'components',
      marginalError: 'error',
      status: {converged: 'converged', iterationLimitReached: 'limited', invalidCount: 'invalid'}
    }
  };
  for (const Operation of [CPUSinkhornDivergenceOperation, GPUSinkhornDivergenceOperation]) {
    for (const scale of [1, 2 ** 126]) {
      const result = await runOperationGraph(device, {
        inputs: {
          source: {format: 'float32', values: Float32Array.of(3 * scale, 2 * scale)},
          target: {format: 'float32', values: Float32Array.of(2 * scale, 3 * scale)},
          parameters: {format: 'float32', values: getGPUSinkhornDivergenceParameterValues(settings)}
        },
        outputs: ['divergence', 'components', 'error', 'converged', 'limited', 'invalid'],
        operations: [new Operation(props)]
      });
      expect(result['converged'][0]).toBe(1);
      expect(result['limited'][0]).toBe(0);
      expect(result['invalid'][0]).toBe(0);
      expect(Number.isFinite(result['error'][0])).toBe(true);
      expect(result['error'][0]).toBeLessThanOrEqual(1e-6);
      expect(Number.isFinite(result['divergence'][0])).toBe(true);
      expect(Math.abs(result['divergence'][0] - reference.divergence)).toBeLessThan(1e-6);
      [reference.crossCost, reference.sourceCost, reference.targetCost].forEach(
        (expected, index) => {
          expect(Number.isFinite(result['components'][index])).toBe(true);
          expect(Math.abs(result['components'][index] - expected)).toBeLessThan(1e-6);
        }
      );
    }
  }
});

for (const scenario of [
  'tiny-mass-limited',
  'tiny-mass-converged',
  'constant-epsilon',
  'decaying-epsilon',
  'unreachable-target'
] as const) {
  it(`Sinkhorn operation status preserves the residual contract: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const tinyMass = scenario.startsWith('tiny-mass');
    const unreachableTarget = scenario === 'unreachable-target';
    const maximumIterations = scenario === 'tiny-mass-converged' ? 3 : 2;
    const shouldConverge = scenario === 'tiny-mass-converged' || scenario === 'decaying-epsilon';
    const props: SinkhornDivergenceOperationProps = {
      grid: {
        width: tinyMass ? 2 : unreachableTarget ? 3 : 1,
        height: 1,
        pixelInterpretation: 'area',
        affine: [1, 0, 0, 0, -1, 0]
      },
      source: 'source',
      target: 'target',
      parameters: 'parameters',
      maximumIterations,
      blurRadius: unreachableTarget ? 1 : undefined,
      output: {
        divergence: 'divergence',
        marginalError: 'error',
        status: {
          converged: 'converged',
          iterationLimitReached: 'limited',
          iterationCount: 'iterations',
          approximation: 'approximation'
        }
      }
    };
    // The CPU twin always uses the full kernel; the unreachable window is GPU-only.
    const operations = unreachableTarget
      ? [GPUSinkhornDivergenceOperation]
      : [CPUSinkhornDivergenceOperation, GPUSinkhornDivergenceOperation];
    for (const Operation of operations) {
      const result = await runOperationGraph(device, {
        inputs: {
          source: {
            format: 'float32',
            values: Float32Array.from(tinyMass ? [1, 1e-37] : unreachableTarget ? [1, 0, 0] : [1])
          },
          target: {
            format: 'float32',
            values: Float32Array.from(tinyMass ? [1, 0] : unreachableTarget ? [0, 0, 1] : [1])
          },
          parameters: {
            format: 'float32',
            values: getGPUSinkhornDivergenceParameterValues({
              cellWidth: tinyMass ? 16 : 1,
              cellHeight: 1,
              epsilon: 1,
              epsilonStart: tinyMass ? 2 : unreachableTarget ? 1 : 1 + 2 ** -23,
              scalingFactor: tinyMass || scenario === 'decaying-epsilon' ? 0.5 : 1,
              tolerance: tinyMass ? 0.01 : 1e-4
            })
          }
        },
        outputs: ['error', 'converged', 'limited', 'iterations', 'approximation'],
        operations: [new Operation(props)]
      });
      expect(result['converged'][0]).toBe(Number(shouldConverge));
      expect(result['limited'][0]).toBe(Number(!shouldConverge));
      expect(result['iterations'][0]).toBe(maximumIterations);
      expect(result['approximation'][0]).toBe(Number(unreachableTarget));
      if (unreachableTarget) expect(result['error'][0]).toBe(1);
    }
  });
}

for (const reverse of [false, true]) {
  it(`Sinkhorn operations require both two-cell self plans to converge (reverse=${reverse})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = {
      width: 2,
      height: 1,
      pixelInterpretation: 'area' as const,
      affine: [1, 0, 0, 0, -1, 0] as [number, number, number, number, number, number]
    };
    const source = reverse ? [1, 0] : [99, 1];
    const target = reverse ? [99, 1] : [1, 0];
    const settings = {cellWidth: 1, cellHeight: 1, epsilon: 1, tolerance: 1e-6};
    const reference = computeSinkhornDivergenceReference(
      {...grid, ...settings, source, target},
      1e-13
    );
    for (const maximumIterations of [2, 100]) {
      const props: SinkhornDivergenceOperationProps = {
        grid,
        source: 'source',
        target: 'target',
        parameters: 'parameters',
        maximumIterations,
        output: {
          divergence: 'divergence',
          components: 'components',
          marginalError: 'error',
          status: {converged: 'converged', iterationLimitReached: 'limited'}
        }
      };
      for (const Operation of [CPUSinkhornDivergenceOperation, GPUSinkhornDivergenceOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            source: {format: 'float32', values: Float32Array.from(source)},
            target: {format: 'float32', values: Float32Array.from(target)},
            parameters: {
              format: 'float32',
              values: getGPUSinkhornDivergenceParameterValues(settings)
            }
          },
          outputs: ['divergence', 'components', 'error', 'converged', 'limited'],
          operations: [new Operation(props)]
        });
        expect(result['error'][0]).toBeLessThan(1e-6);
        expect(result['converged'][0]).toBe(maximumIterations === 2 ? 0 : 1);
        expect(result['limited'][0]).toBe(maximumIterations === 2 ? 1 : 0);
        if (maximumIterations === 100) {
          expect(Number.isFinite(result['divergence'][0])).toBe(true);
          expect(Math.abs(result['divergence'][0] - reference.divergence)).toBeLessThan(1e-7);
          [reference.crossCost, reference.sourceCost, reference.targetCost].forEach(
            (expected, index) => {
              expect(Number.isFinite(result['components'][index])).toBe(true);
              expect(Math.abs(result['components'][index] - expected)).toBeLessThan(
                TWO_CELL_COMPONENT_TOLERANCE
              );
            }
          );
        }
      }
    }
  });
}

for (const frame of ['ordinary', 'limited', 'empty', 'bad-parameters'] as const) {
  it(`Sinkhorn operations agree on every value and diagnostic: ${frame}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SinkhornDivergenceOperationProps = {
      grid: {width: 2, height: 2, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      source: 'source',
      target: 'target',
      parameters: 'parameters',
      maximumIterations: frame === 'limited' ? 1 : 50,
      output: {
        divergence: 'divergence',
        components: 'components',
        marginalError: 'error',
        status: {
          converged: 'converged',
          iterationLimitReached: 'limited',
          iterationCount: 'iterations',
          invalidCount: 'invalid',
          approximation: 'approximation'
        }
      }
    };
    const inputs = {
      source: {format: 'float32' as const, values: new Float32Array([1, 2, 3, 4])},
      target: {
        format: 'float32' as const,
        values: frame === 'empty' ? new Float32Array(4) : new Float32Array([4, 1, 3, 2])
      },
      parameters: {
        format: 'float32' as const,
        values:
          frame === 'bad-parameters'
            ? new Float32Array(6)
            : getGPUSinkhornDivergenceParameterValues({
                cellWidth: 1.5,
                cellHeight: 0.75,
                epsilon: 2,
                tolerance: 0.002,
                epsilonStart: 4,
                scalingFactor: 0.5
              })
      }
    };
    const outputs = [
      'divergence',
      'components',
      'error',
      'converged',
      'limited',
      'iterations',
      'invalid',
      'approximation'
    ];
    const cpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new CPUSinkhornDivergenceOperation(props)]
    });
    const gpu = await runOperationGraph(device, {
      inputs,
      outputs,
      operations: [new GPUSinkhornDivergenceOperation(props)]
    });
    // Positive values: relative 3e-4. The convergence residual approaches zero and has absolute
    // tolerance 3e-6. All integer diagnostics are exact for these threshold-separated scenes.
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let index = 0; index < cpu[output].length; index++) {
        const expected = cpu[output][index];
        if (Number.isNaN(expected)) expect(gpu[output][index]).toBeNaN();
        else if (output === 'error')
          expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-6);
        else if (output === 'divergence' || output === 'components')
          expect(Math.abs(gpu[output][index] - expected)).toBeLessThan(3e-4 * Math.abs(expected));
        else expect(gpu[output][index]).toBe(expected);
      }
    }
    if (frame === 'ordinary') expect(cpu['converged'][0]).toBe(1);
    if (frame === 'limited') expect(cpu['limited'][0]).toBe(1);
  });
}
