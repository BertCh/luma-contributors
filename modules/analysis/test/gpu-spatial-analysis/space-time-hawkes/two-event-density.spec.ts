// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {expectHawkesDensityOutputs} from './hawkes-density-assertions';
import {TWO_EVENT_DENSITY} from './two-event-density-fixture';

it('PTS-140 evaluates both densities with a nonzero parent through GPU public forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const positions = Float32Array.of(0, 0, 1, 0),
      times = Float32Array.of(0, 1);
    const parameters = Float32Array.of(1, 2, 2, 2, 1e-8, 1, 0.5, 1, 1);
    const output = {
      estimates: rig.output('float32', 4),
      logLikelihood: rig.output('float32', 1),
      fittedLogLikelihood: rig.output('float32', 1),
      backgroundProbability: rig.output('float32', 2),
      status: {
        converged: rig.output('uint32', 1),
        iterationLimitReached: rig.output('uint32', 1),
        iterationCount: rig.output('uint32', 1),
        candidateOverflow: rig.output('uint32', 1),
        invalidCount: rig.output('uint32', 1),
        approximation: rig.output('uint32', 1)
      }
    };
    rig.run(
      new GPUSpaceTimeHawkes({
        positions: rig.input(positions, 'float32x2', 2),
        times: rig.input(times, 'float32'),
        parameters: rig.input(parameters, 'float32'),
        maximumIterations: 1,
        maximumParents: 2,
        output
      })
    );
    const direct: Record<string, number[]> = {};
    for (const [port, view] of Object.entries(output.status))
      direct[port] = await rig.readUint(view);
    direct['estimates'] = await rig.readFloat(output.estimates);
    direct['logLikelihood'] = await rig.readFloat(output.logLikelihood);
    direct['fittedLogLikelihood'] = await rig.readFloat(output.fittedLogLikelihood);
    direct['backgroundProbability'] = await rig.readFloat(output.backgroundProbability);
    const operation = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: positions},
        times: {format: 'float32', values: times},
        parameters: {format: 'float32', values: parameters}
      },
      operations: [
        new GPUSpaceTimeHawkesOperation({
          id: 'density',
          rowCount: 2,
          maximumIterations: 1,
          maximumParents: 2,
          positions: 'positions',
          times: 'times',
          parameters: 'parameters',
          output: {
            estimates: 'estimates',
            logLikelihood: 'logLikelihood',
            fittedLogLikelihood: 'fittedLogLikelihood',
            backgroundProbability: 'backgroundProbability',
            status: {
              converged: 'converged',
              iterationLimitReached: 'iterationLimitReached',
              iterationCount: 'iterationCount',
              candidateOverflow: 'candidateOverflow',
              invalidCount: 'invalidCount',
              approximation: 'approximation'
            }
          }
        })
      ],
      outputs: Object.keys(TWO_EVENT_DENSITY)
    });
    for (const actual of [direct, operation]) expectHawkesDensityOutputs(actual, 2e-6);
  } finally {
    rig.destroy();
  }
});
