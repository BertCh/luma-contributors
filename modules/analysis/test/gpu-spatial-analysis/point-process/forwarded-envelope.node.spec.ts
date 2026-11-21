// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  computePointPatternEnvelopeOnCPU,
  CPUPointPatternEnvelopeOperation
} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {GPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {getForwardedEnvelopeCase, MAXIMUM_DEVIATION_CASES} from './forwarded-envelope-fixtures';

for (const {reference, probability} of MAXIMUM_DEVIATION_CASES) {
  it(`TWN-03 CPU maximum deviation with reference ${reference}`, () => {
    const {props, inputs} = getForwardedEnvelopeCase(false);
    const result = computePointPatternEnvelopeOnCPU(
      {
        observed: inputs.observed.values,
        simulations: inputs.simulations.values,
        reference: Float32Array.of(reference)
      },
      props
    );
    expect(Array.from(result.status)).toEqual([0]);
    expect(Array.from(result.statistics)).toEqual([1, probability, 0.125, 7]);
  });
}

for (const integrated of [false, true]) {
  it(`forwarded envelope independent inference and shader audit: integrated=${integrated}`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const {props, inputs} = getForwardedEnvelopeCase(integrated);
      const result = await runOperationGraph(device, {
        inputs,
        operations: [new CPUPointPatternEnvelopeOperation(props)],
        outputs: Object.values(props.output) as string[]
      });
      // Maximum: observed 1+2^-26 > simulated 1-2^-26, giving zero of seven exceedances.
      // Integrated: 1 + 2^-149*(2^63)^2 = 1+2^-23 > 1, zero of three exceedances.
      expect(Array.from(result['statistics'])).toEqual(
        integrated ? [1 + 2 ** -23, 0.25, 0.25, 3] : [1, 0.125, 0.125, 7]
      );
      expect(Array.from(result['status'])).toEqual([0]);
      const graph = new GPUCommandGraph(device);
      const getView = createOperationTestViewGetter(graph, {
        observed: {format: 'float32', length: props.curveRowCount},
        simulations: {format: 'float32', length: props.curveRowCount * props.simulationCount},
        reference: {format: 'float32', length: props.curveRowCount},
        ...(integrated
          ? {integrationWeights: {format: 'float32' as const, length: props.curveRowCount}}
          : {}),
        lower: {format: 'float32', length: props.curveRowCount},
        upper: {format: 'float32', length: props.curveRowCount},
        statistics: {format: 'float32', length: 4},
        status: {format: 'uint32', length: 1}
      });
      const nodes = new GPUPointPatternEnvelopeOperation(props)
        .createContributor(getView)
        .getCommandNodes(graph);
      for (const {id, source, constants} of getParsedKernelSources(nodes).configurations) {
        expect(auditWGSLModule(source, constants), id).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  });
}
