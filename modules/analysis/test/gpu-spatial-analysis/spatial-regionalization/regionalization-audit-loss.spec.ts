// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {CPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createAuditAZPInput, createAuditAZPProps} from './regionalization-audit-azp-fixtures';
import {createAuditMaxPInput, createAuditMaxPProps} from './regionalization-audit-max-p-fixtures';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';

for (const name of ['overflow', 'candidateOverflow'] as const) {
  it(`suppresses every dependent AZP/Max-P CPU/GPU output on adjacency ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const maxP of [false, true]) {
      const baseInput = maxP ? createAuditMaxPInput() : createAuditAZPInput();
      const props = maxP ? createAuditMaxPProps() : createAuditAZPProps();
      const inputs = getRegionalizationOperationInputs({
        ...baseInput,
        status: {[name]: Uint32Array.of(1)}
      });
      const weights = {
        ...props.weights,
        status: {[name]: name === 'overflow' ? 'weightsOverflow' : 'weightsCandidateOverflow'}
      };
      const operations = maxP
        ? [
            new GPUMaxPRegionsOperation({...createAuditMaxPProps(), weights}),
            new CPUMaxPRegionsOperation({...createAuditMaxPProps(), weights})
          ]
        : [
            new GPUAZPRegionsOperation({...createAuditAZPProps(), weights}),
            new CPUAZPRegionsOperation({...createAuditAZPProps(), weights})
          ];
      for (const operation of operations) {
        const result = await runOperationGraph(device, {
          inputs,
          operations: [operation],
          outputs: ['labels', 'objective', 'status']
        });
        expect(Array.from(result['labels'])).toEqual(new Array(props.rowCount).fill(4294967295));
        expect(result['objective']).toHaveLength(1);
        expect(Number.isNaN(result['objective'][0])).toBe(true);
        expect(Array.from(result['status'])).toEqual(new Array(maxP ? 6 : 4).fill(0));
      }
    }
  });
}
