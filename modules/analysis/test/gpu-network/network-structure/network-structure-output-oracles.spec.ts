// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectOracleRows, runPoisonedOperation} from '../alternative-routes/output-oracle-harness';
import {
  createStructureOracleProps,
  structureOracleExpected,
  structureOracleInputs,
  structureOracleSelections
} from './network-structure-output-oracles';

for (const mode of ['exact', 'parallel'] as const)
  it(`pins poisoned resilience ports together, alone and absent (${mode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const selected of structureOracleSelections) {
      const props = createStructureOracleProps(mode, selected);
      const result = await runPoisonedOperation(
        device,
        new GPUNetworkStructureOperation(props),
        structureOracleInputs
      );
      for (const port of Object.keys(props.output) as (keyof typeof structureOracleExpected)[])
        expectOracleRows(result[port], structureOracleExpected[port]);
    }
  });
