// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it} from 'vitest';
import {CPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {expectOracleRows, getOracleCPUVectors} from '../alternative-routes/output-oracle-harness';
import {
  createStructureOracleProps,
  structureOracleExpected,
  structureOracleInputs,
  structureOracleSelections
} from './network-structure-output-oracles';

for (const mode of ['exact', 'parallel'] as const)
  it(`pins triangle, bridge, loop and unused slots for structure (${mode})`, async () => {
    for (const selected of structureOracleSelections) {
      const props = createStructureOracleProps(mode, selected);
      const result = await new CPUNetworkStructureOperation(props).execute({
        inputs: getOracleCPUVectors(structureOracleInputs)
      });
      for (const port of Object.keys(props.output) as (keyof typeof structureOracleExpected)[])
        expectOracleRows(result[port] as Uint32Array, structureOracleExpected[port]);
    }
  });
