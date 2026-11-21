// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {CPUSolarPositionOperation} from '../../../src/gpu-terrain/terrain-illumination/cpu-solar-position-operation';
import {
  GPUSolarPositionOperation,
  type SolarPositionOperationProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position-operation';
import {getGPUSolarPositionParameterValues} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expectCapturedSolarAltitude} from '../terrain-features/terrain-terminal-assertions';

const INPUTS = {
  id: 'solar-terminals',
  positions: 'positions',
  parameters: 'parameters',
  rowCount: 1
};

it('TSC-03 excludes an empty selection from the shared named output type', () => {
  expectTypeOf<{}>().not.toMatchTypeOf<SolarPositionOperationProps['output']>();
});

for (const Operation of [CPUSolarPositionOperation, GPUSolarPositionOperation]) {
  it(`${Operation.name} rejects an empty JavaScript output selection`, () => {
    expect(() => Reflect.construct(Operation, [{...INPUTS, output: {}}])).toThrow(
      /solar-terminals.*output/
    );
  });

  it(`${Operation.name} accepts each singleton and a combined selection`, () => {
    const selections: SolarPositionOperationProps['output'][] = [
      {azimuth: 'azimuth'},
      {altitude: 'altitude'},
      {daylight: 'daylight'},
      {azimuth: 'azimuth', altitude: 'altitude', daylight: 'daylight'}
    ];
    for (const output of selections) {
      const operation = new Operation({...INPUTS, output});
      const declaredOutputs = operation.contract.outputs;
      if (!declaredOutputs || typeof declaredOutputs === 'function')
        throw new Error('Expected explicit operation contract output declarations');
      expect(Object.keys(declaredOutputs)).toEqual(Object.values(output));
    }
  });
}

it('CPU solar operation captures its selected terminal, row count and refraction at construction', async () => {
  const props: SolarPositionOperationProps = {
    ...INPUTS,
    refraction: true,
    output: {altitude: 'result'}
  };
  const operation = new CPUSolarPositionOperation(props);
  props.rowCount = 2;
  props.output.altitude = 'changed';
  props.refraction = false;
  const device = createNullWebGPUDevice();
  try {
    const rows = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(-105.1786, 39.742476)},
        parameters: {
          format: 'float32',
          values: getGPUSolarPositionParameterValues({timestamp: Date.UTC(2003, 9, 17, 19, 30, 30)})
        }
      },
      operations: [operation],
      outputs: ['result']
    });
    const altitude = rows['result'];
    expect(altitude).toBeInstanceOf(Float32Array);
    if (!(altitude instanceof Float32Array)) throw new Error('Expected packed altitude rows');
    expectCapturedSolarAltitude(altitude);
    expect(Object.keys(rows)).toEqual(['result']);
  } finally {
    device.destroy();
  }
});
