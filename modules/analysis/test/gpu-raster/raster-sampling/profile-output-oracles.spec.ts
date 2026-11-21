// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterProfile,
  getGPURasterProfileParameterValues
} from '../../../src/raster-analysis/raster-sampling/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {it} from 'vitest';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  PROFILE_VALUES,
  PROFILE_POSITIONS,
  PROFILE_EXPECTED
} from './profile-output-oracles-fixture';

it('profile optional sample columns and extrema match hand-derived two-line ramps independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = Object.keys(PROFILE_EXPECTED) as (keyof typeof PROFILE_EXPECTED)[];
  const grid = makeGPURasterMetadata({width: 4, height: 2, origin: [0, 0], rowDirection: 'north'});
  for (const selected of [
    names,
    ...names.map(name => [name]),
    ...names.map(name => names.filter(other => other !== name))
  ]) {
    const fixture = new RecipeTestFixture(device, 'profile-output-oracle');
    const outputs = Object.fromEntries(
      selected.map(name => {
        const components = name === 'samplePositions' ? 2 : 1;
        const length = name.startsWith('sample') ? 10 : 2;
        const output = fixture.output(name, components === 2 ? 'float32x2' : 'float32', length);
        output.buffer.write(new Float32Array(length * components).fill(123));
        return [name, output];
      })
    );
    const count = fixture.output('count', 'uint32', 1),
      overflow = fixture.output('overflow', 'uint32', 1);
    count.buffer.write(new Uint32Array([123]));
    overflow.buffer.write(new Uint32Array([123]));
    fixture.graph.add(
      new GPURasterProfile({
        grid,
        sampleCapacity: 10,
        values: fixture.input('values', PROFILE_VALUES, 'float32', 8),
        positions: fixture.input('positions', PROFILE_POSITIONS, 'float32x2', 4),
        lineOffsets: fixture.input('offsets', new Uint32Array([0, 2, 4]), 'uint32', 3),
        parameters: fixture.input(
          'parameters',
          getGPURasterProfileParameterValues({spacing: 1, method: 'nearest'}, grid),
          'float32',
          12
        ),
        output: {
          ...Object.fromEntries(selected.map(name => [name, outputs[name].view])),
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    try {
      fixture.run();
      expectLiteralRows(await fixture.readUint32(count, 1), [8], 0);
      expectLiteralRows(await fixture.readUint32(overflow, 1), [0], 0);
      for (const name of selected) {
        // Positions pad NaN; cumulative columns explicitly leave unused capacity untouched.
        const expected = [
          ...PROFILE_EXPECTED[name],
          ...(name === 'samplePositions'
            ? [NaN, NaN, NaN, NaN]
            : name.startsWith('sample')
              ? [123, 123]
              : [])
        ];
        expectLiteralRows(await fixture.readFloat32(outputs[name], expected.length), expected, 0);
      }
    } finally {
      fixture.destroy();
    }
  }
});
