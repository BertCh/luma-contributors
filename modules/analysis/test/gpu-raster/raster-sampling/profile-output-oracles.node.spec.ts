// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {profileRasterOnCPU, packProfileCase} from './raster-sampling-oracle';
import {
  PROFILE_VALUES,
  PROFILE_POSITIONS,
  PROFILE_EXPECTED
} from './profile-output-oracles-fixture';

it('profile CPU mirror matches independently hand-counted sample columns', () => {
  const result = profileRasterOnCPU(
    {width: 4, height: 2, values: PROFILE_VALUES},
    packProfileCase({width: 4, height: 2, extent: [0, 0, 4, 2], spacing: 1, method: 'nearest'}),
    PROFILE_POSITIONS,
    new Uint32Array([0, 2, 4])
  );
  for (const name of Object.keys(PROFILE_EXPECTED) as (keyof typeof PROFILE_EXPECTED)[])
    expectLiteralRows(result[name], PROFILE_EXPECTED[name], 0);
});
