// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {checkTerrainCPUStatus} from '../terrain-status-oracles-test-utils';
import {getTerrainProcessingStatusOracles} from './terrain-processing-status-oracles-fixtures';

for (const oracle of getTerrainProcessingStatusOracles()) {
  it(`CPU terrain processing independent status oracle: ${oracle.name}`, async () => {
    await checkTerrainCPUStatus(oracle);
  });
}
