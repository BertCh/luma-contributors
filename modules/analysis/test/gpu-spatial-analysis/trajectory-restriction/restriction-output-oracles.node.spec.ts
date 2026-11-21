// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {computeTrajectoryRestrictionOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation';
import {
  createRestrictionOracleScene,
  expectRestrictionCompact
} from './restriction-output-oracles-fixture';

it('restriction compaction matches GEOS and hand-derived source rows and status bits', () => {
  const scene = createRestrictionOracleScene();
  expectRestrictionCompact(computeTrajectoryRestrictionOnCPU(scene.input, scene.options));
});
