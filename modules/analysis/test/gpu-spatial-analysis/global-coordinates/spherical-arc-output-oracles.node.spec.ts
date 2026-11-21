// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {computeSphericalArcOverlayOnCPU} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {expectOctant, OCTANT_POSITIONS, OCTANT_PROPS} from './spherical-arc-output-oracles-fixture';

it('CPU arc overlay preserves the independently defined octant ring', () => {
  const operand = {
    positions: Float32Array.from(OCTANT_POSITIONS),
    ringOffsets: Uint32Array.of(0, 3),
    referenceInside: false
  };
  expectOctant(computeSphericalArcOverlayOnCPU({left: operand, right: operand}, OCTANT_PROPS));
});
