// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';

/** Independent WGS84 midpoint metric, evaluated in /usr/bin/python3 with math:
 * e2=f*(2-f), f=1/298.257223563, a=6378137, D=1-e2*sin(phi)**2.
 * Geographic: phi=radians(45.5), area=(a*pi/180)**2*(1-e2)*cos(phi)/D**2.
 * Mercator: phi=atan(sinh(4600000.5/a)), area=(1-e2)*cos(phi)**2/D**2.
 */
export const AREA_PACKET_CASES = [
  {cellSizeMode: 'geographic', originY: 45, area: 8686608245.953476},
  {cellSizeMode: 'web-mercator', originY: 4600000, area: 0.6175050465639352}
] as const;

export function createAreaPackets(originY: number) {
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, originY] as const,
    pixelInterpretation: 'area' as const
  };
  const legacy = Float32Array.of(0, originY, 1, 0, 0, 1, 1, 123456);
  const rebased = getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [0, originY]});
  rebased[7] = 123456; // Word 7 is padding even in the explicitly discriminated extension.
  // Compile at the equator; every packet places the measured row elsewhere.
  const constructionGrid = {...grid, affine: [1, 0, 0, 0, 1, 0] as const};
  return {grid: constructionGrid, packets: [legacy.subarray(0, 7), legacy, rebased]};
}
