// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {prepareCRSProjection} from '@luma.gl/graph/gpu-project/crs';

const options = {
  from: 'EPSG:4326',
  to: 'EPSG:3857',
  bounds: [-1, -1, 1, 1] as const,
  tolerance: 0.001
};

function requireReady(result: ReturnType<typeof prepareCRSProjection>) {
  if (result.status !== 'ready') throw new Error(JSON.stringify(result.reasons));
  return result;
}

describe('projection spatial frames', () => {
  it('keeps the computed origin when no destination origin is given', () => {
    const baseline = requireReady(prepareCRSProjection(options));
    const noOrigin = requireReady(prepareCRSProjection({...options, destinationOrigin: undefined}));
    expect(noOrigin.program.destinationOrigin).toEqual(baseline.program.destinationOrigin);
  });
});
