// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import type {GPUSpatialFrame} from '../../src/analysis-core/contracts/spatial-context';
import {getGPUProjectionFrameOrigin} from '../../src/geospatial-analysis/projection-spatial-context';
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

function makeFrame(crs: string, extra: Partial<GPUSpatialFrame> = {}): GPUSpatialFrame {
  return {...createGPUSpatialFrameFromCRS(crs), ...extra};
}

describe('projection spatial frames', () => {
  it('keeps the computed origin when no frame origin applies', () => {
    const baseline = requireReady(prepareCRSProjection(options));
    const noOrigin = requireReady(
      prepareCRSProjection({...options, targetFrame: makeFrame('EPSG:3857')})
    );
    const projectionCenter = requireReady(
      prepareCRSProjection({
        ...options,
        targetFrame: makeFrame('EPSG:3857', {origin: [5, 6], anchor: 'projection-center'})
      })
    );
    expect(noOrigin.program.destinationOrigin).toEqual(baseline.program.destinationOrigin);
    expect(projectionCenter.program.destinationOrigin).toEqual(baseline.program.destinationOrigin);
  });

  it('honors precision float32-origin-relative', () => {
    expect(
      getGPUProjectionFrameOrigin(
        makeFrame('EPSG:3857', {origin: [1, 2], precision: 'float32-origin-relative'})
      )
    ).toEqual([1, 2]);
    expect(getGPUProjectionFrameOrigin(makeFrame('EPSG:3857'))).toBeUndefined();
  });
});
