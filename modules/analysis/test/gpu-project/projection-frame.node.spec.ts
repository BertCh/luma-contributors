// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import type {GPUSpatialFrame} from '../../src/analysis-core/contracts/spatial-context';
import {compileProjectionPlan} from '@luma.gl/graph/gpu-project';
import {
  getGPUProjectionFrameOrigin,
  getGPUProjectionSpatialContext
} from '../../src/geospatial-analysis/projection-spatial-context';
import {prepareCRSProjection, ProjectionTableTransform} from '@luma.gl/graph/gpu-project/crs';

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
  it('plans identically from frames and from CRS strings', () => {
    const fromStrings = requireReady(prepareCRSProjection(options));
    const fromFrames = requireReady(
      prepareCRSProjection({
        bounds: options.bounds,
        tolerance: options.tolerance,
        sourceFrame: makeFrame('EPSG:4326'),
        targetFrame: makeFrame('EPSG:3857')
      })
    );
    expect(fromFrames.program).toEqual(fromStrings.program);
    expect(fromFrames.compiled.inputFormat).toBe(fromStrings.compiled.inputFormat);
    expect(fromFrames.program.destinationOrigin).toEqual(fromStrings.program.destinationOrigin);
  });

  it('tags prepared transforms with the target frame context', () => {
    const targetFrame = makeFrame('EPSG:3857');
    const prepared = requireReady(
      prepareCRSProjection({...options, sourceFrame: undefined, targetFrame})
    );
    expect(prepared.spatialContext).toEqual(
      getGPUProjectionSpatialContext('EPSG:3857', targetFrame)
    );
    expect(prepared.spatialContext?.frame).toBe(targetFrame);
    expect(new ProjectionTableTransform(prepared).spatialContext).toBe(prepared.spatialContext);
    // String targets still tag, and match the frame-based context.
    expect(requireReady(prepareCRSProjection(options)).spatialContext?.frame).toMatchObject({
      kind: 'crs',
      crs: 'EPSG:3857'
    });
  });

  it('uses a relative-origin frame origin as the plan destination origin', () => {
    const origin: [number, number] = [1000, 2000];
    const targetFrame = makeFrame('EPSG:3857', {origin, anchor: 'relative-origin'});
    const withFrame = requireReady(prepareCRSProjection({...options, targetFrame}));
    const plan = withFrame.program.operations[0];
    if (plan.type !== 'adaptive') throw new Error('expected adaptive plan');
    expect(plan.plan.destinationOrigin).toEqual(origin);
    expect(withFrame.program.destinationOrigin).toEqual(origin);
    expect(plan.plan.destinationOrigin).not.toBe(origin);
    // Patch offsets are relative to the chosen origin, so patch records stay consistent.
    expect(plan.plan.patches[0].destinationOrigin[0]).not.toBe(origin[0]);
  });

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

  it('honors precision float32-origin-relative and an explicit compile origin', () => {
    expect(
      getGPUProjectionFrameOrigin(
        makeFrame('EPSG:3857', {origin: [1, 2], precision: 'float32-origin-relative'})
      )
    ).toEqual([1, 2]);
    expect(getGPUProjectionFrameOrigin(makeFrame('EPSG:3857'))).toBeUndefined();
    const plan = compileProjectionPlan({
      projection: ([x, y]) => [x + 10, y + 20],
      bounds: [0, 0, 2, 2],
      destinationOrigin: [7, 8]
    });
    expect(plan.destinationOrigin).toEqual([7, 8]);
  });
});
