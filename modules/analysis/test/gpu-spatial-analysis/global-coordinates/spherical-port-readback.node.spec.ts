// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as spherical from '../../../src/gpu-spatial-analysis/global-coordinates';

it('decodes spherical count and status ports together from byte readbacks', () => {
  expect(spherical).toHaveProperty('decodeGPUSphericalConvexIntersectionReadback');
  expect(spherical).toHaveProperty('decodeGPUSphericalArcOverlayReadback');
  const convex = spherical.decodeGPUSphericalConvexIntersectionReadback({
    counts: Uint32Array.of(4, 4).buffer,
    status: new Uint8Array(Uint32Array.of(0).buffer)
  });
  expect(convex.counts).toEqual({writtenVertexCount: 4, requiredVertexCount: 4});
  expect(convex.status.complete).toBe(true);
  const overlay = spherical.decodeGPUSphericalArcOverlayReadback({
    counts: Uint32Array.of(3, 1, 5, 1),
    status: Uint32Array.of(2).buffer
  });
  expect(overlay.counts).toEqual({
    writtenVertexCount: 3,
    writtenRingCount: 1,
    requiredSegmentCount: 5,
    referenceInside: true
  });
  expect(overlay.status.complete).toBe(false);
  expect(overlay.status.outputCapacityExceeded).toBe(true);
  expect(() =>
    spherical.decodeGPUSphericalConvexIntersectionReadback({
      counts: Uint32Array.of(4),
      status: Uint32Array.of(0)
    })
  ).toThrow();
  expect(() =>
    spherical.decodeGPUSphericalArcOverlayReadback({
      counts: Uint32Array.of(3, 1, 5, 1),
      status: new Uint8Array(3)
    })
  ).toThrow();
});
