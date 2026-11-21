// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createSphericalReviewFixture,
  createPlanarIdentityFixture
} from './spherical-overlay-review-fixture';
import {createOracleFixture} from './output-oracle-fixture';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  SPHERICAL_OVERLAY_DISPLAY_COUNT_WGSL,
  getSphericalOverlayStatusLabel,
  selectSphericalOverlayLayers
} from '../../../../../website/src/gpu/runs/entries/cells-index-join/spherical-polygon-overlay-status';

it.each([
  [1, 1, 0, 0],
  [0, 0, 1, 0],
  [1, 0, 0, 1]
])('website suppresses incomplete result count and layers (%s,%s,%s)', async (count, overflow, candidateOverflow, expected) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const displayCount = fixture.output('uint32', 1);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: 'website-display-count',
      operation: 'spherical-overlay-display',
      invocationCount: 1,
      bindings: [
        {name: 'count', view: fixture.input('uint32', [count]), type: 'u32', access: 'read'},
        {name: 'overflow', view: fixture.input('uint32', [overflow]), type: 'u32', access: 'read'},
        {
          name: 'candidateOverflow',
          view: fixture.input('uint32', [candidateOverflow]),
          type: 'u32',
          access: 'read'
        },
        {name: 'displayCount', view: displayCount.view, type: 'u32', access: 'read_write'}
      ],
      body: SPHERICAL_OVERLAY_DISPLAY_COUNT_WGSL
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await displayCount.read()).toEqual([expected]);
    expect(
      selectSphericalOverlayLayers(
        {overflow, candidateOverflow},
        ['left', 'right'],
        ['fill', 'outline']
      )
    ).toEqual(expected ? ['left', 'right', 'fill', 'outline'] : ['left', 'right']);
    expect(getSphericalOverlayStatusLabel({overflow, candidateOverflow})).toBe(
      expected
        ? ''
        : overflow
          ? 'incomplete: output capacity'
          : 'incomplete: input or precision loss'
    );
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('separate complement and feature-source writers preserve disjoint words in one allocation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSphericalReviewFixture(device, {
    ringCapacity: 1,
    leftRow: 1,
    rightRow: 1,
    packFeatureComplement: true
  });
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      fixture.outputs.sourceRows.poison();
      submitGraph(device, compiled, undefined);
      expect(await fixture.outputs.sourceRows.read()).toEqual([1]);
      expect(await fixture.outputs.complement.read()).toEqual([0]);
      expect(await fixture.outputs.overflow.read()).toEqual([0]);
      const positions = await fixture.outputs.positions.read();
      // Lifting chart vertex (1,0) gives (1/sqrt(2),0,1/sqrt(2)); 2e-5 covers float32 chart snapping.
      expect(positions.slice(0, 9).every(Number.isFinite)).toBe(true);
      const vertices = Array.from({length: 3}, (_, row) => positions.slice(row * 3, row * 3 + 3));
      expect(
        vertices.some(
          point =>
            Math.abs(point[0] - Math.SQRT1_2) < 2e-5 &&
            Math.abs(point[1]) < 2e-5 &&
            Math.abs(point[2] - Math.SQRT1_2) < 2e-5
        )
      ).toBe(true);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('planar tagged triangle publishes directed caller IDs and representative feature row one', async () => {
  // The intersection satisfies x>=1, y>=0, x+y<=2. Its bottom and diagonal edges
  // belong to left ID 101; x=1 belongs to right ID 202. Both source features are row 1.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createPlanarIdentityFixture(device);
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.outputs.boundaryCount.read()).toEqual([3]);
    const edges = await fixture.outputs.endpoints.read(),
      operands = await fixture.outputs.operands.read(),
      owners = await fixture.outputs.owners.read();
    const actual = Array.from({length: 3}, (_, row) => [
      ...edges.slice(row * 4, row * 4 + 4),
      operands[row],
      owners[row]
    ]).sort((first, second) => first.join(',').localeCompare(second.join(',')));
    expect(actual).toEqual([
      [1, 0, 2, 0, 0, 101],
      [1, 1, 1, 0, 1, 202],
      [2, 0, 1, 1, 0, 101]
    ]);
    expect((await fixture.outputs.sourceRows.read())[0]).toBe(1);
    expect((await fixture.outputs.sourceOperands.read())[0]).toBe(0);
    expect((await fixture.outputs.mixed.read())[0]).toBe(1);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each([
  false,
  true
])('complemented intersection retains caller feature provenance (mixed=%s)', async mixed => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Complemented intersection is right minus left. Disjoint right triangle survives whole.
  // For overlap, its four-edge remainder has one original-left edge (row 2), which wins
  // the representative ordering over the three original-right edges (row 1).
  const fixture = createSphericalReviewFixture(device, {
    leftComplement: true,
    right: mixed
      ? [
          [1, 0],
          [3, 0],
          [1, 2]
        ]
      : [
          [3, 0],
          [4, 0],
          [3, 1]
        ],
    leftRow: mixed ? 2 : 0,
    rightRow: mixed ? 1 : 0
  });
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.outputs.overflow.read()).toEqual([0]);
    expect(await fixture.outputs.candidate.read()).toEqual([0]);
    expect(await fixture.outputs.boundaryCount.read()).toEqual([mixed ? 4 : 3]);
    const operands = (await fixture.outputs.operands.read()).slice(0, mixed ? 4 : 3).sort();
    expect(operands).toEqual(mixed ? [0, 1, 1, 1] : [1, 1, 1]);
    expect((await fixture.outputs.sourceOperands.read())[0]).toBe(mixed ? 0 : 1);
    expect((await fixture.outputs.sourceRows.read())[0]).toBe(mixed ? 2 : 0);
    expect((await fixture.outputs.mixed.read())[0]).toBe(Number(mixed));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('boundary source IDs distinguish caller identities 101/202 from representative rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSphericalReviewFixture(device, {leftIds: [101], rightIds: [202]});
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.outputs.boundaryCount.read()).toEqual([3]);
    const edges = await fixture.outputs.endpoints.read(),
      owners = await fixture.outputs.owners.read(),
      operands = await fixture.outputs.operands.read();
    const actual = Array.from({length: 3}, (_, row) => [
      ...edges.slice(row * 4, row * 4 + 4),
      operands[row],
      owners[row]
    ]).sort((first, second) => first.join(',').localeCompare(second.join(',')));
    expect(actual).toEqual([
      [1, 0, 2, 0, 0, 101],
      [1, 1, 1, 0, 1, 202],
      [2, 0, 1, 1, 0, 101]
    ]);
    expect((await fixture.outputs.sourceRows.read())[0]).toBe(0);
    expect((await fixture.outputs.sourceOperands.read())[0]).toBe(0);
    expect((await fixture.outputs.mixed.read())[0]).toBe(1);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('boundary capacity two writes two edges but requires three', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSphericalReviewFixture(device, {boundaryCapacity: 2});
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.outputs.boundaryCount.read()).toEqual([2]);
    expect(await fixture.outputs.boundaryRequired.read()).toEqual([3]);
    expect(await fixture.outputs.boundaryOverflow.read()).toEqual([1]);
    expect(await fixture.outputs.boundaryCandidate.read()).toEqual([0]);
    expect(await fixture.outputs.overflow.read()).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('one-ring union capacity writes one disjoint triangle and requires two rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSphericalReviewFixture(device, {
    operation: 'union',
    right: [
      [3, 0],
      [4, 0],
      [3, 1]
    ],
    ringCapacity: 1
  });
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await fixture.outputs.boundaryCount.read()).toEqual([6]);
    expect(await fixture.outputs.boundaryRequired.read()).toEqual([6]);
    expect(await fixture.outputs.ringCount.read()).toEqual([1]);
    expect(await fixture.outputs.requiredRingCount.read()).toEqual([2]);
    expect(await fixture.outputs.overflow.read()).toEqual([1]);
    expect(await fixture.outputs.candidate.read()).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('tagged triangle preserves operand columns packed at zero and 256 bytes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createSphericalReviewFixture(device, {
    leftIds: [101],
    rightIds: [202],
    packBoundary: true
  });
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const edges = await fixture.outputs.endpoints.read(),
      operands = await fixture.outputs.operands.read(),
      owners = await fixture.outputs.owners.read();
    const actual = Array.from({length: 3}, (_, row) => [
      ...edges.slice(row * 4, row * 4 + 4),
      operands[row],
      owners[row]
    ]).sort((first, second) => first.join(',').localeCompare(second.join(',')));
    expect(actual).toEqual([
      [1, 0, 2, 0, 0, 101],
      [1, 1, 1, 0, 1, 202],
      [2, 0, 1, 1, 0, 101]
    ]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
