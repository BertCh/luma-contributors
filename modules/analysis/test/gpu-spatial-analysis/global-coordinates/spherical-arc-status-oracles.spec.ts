// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalArcOverlayOperation} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';
import {
  GPUSphericalArcOverlay,
  type GPUSphericalArcOverlayProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addOracleOperation, createOracleFixture} from './output-oracle-fixture';
import {createArcStatusScenes} from './spherical-arc-status-oracles-fixture';

it('arc caller replacement cannot change the validated square arrangement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const scene = createArcStatusScenes().find(
    entry => entry.name === 'coincident union at capacity four'
  )!;
  const counts = fixture.output('uint32', 4);
  const status = fixture.output('uint32', 1);
  const positions = fixture.output('float32x3', 4);
  const ringOffsets = fixture.output('uint32', 2);
  const referencePoint: [number, number, number] = [0, 0, -1];
  const props: GPUSphericalArcOverlayProps = {
    left: {
      positions: fixture.input('float32x3', Array.from(scene.input.left.positions)),
      ringOffsets: fixture.input('uint32', [0, 4]),
      referenceInside: false
    },
    right: {
      positions: fixture.input('float32x3', Array.from(scene.input.right.positions)),
      ringOffsets: fixture.input('uint32', [0, 4]),
      referenceInside: false
    },
    operation: 'union',
    referencePoint,
    output: {
      positions: positions.view,
      ringOffsets: ringOffsets.view,
      counts: counts.view,
      status: status.view
    }
  };
  const contributor = new GPUSphericalArcOverlay(props);
  // Replacements describe invalid zero vectors and change all semantic settings.
  // They must not alter the originally validated union of the identical squares.
  props.left.positions = fixture.input('float32x3', Array(65 * 3).fill(0));
  props.left.referenceInside = true;
  props.operation = 'difference';
  referencePoint[2] = 1;
  fixture.graph.add(contributor);
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [counts, status, positions, ringOffsets]) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await counts.read()).toEqual([4, 1, 4, 0]);
      expect(await status.read()).toEqual([0]);
      expect(await ringOffsets.read()).toEqual([0, 4]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each(
  createArcStatusScenes()
)('GPU arc status and all four counts overwrite poison: $name', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const operation = new GPUSphericalArcOverlayOperation({
    ...scene.options,
    left: {
      positions: 'leftPositions',
      ringOffsets: 'leftOffsets',
      referenceInside: scene.input.left.referenceInside
    },
    right: {
      positions: 'rightPositions',
      ringOffsets: 'rightOffsets',
      referenceInside: scene.input.right.referenceInside
    },
    output: {positions: 'positions', ringOffsets: 'ringOffsets', counts: 'counts', status: 'status'}
  });
  // This is the operation's deferred GPUSphericalArcOverlay, with caller-owned destinations.
  const outputs = addOracleOperation(fixture, operation, {
    leftPositions: fixture.input('float32x3', Array.from(scene.input.left.positions)),
    rightPositions: fixture.input('float32x3', Array.from(scene.input.right.positions)),
    leftRingOffsets: fixture.input('uint32', Array.from(scene.input.left.ringOffsets)),
    rightRingOffsets: fixture.input('uint32', Array.from(scene.input.right.ringOffsets))
  });
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of Object.values(outputs)) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await outputs['counts'].read()).toEqual(scene.counts);
      expect(await outputs['status'].read()).toEqual(scene.status);
      if (scene.ringOffsets) expect(await outputs['ringOffsets'].read()).toEqual(scene.ringOffsets);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
