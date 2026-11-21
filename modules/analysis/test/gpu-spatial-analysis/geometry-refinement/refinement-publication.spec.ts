// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from './topology-output';

it('GPU loop-removal preserves whole-line first-fit capacity order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const positions = fixture.output('float32x2', 6);
  const lineOffsets = fixture.output('uint32', 4);
  const status = fixture.output('uint32x4', 1);
  fixture.graph.add(
    new GPULineLoopRemoval({
      lines: {
        kind: 'lines',
        positions: fixture.input(
          'float32x2',
          [0, 0, 1, 0, 2, 0, 3, 0, 0, 2, 1, 2, 2, 2, 3, 2, 0, 4, 1, 4]
        ),
        lineOffsets: fixture.input('uint32', [0, 4, 8, 10])
      },
      intersectionCapacity: 8,
      output: {
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        sourceEdges: fixture.output('uint32', 6).view,
        status: status.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await status.read()).toEqual([6, 10, 1, 0]);
    expect(await lineOffsets.read()).toEqual([0, 4, 4, 6]);
    expect(await positions.read()).toEqual([0, 0, 1, 0, 2, 0, 3, 0, 0, 4, 1, 4]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('GPU polygon precision reduces hundreds of empty rings to clean status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const lattice = fixture.output('float32', 4);
  const output = createTopologyOutput((format, length) => fixture.output(format, length).view);
  const count = fixture.output('uint32', 1);
  output.status.count = count.view;
  fixture.graph.add(
    new GPUPolygonPrecisionReduction({
      polygons: {
        kind: 'polygons',
        positions: fixture.input('float32x2', [0, 0]),
        ringOffsets: fixture.input('uint32', Array(514).fill(0)),
        polygonOffsets: fixture.input('uint32', [0, 513]),
        featureOffsets: fixture.input('uint32', [0, 1])
      },
      gridSize: 1,
      lattice: lattice.view,
      intersectionCapacity: 1,
      output
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await count.read()).toEqual([0]);
    expect(await lattice.read()).toEqual([0, 0, 1, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
