// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUCoverageValidity,
  getGPUCoverageValidityParameterValues
} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {flattenPolygons} from '../spatial-weights/spatial-weights-oracle';
import {COVERAGE_VALIDITY_SCENES} from './coverage-topology-scenes';

it('coverage validity pins every flag bit and clears gap diagnostics on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'gap-parameters',
    format: 'float32',
    length: 4
  });
  const flags = rig.output('uint32', 8);
  const invalid = rig.output('uint32', 1);
  const valid = rig.output('uint32', 1);
  rig.graph.add(
    new GPUCoverageValidity({
      positions: rig.input(
        Float32Array.of(0, -5, 10, -5, 10, 0, 0, 0, 2, 0.0625, 8, 0.0625, 8, 5, 2, 5),
        'float32x2',
        8
      ),
      ringOffsets: rig.input(Uint32Array.of(0, 4, 8), 'uint32', 3),
      polygonOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
      featureOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
      parameters: parameters.importToGraph(rig.graph),
      output: {segmentFlags: flags.view, isValid: valid.view, status: {invalidCount: invalid.view}}
    })
  );
  const compiled = rig.graph.compile();
  try {
    // Disjoint rectangles have exactly two facing edges, rows 2 and 4, separated by 1/16.
    // At gapWidth 1/8 only these edges receive gap bit 4; no overlap/crossing/unmatched bits.
    // At width 0 disjoint polygons form a valid (possibly disconnected) coverage.
    for (const gapWidth of [0.125, 0, 0.125]) {
      parameters.write(getGPUCoverageValidityParameterValues({gapWidth}));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(flags.buffer, 8)).toEqual(
        gapWidth ? [0, 0, 4, 0, 4, 0, 0, 0] : [0, 0, 0, 0, 0, 0, 0, 0]
      );
      expect(await readUint32(invalid.buffer, 1)).toEqual([gapWidth ? 2 : 0]);
      expect(await readUint32(valid.buffer, 1)).toEqual([gapWidth ? 0 : 1]);
    }
  } finally {
    compiled.destroy();
    parameters.destroy();
    rig.destroy();
  }
});

for (const name of ['crossing', 'mismatch']) {
  it(`coverage validity pins literal crossing and unmatched bits: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = COVERAGE_VALIDITY_SCENES.find(candidate => candidate.name === name)!;
    const layout = flattenPolygons(scene.polygons);
    const rig = new WeightsRig(device);
    try {
      const vertexCount = layout.positions.length / 2;
      const flags = rig.output('uint32', vertexCount);
      rig.run(
        new GPUCoverageValidity({
          positions: rig.input(layout.positions, 'float32x2', vertexCount),
          ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
          polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
          featureOffsets: rig.input(layout.featureOffsets, 'uint32', layout.featureOffsets.length),
          parameters: rig.input(getGPUCoverageValidityParameterValues({gapWidth: 0}), 'float32', 4),
          output: {segmentFlags: flags.view}
        })
      );
      // Crossing: polygon 0 edges 0/1 cross polygon 1 edges 2/3, all carry literal bit 2.
      // Mismatch: polygon 0's full shared edge 0 has no identical mate; polygon 1 subdivides
      // it into edges 3/4 at y=0.5. These three unmatched edges carry literal bit 1.
      // The expectations deliberately do not import the shader's exported flag constants.
      expect(await readUint32(flags.buffer, vertexCount)).toEqual(
        name === 'crossing' ? [2, 2, 0, 0, 0, 0, 2, 2] : [1, 0, 0, 0, 0, 0, 0, 1, 1]
      );
    } finally {
      rig.destroy();
    }
  });
}
