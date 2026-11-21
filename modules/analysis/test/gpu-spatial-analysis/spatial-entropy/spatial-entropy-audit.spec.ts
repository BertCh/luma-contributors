// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCategoryPairCounts} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-category-pair-counts';
import {GPUSpatialEntropy} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-spatial-entropy';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  CATEGORICAL_AUDIT_CATEGORIES,
  CATEGORICAL_AUDIT_OFFSETS,
  CATEGORICAL_AUDIT_NEIGHBORS,
  CATEGORICAL_AUDIT_WEIGHTS,
  CATEGORICAL_AUDIT_ROW_COUNTS
} from '../categorical-association/categorical-association-audit.test-utils';

it('audit-validation entropy and join counts match every independent SciPy word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 8]) {
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('float32', 8),
        counts = rig.output('uint32', 2),
        pairCounts = rig.output('uint32', 16),
        standalonePairs = rig.output('uint32', 16),
        categoryRowCounts = rig.output('uint32', 4);
      const weights = rig.weights({
        offsets: CATEGORICAL_AUDIT_OFFSETS,
        neighbors: CATEGORICAL_AUDIT_NEIGHBORS,
        weights: CATEGORICAL_AUDIT_WEIGHTS.map(weight => weight * scale)
      });
      const categories = rig.input(Uint32Array.from(CATEGORICAL_AUDIT_CATEGORIES), 'uint32');
      const entropyProps = {
        weights,
        categories,
        categoryCount: 4,
        output: {summary, pairCounts, counts}
      };
      const pairProps = {
        weights,
        categories,
        categoryCount: 4,
        output: {pairCounts: standalonePairs, categoryRowCounts}
      };
      const entropy = new GPUSpatialEntropy(entropyProps);
      const pairs = new GPUCategoryPairCounts(pairProps);
      entropyProps.categoryCount = 1;
      pairProps.categoryCount = 1;
      weights.offsets = rig.input(Uint32Array.of(0, 0), 'uint32');
      weights.weights = rig.input(Float32Array.of(0), 'float32');
      entropyProps.output.summary = rig.output('float32', 1);
      entropyProps.output.counts = rig.output('uint32', 1);
      pairProps.output.pairCounts = rig.output('uint32', 1);
      rig.run(entropy, pairs);
      // categorical-association-audit.oracle.py uses SciPy entropy over a directed
      // count table. 3e-6 absolute covers float32 logarithms and marginal subtraction.
      const expected = [
        1.0789922078775833, 1.0549201679861442, 1.0549201679861442, 1.8866967846580784,
        0.22314355131420993, 0.21152648142104796, 10, 7
      ];
      const actual = await rig.readFloat(summary);
      expected.forEach((value, index) => {
        expect(Number.isFinite(actual[index])).toBe(true);
        expect(Math.abs(actual[index] - value)).toBeLessThan(3e-6);
      });
      expect(await rig.readUint(counts)).toEqual([7, 10]);
      expect(await rig.readUint(categoryRowCounts)).toEqual(CATEGORICAL_AUDIT_ROW_COUNTS);
      for (const port of [pairCounts, standalonePairs])
        expect(await rig.readUint(port)).toEqual([2, 2, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]);
    } finally {
      rig.destroy();
    }
  }
});
