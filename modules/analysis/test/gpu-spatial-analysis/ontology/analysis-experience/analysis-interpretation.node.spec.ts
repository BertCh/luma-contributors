// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GPU_ANALYSIS_CAUTIONS,
  GPU_ANALYSIS_CLASS_SCHEMES,
  GPU_ANALYSIS_SUMMARY_TEMPLATES
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';
import {
  canResolveGPUAnalysisSignificanceLevel,
  classifyGPUAnalysisResultValues,
  deriveGPUAnalysisCautions,
  describeGPUAnalysisPermutationResolution,
  formatGPUAnalysisNumber,
  getGPUAnalysisCautionText,
  formatGPUAnalysisDistinctValues,
  formatGPUAnalysisMeasuredValue,
  getGPUAnalysisSignificanceAttainability,
  getGPUAnalysisResultClasses,
  getGPUAnalysisSmallestPseudoPValue,
  summarizeGPUAnalysisResult,
  validateGPUAnalysisResultEncoding
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';
import {
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_LOCAL_MORAN_QUADRANT
} from '../../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {GPU_LOCAL_GEARY_CLASS} from '../../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {GPU_SPATIAL_CLUSTERING_NOISE} from '../../../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';

function findClass(
  scheme: (typeof GPU_ANALYSIS_CLASS_SCHEMES)[number],
  value: number | string,
  encoding?: 'contributor-integer' | 'catalog-text' | 'catalog-signed-label'
) {
  return getGPUAnalysisResultClasses(scheme, encoding).find(
    resultClass => resultClass.value === value
  );
}

describe('formatGPUAnalysisNumber', () => {
  it('groups thousands, fixes decimals and never prints negative zero', () => {
    expect(formatGPUAnalysisNumber(0)).toBe('0');
    expect(formatGPUAnalysisNumber(-0)).toBe('0');
    expect(formatGPUAnalysisNumber(999)).toBe('999');
    expect(formatGPUAnalysisNumber(1000)).toBe('1,000');
    expect(formatGPUAnalysisNumber(1234567.891, 2)).toBe('1,234,567.89');
    expect(formatGPUAnalysisNumber(-1234.5, 1)).toBe('-1,234.5');
    expect(formatGPUAnalysisNumber(-0.0001, 2)).toBe('0.00');
    expect(formatGPUAnalysisNumber(2, 2, true)).toBe('2');
    expect(formatGPUAnalysisNumber(1.5, 2, true)).toBe('1.5');
    expect(formatGPUAnalysisNumber(100, 0, true)).toBe('100');
  });

  it('rejects non-finite values, huge values and invalid digits', () => {
    expect(() => formatGPUAnalysisNumber(Number.NaN)).toThrow(/formatGPUAnalysisNumber/);
    expect(() => formatGPUAnalysisNumber(Number.POSITIVE_INFINITY)).toThrow(/"value"/);
    expect(() => formatGPUAnalysisNumber(1e15)).toThrow(/magnitude/);
    expect(() => formatGPUAnalysisNumber(1, 13)).toThrow(/"fractionDigits"/);
    expect(() => formatGPUAnalysisNumber(1, 1.5)).toThrow(/"fractionDigits"/);
  });
});

describe('getGPUAnalysisResultClasses', () => {
  it('has classes for every scheme under the default encoding, with unique values', () => {
    for (const scheme of GPU_ANALYSIS_CLASS_SCHEMES) {
      const classes = getGPUAnalysisResultClasses(scheme);
      expect(classes.length).toBeGreaterThanOrEqual(2);
      expect(new Set(classes.map(resultClass => resultClass.value)).size).toBe(classes.length);
      for (const resultClass of classes) expect(resultClass.label.length).toBeGreaterThan(0);
    }
  });

  it('pins the Gi* bins to the contributor: +-3, +-2, +-1 are 99, 95, 90 percent, 0 is not significant', () => {
    // gpu-hot-spot-analysis.ts:66-70 documents `bins` as sint32 in -3..3; the bin level L is the
    // index into GPU_HOT_SPOT_SIGNIFICANCE_LEVELS (spatial-autocorrelation-parameters.ts:69).
    const classes = getGPUAnalysisResultClasses('hot-spot-confidence-bins');
    expect(classes.map(resultClass => resultClass.value)).toEqual([3, 2, 1, 0, -1, -2, -3]);
    GPU_HOT_SPOT_SIGNIFICANCE_LEVELS.forEach((level, index) => {
      const confidence = `${Math.round((1 - level) * 100)}% confidence`;
      expect(findClass('hot-spot-confidence-bins', index + 1)?.label).toBe(
        `Hot spot, ${confidence}`
      );
      expect(findClass('hot-spot-confidence-bins', -(index + 1))?.label).toBe(
        `Cold spot, ${confidence}`
      );
    });
    expect(findClass('hot-spot-confidence-bins', 0)?.tone).toBe('neutral');
    expect(findClass('hot-spot-confidence-bins', 3)?.tone).toBe('strong-high');
    expect(findClass('hot-spot-confidence-bins', 1)?.tone).toBe('weak-high');
    expect(findClass('hot-spot-confidence-bins', -2)?.tone).toBe('low');
    expect(findClass('hot-spot-confidence-bins', -3)?.tone).toBe('strong-low');
  });

  it('pins the Gi* text classes to the Kepler hotspot-gi-star binding', () => {
    // modules/kepler/src/analysis/catalog-platform3-b-statistics.ts:580-589: classification is
    // `hot-99|hot-95|hot-90|cold-90|cold-95|cold-99` or `not-significant`.
    const values = getGPUAnalysisResultClasses('hot-spot-confidence-bins', 'catalog-text').map(
      resultClass => resultClass.value
    );
    expect(values).toEqual([
      'hot-99',
      'hot-95',
      'hot-90',
      'not-significant',
      'cold-90',
      'cold-95',
      'cold-99'
    ]);
    // The two encodings describe the same seven classes in the same order and tone.
    const integerClasses = getGPUAnalysisResultClasses('hot-spot-confidence-bins');
    const textClasses = getGPUAnalysisResultClasses('hot-spot-confidence-bins', 'catalog-text');
    expect(textClasses.map(resultClass => resultClass.label)).toEqual(
      integerClasses.map(resultClass => resultClass.label)
    );
    expect(textClasses.map(resultClass => resultClass.tone)).toEqual(
      integerClasses.map(resultClass => resultClass.tone)
    );
  });

  it('pins the local Moran classes to GPU_LOCAL_MORAN_QUADRANT', () => {
    // spatial-autocorrelation-parameters.ts:71-85 (esda numbering), written by GPULocalMoran.
    expect(findClass('local-moran-clusters', GPU_LOCAL_MORAN_QUADRANT.HIGH_HIGH)?.label).toBe(
      'High-high cluster core'
    );
    expect(findClass('local-moran-clusters', GPU_LOCAL_MORAN_QUADRANT.LOW_LOW)?.label).toBe(
      'Low-low cluster core'
    );
    expect(findClass('local-moran-clusters', GPU_LOCAL_MORAN_QUADRANT.HIGH_LOW)?.tone).toBe(
      'high-outlier'
    );
    expect(findClass('local-moran-clusters', GPU_LOCAL_MORAN_QUADRANT.LOW_HIGH)?.tone).toBe(
      'low-outlier'
    );
    expect(findClass('local-moran-clusters', GPU_LOCAL_MORAN_QUADRANT.NOT_SIGNIFICANT)?.tone).toBe(
      'neutral'
    );
    expect(
      getGPUAnalysisResultClasses('local-moran-clusters')
        .map(resultClass => resultClass.value)
        .sort()
    ).toEqual(Object.values(GPU_LOCAL_MORAN_QUADRANT).sort());
  });

  it('pins the local Moran text classes to the Kepler local-moran-statistics binding', () => {
    // modules/kepler/src/analysis/catalog-platform3-b-statistics.ts:441-455: cluster is
    // HH|HL|LH|LL, `not-significant`, or `not-tested` when there is no p-value.
    const classes = getGPUAnalysisResultClasses('local-moran-clusters', 'catalog-text');
    expect(classes.map(resultClass => resultClass.value).sort()).toEqual(
      ['HH', 'HL', 'LH', 'LL', 'not-significant', 'not-tested'].sort()
    );
    expect(classes.find(resultClass => resultClass.value === 'HH')?.tone).toBe('high');
    expect(classes.find(resultClass => resultClass.value === 'LH')?.tone).toBe('low-outlier');
    expect(classes.find(resultClass => resultClass.value === 'not-tested')?.tone).toBe('excluded');
  });

  it('pins cluster membership: noise is 0xffffffff (contributor) or -1 (Kepler), first cluster 0', () => {
    // spatial-clustering-parameters.ts:14 GPU_SPATIAL_CLUSTERING_NOISE; the Kepler cluster tables
    // convert it to -1 (catalog-platform3-b-clusters.ts:164).
    const contributor = getGPUAnalysisResultClasses('cluster-membership');
    expect(contributor.map(resultClass => resultClass.value)).toEqual([
      GPU_SPATIAL_CLUSTERING_NOISE,
      0
    ]);
    expect(contributor[0].tone).toBe('excluded');
    const catalog = getGPUAnalysisResultClasses('cluster-membership', 'catalog-signed-label');
    expect(catalog.map(resultClass => resultClass.value)).toEqual([-1, 0]);
  });

  it('pins the flag schemes to 1 and 0', () => {
    for (const scheme of ['significance-flag', 'validity-flag'] as const) {
      expect(getGPUAnalysisResultClasses(scheme).map(resultClass => resultClass.value)).toEqual([
        1, 0
      ]);
    }
    expect(findClass('significance-flag', 1)?.label).toBe('Significant');
    expect(findClass('validity-flag', 0)?.tone).toBe('excluded');
  });

  it('throws, naming the scheme and the accepted encodings, for an encoding a scheme lacks', () => {
    expect(() => getGPUAnalysisResultClasses('cluster-membership', 'catalog-text')).toThrow(
      /scheme "cluster-membership" has no "catalog-text" encoding.*catalog-signed-label/
    );
    expect(() => getGPUAnalysisResultClasses('significance-flag', 'catalog-text')).toThrow(
      /accepted: "contributor-integer"/
    );
    expect(() =>
      getGPUAnalysisResultClasses('hot-spot-confidence-bins', 'catalog-signed-label')
    ).toThrow(/getGPUAnalysisResultClasses/);
  });
});

describe('classifyGPUAnalysisResultValues', () => {
  it('tallies Gi* bins and counts non-finite, null and unknown values as unrecognized', () => {
    const tally = classifyGPUAnalysisResultValues('hot-spot-confidence-bins', [
      3,
      3,
      2,
      0,
      -3,
      Number.NaN,
      null,
      undefined,
      7,
      'x',
      Number.POSITIVE_INFINITY,
      true
    ]);
    expect([...tally.counts]).toEqual([
      [3, 2],
      [2, 1],
      [1, 0],
      [0, 1],
      [-1, 0],
      [-2, 0],
      [-3, 1]
    ]);
    expect(tally.unrecognized).toBe(7);
  });

  it('handles empty input, all nulls and typed arrays', () => {
    const empty = classifyGPUAnalysisResultValues('hot-spot-confidence-bins', []);
    expect([...empty.counts.values()]).toEqual([0, 0, 0, 0, 0, 0, 0]);
    expect(empty.unrecognized).toBe(0);
    const nulls = classifyGPUAnalysisResultValues('local-moran-clusters', [null, null, undefined]);
    expect(nulls.unrecognized).toBe(3);
    expect([...nulls.counts.values()].every(count => count === 0)).toBe(true);
    const typed = classifyGPUAnalysisResultValues(
      'hot-spot-confidence-bins',
      new Int32Array([1, 1, -1, 0])
    );
    expect(typed.counts.get(1)).toBe(2);
    expect(typed.counts.get(-1)).toBe(1);
    expect(typed.unrecognized).toBe(0);
  });

  it('tallies Kepler text classes', () => {
    const tally = classifyGPUAnalysisResultValues(
      'hot-spot-confidence-bins',
      ['hot-99', 'cold-90', 'not-significant', 'not-significant', '', null, 3],
      'catalog-text'
    );
    expect(tally.counts.get('hot-99')).toBe(1);
    expect(tally.counts.get('cold-90')).toBe(1);
    expect(tally.counts.get('not-significant')).toBe(2);
    expect(tally.unrecognized).toBe(3);
    const moran = classifyGPUAnalysisResultValues(
      'local-moran-clusters',
      ['HH', 'HH', 'LL', 'not-tested', 1],
      'catalog-text'
    );
    expect(moran.counts.get('HH')).toBe(2);
    expect(moran.counts.get('not-tested')).toBe(1);
    expect(moran.unrecognized).toBe(1);
  });

  it('tallies local Moran quadrants and rejects codes outside 0 to 4', () => {
    const tally = classifyGPUAnalysisResultValues(
      'local-moran-clusters',
      [0, 1, 1, 2, 3, 4, 5, -1]
    );
    expect(tally.counts.get(1)).toBe(2);
    expect(tally.counts.get(0)).toBe(1);
    expect(tally.unrecognized).toBe(2);
  });

  it('tallies cluster membership with noise first and clusters ascending', () => {
    const tally = classifyGPUAnalysisResultValues('cluster-membership', [
      3,
      0,
      0,
      1,
      GPU_SPATIAL_CLUSTERING_NOISE,
      GPU_SPATIAL_CLUSTERING_NOISE,
      -1,
      1.5,
      Number.NaN,
      GPU_SPATIAL_CLUSTERING_NOISE + 1
    ]);
    expect([...tally.counts]).toEqual([
      [GPU_SPATIAL_CLUSTERING_NOISE, 2],
      [0, 2],
      [1, 1],
      [3, 1]
    ]);
    expect(tally.unrecognized).toBe(4);
    const empty = classifyGPUAnalysisResultValues('cluster-membership', []);
    expect([...empty.counts]).toEqual([[GPU_SPATIAL_CLUSTERING_NOISE, 0]]);
  });

  it('reads -1 as noise only under the signed encoding', () => {
    const tally = classifyGPUAnalysisResultValues(
      'cluster-membership',
      [-1, 0, 0, -1, -2, GPU_SPATIAL_CLUSTERING_NOISE],
      'catalog-signed-label'
    );
    expect([...tally.counts]).toEqual([
      [-1, 2],
      [0, 2]
    ]);
    expect(tally.unrecognized).toBe(2);
  });

  it('reads booleans as flags for the flag schemes only', () => {
    const tally = classifyGPUAnalysisResultValues('significance-flag', [
      true,
      false,
      true,
      1,
      0,
      2,
      '1'
    ]);
    expect(tally.counts.get(1)).toBe(3);
    expect(tally.counts.get(0)).toBe(2);
    expect(tally.unrecognized).toBe(2);
    expect(classifyGPUAnalysisResultValues('validity-flag', [true, null]).unrecognized).toBe(1);
  });
});

describe('summarizeGPUAnalysisResult: significant-hot-cold-count', () => {
  const template = 'significant-hot-cold-count';

  it('states typical values', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1840,
        hotCount: 212,
        coldCount: 37,
        confidencePercent: 95,
        correction: 'false-discovery-rate',
        featureNoun: {singular: 'cell', plural: 'cells'}
      })
    ).toBe(
      '212 of 1,840 cells are significant hot spots and 37 are significant cold spots at 95% confidence after false discovery rate correction.'
    );
  });

  it('states zero, one and large counts', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1840,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'none'
      })
    ).toBe(
      'None of the 1,840 features is a significant hot spot or cold spot at 95% confidence without correction for multiple testing.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1,
        hotCount: 1,
        coldCount: 0,
        confidencePercent: 99,
        correction: 'bonferroni'
      })
    ).toBe(
      'The only feature is a significant hot spot at 99% confidence after Bonferroni correction.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 12,
        hotCount: 1,
        coldCount: 1,
        confidencePercent: 90,
        correction: 'holm',
        featureNoun: {singular: 'cell', plural: 'cells'}
      })
    ).toBe(
      '1 of 12 cells is a significant hot spot and 1 is a significant cold spot at 90% confidence after Holm correction.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1234567,
        hotCount: 12345,
        coldCount: 6789,
        confidencePercent: 95,
        correction: 'false-discovery-rate'
      })
    ).toBe(
      '12,345 of 1,234,567 features are significant hot spots and 6,789 are significant cold spots at 95% confidence after false discovery rate correction.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'false-discovery-rate'
      })
    ).toBe(
      'The only feature is not a significant hot spot or cold spot at 95% confidence after false discovery rate correction.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 0,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'none'
      })
    ).toBe('There are no features to test.');
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 40,
        hotCount: 3,
        coldCount: 2,
        confidencePercent: 97.5,
        correction: 'none'
      })
    ).toContain('at 97.5% confidence');
  });

  it('throws on counts the data cannot hold', () => {
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        featureCount: 10,
        hotCount: 6,
        coldCount: 5,
        confidencePercent: 95,
        correction: 'none'
      })
    ).toThrow(/sum to at most "featureCount" \(10\)/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        featureCount: 10,
        hotCount: -1,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'none'
      })
    ).toThrow(/"hotCount"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        featureCount: 10,
        hotCount: 1.5,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'none'
      })
    ).toThrow(/"hotCount"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        featureCount: 10,
        hotCount: 1,
        coldCount: 0,
        confidencePercent: Number.NaN,
        correction: 'none'
      })
    ).toThrow(/"confidencePercent"/);
  });
});

describe('summarizeGPUAnalysisResult: local-cluster-count', () => {
  const template = 'local-cluster-count';

  it('states typical values with cores and outliers', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1840,
        highHighCount: 120,
        lowLowCount: 85,
        highLowCount: 12,
        lowHighCount: 9,
        significanceLevel: 0.05,
        correction: 'false-discovery-rate'
      })
    ).toBe(
      '226 of 1,840 features are significant at a pseudo p-value of 0.05 or less after false discovery rate correction: 120 high-high cores, 85 low-low cores, 12 high-low outliers and 9 low-high outliers.'
    );
  });

  it('states zero, one and large counts', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1840,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        correction: 'none'
      })
    ).toBe(
      'None of the 1,840 features is significant at a pseudo p-value of 0.05 or less without correction for multiple testing.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1,
        highHighCount: 1,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.01,
        pValueKind: 'analytic',
        correction: 'none'
      })
    ).toBe(
      '1 of 1 feature is significant at a p-value of 0.01 or less without correction for multiple testing: 1 high-high core.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        correction: 'holm'
      })
    ).toBe(
      'The only feature is not significant at a pseudo p-value of 0.05 or less after Holm correction.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 1234567,
        highHighCount: 100000,
        lowLowCount: 0,
        highLowCount: 1,
        lowHighCount: 0,
        significanceLevel: 0.001,
        correction: 'bonferroni'
      })
    ).toBe(
      '100,001 of 1,234,567 features are significant at a pseudo p-value of 0.001 or less after Bonferroni correction: 100,000 high-high cores and 1 high-low outlier.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        featureCount: 0,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        correction: 'none'
      })
    ).toBe('There are no features to test.');
  });

  it('throws on counts that exceed the features and on a level outside 0 to 1', () => {
    const base = {
      featureCount: 3,
      highHighCount: 2,
      lowLowCount: 2,
      highLowCount: 0,
      lowHighCount: 0,
      significanceLevel: 0.05,
      correction: 'none'
    } as const;
    expect(() => summarizeGPUAnalysisResult(template, base)).toThrow(/sum to at most/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {...base, lowLowCount: 0, significanceLevel: 1})
    ).toThrow(/"significanceLevel"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {...base, lowLowCount: 0, significanceLevel: 0})
    ).toThrow(/"significanceLevel"/);
  });
});

describe('summarizeGPUAnalysisResult: cluster-count-and-noise', () => {
  const template = 'cluster-count-and-noise';

  it('states typical values', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        pointCount: 5000,
        clusterCount: 14,
        noiseCount: 312,
        largestClusterCount: 920
      })
    ).toBe(
      'Found 14 clusters; 312 of 5,000 points (6.2%) are noise and the largest cluster holds 18.4% of all points.'
    );
  });

  it('states zero, one and extreme shares', () => {
    expect(
      summarizeGPUAnalysisResult(template, {pointCount: 1000, clusterCount: 0, noiseCount: 1000})
    ).toBe('Found no clusters; all 1,000 points are noise.');
    expect(
      summarizeGPUAnalysisResult(template, {pointCount: 1, clusterCount: 0, noiseCount: 1})
    ).toBe('Found no clusters; the only point is noise.');
    expect(
      summarizeGPUAnalysisResult(template, {
        pointCount: 50,
        clusterCount: 1,
        noiseCount: 0,
        largestClusterCount: 50
      })
    ).toBe('Found 1 cluster; no points are noise and it holds 100% of all points.');
    expect(
      summarizeGPUAnalysisResult(template, {
        pointCount: 100,
        clusterCount: 3,
        noiseCount: 1,
        largestClusterCount: 40
      })
    ).toBe(
      'Found 3 clusters; 1 of 100 points (1.0%) is noise and the largest cluster holds 40.0% of all points.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        pointCount: 1000000,
        clusterCount: 2,
        noiseCount: 1,
        largestClusterCount: 500000
      })
    ).toBe(
      'Found 2 clusters; 1 of 1,000,000 points (less than 0.1%) is noise and the largest cluster holds 50.0% of all points.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        pointCount: 1000000,
        clusterCount: 1,
        noiseCount: 0,
        largestClusterCount: 999999
      })
    ).toContain('more than 99.9%');
    expect(
      summarizeGPUAnalysisResult(template, {pointCount: 0, clusterCount: 0, noiseCount: 0})
    ).toBe('There are no points to cluster.');
  });

  it('throws on inconsistent facts', () => {
    expect(() =>
      summarizeGPUAnalysisResult(template, {pointCount: 10, clusterCount: 2, noiseCount: 1})
    ).toThrow(/"largestClusterCount" is required/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {pointCount: 10, clusterCount: 0, noiseCount: 11})
    ).toThrow(/"noiseCount" must be at most/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {pointCount: 10, clusterCount: 0, noiseCount: 4})
    ).toThrow(/every point is noise/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        pointCount: 10,
        clusterCount: 2,
        noiseCount: 4,
        largestClusterCount: 7
      })
    ).toThrow(/"largestClusterCount" must be from 1 to the clustered points \(6\)/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {
        pointCount: 10,
        clusterCount: 2,
        noiseCount: 0,
        largestClusterCount: 0
      })
    ).toThrow(/"largestClusterCount"/);
  });
});

describe('summarizeGPUAnalysisResult: global-statistic', () => {
  const template = 'global-statistic';
  const base = {statisticName: "Moran's I", value: 0.412, pValue: 0.001, permutationCount: 999};

  it('defines a two-sided pseudo p-value and states the sign meaning when given', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        ...base,
        positiveMeaning: 'similar values near each other',
        negativeMeaning: 'dissimilar values near each other'
      })
    ).toBe(
      "Moran's I is 0.412 (a positive value means similar values near each other), with pseudo p-value 0.001 from 999 permutations (the number of permutations with a value at least as extreme as the observed one in either direction, plus one, divided by 999 plus one); a result this extreme would be rare under a random pattern at the 0.05 level."
    );
  });

  it('words one-sided p-values by direction', () => {
    expect(summarizeGPUAnalysisResult(template, {...base, sidedness: 'greater'})).toBe(
      "Moran's I is 0.412, with pseudo p-value 0.001 from 999 permutations (the number of permutations with a value at least as large as the observed one, plus one, divided by 999 plus one); a result this extreme would be rare under a random pattern at the 0.05 level."
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        ...base,
        sidedness: 'less',
        value: -0.1,
        pValue: 0.4,
        negativeMeaning: 'dissimilar values near each other'
      })
    ).toBe(
      "Moran's I is -0.1 (a negative value means dissimilar values near each other), with pseudo p-value 0.400 from 999 permutations (the number of permutations with a value at least as small as the observed one, plus one, divided by 999 plus one); this is consistent with a random pattern at the 0.05 level."
    );
  });

  it('does not state a sign meaning for a value of zero or when none is given', () => {
    const sentence = summarizeGPUAnalysisResult(template, {
      ...base,
      value: 0,
      positiveMeaning: 'a',
      negativeMeaning: 'b'
    });
    expect(sentence).not.toMatch(/value means/);
    expect(summarizeGPUAnalysisResult(template, {...base, value: -0.2})).not.toMatch(/value means/);
  });

  it('reads the p-value against the fact-supplied null, so Knox is not called a random pattern', () => {
    const sentence = summarizeGPUAnalysisResult(template, {
      ...base,
      statisticName: 'Knox statistic',
      nullDescription: 'no space-time interaction',
      sidedness: 'greater',
      pValue: 0.2
    });
    expect(sentence).toBe(
      'Knox statistic is 0.412, with pseudo p-value 0.200 from 999 permutations (the number of permutations with a value at least as large as the observed one, plus one, divided by 999 plus one); this is consistent with no space-time interaction at the 0.05 level.'
    );
    expect(sentence).not.toMatch(/random pattern/);
    expect(() => summarizeGPUAnalysisResult(template, {...base, nullDescription: '  '})).toThrow(
      /"nullDescription"/
    );
  });

  it('reads at the level from the facts and treats p equal to the level as significant', () => {
    expect(
      summarizeGPUAnalysisResult(template, {...base, pValue: 0.01, significanceLevel: 0.01})
    ).toBe(
      "Moran's I is 0.412, with pseudo p-value 0.010 from 999 permutations (the number of permutations with a value at least as extreme as the observed one in either direction, plus one, divided by 999 plus one); a result this extreme would be rare under a random pattern at the 0.01 level."
    );
    expect(
      summarizeGPUAnalysisResult(template, {...base, pValue: 0.02, significanceLevel: 0.01})
    ).toContain('consistent with a random pattern at the 0.01 level');
  });

  it('shows the resolution of a pseudo p-value from the permutation count', () => {
    expect(
      summarizeGPUAnalysisResult(template, {...base, pValue: 0.0001, permutationCount: 9999})
    ).toContain('pseudo p-value 0.0001 from 9,999 permutations');
    expect(
      summarizeGPUAnalysisResult(template, {...base, pValue: 1, permutationCount: 1})
    ).toContain('pseudo p-value 1.000 from 1 permutation (');
  });

  it('prints two p-values on different sides of the level differently', () => {
    const analytic = (pValue: number, significanceLevel?: number) =>
      summarizeGPUAnalysisResult(template, {
        ...base,
        permutationCount: 0,
        pValue,
        ...(significanceLevel === undefined ? {} : {significanceLevel})
      });
    expect(analytic(0.0501)).toContain('with p-value 0.0501 (');
    expect(analytic(0.0501)).toContain('consistent with a random pattern');
    expect(analytic(0.0499)).toContain('with p-value 0.050 (');
    expect(analytic(0.0499)).toContain('would be rare');
    expect(analytic(0.0005, 0.0001)).toContain('with p-value 0.0005 (');
    expect(analytic(0.0005, 0.0001)).toContain('consistent with');
    expect(analytic(0.00005, 0.0001)).toContain('with p-value less than 0.0001 (');
  });

  it('prints an analytic p-value of 0 or below display precision as less than', () => {
    expect(analytic0(0)).toContain(
      'with p-value less than 0.001 (the chance under a random pattern of a value at least as extreme as the observed one in either direction)'
    );
    expect(analytic0(0.0004)).toContain('p-value less than 0.001');
    expect(analytic0(0.0321)).toContain('with p-value 0.032 (');
    function analytic0(pValue: number) {
      return summarizeGPUAnalysisResult(template, {...base, permutationCount: 0, pValue});
    }
  });

  it('throws on a p-value outside 0 to 1 or a non-finite statistic', () => {
    expect(() => summarizeGPUAnalysisResult(template, {...base, pValue: 1.1})).toThrow(/"pValue"/);
    expect(() => summarizeGPUAnalysisResult(template, {...base, value: Number.NaN})).toThrow(
      /"value"/
    );
    expect(() => summarizeGPUAnalysisResult(template, {...base, permutationCount: 9.5})).toThrow(
      /"permutationCount"/
    );
  });
});

describe('summarizeGPUAnalysisResult: regression-fit', () => {
  const template = 'regression-fit';

  it('reports R squared and the observation count without a causal claim', () => {
    const sentence = summarizeGPUAnalysisResult(template, {
      rSquared: 0.62,
      observationCount: 1840,
      adjustedRSquared: 0.61
    });
    expect(sentence).toBe(
      'R² is 0.62 on 1,840 observations (adjusted R² 0.61), a measure of how closely the model fits the data rather than of cause.'
    );
    expect(sentence).not.toMatch(/explain/i);
  });

  it('states one, zero, negative and near-one values', () => {
    expect(summarizeGPUAnalysisResult(template, {rSquared: 0.5, observationCount: 1})).toBe(
      'R² is 0.50 on 1 observation, a measure of how closely the model fits the data rather than of cause.'
    );
    expect(summarizeGPUAnalysisResult(template, {rSquared: 0.5, observationCount: 0})).toBe(
      'No observations were fitted, so R² is not defined.'
    );
    expect(summarizeGPUAnalysisResult(template, {rSquared: -0.1, observationCount: 30})).toContain(
      'R² is -0.10 on 30 observations'
    );
    expect(
      summarizeGPUAnalysisResult(template, {rSquared: 0.999, observationCount: 1234567})
    ).toContain('R² is just under 1 on 1,234,567 observations');
    expect(summarizeGPUAnalysisResult(template, {rSquared: 1, observationCount: 5})).toContain(
      'R² is 1.00 on 5 observations'
    );
  });

  it('throws on R squared above one or not finite', () => {
    expect(() =>
      summarizeGPUAnalysisResult(template, {rSquared: 1.01, observationCount: 5})
    ).toThrow(/"rSquared"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {rSquared: Number.NaN, observationCount: 5})
    ).toThrow(/"rSquared"/);
  });
});

describe('summarizeGPUAnalysisResult: value-range', () => {
  const template = 'value-range';

  it('states typical, labelled, fractional and constant ranges', () => {
    expect(
      summarizeGPUAnalysisResult(template, {minimum: 3, maximum: 1240, valueCount: 1840})
    ).toBe('The 1,840 values range from 3 to 1,240.');
    expect(
      summarizeGPUAnalysisResult(template, {
        minimum: 3,
        maximum: 1240,
        valueCount: 1840,
        valueLabel: 'Median income'
      })
    ).toBe('Median income ranges from 3 to 1,240 across 1,840 values.');
    expect(
      summarizeGPUAnalysisResult(template, {minimum: 0.5, maximum: 12.25, valueCount: 10})
    ).toBe('The 10 values range from 0.5 to 12.3.');
    expect(summarizeGPUAnalysisResult(template, {minimum: 5, maximum: 5, valueCount: 12})).toBe(
      'All 12 values are 5.'
    );
    expect(summarizeGPUAnalysisResult(template, {minimum: 5, maximum: 5, valueCount: 1})).toBe(
      'The only value is 5.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        minimum: 5,
        maximum: 5,
        valueCount: 1,
        valueLabel: 'Age'
      })
    ).toBe('Age is 5 in the only value.');
    expect(summarizeGPUAnalysisResult(template, {minimum: 0, maximum: 0, valueCount: 0})).toBe(
      'There are no values to summarize.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {minimum: -1234567, maximum: 1234567, valueCount: 3})
    ).toBe('The 3 values range from -1,234,567 to 1,234,567.');
  });

  it('throws on a reversed or non-finite range', () => {
    expect(() =>
      summarizeGPUAnalysisResult(template, {minimum: 5, maximum: 1, valueCount: 3})
    ).toThrow(/"minimum" must be at most "maximum"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {minimum: Number.NaN, maximum: 1, valueCount: 3})
    ).toThrow(/"minimum"/);
    expect(() =>
      summarizeGPUAnalysisResult(template, {minimum: 1, maximum: 2, valueCount: 1})
    ).toThrow(/valueCount" 1/);
  });
});

describe('summarizeGPUAnalysisResult: count-per-area', () => {
  const template = 'count-per-area';

  it('states typical values', () => {
    expect(
      summarizeGPUAnalysisResult(template, {
        totalCount: 1840,
        areaCount: 212,
        maximumCount: 134,
        itemNoun: {singular: 'event', plural: 'events'}
      })
    ).toBe('1,840 events across 212 areas: 8.68 per area on average, up to 134 in one area.');
  });

  it('states zero, one and large values', () => {
    expect(summarizeGPUAnalysisResult(template, {totalCount: 100, areaCount: 50})).toBe(
      '100 items across 50 areas: 2 per area on average.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 1234567, areaCount: 1000})).toBe(
      '1,234,567 items across 1,000 areas: 1,235 per area on average.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 1, areaCount: 3})).toBe(
      '1 item across 3 areas: 0.333 per area on average.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 1, areaCount: 1})).toBe(
      '1 item in 1 area.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 5, areaCount: 1})).toBe(
      '5 items in 1 area.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 0, areaCount: 212})).toBe(
      'No items fall in any of the 212 areas.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 0, areaCount: 1})).toBe(
      'No items fall in the only area.'
    );
    expect(summarizeGPUAnalysisResult(template, {totalCount: 0, areaCount: 0})).toBe(
      'There are no areas to count in.'
    );
  });

  it('throws when the maximum exceeds the total', () => {
    expect(() =>
      summarizeGPUAnalysisResult(template, {totalCount: 5, areaCount: 2, maximumCount: 6})
    ).toThrow(/"maximumCount" must be at most "totalCount" \(5\)/);
    expect(() => summarizeGPUAnalysisResult(template, {totalCount: -5, areaCount: 2})).toThrow(
      /"totalCount"/
    );
  });
});

describe('summarizeGPUAnalysisResult: row-count', () => {
  const template = 'row-count';

  it('states zero, one, typical and large counts', () => {
    expect(summarizeGPUAnalysisResult(template, {rowCount: 0})).toBe('The result has no rows.');
    expect(summarizeGPUAnalysisResult(template, {rowCount: 1})).toBe('The result has 1 row.');
    expect(summarizeGPUAnalysisResult(template, {rowCount: 1840})).toBe(
      'The result has 1,840 rows.'
    );
    expect(summarizeGPUAnalysisResult(template, {rowCount: 12345678})).toBe(
      'The result has 12,345,678 rows.'
    );
    expect(
      summarizeGPUAnalysisResult(template, {
        rowCount: 2,
        rowNoun: {singular: 'cell', plural: 'cells'}
      })
    ).toBe('The result has 2 cells.');
  });

  it('throws on a negative or fractional count', () => {
    expect(() => summarizeGPUAnalysisResult(template, {rowCount: -1})).toThrow(/"rowCount"/);
    expect(() => summarizeGPUAnalysisResult(template, {rowCount: 1.5})).toThrow(/"rowCount"/);
  });

  it('covers every summary template', () => {
    expect(GPU_ANALYSIS_SUMMARY_TEMPLATES.length).toBe(8);
    expect(() => summarizeGPUAnalysisResult('nonsense' as never, {} as never)).toThrow(
      /"template"/
    );
  });
});

describe('cautions', () => {
  it('has a title and text for every caution id', () => {
    const titles = new Set<string>();
    for (const caution of GPU_ANALYSIS_CAUTIONS) {
      const {title, text} = getGPUAnalysisCautionText(caution);
      expect(title.length).toBeGreaterThan(0);
      expect(text.length).toBeGreaterThan(40);
      titles.add(title);
    }
    expect(titles.size).toBe(GPU_ANALYSIS_CAUTIONS.length);
  });

  it('states what each caution is about', () => {
    expect(getGPUAnalysisCautionText('fewer-than-thirty-features').text).toMatch(/30 features/);
    expect(getGPUAnalysisCautionText('pseudo-p-values').text).toMatch(
      /vary with the random seed.*1 \/ \(permutations \+ 1\)/
    );
    expect(getGPUAnalysisCautionText('multiple-testing-uncorrected').text).toMatch(
      /expected share.*even when tests depend.*joint behaviour.*Where the tool offers one/
    );
    expect(getGPUAnalysisCautionText('multiple-testing-uncorrected').text).not.toMatch(/\d/);
    expect(getGPUAnalysisCautionText('fewer-than-thirty-features').text).toMatch(
      /hot spot analysis.*cluster and outlier.*rule of thumb.*by analogy.*usually given/
    );
    expect(getGPUAnalysisCautionText('multiple-testing-uncorrected').text).toMatch(
      /false discovery rate/
    );
    expect(getGPUAnalysisCautionText('counts-where-rates-expected').text).toMatch(/where risk is/);
    expect(getGPUAnalysisCautionText('default-from-fallback').text).toMatch(
      /not derived from this data/
    );
    expect(getGPUAnalysisCautionText('exploratory-parameter-search').text).toMatch(
      /does not control this/
    );
  });

  it('throws for an unknown caution id', () => {
    expect(() => getGPUAnalysisCautionText('nonsense' as never)).toThrow(/"caution"/);
  });
});

describe('deriveGPUAnalysisCautions', () => {
  it('returns nothing for no facts and for facts that raise no caution', () => {
    expect(deriveGPUAnalysisCautions({})).toEqual([]);
    expect(
      deriveGPUAnalysisCautions({
        featureCount: 30,
        corrected: true,
        permutationCount: 0,
        rowsWithoutNeighbors: 0,
        sampledStatistics: false,
        guessedCoordinates: false,
        usedFallbackDefault: false,
        measurementMismatch: false,
        parameterSearch: false
      })
    ).toEqual([]);
  });

  it('raises fewer-than-thirty-features below 30, including 0', () => {
    expect(deriveGPUAnalysisCautions({featureCount: 29})).toEqual(['fewer-than-thirty-features']);
    expect(deriveGPUAnalysisCautions({featureCount: 0})).toEqual(['fewer-than-thirty-features']);
    expect(deriveGPUAnalysisCautions({featureCount: 30})).toEqual([]);
  });

  it('raises multiple-testing-uncorrected only when corrected is explicitly false', () => {
    expect(deriveGPUAnalysisCautions({corrected: false})).toEqual(['multiple-testing-uncorrected']);
    expect(deriveGPUAnalysisCautions({corrected: true})).toEqual([]);
    expect(deriveGPUAnalysisCautions({corrected: undefined})).toEqual([]);
  });

  it('raises pseudo-p-values for any positive permutation count', () => {
    expect(deriveGPUAnalysisCautions({permutationCount: 1})).toEqual(['pseudo-p-values']);
    expect(deriveGPUAnalysisCautions({permutationCount: 999})).toEqual(['pseudo-p-values']);
    expect(deriveGPUAnalysisCautions({permutationCount: 0})).toEqual([]);
  });

  it('raises each flag caution on its own', () => {
    expect(deriveGPUAnalysisCautions({rowsWithoutNeighbors: 1})).toEqual([
      'rows-without-neighbors'
    ]);
    expect(deriveGPUAnalysisCautions({sampledStatistics: true})).toEqual(['sampled-statistics']);
    expect(deriveGPUAnalysisCautions({guessedCoordinates: true})).toEqual(['guessed-coordinates']);
    expect(deriveGPUAnalysisCautions({usedFallbackDefault: true})).toEqual([
      'default-from-fallback'
    ]);
    expect(deriveGPUAnalysisCautions({measurementMismatch: true})).toEqual([
      'counts-where-rates-expected'
    ]);
    expect(deriveGPUAnalysisCautions({parameterSearch: true})).toEqual([
      'exploratory-parameter-search'
    ]);
  });

  it('returns every caution in the order of GPU_ANALYSIS_CAUTIONS when every fact applies', () => {
    const cautions = deriveGPUAnalysisCautions({
      parameterSearch: true,
      usedFallbackDefault: true,
      guessedCoordinates: true,
      sampledStatistics: true,
      rowsWithoutNeighbors: 4,
      corrected: false,
      permutationCount: 99,
      measurementMismatch: true,
      featureCount: 10,
      significanceLevel: 0.001,
      smallDenominatorAreaCount: 3,
      poorModelFitReason: 'the residuals are strongly clustered',
      rowOrderDependentInference: true
    });
    expect(cautions).toEqual([...GPU_ANALYSIS_CAUTIONS]);
  });

  it('throws on invalid counts and levels', () => {
    expect(() => deriveGPUAnalysisCautions({featureCount: -1})).toThrow(/"featureCount"/);
    expect(() => deriveGPUAnalysisCautions({permutationCount: 1.5})).toThrow(/"permutationCount"/);
    expect(() => deriveGPUAnalysisCautions({rowsWithoutNeighbors: -2})).toThrow(
      /"rowsWithoutNeighbors"/
    );
    expect(() => deriveGPUAnalysisCautions({significanceLevel: 0})).toThrow(/"significanceLevel"/);
  });
});

describe('permutation resolution', () => {
  it('computes the smallest attainable pseudo p-value as 1 / (permutations + 1)', () => {
    expect(getGPUAnalysisSmallestPseudoPValue(999)).toBe(0.001);
    expect(getGPUAnalysisSmallestPseudoPValue(0)).toBe(1);
    expect(getGPUAnalysisSmallestPseudoPValue(19)).toBe(0.05);
    expect(() => getGPUAnalysisSmallestPseudoPValue(-1)).toThrow(/"permutationCount"/);
  });

  it('resolves 0.001 but not 0.0005 with 999 permutations', () => {
    expect(canResolveGPUAnalysisSignificanceLevel(999, 0.001)).toBe(true);
    expect(canResolveGPUAnalysisSignificanceLevel(999, 0.0005)).toBe(false);
    expect(canResolveGPUAnalysisSignificanceLevel(998, 0.001)).toBe(false);
    expect(canResolveGPUAnalysisSignificanceLevel(1999, 0.0005)).toBe(true);
  });

  it('is exact at the common levels', () => {
    expect(canResolveGPUAnalysisSignificanceLevel(19, 0.05)).toBe(true);
    expect(canResolveGPUAnalysisSignificanceLevel(18, 0.05)).toBe(false);
    expect(canResolveGPUAnalysisSignificanceLevel(99, 0.01)).toBe(true);
    expect(canResolveGPUAnalysisSignificanceLevel(98, 0.01)).toBe(false);
    expect(canResolveGPUAnalysisSignificanceLevel(0, 0.05)).toBe(false);
    expect(() => canResolveGPUAnalysisSignificanceLevel(999, 0)).toThrow(/"significanceLevel"/);
  });

  it('describes the shortfall and the count that would resolve the level', () => {
    expect(describeGPUAnalysisPermutationResolution(999, 0.001)).toBeUndefined();
    expect(describeGPUAnalysisPermutationResolution(999, 0.0005)).toBe(
      '999 permutations cannot give a pseudo p-value below 0.001, so no location can be significant at 0.0005; at least 1,999 permutations are needed.'
    );
    expect(describeGPUAnalysisPermutationResolution(0, 0.05)).toBe(
      '0 permutations cannot give a pseudo p-value below 1, so no location can be significant at 0.05; at least 19 permutations are needed.'
    );
  });
});

describe('validateGPUAnalysisResultEncoding', () => {
  const samples = {
    'hot-spot-confidence-bins': {
      'contributor-integer': [3, 2, 1, 0, -1, -2, -3],
      'catalog-text': [
        'hot-99',
        'hot-95',
        'hot-90',
        'not-significant',
        'cold-90',
        'cold-95',
        'cold-99'
      ]
    },
    'local-moran-clusters': {
      'contributor-integer': [0, 1, 2, 3, 4],
      'catalog-text': ['HH', 'HL', 'LH', 'LL', 'not-significant', 'not-tested']
    },
    'cluster-membership': {
      'contributor-integer': [0, 1, 2, GPU_SPATIAL_CLUSTERING_NOISE],
      'catalog-signed-label': [0, 1, 2, -1]
    },
    'significance-flag': {'contributor-integer': [0, 1, true, false]},
    'validity-flag': {'contributor-integer': [0, 1, true, false]}
  } as const;

  it('scores 1 for a column written under its own scheme and encoding', () => {
    for (const [scheme, encodings] of Object.entries(samples)) {
      for (const [encoding, values] of Object.entries(encodings)) {
        expect(
          validateGPUAnalysisResultEncoding(scheme as never, encoding as never, [...values])
        ).toBe(1);
      }
    }
  });

  it('scores 0 for a text status column given validity-flag and for the wrong encoding', () => {
    expect(
      validateGPUAnalysisResultEncoding('validity-flag', 'contributor-integer', [
        'ok',
        'failed',
        'pending'
      ])
    ).toBe(0);
    expect(
      validateGPUAnalysisResultEncoding('hot-spot-confidence-bins', 'catalog-text', [3, 2, 0, -1])
    ).toBe(0);
    expect(
      validateGPUAnalysisResultEncoding('hot-spot-confidence-bins', 'contributor-integer', [
        'hot-99',
        'not-significant'
      ])
    ).toBe(0);
    expect(
      validateGPUAnalysisResultEncoding('cluster-membership', 'catalog-signed-label', [
        GPU_SPATIAL_CLUSTERING_NOISE,
        GPU_SPATIAL_CLUSTERING_NOISE
      ])
    ).toBe(0);
  });

  it('scores the share for a partly matching column and ignores nulls', () => {
    expect(
      validateGPUAnalysisResultEncoding('local-moran-clusters', 'contributor-integer', [
        0,
        1,
        2,
        9,
        null,
        undefined
      ])
    ).toBe(0.75);
    // Local Geary class 4 means "not significant" there but is high-low here: a Geary column
    // with codes 0 to 4 still scores 1, which is why the doc comment warns against it.
    expect(
      validateGPUAnalysisResultEncoding('significance-flag', 'contributor-integer', [
        1,
        0,
        Number.NaN,
        2
      ])
    ).toBe(0.5);
  });

  it('scores 0 for empty and all-null columns', () => {
    expect(validateGPUAnalysisResultEncoding('significance-flag', 'contributor-integer', [])).toBe(
      0
    );
    expect(
      validateGPUAnalysisResultEncoding('significance-flag', 'contributor-integer', [
        null,
        undefined
      ])
    ).toBe(0);
  });

  it('throws for an encoding the scheme lacks', () => {
    expect(() => validateGPUAnalysisResultEncoding('validity-flag', 'catalog-text', [1])).toThrow(
      /has no "catalog-text" encoding/
    );
  });
});

describe('formatGPUAnalysisMeasuredValue', () => {
  it('shows three significant digits below 1', () => {
    expect(formatGPUAnalysisMeasuredValue(0.000213)).toBe('0.000213');
    expect(formatGPUAnalysisMeasuredValue(0.04567)).toBe('0.0457');
    expect(formatGPUAnalysisMeasuredValue(0.5)).toBe('0.5');
    expect(formatGPUAnalysisMeasuredValue(0.412)).toBe('0.412');
    expect(formatGPUAnalysisMeasuredValue(0.0002)).toBe('0.0002');
    expect(formatGPUAnalysisMeasuredValue(0.001)).toBe('0.001');
    expect(formatGPUAnalysisMeasuredValue(0.00000123456)).toBe('0.00000123');
  });

  it('shows at most two decimals with separators from 1 up', () => {
    expect(formatGPUAnalysisMeasuredValue(1)).toBe('1');
    expect(formatGPUAnalysisMeasuredValue(1.234)).toBe('1.23');
    expect(formatGPUAnalysisMeasuredValue(1234.5678)).toBe('1,235');
    expect(formatGPUAnalysisMeasuredValue(12345678901)).toBe('12,345,678,901');
  });

  it('handles zero, negative zero, mixed signs and exponent notation', () => {
    expect(formatGPUAnalysisMeasuredValue(0)).toBe('0');
    expect(formatGPUAnalysisMeasuredValue(-0)).toBe('0');
    expect(formatGPUAnalysisMeasuredValue(-0.000213)).toBe('-0.000213');
    expect(formatGPUAnalysisMeasuredValue(-1234.5678)).toBe('-1,235');
    expect(formatGPUAnalysisMeasuredValue(2.134e-12)).toBe('2.13e-12');
    expect(formatGPUAnalysisMeasuredValue(1.2345e20)).toBe('1.23e+20');
    expect(formatGPUAnalysisMeasuredValue(-1.2345e20)).toBe('-1.23e+20');
    expect(() => formatGPUAnalysisMeasuredValue(Number.NaN)).toThrow(/"value"/);
  });

  it('prints values that differ in the first three significant digits differently', () => {
    expect(formatGPUAnalysisMeasuredValue(0.00021)).not.toBe(
      formatGPUAnalysisMeasuredValue(0.00029)
    );
    for (const scale of [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 1e3, 1e6]) {
      for (let mantissa = 100; mantissa < 999; mantissa++) {
        const lower = (mantissa / 100) * scale;
        const upper = ((mantissa + 1) / 100) * scale;
        expect(formatGPUAnalysisMeasuredValue(lower)).not.toBe(
          formatGPUAnalysisMeasuredValue(upper)
        );
      }
    }
  });

  it('is used by value range, global statistic and count per area', () => {
    expect(
      summarizeGPUAnalysisResult('value-range', {
        minimum: 0.00021,
        maximum: 0.00029,
        valueCount: 40
      })
    ).toBe('The 40 values range from 0.00021 to 0.00029.');
    expect(
      summarizeGPUAnalysisResult('value-range', {minimum: 0.0002, maximum: 0.0002, valueCount: 3})
    ).toBe('All 3 values are 0.0002.');
    expect(
      summarizeGPUAnalysisResult('value-range', {
        minimum: -0.000213,
        maximum: 1234.5678,
        valueCount: 9
      })
    ).toBe('The 9 values range from -0.000213 to 1,235.');
    expect(
      summarizeGPUAnalysisResult('value-range', {
        minimum: -1e-12,
        maximum: 12345678901,
        valueCount: 9
      })
    ).toBe('The 9 values range from -1.00e-12 to 12,345,678,901.');
    expect(
      summarizeGPUAnalysisResult('global-statistic', {
        statisticName: 'I',
        value: 0.000213,
        pValue: 0.5,
        permutationCount: 999
      })
    ).toContain('I is 0.000213, with pseudo p-value');
    expect(summarizeGPUAnalysisResult('count-per-area', {totalCount: 1, areaCount: 10000})).toBe(
      '1 item across 10,000 areas: 0.0001 per area on average.'
    );
  });
});

describe('formatGPUAnalysisMeasuredValue significant digits and one string per value', () => {
  it('rounds to three significant digits without dropping integer digits', () => {
    expect(formatGPUAnalysisMeasuredValue(12345.678)).toBe('12,346');
    expect(formatGPUAnalysisMeasuredValue(12.345)).toBe('12.3');
    expect(formatGPUAnalysisMeasuredValue(999.6)).toBe('1,000');
    expect(formatGPUAnalysisMeasuredValue(0.9996)).toBe('1');
  });

  it('takes the number of significant digits as an argument', () => {
    expect(formatGPUAnalysisMeasuredValue(12345.678, 7)).toBe('12,345.68');
    expect(formatGPUAnalysisMeasuredValue(0.04567, 2)).toBe('0.046');
    expect(formatGPUAnalysisMeasuredValue(1.5, 1)).toBe('2');
    expect(formatGPUAnalysisMeasuredValue(2.134e-12, 5)).toBe('2.1340e-12');
    expect(() => formatGPUAnalysisMeasuredValue(1, 0)).toThrow(/"significantDigits"/);
    expect(() => formatGPUAnalysisMeasuredValue(1, 13)).toThrow(/"significantDigits"/);
    expect(() => formatGPUAnalysisMeasuredValue(1, 2.5)).toThrow(/"significantDigits"/);
  });
});

describe('formatGPUAnalysisDistinctValues', () => {
  it('uses the default digits when they already separate the values', () => {
    expect(formatGPUAnalysisDistinctValues([10, 20, 30])).toEqual(['10', '20', '30']);
    expect(formatGPUAnalysisDistinctValues([0.000213, 0.04567, 12345.678])).toEqual([
      '0.000213',
      '0.0457',
      '12,346'
    ]);
  });

  it('adds the fewest digits that separate neighbouring breaks', () => {
    expect(formatGPUAnalysisDistinctValues([1000, 1000.4])).toEqual(['1,000', '1,000.4']);
    expect(formatGPUAnalysisDistinctValues([0.00021, 0.000214])).toEqual(['0.00021', '0.000214']);
    expect(formatGPUAnalysisDistinctValues([12345.1, 12345.2, 12345.9])).toEqual([
      '12,345.1',
      '12,345.2',
      '12,345.9'
    ]);
  });

  it('prints equal values alike, handles empty and single lists, and rejects non-finite values', () => {
    expect(formatGPUAnalysisDistinctValues([5, 5, 7])).toEqual(['5', '5', '7']);
    expect(formatGPUAnalysisDistinctValues([])).toEqual([]);
    expect(formatGPUAnalysisDistinctValues([0])).toEqual(['0']);
    expect(formatGPUAnalysisDistinctValues([-0.5, 0, 0.5])).toEqual(['-0.5', '0', '0.5']);
    expect(() => formatGPUAnalysisDistinctValues([1, Number.NaN])).toThrow(/index 1/);
  });

  it('keeps distinct values distinct for a sweep of near neighbours', () => {
    for (const base of [1e-6, 0.003, 0.5, 1, 77, 1234, 98765.4, 1e9]) {
      for (const gap of [1e-1, 1e-2, 1e-3, 1e-4]) {
        const values = [base, base * (1 + gap), base * (1 + 2 * gap)];
        const strings = formatGPUAnalysisDistinctValues(values);
        expect(new Set(strings).size).toBe(3);
      }
    }
  });
});

describe('getGPUAnalysisSignificanceAttainability', () => {
  // Oracles (/usr/bin/python3): ceil(144 / 0.05) - 1 = 2879; sidak threshold
  // 1 - (1 - 0.05) ** (1 / 144) = 0.00035614000033590276, ceil(1 / threshold) - 1 = 2807;
  // BH tests at the minimum: ceil(144 / (999 + 1) / 0.05) = 3 and ceil(144 / (99 + 1) / 0.05) = 29.
  const base = {testCount: 144, permutationCount: 999, significanceLevel: 0.05};

  it('says Bonferroni over 144 tests cannot be significant with 999 permutations', () => {
    const result = getGPUAnalysisSignificanceAttainability({...base, correction: 'bonferroni'});
    expect(result.attainable).toBe(false);
    expect(result.smallestPValue).toBe(0.001);
    expect(result.threshold).toBeCloseTo(0.05 / 144, 15);
    expect(result.requiredPermutationCount).toBe(2879);
    expect(result.reason).toBe(
      'with 999 permutations the smallest pseudo p-value is 0.001, but a test needs 0.000347 or less to be declared significant under Bonferroni correction over 144 tests (0.05 / 144), so none can be; at least 2,879 permutations are needed'
    );
  });

  it('is exact at the Bonferroni boundary', () => {
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'bonferroni',
        permutationCount: 2879
      }).attainable
    ).toBe(true);
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'bonferroni',
        permutationCount: 2878
      }).attainable
    ).toBe(false);
  });

  it('treats Holm like Bonferroni for the first step', () => {
    const result = getGPUAnalysisSignificanceAttainability({...base, correction: 'holm'});
    expect(result.attainable).toBe(false);
    expect(result.requiredPermutationCount).toBe(2879);
    expect(result.reason).toContain('under Holm correction over 144 tests');
  });

  it('derives the Sidak threshold', () => {
    const result = getGPUAnalysisSignificanceAttainability({...base, correction: 'sidak'});
    expect(result.threshold).toBeCloseTo(0.00035614000033590276, 15);
    expect(result.attainable).toBe(false);
    expect(result.requiredPermutationCount).toBe(2807);
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'sidak',
        permutationCount: 2807
      }).attainable
    ).toBe(true);
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'sidak',
        permutationCount: 2806
      }).attainable
    ).toBe(false);
  });

  it('reports how many tests Benjamini-Hochberg needs at the smallest value', () => {
    const demanding = getGPUAnalysisSignificanceAttainability({
      ...base,
      correction: 'false-discovery-rate'
    });
    expect(demanding.attainable).toBe(true);
    expect(demanding.testsNeededAtMinimum).toBe(3);
    expect(demanding.reason).toBe(
      'with 999 permutations the smallest pseudo p-value is 0.001; Benjamini-Hochberg over 144 tests rejects through the largest rank k whose k-th smallest p-value is at most k times 0.05 / 144; 3 tests tied at the minimum would suffice, even if all remaining p-values were larger'
    );
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'false-discovery-rate',
        permutationCount: 99
      }).testsNeededAtMinimum
    ).toBe(29);
    const impossible = getGPUAnalysisSignificanceAttainability({
      ...base,
      correction: 'false-discovery-rate',
      permutationCount: 18
    });
    expect(impossible.attainable).toBe(false);
    expect(impossible.requiredPermutationCount).toBe(19);
    const easy = getGPUAnalysisSignificanceAttainability({
      ...base,
      correction: 'false-discovery-rate',
      permutationCount: 99999
    });
    expect(easy.testsNeededAtMinimum).toBe(1);
    expect(easy.reason).toContain('1 test tied at the minimum would suffice');
  });

  it('handles no correction, zero permutations and a single test', () => {
    expect(
      getGPUAnalysisSignificanceAttainability({...base, correction: 'none', permutationCount: 19})
        .attainable
    ).toBe(true);
    const nine = getGPUAnalysisSignificanceAttainability({
      ...base,
      correction: 'none',
      permutationCount: 9
    });
    expect(nine.attainable).toBe(false);
    expect(nine.requiredPermutationCount).toBe(19);
    const zero = getGPUAnalysisSignificanceAttainability({
      ...base,
      correction: 'none',
      permutationCount: 0
    });
    expect(zero.attainable).toBe(false);
    expect(zero.reason).toMatch(/^with 0 permutations no pseudo p-value is computed/);
    expect(
      getGPUAnalysisSignificanceAttainability({
        ...base,
        correction: 'bonferroni',
        testCount: 1,
        permutationCount: 19
      }).attainable
    ).toBe(true);
    // The test count does not matter without a correction.
    expect(
      getGPUAnalysisSignificanceAttainability({...base, correction: 'none', testCount: 100000})
        .attainable
    ).toBe(true);
  });

  it('throws on invalid facts', () => {
    expect(() =>
      getGPUAnalysisSignificanceAttainability({...base, correction: 'bonferroni', testCount: 0})
    ).toThrow(/"testCount"/);
    expect(() =>
      getGPUAnalysisSignificanceAttainability({...base, correction: 'none', permutationCount: -1})
    ).toThrow(/"permutationCount"/);
    expect(() =>
      getGPUAnalysisSignificanceAttainability({...base, correction: 'none', significanceLevel: 1})
    ).toThrow(/"significanceLevel"/);
  });
});

describe('sentences that report unattainable significance', () => {
  const cell = {singular: 'cell', plural: 'cells'};

  it('does not present none significant as a finding for a hot spot run', () => {
    expect(
      summarizeGPUAnalysisResult('significant-hot-cold-count', {
        featureCount: 144,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'bonferroni',
        permutationCount: 999,
        featureNoun: cell
      })
    ).toBe(
      'No cell could be a significant hot spot or cold spot at 95% confidence after Bonferroni correction: with 999 permutations the smallest pseudo p-value is 0.001, but a test needs 0.000347 or less to be declared significant under Bonferroni correction over 144 tests (0.05 / 144), so none can be; at least 2,879 permutations are needed.'
    );
    expect(
      summarizeGPUAnalysisResult('significant-hot-cold-count', {
        featureCount: 144,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'sidak',
        permutationCount: 999
      })
    ).toContain('at least 2,807 permutations are needed');
  });

  it('keeps the plain sentence when significance can be reached or the p-values are analytic', () => {
    const facts = {
      featureCount: 144,
      hotCount: 0,
      coldCount: 0,
      confidencePercent: 95,
      correction: 'bonferroni' as const
    };
    expect(
      summarizeGPUAnalysisResult('significant-hot-cold-count', {...facts, permutationCount: 99999})
    ).toBe(
      'None of the 144 features is a significant hot spot or cold spot at 95% confidence after Bonferroni correction.'
    );
    expect(summarizeGPUAnalysisResult('significant-hot-cold-count', facts)).toContain(
      'None of the 144'
    );
    expect(
      summarizeGPUAnalysisResult('significant-hot-cold-count', {...facts, permutationCount: 0})
    ).toContain('None of the 144');
  });

  it('adds the demanding note for Benjamini-Hochberg', () => {
    expect(
      summarizeGPUAnalysisResult('significant-hot-cold-count', {
        featureCount: 144,
        hotCount: 0,
        coldCount: 0,
        confidencePercent: 95,
        correction: 'false-discovery-rate',
        permutationCount: 99
      })
    ).toBe(
      'None of the 144 features is a significant hot spot or cold spot at 95% confidence after false discovery rate correction; with 99 permutations the smallest pseudo p-value is 0.01; Benjamini-Hochberg over 144 tests rejects through the largest rank k whose k-th smallest p-value is at most k times 0.05 / 144; 29 tests tied at the minimum would suffice, even if all remaining p-values were larger.'
    );
  });

  it('applies the same rule to local clusters', () => {
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {
        featureCount: 144,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        correction: 'bonferroni',
        permutationCount: 999
      })
    ).toBe(
      'No feature could be significant at a pseudo p-value of 0.05 or less after Bonferroni correction: with 999 permutations the smallest pseudo p-value is 0.001, but a test needs 0.000347 or less to be declared significant under Bonferroni correction over 144 tests (0.05 / 144), so none can be; at least 2,879 permutations are needed.'
    );
  });

  it('says no p-values were computed for a local cluster result with zero permutations', () => {
    const facts = {
      featureCount: 0,
      highHighCount: 0,
      lowLowCount: 0,
      highLowCount: 0,
      lowHighCount: 0,
      significanceLevel: 0.05,
      correction: 'none' as const,
      permutationCount: 0,
      analyzedCount: 144
    };
    expect(summarizeGPUAnalysisResult('local-cluster-count', facts)).toBe(
      'No p-values were computed because the number of permutations is 0, so none of the 144 features was tested; at least 19 permutations are needed to reach a pseudo p-value of 0.05 or less without correction for multiple testing.'
    );
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {...facts, correction: 'bonferroni'})
    ).toContain(
      'at least 2,879 permutations are needed to reach a pseudo p-value of 0.05 or less after Bonferroni correction'
    );
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {...facts, analyzedCount: 1})
    ).toContain('so the only feature was not tested');
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {
        ...facts,
        analyzedCount: undefined,
        permutationCount: undefined,
        pValuesComputed: false
      })
    ).toContain('so no feature was tested');
    // With an analytic p-value, zero permutations is not a missing test.
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {
        ...facts,
        featureCount: 10,
        highHighCount: 2,
        pValueKind: 'analytic'
      })
    ).toBe(
      '2 of 10 features are significant at a p-value of 0.05 or less without correction for multiple testing: 2 high-high cores.'
    );
  });

  it('prints a level of 1e-5 and one with more digits correctly', () => {
    const facts = {
      featureCount: 5,
      highHighCount: 1,
      lowLowCount: 0,
      highLowCount: 0,
      lowHighCount: 0,
      correction: 'none' as const
    };
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {...facts, significanceLevel: 0.00001})
    ).toContain('a pseudo p-value of 0.00001 or less');
    expect(
      summarizeGPUAnalysisResult('local-cluster-count', {...facts, significanceLevel: 0.0001234})
    ).toContain('a pseudo p-value of 0.000123 or less');
  });
});

describe('hot spot sentence grammar for small counts', () => {
  const run = (featureCount: number, hotCount: number, coldCount: number) =>
    summarizeGPUAnalysisResult('significant-hot-cold-count', {
      featureCount,
      hotCount,
      coldCount,
      confidencePercent: 90,
      correction: 'none'
    });

  it('agrees the verb and says significant cold spot', () => {
    const tail = ' at 90% confidence without correction for multiple testing.';
    expect(run(1, 0, 1)).toBe(`The only feature is a significant cold spot${tail}`);
    expect(run(1, 1, 0)).toBe(`The only feature is a significant hot spot${tail}`);
    expect(run(5, 0, 1)).toBe(
      `0 of 5 features are significant hot spots and 1 is a significant cold spot${tail}`
    );
    expect(run(2, 2, 0)).toBe(
      `2 of 2 features are significant hot spots and 0 are significant cold spots${tail}`
    );
  });
});

describe('cautions with facts', () => {
  it('states the level of the uncorrected tests, and no number without it', () => {
    const generic = getGPUAnalysisCautionText('multiple-testing-uncorrected').text;
    expect(generic).not.toMatch(/\d/);
    expect(generic).toContain('Where the tool offers one');
    expect(generic).toContain('even when tests depend on each other');
    const at005 = getGPUAnalysisCautionText('multiple-testing-uncorrected', {
      significanceLevel: 0.05
    });
    expect(at005.text).toContain('at the 0.05 level, about 5 in 100 locations with no pattern');
    expect(
      getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: 0.1}).text
    ).toContain('at the 0.1 level, about 10 in 100 locations');
    expect(
      getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: 0.01}).text
    ).toContain('at the 0.01 level, about 1 in 100 locations');
    expect(
      getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: 0.01}).text
    ).toContain('Dependence changes the joint behaviour');
    expect(() =>
      getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: 2})
    ).toThrow(/"significanceLevel"/);
  });

  it('words rows without neighbours for both cases', () => {
    expect(getGPUAnalysisCautionText('rows-without-neighbors').text).toContain(
      'either left out of the classes or classified from their own value alone'
    );
    expect(getGPUAnalysisCautionText('rows-without-neighbors', {rowsExcluded: true}).text).toBe(
      'Some rows have no neighbours under the chosen spatial weights, so they get no statistic and are left out of the classes. Widen the distance band or use more neighbours if they matter.'
    );
    expect(getGPUAnalysisCautionText('rows-without-neighbors', {rowsExcluded: false}).text).toBe(
      'Some rows have no neighbours under the chosen spatial weights. They are still classified, from their own value alone, which says nothing about clustering. Widen the distance band or use more neighbours if they matter.'
    );
  });

  it('adds the reason to significance-not-attainable', () => {
    const generic = getGPUAnalysisCautionText('significance-not-attainable');
    expect(generic.title).toBe('Significance cannot be reached');
    const detailed = getGPUAnalysisCautionText('significance-not-attainable', {
      correction: 'bonferroni',
      testCount: 144,
      permutationCount: 999,
      significanceLevel: 0.05
    });
    expect(detailed.text).toContain(
      'A result of none significant is then not a finding about the data.'
    );
    expect(detailed.text).toContain('With 999 permutations the smallest pseudo p-value is 0.001');
    expect(detailed.text).toContain('at least 2,879 permutations are needed.');
  });

  it('adds the count and reason to small-denominators and poor-model-fit', () => {
    expect(
      getGPUAnalysisCautionText('small-denominators', {
        smallDenominatorAreaCount: 12,
        expectedCountLimit: 5
      }).text.startsWith('12 areas have an expected count below 5. Ratios of observed')
    ).toBe(true);
    expect(
      getGPUAnalysisCautionText('small-denominators', {smallDenominatorAreaCount: 1}).text
    ).toMatch(/^1 area has an expected count below the stated limit\./);
    expect(
      getGPUAnalysisCautionText('poor-model-fit', {reason: 'the residuals are clustered'}).text
    ).toMatch(/Reason given: the residuals are clustered\.$/);
    expect(getGPUAnalysisCautionText('poor-model-fit', {reason: '  '}).text).not.toMatch(/Reason/);
    expect(getGPUAnalysisCautionText('row-order-dependent-inference').text).toMatch(
      /depend on the order of the input rows/
    );
  });

  it('says in counts-where-rates-expected that it is about the measurement level', () => {
    expect(getGPUAnalysisCautionText('counts-where-rates-expected').text).toMatch(
      /count or an amount, where this method is meant for rates/
    );
  });
});

describe('deriveGPUAnalysisCautions for the new facts', () => {
  it('raises significance-not-attainable when the correction cannot be satisfied', () => {
    expect(
      deriveGPUAnalysisCautions({correction: 'bonferroni', testCount: 144, permutationCount: 999})
    ).toEqual(['pseudo-p-values', 'significance-not-attainable']);
    expect(
      deriveGPUAnalysisCautions({correction: 'bonferroni', testCount: 144, permutationCount: 2879})
    ).toEqual(['pseudo-p-values']);
    expect(
      deriveGPUAnalysisCautions({correction: 'holm', testCount: 144, permutationCount: 999})
    ).toContain('significance-not-attainable');
    expect(
      deriveGPUAnalysisCautions({correction: 'sidak', testCount: 144, permutationCount: 2806})
    ).toContain('significance-not-attainable');
    expect(
      deriveGPUAnalysisCautions({
        correction: 'false-discovery-rate',
        testCount: 144,
        permutationCount: 999
      })
    ).toEqual(['pseudo-p-values']);
    expect(
      deriveGPUAnalysisCautions({
        correction: 'false-discovery-rate',
        testCount: 144,
        permutationCount: 18
      })
    ).toContain('significance-not-attainable');
  });

  it('reads corrected false as no correction and uses the level fact', () => {
    expect(deriveGPUAnalysisCautions({corrected: false, permutationCount: 9})).toEqual([
      'pseudo-p-values',
      'multiple-testing-uncorrected',
      'significance-not-attainable'
    ]);
    expect(deriveGPUAnalysisCautions({corrected: false, permutationCount: 19})).toEqual([
      'pseudo-p-values',
      'multiple-testing-uncorrected'
    ]);
    expect(
      deriveGPUAnalysisCautions({
        correction: 'none',
        permutationCount: 999,
        significanceLevel: 0.0005
      })
    ).toEqual(['pseudo-p-values', 'multiple-testing-uncorrected', 'significance-not-attainable']);
  });

  it('does not raise it without the facts it needs', () => {
    expect(deriveGPUAnalysisCautions({correction: 'bonferroni', permutationCount: 999})).toEqual([
      'pseudo-p-values'
    ]);
    expect(deriveGPUAnalysisCautions({corrected: true, permutationCount: 999})).toEqual([
      'pseudo-p-values'
    ]);
    expect(deriveGPUAnalysisCautions({correction: 'bonferroni', testCount: 144})).toEqual([]);
  });

  it('raises it for zero permutations only when permutation p-values are all there is', () => {
    expect(deriveGPUAnalysisCautions({permutationCount: 0, permutationsRequired: true})).toEqual([
      'significance-not-attainable'
    ]);
    expect(deriveGPUAnalysisCautions({permutationCount: 0})).toEqual([]);
    expect(deriveGPUAnalysisCautions({permutationCount: 0, permutationsRequired: false})).toEqual(
      []
    );
  });

  it('raises the three fact-driven cautions', () => {
    expect(deriveGPUAnalysisCautions({smallDenominatorAreaCount: 3})).toEqual([
      'small-denominators'
    ]);
    expect(deriveGPUAnalysisCautions({smallDenominatorAreaCount: 0})).toEqual([]);
    expect(deriveGPUAnalysisCautions({poorModelFitReason: 'residuals clustered'})).toEqual([
      'poor-model-fit'
    ]);
    expect(deriveGPUAnalysisCautions({poorModelFitReason: '  '})).toEqual([]);
    expect(deriveGPUAnalysisCautions({rowOrderDependentInference: true})).toEqual([
      'row-order-dependent-inference'
    ]);
    expect(deriveGPUAnalysisCautions({rowOrderDependentInference: false})).toEqual([]);
    expect(() => deriveGPUAnalysisCautions({smallDenominatorAreaCount: -1})).toThrow(
      /"smallDenominatorAreaCount"/
    );
  });

  it('takes the feature count as features analysed, so a zero tested count does not hide it', () => {
    expect(deriveGPUAnalysisCautions({featureCount: 144, permutationCount: 0})).toEqual([]);
    expect(deriveGPUAnalysisCautions({featureCount: 12, permutationCount: 0})).toEqual([
      'fewer-than-thirty-features'
    ]);
  });
});

describe('local-geary-classes', () => {
  it('pins the values to GPU_LOCAL_GEARY_CLASS', () => {
    // gpu-local-permutation-test.ts:119-131: undefined 0, highHigh 1, lowLow 2,
    // negativeAssociation 3, notSignificant 4.
    const classes = getGPUAnalysisResultClasses('local-geary-classes');
    expect(classes.map(resultClass => resultClass.value)).toEqual([
      GPU_LOCAL_GEARY_CLASS.highHigh,
      GPU_LOCAL_GEARY_CLASS.lowLow,
      GPU_LOCAL_GEARY_CLASS.negativeAssociation,
      GPU_LOCAL_GEARY_CLASS.notSignificant,
      GPU_LOCAL_GEARY_CLASS.undefined
    ]);
    expect(findClass('local-geary-classes', GPU_LOCAL_GEARY_CLASS.undefined)?.tone).toBe(
      'excluded'
    );
    expect(findClass('local-geary-classes', GPU_LOCAL_GEARY_CLASS.notSignificant)?.tone).toBe(
      'neutral'
    );
    expect(findClass('local-geary-classes', GPU_LOCAL_GEARY_CLASS.highHigh)?.tone).toBe('high');
    expect(findClass('local-geary-classes', GPU_LOCAL_GEARY_CLASS.lowLow)?.tone).toBe('low');
    expect(() => getGPUAnalysisResultClasses('local-geary-classes', 'catalog-text')).toThrow(
      /accepted: "contributor-integer"/
    );
  });

  it('numbers the classes differently from local Moran', () => {
    // The same code means different things: 4 is not significant for Geary, high-low for Moran.
    expect(findClass('local-geary-classes', 4)?.label).toBe('Not significant');
    expect(findClass('local-moran-clusters', 4)?.label).toBe('High value among low values');
    expect(findClass('local-geary-classes', 0)?.label).toBe('Undefined or not tested');
    expect(findClass('local-moran-clusters', 0)?.label).toBe('Not significant or not analysed');
  });

  it('is tallied and validated, but the validator cannot tell it from local Moran', () => {
    const column = [0, 1, 2, 3, 4, 4, 4];
    expect(classifyGPUAnalysisResultValues('local-geary-classes', column).counts.get(4)).toBe(3);
    expect(
      validateGPUAnalysisResultEncoding('local-geary-classes', 'contributor-integer', column)
    ).toBe(1);
    // Overlap: the same column scores 1 under local Moran's scheme too.
    expect(
      validateGPUAnalysisResultEncoding('local-moran-clusters', 'contributor-integer', column)
    ).toBe(1);
    // A 0/1 column scores 1 under both as well; text and out-of-range codes do not.
    expect(
      validateGPUAnalysisResultEncoding('local-geary-classes', 'contributor-integer', [0, 1, 1])
    ).toBe(1);
    expect(
      validateGPUAnalysisResultEncoding('local-geary-classes', 'contributor-integer', ['HH', 'LL'])
    ).toBe(0);
    expect(
      validateGPUAnalysisResultEncoding('local-geary-classes', 'contributor-integer', [0, 1, 7, 8])
    ).toBe(0.5);
  });
});

describe('review regressions: inference and floating-point labels', () => {
  it('states the BH rank rule and a sufficient tied-minimum case without claiming necessity', () => {
    const facts = getGPUAnalysisSignificanceAttainability({
      testCount: 100,
      permutationCount: 99,
      significanceLevel: 0.05,
      correction: 'false-discovery-rate'
    });
    expect(facts.testsNeededAtMinimum).toBe(20);
    expect(facts.reason).toContain(
      'largest rank k whose k-th smallest p-value is at most k times 0.05 / 100'
    );
    expect(facts.reason).toContain('20 tests tied at the minimum would suffice');
    expect(facts.reason).not.toContain('must');
    // All 100 values of 0.02 pass at rank 100, without any attaining the minimum 0.01.
    expect(0.02).toBeLessThanOrEqual((100 * 0.05) / 100);
  });

  it('round-trips adjacent floats, keeping duplicates equal and the range distinct', () => {
    const values = [1, 1 + Number.EPSILON, 1 + 2 * Number.EPSILON];
    expect(formatGPUAnalysisDistinctValues(values)).toEqual(values.map(String));
    expect(formatGPUAnalysisDistinctValues([...values, values[0]])).toEqual(
      [...values, values[0]].map(String)
    );
    expect(
      summarizeGPUAnalysisResult('value-range', {
        minimum: values[0],
        maximum: values[2],
        valueCount: 3
      })
    ).toContain('from 1 to 1.0000000000000004');
  });

  it('describes insufficient uncorrected resolution without inventing a correction', () => {
    const text = getGPUAnalysisCautionText('significance-not-attainable', {
      correction: 'none',
      testCount: 63,
      permutationCount: 9,
      significanceLevel: 0.05
    }).text;
    expect(text).toContain('smallest pseudo p-value is 0.1');
    expect(text).toContain('at least 19 permutations');
    expect(text).not.toContain('correction');
    expect(
      deriveGPUAnalysisCautions({
        correction: 'none',
        testCount: 0,
        permutationCount: 0,
        permutationsRequired: true
      })
    ).not.toContain('multiple-testing-uncorrected');
    const marginal = getGPUAnalysisCautionText('multiple-testing-uncorrected', {
      significanceLevel: 0.1
    }).text;
    expect(marginal).toContain('about 10 in 100');
    expect(marginal).toContain('even when tests depend on each other');
    expect(marginal).toContain('joint behaviour');
  });
});
