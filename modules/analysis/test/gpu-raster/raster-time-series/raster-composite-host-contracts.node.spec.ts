// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {GPUCubeLayoutTranspose} from '../../../src/dataframe-analysis/cube-layout/gpu-cube-layout-transpose';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterTemporalResample,
  getGPURasterTemporalResampleParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {
  GPURasterSnapshotCovariance,
  getGPURasterSnapshotCovarianceParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance';
import {GPURasterEOFProjection} from '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection';
import {GPURasterMedoidComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite';
import {
  GPURasterGeometricMedianComposite,
  getGPURasterGeometricMedianCompositeParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {
  GPURasterBestAvailablePixel,
  getGPURasterBestAvailablePixelParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel';
import {
  GPURasterPhenologyMetrics,
  getGPURasterPhenologyMetricsParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-phenology-metrics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('PCK-14 accepts all ordinary hundredths with an encoded default threshold', () => {
  for (let hundredths = 1; hundredths < 100; hundredths++) {
    const value = hundredths / 100;
    const packed = getGPURasterPhenologyMetricsParameterValues({
      startFraction: value,
      endFraction: value,
      minimumAmplitude: 0
    });
    expect(packed[3]).toBe(Math.fround(value));
  }
  const packed = getGPURasterPhenologyMetricsParameterValues({
    startFraction: 0.7,
    endFraction: 0.2,
    minimumAmplitude: 0
  });
  expect(new Uint32Array(packed.buffer)[3]).toBe(0x3f333333);
  // Requested order must hold even when encoding makes the endpoints equal.
  expect(() =>
    getGPURasterPhenologyMetricsParameterValues({
      startFraction: 0.7,
      endFraction: 0.2,
      minimumAmplitude: 0,
      secondSeasonMinimumFraction: 0.7 - 1e-10
    })
  ).toThrow(/secondSeasonMinimumFraction/);
});

const CLOUD_SETTINGS = {
  targetDayOfYear: 180,
  dayOfYearSigma: 20,
  dayOfYearWeight: 0.5,
  targetYear: 2020,
  maximumYearDistance: 5,
  yearWeight: 0.2,
  minimumCloudDistance: 10,
  maximumCloudDistance: 100,
  cloudWeight: 0.3
};
it('W3-02 rejects collapsed cloud limits without mutating the target', () => {
  const target = new Float32Array(9).fill(77);
  expect(() =>
    getGPURasterBestAvailablePixelParameterValues(
      {...CLOUD_SETTINGS, maximumCloudDistance: 10.0000001},
      target
    )
  ).toThrow(/minimumCloudDistance.*10.*maximumCloudDistance.*10\.0000001.*encoded.*10/);
  expect(Array.from(target)).toEqual(Array(9).fill(77));
  expect(() =>
    getGPURasterBestAvailablePixelParameterValues({...CLOUD_SETTINGS, maximumCloudDistance: 10})
  ).toThrow(/CloudDistance/);
});

// Independent packet words: 400 has binary exponent 8; unit = 8+1+1-95 = -85.
for (const [name, length, expected, pack] of [
  [
    'resample',
    3,
    [0, 1, 7],
    (target: Float32Array) =>
      getGPURasterTemporalResampleParameterValues(
        {gridStart: 0, gridStep: 1, maximumGap: 7},
        target
      )
  ],
  [
    'covariance',
    2,
    [-85, 10],
    (target: Float32Array) =>
      getGPURasterSnapshotCovarianceParameterValues({maximumValueMagnitude: 10}, 1, target)
  ],
  [
    'geometric',
    1,
    [0.000009999999747378752],
    (target: Float32Array) => getGPURasterGeometricMedianCompositeParameterValues({}, target)
  ]
] as const) {
  it(`W3-04 ${name} rejects empty/short targets and preserves padded storage`, () => {
    for (const size of new Set([0, length - 1])) {
      const target = new Float32Array(size).fill(77);
      expect(() => pack(target)).toThrow(/target/);
      expect(Array.from(target)).toEqual(Array(size).fill(77));
    }
    expect(Array.from(pack(new Float32Array(length)))).toEqual(expected);
    const storage = new Float32Array(length + 4).fill(77);
    const target = storage.subarray(1, length + 3);
    expect(pack(target)).toBe(target);
    expect(Array.from(target.subarray(0, length))).toEqual(expected);
    expect(storage[0]).toBe(77);
    expect(target[length]).toBe(77);
    expect(target[length + 1]).toBe(77);
    if (name === 'resample') expect(target[2]).toBe(7);
  });
}

for (const kind of [
  'resample',
  'covariance',
  'eof',
  'medoid',
  'geometric',
  'best',
  'phenology'
] as const) {
  it(`W3-03 ${kind} validates named formats and required ports before lengths`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = (format: 'float32' | 'uint32' | 'uint32x2') =>
        createTransientView(graph, `port-${serial++}`, format, 100);
      const output: Record<string, unknown> = {};
      const props: Record<string, unknown> = {
        id: 'contract',
        cellCount: 1,
        sliceCount: 2,
        bandCount: 1,
        iterationCount: 1,
        modeCount: 1,
        gridCount: 1,
        maximumSeasonCount: 1,
        output
      };
      const ports: Record<string, 'float32' | 'uint32' | 'uint32x2'> =
        kind === 'resample'
          ? {
              values: 'float32',
              parameters: 'float32',
              mask: 'uint32',
              targetTimes: 'uint32x2',
              'dateAxis.acquisitionTimes': 'uint32x2',
              'output.values': 'float32',
              'output.status': 'uint32'
            }
          : kind === 'covariance'
            ? {
                values: 'float32',
                mask: 'uint32',
                weights: 'float32',
                parameters: 'float32',
                'output.covarianceWords': 'uint32',
                'output.keptCellCount': 'uint32'
              }
            : kind === 'eof'
              ? {
                  values: 'float32',
                  mask: 'uint32',
                  weights: 'float32',
                  parameters: 'float32',
                  temporalModes: 'float32',
                  singularValues: 'float32',
                  'output.eofs': 'float32',
                  'output.status': 'uint32'
                }
              : kind === 'medoid'
                ? {
                    values: 'float32',
                    mask: 'uint32',
                    'output.composite': 'float32',
                    'output.sliceIndex': 'uint32',
                    'output.status': 'uint32'
                  }
                : kind === 'geometric'
                  ? {
                      values: 'float32',
                      mask: 'uint32',
                      parameters: 'float32',
                      'output.composite': 'float32',
                      'output.lastStep': 'float32',
                      'output.status': 'uint32'
                    }
                  : kind === 'best'
                    ? {
                        values: 'float32',
                        mask: 'uint32',
                        cloudDistance: 'float32',
                        sliceAttributes: 'float32',
                        parameters: 'float32',
                        'output.composite': 'float32',
                        'output.bestSlice': 'uint32',
                        'output.bestScore': 'float32'
                      }
                    : {
                        'dateAxis.acquisitionTimes': 'uint32x2',
                        smoothed: 'float32',
                        parameters: 'float32',
                        'output.metrics': 'float32',
                        'output.flags': 'uint32',
                        'output.seasonCount': 'uint32',
                        'output.status': 'uint32'
                      };
      const set = (path: string, value: unknown) => {
        if (path.startsWith('output.')) output[path.slice(7)] = value;
        else if (path.startsWith('dateAxis.'))
          (props['dateAxis'] as Record<string, unknown>)[path.slice(9)] = value;
        else props[path] = value;
      };
      props['dateAxis'] = {
        acquisitionTimes: view('uint32x2'),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        }
      };
      for (const [path, format] of Object.entries(ports)) set(path, view(format));
      const construct = () => {
        switch (kind) {
          case 'resample':
            return new GPURasterTemporalResample(
              props as unknown as ConstructorParameters<typeof GPURasterTemporalResample>[0]
            );
          case 'covariance':
            return new GPURasterSnapshotCovariance(
              props as unknown as ConstructorParameters<typeof GPURasterSnapshotCovariance>[0]
            );
          case 'eof':
            return new GPURasterEOFProjection(
              props as unknown as ConstructorParameters<typeof GPURasterEOFProjection>[0]
            );
          case 'medoid':
            return new GPURasterMedoidComposite(
              props as unknown as ConstructorParameters<typeof GPURasterMedoidComposite>[0]
            );
          case 'geometric':
            return new GPURasterGeometricMedianComposite(
              props as unknown as ConstructorParameters<typeof GPURasterGeometricMedianComposite>[0]
            );
          case 'best':
            return new GPURasterBestAvailablePixel(
              props as unknown as ConstructorParameters<typeof GPURasterBestAvailablePixel>[0]
            );
          case 'phenology':
            return new GPURasterPhenologyMetrics(
              props as unknown as ConstructorParameters<typeof GPURasterPhenologyMetrics>[0]
            );
        }
      };
      expect(construct).not.toThrow();
      for (const [path, format] of Object.entries(ports)) {
        set(path, view(format === 'float32' ? 'uint32' : 'float32'));
        expect(construct, path).toThrow(
          new RegExp(`contract ${path.replaceAll('.', '\\.')}.*${format}`)
        );
        set(path, view(format));
      }
      if (kind === 'phenology') {
        set('dateAxis.acquisitionTimes', undefined);
        expect(construct).toThrow(/contract dateAxis.acquisitionTimes.*uint32x2.*undefined/);
        props['dateAxis'] = undefined;
        expect(construct).not.toThrow();
      }
      if (kind === 'resample')
        for (const path of ['values', 'parameters']) {
          set(path, undefined);
          expect(construct).toThrow(new RegExp(`contract ${path}.*float32.*undefined`));
          set(path, view('float32'));
        }
    } finally {
      device.destroy();
    }
  });
}

it('W3-04 EOF word decoder respects bounded byte views and refuses missing upper-triangle entries', async () => {
  const {getGPURasterSnapshotCovarianceMatrix} = await import(
    '../../../src/raster-analysis/raster-time-series/raster-snapshot-eof-decomposition'
  );
  // Signed 96-bit integers [1,-1,1] at unit 2^-2 encode [.25,-.25,.25].
  const words = Uint32Array.of(77, 1, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 1, 0, 0, 88);
  const bytes = new Uint8Array(words.buffer, 4, 36);
  expect(Array.from(getGPURasterSnapshotCovarianceMatrix(bytes, 2, -2))).toEqual([
    0.25, -0.25, -0.25, 0.25
  ]);
  expect(Array.from(getGPURasterSnapshotCovarianceMatrix(bytes.slice().buffer, 2, -2))).toEqual([
    0.25, -0.25, -0.25, 0.25
  ]);
  expect(() => getGPURasterSnapshotCovarianceMatrix(bytes.subarray(0, 35), 2, -2)).toThrow();
  expect(() => getGPURasterSnapshotCovarianceMatrix(words.subarray(1, 9), 2, -2)).toThrow();
});

it('review B2 the documented NDVI and QA transposes coexist in one graph', () => {
  const guide = readFileSync('docs/api/gpu-raster/raster-time-series.md', 'utf8');
  const blocks = [
    ...guide.matchAll(/graph\.add\(new GPUCubeLayoutTranspose\(\{([\s\S]*?)\}\)\);/g)
  ];
  expect(blocks).toHaveLength(2);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    for (const [index, block] of blocks.entries()) {
      const format = index === 0 ? 'float32' : 'uint32';
      const id = block[1].match(/id:\s*'([^']+)'/)?.[1];
      graph.add(
        new GPUCubeLayoutTranspose({
          id,
          cellCount: 2,
          periodCount: 2,
          from: 'period-major',
          values: createTransientView(graph, `source-${index}`, format, 4),
          output: createTransientView(graph, `destination-${index}`, format, 4)
        })
      );
    }
    expect(guide).toContain('id: `ndvi-${slice}`');
  } finally {
    device.destroy();
  }
});
