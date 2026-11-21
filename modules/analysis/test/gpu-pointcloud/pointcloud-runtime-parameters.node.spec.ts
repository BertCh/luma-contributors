// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudChange,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudICP,
  GPUPointCloudSMRF,
  GPUPointCloudStatisticalFilter,
  getGPUPointCloudChangeParameterValues,
  getGPUPointCloudFeaturesParameterValues,
  getGPUPointCloudGroundParameterLength,
  getGPUPointCloudGroundParameterValues,
  getGPUPointCloudICPParameterValues,
  getGPUPointCloudSMRFParameterLength,
  getGPUPointCloudSMRFParameterValues,
  getGPUPointCloudStatisticalFilterParameterValues
} from '@luma-contributors/analysis/pointcloud-analysis';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

const GRID = {origin: [0, 0] as const, cellSize: 0.7, width: 3, height: 3};

function makeDeclaration() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `pointcloud-parameters-${sequence++}`, format, length);
  const positions = view('float32x3', 9);
  const counts = view('uint32', 9);
  const classifications = view('uint32', 9);
  const status = view('uint32', 9);
  const features = {
    positions,
    counts,
    indices: view('uint32', 36),
    capacity: 4,
    output: {
      normals: view('float32x3', 9),
      eigenvalues: view('float32x3', 9),
      shapes: view('float32x3', 9),
      status
    }
  };
  const ground = {
    positions,
    classifications,
    grid: GRID,
    radii: [1, 3],
    initialThreshold: 0.2,
    slope: 0.3,
    maximumThreshold: 2,
    minimumSurface: view('float32', 9),
    coverage: view('uint32', 9),
    output: {groundMask: view('uint32', 9), outputClassifications: view('uint32', 9), status}
  };
  const smrf = {
    positions,
    classifications,
    grid: GRID,
    window: GRID.cellSize * 3,
    output: {
      groundSurface: view('float32', 9),
      groundMask: view('uint32', 9),
      outputClassifications: view('uint32', 9),
      status
    }
  };
  const filter = {
    counts,
    squaredDistances: view('float32', 36),
    capacity: 4,
    output: {meanDistances: view('float32', 9), validity: view('uint32', 9)}
  };
  return {graph, view, features, ground, smrf, filter};
}

it('packs orientation and population-standard-deviation settings', () => {
  expect([...getGPUPointCloudFeaturesParameterValues()]).toEqual([0, 0, 0, 0]);
  expect([...getGPUPointCloudFeaturesParameterValues({viewpoint: [1, -2, 3]})]).toEqual([
    1, -2, 3, 1
  ]);
  expect([...getGPUPointCloudStatisticalFilterParameterValues()]).toEqual([2]);
  expect([...getGPUPointCloudStatisticalFilterParameterValues({standardDeviations: 0})]).toEqual([
    0
  ]);
  for (const value of [-1, Infinity, NaN, 1e40]) {
    expect(() =>
      getGPUPointCloudStatisticalFilterParameterValues({standardDeviations: value})
    ).toThrow(/^gpu-pointcloud-statistical-filter standardDeviations/);
  }
  expect(() => getGPUPointCloudFeaturesParameterValues({viewpoint: [0, Infinity, 0]})).toThrow(
    /^gpu-pointcloud-features viewpoint\[1\]/
  );
});

it('packs exact uint32 M3C2 support and float32 uncertainty bit patterns', () => {
  const words = getGPUPointCloudChangeParameterValues({
    minimumSupport: 0xffffffff,
    registrationError: 2,
    confidenceMultiplier: 3,
    registrationCovariance: [1, 0, 0, 4, 0, 9],
    samplingCovariance: {referenceVariance: 2, comparisonVariance: 3, correlationLength: 0.5}
  });
  expect(words[0]).toBe(0xffffffff);
  expect([...new Float32Array(words.buffer).slice(1)]).toEqual([4, 3, 1, 0, 0, 4, 0, 9, 2, 3, 0.5]);
  expect(() => getGPUPointCloudChangeParameterValues({minimumSupport: 1})).toThrow(
    /^gpu-pointcloud-change minimumSupport/
  );
  expect(() =>
    getGPUPointCloudChangeParameterValues({registrationCovariance: [1, 2, 0, 1, 0, 0]})
  ).toThrow(/positive semidefinite/);
  expect(() => getGPUPointCloudChangeParameterValues({confidenceMultiplier: 0})).toThrow(
    /confidenceMultiplier/
  );
});

it('packs ICP distance, robust scale, tolerances and an origin-adjusted initial transform', () => {
  const values = getGPUPointCloudICPParameterValues({maximumDistance: 2, robustScale: 0.5});
  expect([...values.slice(0, 6)]).toEqual([2, 4, 0.5, 0.25, Math.fround(1e-5), Math.fround(1e-5)]);
  expect([...values.slice(6)]).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const rotated = getGPUPointCloudICPParameterValues(
    {maximumDistance: 2, initialTransform: [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
    [100, 200, 0]
  );
  expect([...rotated.slice(18, 21)]).toEqual([-300, -100, 0]);
  for (const maximumDistance of [0, -1, 1e-30, 1e30]) {
    expect(() => getGPUPointCloudICPParameterValues({maximumDistance})).toThrow(
      /^gpu-pointcloud-icp maximumDistance/
    );
  }
  expect(() => getGPUPointCloudICPParameterValues({maximumDistance: 1, robustScale: 0})).toThrow(
    /^gpu-pointcloud-icp robustScale/
  );
  expect(() =>
    getGPUPointCloudICPParameterValues({maximumDistance: 1, initialTransform: [1]})
  ).toThrow(/^gpu-pointcloud-icp initialTransform/);
});

it('matches Huber and Tukey constructor scale limits when packing ICP parameters', () => {
  const {features, view} = makeDeclaration();
  for (const method of ['huber', 'tukey'] as const) {
    const construct = (scale: number) =>
      new GPUPointCloudICP({
        source: features.positions,
        target: features.positions,
        maximumDistance: 1,
        robust: {method, scale},
        maximumIterations: 1,
        output: {
          transform: view('float32', 16),
          history: view('float32x3', 1),
          inlierMask: view('uint32', 9),
          status: view('uint32', 1)
        }
      });
    for (const robustScale of [1e20, 1e-30]) {
      const settings = {maximumDistance: 1, robustScale, robustMethod: method};
      if (method === 'huber') {
        expect(construct(robustScale).props.robust?.scale).toBe(robustScale);
        const values = getGPUPointCloudICPParameterValues(settings);
        expect(values[2]).toBe(Math.fround(robustScale));
        // The square is unused by Huber; keep the packed buffer finite and nonzero.
        expect(values[3]).toBe(1);
        expect([...values].every(Number.isFinite)).toBe(true);
      } else {
        expect(() => construct(robustScale)).toThrow(/^gpu-pointcloud-icp robust.scale/);
        expect(() => getGPUPointCloudICPParameterValues(settings)).toThrow(
          /^gpu-pointcloud-icp robustScale.*tukey/
        );
      }
    }
    for (const robustScale of [0, -1, 1e-50, 1e40, NaN, Infinity]) {
      const settings = {maximumDistance: 1, robustScale, robustMethod: method};
      expect(() => construct(robustScale)).toThrow(/^gpu-pointcloud-icp robust.scale/);
      expect(() => getGPUPointCloudICPParameterValues(settings)).toThrow(
        /^gpu-pointcloud-icp robustScale/
      );
    }
    const settings = {maximumDistance: 1, robustScale: 0.5, robustMethod: method};
    expect(construct(0.5).props.robust?.scale).toBe(0.5);
    expect([...getGPUPointCloudICPParameterValues(settings).slice(2, 4)]).toEqual([0.5, 0.25]);
  }
});

it('keeps Tukey scale validation as the default packing behavior', () => {
  for (const robustScale of [1e20, 1e-30]) {
    expect(() => getGPUPointCloudICPParameterValues({maximumDistance: 1, robustScale})).toThrow(
      /^gpu-pointcloud-icp robustScale.*tukey/
    );
  }
});

it('rejects ICP local transforms that overflow float32 after origin conversion', () => {
  const initialTransform = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const {features, view} = makeDeclaration();
  for (const horizontalOrigin of [2e38, -2e38]) {
    const origin = [horizontalOrigin, 0, 0] as const;
    expect(() =>
      getGPUPointCloudICPParameterValues({maximumDistance: 1, initialTransform}, origin)
    ).toThrow(/^gpu-pointcloud-icp initialTransform.*local.*finite float32/);
    expect(
      () =>
        new GPUPointCloudICP({
          id: 'test-local-icp',
          source: features.positions,
          target: features.positions,
          initialTransform,
          origin,
          maximumDistance: 1,
          maximumIterations: 1,
          output: {
            transform: view('float32', 16),
            history: view('float32x3', 1),
            inlierMask: view('uint32', 9),
            status: view('uint32', 1)
          }
        })
    ).toThrow(/^test-local-icp initialTransform.*local.*finite float32/);
  }
  // A large but representable local translation remains valid.
  const values = getGPUPointCloudICPParameterValues(
    {maximumDistance: 1, initialTransform},
    [1e38, 0, 0]
  );
  expect(values[18]).toBe(Math.fround(-2e38));
  expect([...values].every(Number.isFinite)).toBe(true);
});

it('rounds derived morphology thresholds once, retaining the constructor convention', () => {
  const settings = {initialThreshold: 0.2, slope: 0.15, maximumThreshold: 0.5};
  const radii = [1, 2, 5];
  expect(getGPUPointCloudGroundParameterLength(radii)).toBe(3);
  expect([...getGPUPointCloudGroundParameterValues(settings, GRID, radii)]).toEqual([
    Math.fround(0.2 + 0.15 * 0.7),
    Math.fround(0.2 + 0.15 * 0.7),
    0.5
  ]);
  expect(getGPUPointCloudSMRFParameterLength(3)).toBe(5);
  expect([...getGPUPointCloudSMRFParameterValues({}, GRID, 3)]).toEqual([
    1.25,
    0.5,
    ...[1, 2, 3].map(radius => Math.fround(0.15 * 0.7 * radius))
  ]);
  expect(() =>
    getGPUPointCloudGroundParameterValues({...settings, maximumThreshold: 0.1}, GRID, radii)
  ).toThrow(/^gpu-pointcloud-ground maximumThreshold/);
  expect(() => getGPUPointCloudSMRFParameterValues({slope: -1}, GRID, 3)).toThrow(
    /^gpu-pointcloud-smrf slope/
  );
  expect(() => getGPUPointCloudSMRFParameterValues({}, GRID, 65)).toThrow(/maximumRadius/);
});

it('validates every parameter view length at construction', () => {
  const {view, features, ground, smrf, filter} = makeDeclaration();
  expect(
    () =>
      new GPUPointCloudFeatures({
        ...features,
        parameters: view('float32', 3)
      })
  ).toThrow(/^gpu-pointcloud-features parameters/);
  expect(
    () =>
      new GPUPointCloudGround({
        ...ground,
        parameters: view('float32', 1)
      })
  ).toThrow(/^gpu-pointcloud-ground parameters/);
  expect(
    () =>
      new GPUPointCloudSMRF({
        ...smrf,
        parameters: view('float32', 4)
      })
  ).toThrow(/^gpu-pointcloud-smrf parameters/);
  expect(
    () =>
      new GPUPointCloudStatisticalFilter({
        ...filter,
        parameters: view('float32', 2)
      })
  ).toThrow(/^gpu-pointcloud-statistical-filter parameters/);
});

it('reads resident parameters and audits the feature and ICP shared eigensolvers', () => {
  const {graph, view, features, ground, smrf, filter} = makeDeclaration();
  const change = {
    reference: features.positions,
    comparison: features.positions,
    cores: features.positions,
    normals: features.output.normals,
    alignment: 'already-aligned' as const,
    radius: 1,
    halfDepth: 1,
    samplingCovariance: {referenceVariance: 1, comparisonVariance: 1, correlationLength: 1},
    output: {
      changes: view('float32x3', 9),
      roughness: view('float32x3', 9),
      counts: view('uint32', 18),
      status: view('uint32', 9)
    }
  };
  expect(
    () =>
      new GPUPointCloudChange({
        ...change,
        parameters: view('uint32', 11)
      })
  ).toThrow(/^gpu-pointcloud-change parameters/);
  const contributors = [
    new GPUPointCloudChange({
      ...change,
      parameters: view('uint32', 12)
    }),
    new GPUPointCloudChange({
      ...change,
      id: 'indexed-change',
      parameters: view('uint32', 12),
      grid: {origin: [-2, -2, -2], cellSize: 1, dimensions: [6, 6, 6]}
    }),
    new GPUPointCloudFeatures({
      ...features,
      parameters: view('float32', 4),
      output: {...features.output, covariance: view('float32', 54)}
    }),
    new GPUPointCloudGround({
      ...ground,
      parameters: view('float32', 2)
    }),
    new GPUPointCloudSMRF({
      ...smrf,
      mask: view('uint32', 9),
      parameters: view('float32', 5)
    }),
    new GPUPointCloudStatisticalFilter({
      ...filter,
      parameters: view('float32', 1)
    }),
    ...(['point-to-point', 'point-to-plane'] as const).map(
      metric =>
        new GPUPointCloudICP({
          id: `icp-${metric}`,
          metric,
          targetNormals: view('float32x3', 9),
          source: features.positions,
          target: view('float32x3', 9),
          maximumDistance: 1,
          robust: {method: 'tukey', scale: 1},
          targetGrid: {origin: [-1, -1, -1], cellSize: 1, dimensions: [5, 5, 5]},
          parameters: view('float32', 22),
          maximumIterations: 1,
          output: {
            transform: view('float32', 16),
            history: view('float32x3', 1),
            inlierMask: view('uint32', 9),
            status: view('uint32', 1)
          }
        })
    )
  ];
  // Chunk kernels intentionally share a pipeline ID. Retain every node's source so a later
  // dispatch cannot hide the feature eigensolver or an earlier parameter reader in this audit.
  const sources = new Map<string, string>();
  for (const contributor of contributors) {
    for (const node of contributor.getCommandNodes(graph)) {
      for (const {source, constants} of getParsedKernelSources([node]).configurations) {
        expect(auditWGSLModule(source, constants), node.id).toEqual([]);
        sources.set(node.id, source);
      }
    }
  }
  expect(sources.get('gpu-pointcloud-change')).toContain('readParameters(0u)');
  expect(sources.get('indexed-change-reference')).toContain('bitcast<f32>(readParameters(11u))');
  const featureSource = sources.get('gpu-pointcloud-features-chunk-0')!;
  expect(featureSource).toContain('getSymmetricEigen_3(&matrix, &vectors, 3u, 8u, 1e-6)');
  const icpSource = sources.get('icp-point-to-point-solve-0-chunk-0')!;
  expect(icpSource).toContain('getSymmetricEigen_4(&matrix, &vectors, 4u, 8u, 1e-6)');
  expect(featureSource + icpSource).not.toContain('atan2(');
  expect(sources.get('icp-point-to-plane-solve-0-chunk-0')).toContain('readParameters(4u)');
  expect(sources.get('icp-point-to-point-match-0')).toContain('readParameters(1u)');
  expect(sources.get('icp-point-to-point-initialize')).toContain('readParameters(entry + 6u)');
  for (const identifier of [
    'gpu-pointcloud-features-orient-chunk-0',
    'gpu-pointcloud-ground-classify-0-chunk-0',
    'gpu-pointcloud-smrf-object-mask-1-chunk-0',
    'gpu-pointcloud-smrf-classify',
    'gpu-pointcloud-statistical-filter-threshold'
  ]) {
    expect(sources.get(identifier), identifier).toContain('readParameters(');
  }
});
