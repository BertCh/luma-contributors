// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';
import {
  GEOGRAPHIC_INPUT,
  GEOGRAPHIC_EXPECTED,
  AXIAL_DIRECTIONAL_EXPECTED,
  COINCIDENT_INPUT
} from './geographic-distribution-audit-fixtures';

function createPairView(
  rig: AnalysisRig,
  storage: GraphDataView<'float32'>
): GraphDataView<'float32x2'> {
  return rig.graph.createDataView(storage.buffer, {
    format: 'float32x2',
    length: storage.length / 2
  });
}

for (const reduction of ['chunked', 'sorted'] as const) {
  it(`retains a rotated narrow ellipse on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const ellipses = rig.output('float32', 3);
      rig.run(
        new GPUGeographicDistribution({
          // +/- (100,99) +/- (1,1) has exact covariance [[10001,9901],[9901,9802]].
          // det=1, trace=19803: lambdaSmall=1/lambdaLarge.
          positions: rig.input(
            Float32Array.of(101, 100, 99, 98, -99, -98, -101, -100),
            'float32x2',
            4
          ),
          reduction,
          parameters: rig.input(
            getGPUGeographicDistributionParameterValues({ellipseConvention: 'standard'}),
            'float32'
          ),
          output: {ellipses}
        })
      );
      const actual = await rig.readFloat(ellipses);
      expectOracleRows(actual.slice(1, 2), [0.0071061522297724144], 2e-8);
      expectOracleRows(actual.slice(2), [140.72313224734092], 2e-5);
    } finally {
      rig.destroy();
    }
  });
  for (const [label, coordinates, shortAxis, longAxis] of [
    [
      'small transverse variance',
      [
        100 + 1 / 64,
        99 + 1 / 64,
        100 - 1 / 64,
        99 - 1 / 64,
        -100 + 1 / 64,
        -99 + 1 / 64,
        -100 - 1 / 64,
        -99 - 1 / 64
      ],
      0.0001110392344137977,
      140.71602783005538
    ],
    ['exactly collinear', [1, Math.fround(0.1), -1, -Math.fround(0.1)], 0, 1.004987562260361]
  ] as const) {
    it(`retains ${label} on the ${reduction} route`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const ellipses = rig.output('float32', 3);
        rig.run(
          new GPUGeographicDistribution({
            positions: rig.input(
              Float32Array.from(coordinates),
              'float32x2',
              coordinates.length / 2
            ),
            reduction,
            parameters: rig.input(
              getGPUGeographicDistributionParameterValues({ellipseConvention: 'standard'}),
              'float32'
            ),
            output: {ellipses}
          })
        );
        const actual = await rig.readFloat(ellipses);
        expectOracleRows(actual.slice(1, 2), [shortAxis], shortAxis === 0 ? 0 : 2e-8);
        expectOracleRows(actual.slice(2), [longAxis], 2e-5);
      } finally {
        rig.destroy();
      }
    });
  }
  it(`retains a narrow nondegenerate ellipse on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const ellipses = rig.output('float32', 3);
      const vertices = rig.output('float32', 8);
      rig.run(
        new GPUGeographicDistribution({
          // Population covariance diag(1e8,1), hence exact standard axes [1,10000].
          positions: rig.input(
            Float32Array.of(
              -10000,
              -1,
              -10000,
              1,
              10000,
              -1,
              10000,
              1,
              -10000,
              -1,
              -10000,
              1,
              10000,
              -1,
              10000,
              1
            ),
            'float32x2',
            8
          ),
          reduction,
          polygonVertexCount: 4,
          parameters: rig.input(
            getGPUGeographicDistributionParameterValues({ellipseConvention: 'standard'}),
            'float32'
          ),
          output: {ellipses, ellipseVertices: createPairView(rig, vertices)}
        })
      );
      expectOracleRows(await rig.readFloat(ellipses), [Math.PI / 2, 1, 10000], 1e-6);
      expectOracleRows((await rig.readFloat(vertices)).slice(0, 2), [0, 1], 1e-6);
    } finally {
      rig.destroy();
    }
  });
  it(`distinguishes the weighted mean and median iterate on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const meanCenters = rig.output('float32', 2);
      const medianCenters = rig.output('float32', 2);
      const medianConverged = rig.output('uint32', 1);
      rig.run(
        new GPUGeographicDistribution({
          positions: rig.input(
            Float32Array.of(10, 0, 14, 0, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
            'float32x2',
            8
          ),
          weights: rig.input(Float32Array.of(1, 3, 1, 1, 1, 1, 1, 1), 'float32'),
          mask: rig.input(Uint32Array.of(1, 1, 0, 0, 0, 0, 0, 0), 'uint32'),
          maximumMedianIterations: 1,
          reduction,
          parameters: rig.input(getGPUGeographicDistributionParameterValues(), 'float32'),
          output: {
            meanCenters: createPairView(rig, meanCenters),
            medianCenters: createPairView(rig, medianCenters),
            medianConverged
          }
        })
      );
      // Independent derivation in geographic-distribution-audit.oracle.py:
      // mean=13; pull=2, inverse-distance mass=10/3, first median=13.6.
      expectOracleRows(await rig.readFloat(meanCenters), [13, 0], 1e-6);
      expectOracleRows(await rig.readFloat(medianCenters), [13.6, 0], 2e-6);
      expectOracleRows(await rig.readUint(medianConverged), [0], 0);
    } finally {
      rig.destroy();
    }
  });

  it(`checks a nonzero covariance and ellipse-angle sign on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const rotate = (values: number[]) => {
        const transformed: number[] = [];
        for (let index = 0; index < values.length; index += 2)
          transformed.push(values[index] - values[index + 1], values[index] + values[index + 1]);
        return new Float32Array(transformed);
      };
      const meanCenters = rig.output('float32', 6);
      const ellipses = rig.output('float32', 9);
      const standardDistances = rig.output('float32', 3);
      const directionalMeans = rig.output('float32', 9);
      rig.run(
        new GPUGeographicDistribution({
          positions: rig.input(rotate(GEOGRAPHIC_INPUT.positions), 'float32x2', 9),
          lineEnds: rig.input(rotate(GEOGRAPHIC_INPUT.lineEnds), 'float32x2', 9),
          weights: rig.input(new Float32Array(GEOGRAPHIC_INPUT.weights), 'float32'),
          groupIds: rig.input(new Uint32Array(GEOGRAPHIC_INPUT.groupIds), 'uint32'),
          mask: rig.input(new Uint32Array(GEOGRAPHIC_INPUT.mask), 'uint32'),
          groupCount: 3,
          reduction,
          parameters: rig.input(
            getGPUGeographicDistributionParameterValues({ellipseConvention: 'standard'}),
            'float32'
          ),
          output: {
            meanCenters: createPairView(rig, meanCenters),
            ellipses,
            standardDistances,
            directionalMeans
          }
        })
      );
      // Integer transform [[1,-1],[1,1]] gives group 0 covariance
      // [[1.25,.75],[.75,1.25]], eigenvalues [.5,2], short angle -pi/4.
      expectOracleRows(await rig.readFloat(meanCenters), [0, 0, 12, 12, NaN, NaN], 2e-6);
      expectOracleRows(
        await rig.readFloat(ellipses),
        [
          -0.7853981633974483,
          0.7071067811865476,
          1.4142135623730951,
          -0.7853981633974483,
          0,
          2.8284271247461903,
          NaN,
          NaN,
          NaN
        ],
        2e-6
      );
      expectOracleRows(
        await rig.readFloat(standardDistances),
        [1.5811388300841898, 2.8284271247461903, NaN],
        2e-6
      );
      expectOracleRows(
        await rig.readFloat(directionalMeans),
        [
          -2.677945044588987,
          0.4729537233052701,
          4.949747468305833,
          NaN,
          1,
          1.4142135623730951,
          NaN,
          NaN,
          NaN
        ],
        2e-6
      );
    } finally {
      rig.destroy();
    }
  });

  for (const ellipseConvention of ['standard', 'root-two'] as const) {
    it(`checks every ${reduction}/${ellipseConvention} geographic output against the NumPy and analytic oracle`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const orientationOnly of [false, true]) {
        for (const transformed of [false, true]) {
          const rig = new AnalysisRig(device);
          try {
            const scale = transformed ? 3 : 1;
            const standardDeviations = transformed ? 2 : 1;
            const origin: [number, number] = transformed ? [32, -16] : [0, 0];
            const transform = (values: number[]) =>
              Float32Array.from(values, (value, index) => value * scale + origin[index % 2]);
            const output = {
              counts: rig.output('uint32', 3),
              weightSums: rig.output('float32', 3),
              meanCenters: rig.output('float32', 6),
              medianCenters: rig.output('float32', 6),
              medianConverged: rig.output('uint32', 3),
              standardDistances: rig.output('float32', 3),
              ellipses: rig.output('float32', 9),
              directionalMeans: rig.output('float32', 9),
              ellipseVertices: rig.output('float32', 24),
              circleVertices: rig.output('float32', 24)
            };
            const props = {
              positions: rig.input(transform(GEOGRAPHIC_INPUT.positions), 'float32x2', 9),
              lineEnds: rig.input(transform(GEOGRAPHIC_INPUT.lineEnds), 'float32x2', 9),
              weights: rig.input(new Float32Array(GEOGRAPHIC_INPUT.weights), 'float32'),
              groupIds: rig.input(new Uint32Array(GEOGRAPHIC_INPUT.groupIds), 'uint32'),
              mask: rig.input(new Uint32Array(GEOGRAPHIC_INPUT.mask), 'uint32'),
              groupCount: 3,
              polygonVertexCount: 4,
              maximumMedianIterations: 1,
              reduction,
              parameters: rig.input(
                getGPUGeographicDistributionParameterValues({
                  origin,
                  standardDeviations,
                  ellipseConvention,
                  orientationOnly
                }),
                'float32'
              ),
              output: {
                ...output,
                meanCenters: createPairView(rig, output.meanCenters),
                medianCenters: createPairView(rig, output.medianCenters),
                ellipseVertices: createPairView(rig, output.ellipseVertices),
                circleVertices: createPairView(rig, output.circleVertices)
              }
            };
            const contributor = new GPUGeographicDistribution(props);
            // Mutating the caller's props after validation must not change the model.
            props.weights = rig.input(new Float32Array(9), 'float32');
            rig.run(contributor);
            const {counts, medianConverged, ...floating} = output;
            expectOracleRows(await rig.readUint(counts), GEOGRAPHIC_EXPECTED.counts, 0);
            expectOracleRows(
              await rig.readUint(medianConverged),
              GEOGRAPHIC_EXPECTED.medianConverged,
              0
            );
            const ellipseScale = ellipseConvention === 'root-two' ? Math.SQRT2 : 1;
            for (const [port, view] of Object.entries(floating)) {
              let expected = GEOGRAPHIC_EXPECTED[port as keyof typeof GEOGRAPHIC_EXPECTED];
              if (port === 'meanCenters' || port === 'medianCenters') {
                expected = expected.map((value, index) => value * scale + origin[index % 2]);
              } else if (port === 'standardDistances')
                expected = expected.map(value => value * scale * standardDeviations);
              else if (port === 'ellipses')
                expected = expected.map((value, index) =>
                  index % 3 === 0 ? value : value * scale * ellipseScale * standardDeviations
                );
              else if (port === 'directionalMeans')
                expected = (orientationOnly ? AXIAL_DIRECTIONAL_EXPECTED : expected).map(
                  (value, index) => (index % 3 === 2 ? value * scale : value)
                );
              else if (port === 'ellipseVertices' || port === 'circleVertices')
                expected = expected.map((value, index) => {
                  const component = index % 2;
                  const center =
                    GEOGRAPHIC_EXPECTED.meanCenters[Math.floor(index / 8) * 2 + component];
                  const shapeScale = port === 'ellipseVertices' ? ellipseScale : 1;
                  return (
                    (center + (value - center) * shapeScale * standardDeviations) * scale +
                    origin[component]
                  );
                });
              // <=85 coordinate magnitude; 2e-5 covers exp-free float32 reductions,
              // eigenvalue arithmetic and ring trigonometry while detecting wrong ports/divisors.
              expectOracleRows(await rig.readFloat(view), expected, 2e-5);
            }
          } finally {
            rig.destroy();
          }
        }
      }
    });
  }

  it(`keeps the exact coincident weighted median on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const medianCenters = rig.output('float32', 2);
      const medianConverged = rig.output('uint32', 1);
      rig.run(
        new GPUGeographicDistribution({
          positions: rig.input(new Float32Array(COINCIDENT_INPUT.positions), 'float32x2', 8),
          weights: rig.input(new Float32Array(COINCIDENT_INPUT.weights), 'float32'),
          mask: rig.input(new Uint32Array(COINCIDENT_INPUT.mask), 'uint32'),
          maximumMedianIterations: 1,
          reduction,
          parameters: rig.input(
            getGPUGeographicDistributionParameterValues({medianTolerance: 1e-6}),
            'float32'
          ),
          output: {medianCenters: createPairView(rig, medianCenters), medianConverged}
        })
      );
      expectOracleRows(await rig.readFloat(medianCenters), [0, 0], 0);
      expectOracleRows(await rig.readUint(medianConverged), [1], 0);
    } finally {
      rig.destroy();
    }
  });

  it(`keeps finite ellipse axes when squared variances would overflow on the ${reduction} route`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const ellipses = rig.output('float32', 3);
      const standardDistances = rig.output('float32', 1);
      rig.run(
        new GPUGeographicDistribution({
          // Eight points: each extreme duplicated, Sxx=2e20 and Syy=5e19.
          positions: rig.input(
            Float32Array.of(
              -2e10,
              0,
              2e10,
              0,
              0,
              -1e10,
              0,
              1e10,
              -2e10,
              0,
              2e10,
              0,
              0,
              -1e10,
              0,
              1e10
            ),
            'float32x2',
            8
          ),
          parameters: rig.input(getGPUGeographicDistributionParameterValues(), 'float32'),
          reduction,
          output: {ellipses, standardDistances}
        })
      );
      const actual = await rig.readFloat(ellipses);
      expectOracleRows(actual.slice(0, 1), [1.5707963267948966], 1e-6);
      // Root-two axes [1e10,2e10]; 1e-6 relative allows a few f32 ulps.
      expectOracleRows(
        actual.slice(1).map(value => value / 1e10),
        [1, 2],
        1e-6
      );
      expectOracleRows(
        (await rig.readFloat(standardDistances)).map(value => value / 1e10),
        [1.5811388300841898],
        1e-6
      );
    } finally {
      rig.destroy();
    }
  });
}
