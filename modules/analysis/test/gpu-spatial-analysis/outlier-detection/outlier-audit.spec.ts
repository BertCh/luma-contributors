// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {getGPULocalOutlierFactorParameterValues} from '../../../src/gpu-spatial-analysis/outlier-detection/outlier-detection-parameters';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';

const OFFSETS = [0, 3, 5, 7, 9, 11, 13, 15, 15];
const NEIGHBORS = [1, 2, 99, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 0, 1];
const DISTANCES = [1, 1, 1000, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 8, 9];
// Exact Fractions from outlier-audit.oracle.py, including a tied pair, duplicate
// cluster, an empty/masked row and an invalid neighbor excluded at every stage.
const EXPECTED = {
  kDistance: [1, 2, 2, 0, 0, 0, 9, 0],
  localReachabilityDensity: [0.4, 0.5, 0.5, 2, 2, 2, 0.1111111111111111, 0],
  lof: [1.25, 0.9, 0.9, 1, 1, 1, 4.05, 1]
};

for (const field of ['complete', 'requiredCount'] as const) {
  it(`review B1/S3 restores actual LOF scores after standalone ${field} loss`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const clean = field === 'complete' ? 1 : 3;
    const loss = field === 'complete' ? 0 : 4;
    const evidenceBuffer = device.createBuffer({
      data: Uint32Array.of(clean),
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    try {
      const evidence = importGraphBuffer(
        rig.graph,
        'standalone-evidence',
        evidenceBuffer,
        'uint32',
        1
      );
      const output = {
        kDistance: rig.output('float32', 3),
        localReachabilityDensity: rig.output('float32', 3),
        lof: rig.output('float32', 3),
        outlier: rig.output('uint32', 3),
        outlierCount: rig.output('uint32', 1)
      };
      rig.graph.add(
        new GPULocalOutlierFactor({
          neighbors: {
            offsets: rig.input(Uint32Array.of(0, 1, 2, 3), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 0, 1), 'uint32'),
            weights: rig.input(Float32Array.of(1, 1, 1), 'float32'),
            distances: rig.input(Float32Array.of(1, 1, 2), 'float32'),
            status: {[field]: evidence}
          },
          parameters: rig.input(
            getGPULocalOutlierFactorParameterValues({threshold: 1.5, densityFloor: 0.5}),
            'float32'
          ),
          output
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const word of [clean, loss, clean]) {
          evidenceBuffer.write(Uint32Array.of(word));
          submitGraph(device, compiled, undefined);
          if (word === loss) {
            for (const result of [output.kDistance, output.localReachabilityDensity, output.lof]) {
              expect(await rig.readFloat(result)).toEqual([NaN, NaN, NaN]);
            }
            expect(await rig.readUint(output.outlier)).toEqual([0, 0, 0]);
            expect(await rig.readUint(output.outlierCount)).toEqual([0]);
          } else {
            // Mean reachabilities [1,1,2] plus floor .5 give [2/3,2/3,2/5];
            // the third density ratio is (2/3)/(2/5) = 5/3.
            expectOracleRows(await rig.readFloat(output.kDistance), [1, 1, 2], 0);
            expectOracleRows(
              await rig.readFloat(output.localReachabilityDensity),
              [2 / 3, 2 / 3, 2 / 5],
              1e-6
            );
            expectOracleRows(await rig.readFloat(output.lof), [1, 1, 5 / 3], 1e-6);
            expect(await rig.readUint(output.outlier)).toEqual([0, 0, 1]);
            expect(await rig.readUint(output.outlierCount)).toEqual([1]);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      evidenceBuffer.destroy();
      rig.destroy();
    }
  });
}

for (const packedPair of ['k-distance/density', 'density/lof'] as const) {
  it(`W5-09 computes every output with disjoint aligned ${packedPair} in one physical buffer`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const packed = rig.output('uint32', 128);
      const first = rig.graph.createDataView(packed.buffer, {format: 'float32', length: 2});
      const second = rig.graph.createDataView(packed.buffer, {
        format: 'float32',
        length: 2,
        byteOffset: 256
      });
      const separate = rig.output('float32', 2);
      const kDistance = packedPair === 'k-distance/density' ? first : separate;
      const density = packedPair === 'k-distance/density' ? second : first;
      const lof = packedPair === 'density/lof' ? second : separate;
      rig.run(
        new GPULocalOutlierFactor({
          neighbors: {
            offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
            weights: rig.input(Float32Array.of(1, 1), 'float32'),
            distances: rig.input(Float32Array.of(1, 1), 'float32')
          },
          parameters: rig.input(
            getGPULocalOutlierFactorParameterValues({densityFloor: 0.5}),
            'float32'
          ),
          output: {kDistance, localReachabilityDensity: density, lof}
        })
      );
      // Reciprocal unit-distance neighbors have reachability 1: density=1/(1+.5),
      // and dividing either density by the other gives LOF=1.
      const words = await rig.readUint(packed);
      const firstValues = new Float32Array(Uint32Array.from(words.slice(0, 2)).buffer);
      const secondValues = new Float32Array(Uint32Array.from(words.slice(64, 66)).buffer);
      const separateValues = await rig.readFloat(separate);
      expectOracleRows(
        packedPair === 'k-distance/density' ? firstValues : separateValues,
        [1, 1],
        0
      );
      expectOracleRows(
        packedPair === 'k-distance/density' ? secondValues : firstValues,
        [2 / 3, 2 / 3],
        1e-6
      );
      expectOracleRows(packedPair === 'density/lof' ? secondValues : separateValues, [1, 1], 1e-6);
    } finally {
      rig.destroy();
    }
  });
}

for (const packedPort of ['density', 'score'] as const) {
  it(`classifies mutual neighbors with ${packedPort} and flags in one packed physical buffer`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const packed = rig.output('uint32', 128);
      const packedView = rig.graph.createDataView(packed.buffer, {format: 'float32', length: 2});
      const unpacked = rig.output('float32', 2);
      const density = packedPort === 'density' ? packedView : unpacked;
      const lof = packedPort === 'score' ? packedView : unpacked;
      const flags = rig.graph.createDataView(packed.buffer, {
        format: 'uint32',
        length: 2,
        byteOffset: 256
      });
      const kDistance = rig.output('float32', 2),
        count = rig.output('uint32', 1);
      rig.run(
        new GPULocalOutlierFactor({
          neighbors: {
            offsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
            distances: rig.input(Float32Array.of(1, 1), 'float32'),
            weights: rig.input(Float32Array.of(1, 1), 'float32')
          },
          parameters: rig.input(
            getGPULocalOutlierFactorParameterValues({threshold: 0.5, densityFloor: 0.5}),
            'float32'
          ),
          output: {
            kDistance,
            localReachabilityDensity: density,
            lof,
            outlier: flags,
            outlierCount: count
          }
        })
      );
      const words = await rig.readUint(packed);
      // Mutual distance 1 => density 1/(1+.5), equal densities => both scores 1.
      expectOracleRows(
        new Float32Array(Uint32Array.from(words.slice(0, 2)).buffer),
        packedPort === 'density' ? [2 / 3, 2 / 3] : [1, 1],
        1e-6
      );
      expectOracleRows(words.slice(64, 66), [1, 1], 0);
      expectOracleRows(await rig.readFloat(kDistance), [1, 1], 0);
      expectOracleRows(
        await rig.readFloat(unpacked),
        packedPort === 'density' ? [1, 1] : [2 / 3, 2 / 3],
        1e-6
      );
      expectOracleRows(await rig.readUint(count), [2], 0);
    } finally {
      rig.destroy();
    }
  });
}

for (const threshold of [undefined, 1.25, 0.5]) {
  it(`checks every LOF output against the exact-rational oracle at threshold ${threshold}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const scale of [1, 2]) {
      const rig = new AnalysisRig(device);
      try {
        const output = {
          kDistance: rig.output('float32', 8),
          localReachabilityDensity: rig.output('float32', 8),
          lof: rig.output('float32', 8),
          outlier: rig.output('uint32', 8),
          outlierCount: rig.output('uint32', 1)
        };
        rig.run(
          new GPULocalOutlierFactor({
            neighbors: {
              offsets: rig.input(new Uint32Array(OFFSETS), 'uint32'),
              neighbors: rig.input(new Uint32Array(NEIGHBORS), 'uint32'),
              distances: rig.input(
                Float32Array.from(DISTANCES, value => value * scale),
                'float32'
              ),
              weights: rig.input(new Float32Array(15).fill(1), 'float32')
            },
            parameters: rig.input(
              getGPULocalOutlierFactorParameterValues({threshold, densityFloor: 0.5 * scale}),
              'float32'
            ),
            output
          })
        );
        // 2e-6 absolute: at most three additions/divisions of quantities <=9.
        expectOracleRows(
          await rig.readFloat(output.kDistance),
          EXPECTED.kDistance.map(value => value * scale),
          0
        );
        expectOracleRows(
          await rig.readFloat(output.localReachabilityDensity),
          EXPECTED.localReachabilityDensity.map(value => value / scale),
          2e-6
        );
        expectOracleRows(await rig.readFloat(output.lof), EXPECTED.lof, 2e-6);
        const outlier =
          threshold === undefined
            ? [0, 0, 0, 0, 0, 0, 0, 0]
            : threshold === 1.25
              ? [0, 0, 0, 0, 0, 0, 1, 0]
              : [1, 1, 1, 1, 1, 1, 1, 0];
        expectOracleRows(await rig.readUint(output.outlier), outlier, 0);
        expectOracleRows(
          await rig.readUint(output.outlierCount),
          [threshold === undefined ? 0 : threshold === 1.25 ? 1 : 7],
          0
        );
      } finally {
        rig.destroy();
      }
    }
  });
}

for (const field of [
  'overflow',
  'candidateOverflow',
  'iterationLimitReached',
  'converged'
] as const) {
  for (const complete of [false, true]) {
    it(`checks every LOF output with neighbors.${field} reporting complete=${complete}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const output = {
          kDistance: rig.output('float32', 8),
          localReachabilityDensity: rig.output('float32', 8),
          lof: rig.output('float32', 8),
          outlier: rig.output('uint32', 8),
          outlierCount: rig.output('uint32', 1)
        };
        rig.run(
          new GPULocalOutlierFactor({
            neighbors: {
              offsets: rig.input(new Uint32Array(OFFSETS), 'uint32'),
              neighbors: rig.input(new Uint32Array(NEIGHBORS), 'uint32'),
              distances: rig.input(new Float32Array(DISTANCES), 'float32'),
              weights: rig.input(new Float32Array(15).fill(1), 'float32'),
              status: {
                [field]: rig.input(
                  Uint32Array.of(field === 'converged' ? Number(complete) : Number(!complete)),
                  'uint32'
                )
              }
            },
            parameters: rig.input(
              getGPULocalOutlierFactorParameterValues({threshold: 0.5, densityFloor: 0.5}),
              'float32'
            ),
            output
          })
        );
        for (const port of ['kDistance', 'localReachabilityDensity', 'lof'] as const)
          expectOracleRows(
            await rig.readFloat(output[port]),
            complete ? EXPECTED[port] : new Array<number>(8).fill(NaN),
            complete ? 2e-6 : 0
          );
        expectOracleRows(
          await rig.readUint(output.outlier),
          complete ? [1, 1, 1, 1, 1, 1, 1, 0] : [0, 0, 0, 0, 0, 0, 0, 0],
          0
        );
        expectOracleRows(await rig.readUint(output.outlierCount), [complete ? 7 : 0], 0);
      } finally {
        rig.destroy();
      }
    });
  }
}
