// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSimilarLocations} from '../../../src/gpu-spatial-analysis/similar-locations/gpu-similar-locations';
import {getGPUSimilarLocationsParameterValues} from '../../../src/gpu-spatial-analysis/similar-locations/similar-locations-parameters';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';

const VALUES = [0, 2, 7, 2, 0, 7, 1, 1, 7, 1, 1, 7, 4, 3, 7, 99, 99, 7, NaN, 2, 7, -2, -1, 7];
const NO_RANK = 4294967295;

it('preserves the unique nearest row when an unrelated anchor merges adjacent values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const output = {
      ranks: rig.output('uint32', 4),
      distances: rig.output('float32', 4),
      topIds: rig.output('uint32', 4),
      status: {count: rig.output('uint32', 1)}
    };
    rig.run(
      new GPUSimilarLocations({
        values: rig.input(Float32Array.of(1, 1, -1, -1 + 2 ** -24), 'float32'),
        columnCount: 1,
        referenceMask: rig.input(Uint32Array.of(0, 0, 0, 1), 'uint32'),
        resultCapacity: 4,
        parameters: rig.input(
          getGPUSimilarLocationsParameterValues({resultCount: 4, excludeReference: false}, 1),
          'float32'
        ),
        output
      })
    );
    // Direct standardized differences to row 3: zero only at row 3, row 2 differs by 2^-24.
    // SciPy population deviation=0.999999985098839, giving distance 5.960464566356904e-8.
    expectOracleRows(await rig.readUint(output.topIds), [3, 2, 0, 1], 0);
    expectOracleRows(await rig.readUint(output.ranks), [2, 3, 1, 0], 0);
    const distances = await rig.readFloat(output.distances);
    expectOracleRows(distances.slice(0, 2), [1.9999999701976776, 1.9999999701976776], 1e-7);
    expectOracleRows(distances.slice(2), [5.960464566356904e-8, 0], 1e-14);
    expectOracleRows(await rig.readUint(output.status.count), [4], 0);
  } finally {
    rig.destroy();
  }
});

it('clears every word of an oversized topIds view with no selected reference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const buffer = device.createBuffer({
    data: new Uint32Array(5).fill(17),
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    const topIds = importGraphBuffer(rig.graph, 'oversized-top-ids', buffer, 'uint32', 5);
    const ranks = rig.output('uint32', 8),
      distances = rig.output('float32', 8),
      count = rig.output('uint32', 1);
    rig.run(
      new GPUSimilarLocations({
        values: rig.input(Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7), 'float32'),
        columnCount: 1,
        referenceMask: rig.input(new Uint32Array(8), 'uint32'),
        resultCapacity: 2,
        parameters: rig.input(
          getGPUSimilarLocationsParameterValues({resultCount: 2}, 1),
          'float32'
        ),
        output: {ranks, distances, topIds, status: {count}}
      })
    );
    expectOracleRows(await readUint32(buffer), [NO_RANK, NO_RANK, NO_RANK, NO_RANK, NO_RANK], 0);
    expectOracleRows(await rig.readUint(ranks), new Array<number>(8).fill(NO_RANK), 0);
    expectOracleRows(await rig.readFloat(distances), new Array<number>(8).fill(NaN), 0);
    expectOracleRows(await rig.readUint(count), [0], 0);
  } finally {
    rig.destroy();
    buffer.destroy();
  }
});

it('keeps exact zscore distances after an exactly representable large translation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [0, 8192]) {
    for (const meanReference of [false, true]) {
      const rig = new AnalysisRig(device);
      try {
        const output = {
          ranks: rig.output('uint32', 8),
          distances: rig.output('float32', 8),
          topIds: rig.output('uint32', 8),
          status: {count: rig.output('uint32', 1)}
        };
        rig.run(
          new GPUSimilarLocations({
            // Four of each exact f32 value: population zscores [-1]*4+[1]*4.
            values: rig.input(
              Float32Array.from(
                [0, 0, 0, 0, 1 / 1024, 1 / 1024, 1 / 1024, 1 / 1024],
                value => origin + value
              ),
              'float32'
            ),
            columnCount: 1,
            referenceMask: rig.input(
              Uint32Array.of(1, 0, 0, 0, Number(meanReference), 0, 0, 0),
              'uint32'
            ),
            parameters: rig.input(
              getGPUSimilarLocationsParameterValues({resultCount: 8, excludeReference: false}, 1),
              'float32'
            ),
            output
          })
        );
        // A low/high reference pair has centroid halfway between f32 grid points at 8192:
        // retaining its low word gives distance 1 on both sides, rather than 0 and 2.
        expectOracleRows(
          await rig.readFloat(output.distances),
          meanReference ? [1, 1, 1, 1, 1, 1, 1, 1] : [0, 0, 0, 0, 2, 2, 2, 2],
          1e-6
        );
        expectOracleRows(await rig.readUint(output.ranks), [0, 1, 2, 3, 4, 5, 6, 7], 0);
        expectOracleRows(await rig.readUint(output.topIds), [0, 1, 2, 3, 4, 5, 6, 7], 0);
        expectOracleRows(await rig.readUint(output.status.count), [8], 0);
      } finally {
        rig.destroy();
      }
    }
  }
});
// Independent scipy zscore/rankdata and NumPy norms/lexsort, in similar-audit.oracle.py.
const ORDERS = {
  'exclude/most': [2, 3, 4, 7],
  'exclude/least': [4, 7, 2, 3],
  'include/most': [2, 3, 0, 1, 4, 7],
  'include/least': [4, 7, 0, 1, 2, 3]
};
const RANKS = {
  'exclude/most': [NO_RANK, NO_RANK, 0, 1, 2, NO_RANK, NO_RANK, 3],
  'exclude/least': [NO_RANK, NO_RANK, 2, 3, 0, NO_RANK, NO_RANK, 1],
  'include/most': [2, 3, 0, 1, 4, NO_RANK, NO_RANK, 5],
  'include/least': [2, 3, 4, 5, 0, NO_RANK, NO_RANK, 1]
};

it('checks the sorted-column rank route against the same eight-row SciPy oracle with masked padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    // 4096 is the fixed route threshold, not a device-derived size. Only the
    // original eight-row oracle fixture can participate; all padding is masked.
    const rows = 4096;
    const values = new Float32Array(rows * 3);
    values.set(VALUES);
    const mask = new Uint32Array(rows);
    mask.set([1, 1, 1, 1, 1, 0, 1, 1]);
    const referenceMask = new Uint32Array(rows);
    referenceMask.set([1, 1]);
    const output = {
      ranks: rig.output('uint32', rows),
      distances: rig.output('float32', rows),
      topIds: rig.output('uint32', 8),
      status: {count: rig.output('uint32', 1)}
    };
    rig.run(
      new GPUSimilarLocations({
        values: rig.input(values, 'float32'),
        columnCount: 3,
        mask: rig.input(mask, 'uint32'),
        referenceMask: rig.input(referenceMask, 'uint32'),
        standardization: 'rank',
        resultCapacity: 8,
        parameters: rig.input(
          getGPUSimilarLocationsParameterValues({resultCount: 8, weights: [2, 0.5, 0]}, 3),
          'float32'
        ),
        output
      })
    );
    expectOracleRows(
      await rig.readFloat(output.distances),
      [
        NaN,
        NaN,
        0,
        0,
        0.7905694150420949,
        NaN,
        NaN,
        0.7905694150420949,
        ...new Array<number>(rows - 8).fill(NaN)
      ],
      2e-6
    );
    expectOracleRows(
      await rig.readUint(output.ranks),
      [...RANKS['exclude/most'], ...new Array<number>(rows - 8).fill(NO_RANK)],
      0
    );
    expectOracleRows(
      await rig.readUint(output.topIds),
      [2, 3, 4, 7, NO_RANK, NO_RANK, NO_RANK, NO_RANK],
      0
    );
    expectOracleRows(await rig.readUint(output.status.count), [4], 0);
  } finally {
    rig.destroy();
  }
});

for (const standardization of ['z-score', 'rank'] as const) {
  for (const excludeReference of [true, false]) {
    for (const direction of ['most', 'least'] as const) {
      it(`checks all ${standardization}/${direction}/exclude=${excludeReference} similarity ports against the independent oracle`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        for (const transformed of [false, true]) {
          const rig = new AnalysisRig(device);
          try {
            const output = {
              ranks: rig.output('uint32', 8),
              distances: rig.output('float32', 8),
              topIds: rig.output('uint32', 8),
              status: {count: rig.output('uint32', 1)}
            };
            const props = {
              values: rig.input(
                Float32Array.from(VALUES, value => (transformed ? value * 4 + 32 : value)),
                'float32'
              ),
              columnCount: 3,
              mask: rig.input(Uint32Array.of(1, 1, 1, 1, 1, 0, 1, 1), 'uint32'),
              referenceMask: rig.input(Uint32Array.of(1, 1, 0, 0, 0, 0, 0, 0), 'uint32'),
              parameters: rig.input(
                getGPUSimilarLocationsParameterValues(
                  {
                    resultCount: transformed ? 2 : 8,
                    weights: [2, 0.5, 0],
                    direction,
                    excludeReference
                  },
                  3
                ),
                'float32'
              ),
              standardization,
              resultCapacity: 8,
              output
            };
            const contributor = new GPUSimilarLocations(props);
            props.standardization = standardization === 'rank' ? 'z-score' : 'rank';
            rig.run(contributor);
            const key =
              `${excludeReference ? 'exclude' : 'include'}/${direction}` as keyof typeof ORDERS;
            const near = standardization === 'z-score' ? 0.9486832980505139 : 0.47434164902525694;
            const far = standardization === 'z-score' ? 2.569046515733026 : 0.7905694150420949;
            const distances = [
              excludeReference ? NaN : near,
              excludeReference ? NaN : near,
              0,
              0,
              far,
              NaN,
              NaN,
              far
            ];
            expectOracleRows(await rig.readFloat(output.distances), distances, 2e-6);
            expectOracleRows(await rig.readUint(output.ranks), RANKS[key], 0);
            // A smaller per-frame request keeps the independent ranks/distances
            // and takes only the first two IDs of the fixed oracle order.
            const topCount = transformed ? 2 : excludeReference ? 4 : 6;
            expectOracleRows(
              await rig.readUint(output.topIds),
              [...ORDERS[key].slice(0, topCount), ...new Array<number>(8 - topCount).fill(NO_RANK)],
              0
            );
            expectOracleRows(await rig.readUint(output.status.count), [topCount], 0);
          } finally {
            rig.destroy();
          }
        }
      });
    }
  }
  it(`publishes every ${standardization} port with no selected reference`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = {
        ranks: rig.output('uint32', 8),
        distances: rig.output('float32', 8),
        topIds: rig.output('uint32', 8),
        status: {count: rig.output('uint32', 1)}
      };
      rig.run(
        new GPUSimilarLocations({
          values: rig.input(new Float32Array(VALUES), 'float32'),
          columnCount: 3,
          referenceMask: rig.input(new Uint32Array(8), 'uint32'),
          standardization,
          resultCapacity: 8,
          parameters: rig.input(
            getGPUSimilarLocationsParameterValues({resultCount: 8}, 3),
            'float32'
          ),
          output
        })
      );
      expectOracleRows(await rig.readFloat(output.distances), new Array<number>(8).fill(NaN), 0);
      expectOracleRows(await rig.readUint(output.ranks), new Array<number>(8).fill(NO_RANK), 0);
      expectOracleRows(await rig.readUint(output.topIds), new Array<number>(8).fill(NO_RANK), 0);
      expectOracleRows(await rig.readUint(output.status.count), [0], 0);
    } finally {
      rig.destroy();
    }
  });
}
