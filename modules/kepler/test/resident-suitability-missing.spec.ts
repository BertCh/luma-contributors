// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src';
import {
  computeNativeSuitabilityScores,
  createResidentSuitabilityResultPlan,
  getResidentSuitabilityEffectiveWeights,
  type ResidentSuitabilityOperation
} from '../../../examples/deck/kepler-compute/src/resident-suitability-plan';
import {executeResidentSuitability} from '../../../examples/deck/kepler-compute/src/resident-suitability-runtime';

// ANALYSIS4-7: the resident GPU scorer on rows that are missing in only some weighted fields. The
// independent CPU oracle (computeNativeSuitabilityScores) must match under the exclude and
// renormalize policies, the default policy must refuse, and partialMissingRowCount must count the
// partially missing rows.

const cost = [10, 20, 15, null, 40, 25, null, 5];
const access = [3, null, 7, null, 5, 9, 2, null];
const comfort = [1, 2, null, null, 4, 8, 6, 3];
const columns = {cost, access, comfort};
// Rows 1, 2, 6 and 7 are missing in some but not all weighted fields; row 3 is missing in all.
const PARTIAL_ROW_COUNT = 4;

const baseOperation: ResidentSuitabilityOperation = {
  id: 'op',
  dataId: 'suitability-source',
  weights: {cost: 0.25, access: 1, comfort: 0.5},
  weightStandardization: 'normalize',
  dataStandardization: 'range',
  outputFieldName: 'score',
  resultId: 'suitability-result',
  resultLabel: 'Suitability result'
};

function importSource(session: KeplerComputeSession) {
  const table = arrow.tableFromArrays({
    id: Int32Array.from({length: cost.length}, (_, index) => index),
    cost: arrow.vectorFromArray(cost, new arrow.Float32()),
    access: arrow.vectorFromArray(access, new arrow.Float32()),
    comfort: arrow.vectorFromArray(comfort, new arrow.Float32())
  });
  return session.importArrowDataset({id: 'suitability-source', table});
}

function describeSource(source: ReturnType<typeof importSource>) {
  return {
    datasetId: source.id,
    label: 'Suitability source',
    revision: source.revision,
    rowCount: source.rowCount,
    fields: ['id', 'cost', 'access', 'comfort'].map(name => ({
      name,
      type: name === 'id' ? 'integer' : 'real',
      analyzerType: name === 'id' ? 'INT' : 'FLOAT'
    })),
    columns: source.value.columns,
    dictionaries: source.value.prepared.dictionaries
  };
}

for (const missingValues of ['exclude', 'renormalize'] as const) {
  test(`resident suitability ${missingValues} matches the CPU oracle on partially missing rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
    const source = importSource(session);
    const plan = createResidentSuitabilityResultPlan(
      {...baseOperation, missingValues},
      describeSource(source) as never
    );
    const result = await executeResidentSuitability({session, plan, isCurrent: () => true});
    try {
      expect(result.partialMissingRowCount).toBe(PARTIAL_ROW_COUNT);
      const effectiveWeights = getResidentSuitabilityEffectiveWeights(
        plan.scoring,
        plan.provenance.weightStandardization
      );
      const expected = computeNativeSuitabilityScores(
        plan.scoring.indicators.map(({column}) => columns[column as keyof typeof columns]),
        effectiveWeights,
        {
          dataStandardization: plan.provenance.dataStandardization,
          weightStandardization: 'raw',
          missingValues
        }
      );
      const scores = result.dataset.value.source.getChild('score')!;
      expect(scores.length).toBe(cost.length);
      let definedRows = 0;
      for (let row = 0; row < cost.length; row++) {
        const actual = scores.get(row) as number | null;
        const oracle = expected[row];
        if (oracle === null) expect(actual, `row ${row} is null`).toBeNull();
        else {
          definedRows++;
          expect(actual, `row ${row} is defined`).not.toBeNull();
          expect(Math.abs((actual as number) - oracle), `row ${row}`).toBeLessThan(1e-5);
        }
      }
      // Exclude drops every partial row; renormalize keeps the ones with a defined indicator.
      expect(definedRows).toBe(missingValues === 'exclude' ? 3 : 7);
    } finally {
      result.transfer();
      await result.destroy();
    }
  }, 60_000);
}

test('resident suitability refuses partially missing rows without a missing-value policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const source = importSource(session);
  const plan = createResidentSuitabilityResultPlan(baseOperation, describeSource(source) as never);
  await expect(executeResidentSuitability({session, plan, isCurrent: () => true})).rejects.toThrow(
    /choose a missing-value policy/
  );
}, 60_000);
