// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';

// These are the current scenario outputs, not aliases for legacy validity or overflow words.
const CONTRACTS = [
  {
    scenario: 'gpu-spatial-analysis.population-support-alignment-recipe',
    outputs: {'output-alignmentCode': [0, 0], 'output-matchedMask': [1, 1]}
  },
  {
    scenario: 'gpu-spatial-analysis.pair-gather',
    outputs: {pairCandidateOverflow: [0], candidateOverflow: [0]}
  }
] as const;

for (const contract of CONTRACTS) {
  it(`final1 catalog-docs:F10/core:F9 ${contract.scenario} captures current outputs`, () => {
    const {record} = fromRecordedLensFile(
      JSON.parse(readFileSync(`docs/lens-records/${contract.scenario}.json`, 'utf8'))
    );
    const snapshots = record.captures?.snapshots ?? [];
    expect(snapshots.map(snapshot => snapshot.name)).toEqual(
      expect.arrayContaining(Object.keys(contract.outputs))
    );
    for (const [name, expected] of Object.entries(contract.outputs)) {
      const snapshot = snapshots.find(candidate => candidate.name === name)!;
      expect(Array.from(snapshot.values, Number), name).toEqual(expected);
    }
  });
}
