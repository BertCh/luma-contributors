// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {fromRecordedLensFile} from '../../src/computation-lenses/index';
import {buildDecodedLensTwin} from '../../src/computation-views/index';
import {TERRAIN_DECODE_OPERATORS} from '../../src/gpu-terrain/computation-pack/operators/decode';
import '../../src/gpu-terrain/computation-pack/index';
import '../../src/computation-lenses/scenarios/index';

const SCENARIO = 'gpu-terrain.terrain-spike-repair';

it('published spike-repair record and rendered formula reserve the unresolved sentinel for labels', () => {
  const recorded = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/${SCENARIO}.json`, import.meta.url),
        'utf8'
      )
    )
  );
  const formula = recorded.record.refs.formulas.find(
    entry => entry.id === 'gpu-terrain.terrain-spike-repair.component-shift'
  );
  // Independent usability contract: +Infinity is unusable (validity 0). Only labels
  // carry the nonzero diagnostic sentinel; the published HTML must teach that rule.
  expect(formula?.ascii).toContain('+Infinity, validity 0, labels 0xffffffff');
  const ownerFormula = TERRAIN_DECODE_OPERATORS['GPUTerrainSpikeRepair'].formula;
  expect(formula?.ascii).toBe(
    ownerFormula && 'ascii' in ownerFormula ? ownerFormula.ascii : undefined
  );
  const validity = recorded.record.captures?.snapshots.find(
    entry => entry.name === 'repairValidity'
  );
  expect(validity?.meaning).toContain('0 while component labelling is unresolved');
  expect(validity?.meaning).not.toContain('0xffffffff');
  const rendered = buildDecodedLensTwin(recorded, {scenario: SCENARIO, rung: 'formula'});
  const committed = JSON.parse(
    readFileSync(
      new URL(`../../../../docs/lens-records/twins/${SCENARIO}~formula.json`, import.meta.url),
      'utf8'
    )
  );
  expect(committed).toEqual({...rendered, schema: 'luma.lens-twin/1', scenario: SCENARIO});
  for (const twin of [rendered, committed]) {
    expect(twin.figure?.html).toContain('+Infinity, validity 0, labels 0xffffffff');
    expect(twin.formula?.ascii).toContain('+Infinity, validity 0, labels 0xffffffff');
    expect(twin.llms).not.toContain('validity and labels 0xffffffff');
  }
});
