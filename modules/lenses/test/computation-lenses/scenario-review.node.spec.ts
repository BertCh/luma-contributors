// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {createGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {
  GPUCommandGraph,
  createGPUCopyCommandNode,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {
  fromRecordedLensFile,
  toRecordedLensFile,
  listLensScenarios
} from '../../src/computation-lenses/index';
import {
  createDataFrameLensGraph,
  runDataFrameContributorLens
} from '../../src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support';
import {
  declareLensScenario,
  getMissingScenarioChoices,
  getRecordedCaptureNames,
  getStructureSignature
} from './scenario-declaration.test-utils';
import '../../src/computation-lenses/scenarios/index';

const scenarios = listLensScenarios();
const focal = scenarios.find(scenario => scenario.id === 'gpu-raster.raster-focal-pairs')!;

test('X10 rejects an archived compiled graph supplied without any actual compilation', async () => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-spatial-analysis.pair-gather'
  )!;
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/${scenario.id}.json`, import.meta.url),
        'utf8'
      )
    )
  ).record;
  await expect(
    declareLensScenario({
      ...scenario,
      load: async () => ({
        run: async () =>
          createGPUComputationRecord({
            ...record,
            sources: record.refs.sources,
            formulas: record.refs.formulas
          })
      })
    })
  ).rejects.toThrow(/compiled graph.*not compiled/);
});

test('R2 focal-pairs gate names a deleted radius choice', async () => {
  const declared = await declareLensScenario(focal);
  expect(
    Object.fromEntries(declared.record.choices.map(choice => [choice.name, choice.value]))
  ).toMatchObject({radius: 1, covarianceNormalization: 'sample'});
  expect(
    getMissingScenarioChoices(focal, {
      ...declared.record,
      choices: declared.record.choices.filter(choice => choice.name !== 'radius')
    })
  ).toEqual(['radius']);
});

test('R2 dataframe string templates require a deleted noDataColor choice', async () => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-dataframe.bivariate-classification'
  )!;
  const declared = await declareLensScenario(scenario);
  expect(
    getMissingScenarioChoices(scenario, {
      ...declared.record,
      choices: declared.record.choices.filter(choice => choice.name !== 'noDataColor')
    })
  ).toEqual(['noDataColor']);
});

test('R2 low-pass filtering marks the Gaussian sigma choice inapplicable', async () => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-raster.raster-spectral-filter'
  )!;
  const declared = await declareLensScenario(scenario);
  expect(declared.record.choices.find(choice => choice.name === 'sigma')).toMatchObject({
    value: 'disabled: low-pass filter (packed default 0.1)',
    sweepable: false
  });
});

test('R2 registration receipt records the executed integer upsampling factor', async () => {
  const scenario = scenarios.find(candidate => candidate.id === 'gpu-raster.raster-registration')!;
  const declared = await declareLensScenario(scenario);
  expect(declared.record.choices.find(choice => choice.name === 'upsampleFactor')?.value).toBe(1);
});

test('R3 gate rejects a compiled graph omitted from the assembled record', async () => {
  await expect(
    declareLensScenario({
      ...focal,
      load: async () => {
        const loaded = await focal.load();
        return {
          run: async device => {
            const omitted = new GPUCommandGraph(device, {id: 'omitted-observation'});
            omitted.add(
              createGPUCopyCommandNode({
                id: 'omitted-copy',
                resources: [],
                compile: () => ({encode() {}})
              })
            );
            omitted.compile();
            return loaded.run(device);
          }
        };
      }
    })
  ).rejects.toThrow(/omitted-observation.*omitted-copy/);
});

test('R4 key-join declaration rejects an unknown datum key space before readback', async () => {
  const keyJoin = scenarios.find(scenario => scenario.id === 'gpu-dataframe.key-join')!;
  await expect(
    declareLensScenario({
      ...keyJoin,
      load: async () => ({
        run: async device => {
          const graph = createDataFrameLensGraph<undefined>(device, 'invalid-key-join');
          const values = createTransientView(
            graph,
            'values',
            'float32',
            1,
            Buffer.STORAGE | Buffer.COPY_SRC
          );
          return runDataFrameContributorLens(device, {
            scenarioId: keyJoin.id,
            element: keyJoin.element,
            graph,
            specs: {values: {view: values, role: 'output', keySpace: 'row'}},
            keySpaces: [{id: 'row', label: 'row', size: 1, indexSpace: '1d'}],
            datumKeySpace: 'undeclared-domain',
            choices: [],
            datasetVersion: 'fixture'
          });
        }
      })
    })
  ).rejects.toThrow(/keySpace.*Unknown reference/);
});

test('R5 deleting a named polygon alias changes decoded capture parity', async () => {
  const scenario = scenarios.find(
    candidate => candidate.id === 'gpu-spatial-analysis.assuncao-reis-standardization'
  )!;
  const file = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../docs/lens-records/${scenario.id}.json`, import.meta.url),
        'utf8'
      )
    )
  );
  const record = file.record;
  const changed = {
    ...record,
    captures: {
      ...record.captures!,
      snapshots: record.captures!.snapshots.filter(
        capture => capture.name !== 'standardizedRates.geometry'
      )
    }
  };
  const roundTrip = fromRecordedLensFile(toRecordedLensFile(changed, file)).record;
  expect(roundTrip.graphs).toEqual(record.graphs);
  expect(roundTrip.captures!.snapshots).toEqual(changed.captures.snapshots);
  expect(getStructureSignature(roundTrip, getRecordedCaptureNames(roundTrip))).not.toEqual(
    getStructureSignature(record, getRecordedCaptureNames(record))
  );
  const live = await declareLensScenario(scenario);
  expect(live.captureNames).toContain('standardizedRates.geometry');
});
