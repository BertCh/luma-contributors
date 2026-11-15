// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {SampleDatasetAttributions} from '../../../examples/deck/kepler-compute/src/sample-map-attribution';
import * as arrow from 'apache-arrow';
import {
  createSampleMapProvenance,
  getSampleMapProvenance,
  listSampleMapAttributions,
  SAMPLE_MAP_METADATA_KEY
} from '../../../examples/deck/kepler-compute/src/sample-map-provenance';
import {
  getSandboxSources,
  readPublishedSandboxSource
} from '../../../examples/deck/kepler-compute/src/sandbox-state';
import {
  collectNativeOnlySources,
  readEmbeddedNativeSources,
  getStagedNativeDatasets,
  stageNativeSources,
  clearStagedNativeSources
} from '../../../examples/deck/kepler-compute/src/bundle-native-sources';
import type {ComputeRuntime} from '../../../examples/deck/kepler-compute/src/compute-runtime';
import type {KeplerTable} from '@kepler.gl/table';

const sample = {
  id: 'movement_pittsburgh',
  label: 'Movement',
  detail: 'Creative Commons Noncommercial Attribution License',
  description: '',
  visible: true,
  dataUrl: 'https://example.com/data.csv',
  configUrl: 'https://example.com/config.json'
};
const provenance = createSampleMapProvenance(sample);
const metadata = {[SAMPLE_MAP_METADATA_KEY]: provenance};

test('retained sample attribution remains per dataset when another sample is selected', () => {
  const records = listSampleMapAttributions([
    {id: 'movement', metadata},
    {
      id: 'earthquakes',
      metadata: {
        [SAMPLE_MAP_METADATA_KEY]: createSampleMapProvenance({
          ...sample,
          id: 'earthquakes',
          detail: 'USGS'
        })
      }
    }
  ]);
  expect(records.map(record => record.datasetId)).toEqual(['movement', 'earthquakes']);
  expect(records[0]!.licenseNotice).toContain('Noncommercial');
  const html = renderToStaticMarkup(
    createElement(SampleDatasetAttributions, {
      datasets: [
        {id: 'movement', metadata},
        {
          id: 'remote',
          metadata: {
            [SAMPLE_MAP_METADATA_KEY]: createSampleMapProvenance({
              ...sample,
              id: 'remote',
              dataUrl: undefined
            })
          }
        }
      ]
    })
  );
  expect(html).toContain('data-dataset-id="movement"');
  expect(html).toContain('Noncommercial');
  expect(html).toContain('sample-recovery-notice');
  expect(html).toContain('Remote sample maps are not recoverable');
});

test('sample provenance survives the actual published-source save/read manifest boundary', () => {
  const table = arrow.tableFromArrays({value: [1, 2]});
  const dataset = {
    id: sample.id,
    revision: 1,
    rowCount: 2,
    value: {source: table},
    source: {metadata}
  };
  const fixture = {...dataset, id: 'resident-observations'};
  const runtime = {
    explorer: {dataset: fixture},
    publications: new Map([[sample.id, {dataset, label: sample.label}]])
  } as unknown as ComputeRuntime;
  const descriptors = JSON.parse(JSON.stringify(getSandboxSources(runtime)));
  const restored = readPublishedSandboxSource(descriptors[1]);
  expect(restored.sampleMap).toEqual(provenance);
  expect(restored.sampleMap?.licenseNotice).toContain('Noncommercial');
});

test('native-only portable bundle provenance survives encode, validate and staged replay', async () => {
  const table = arrow.tableFromArrays({value: [1, 2]});
  const native = {
    id: sample.id,
    label: sample.label,
    metadata,
    dataRevision: 1,
    fields: [{name: 'value', type: 'real', format: ''}],
    dataContainer: {numRows: () => table.numRows, getTable: () => table}
  } as unknown as KeplerTable;
  const collected = await collectNativeOnlySources([native]);
  expect(collected.skipped).toEqual([]);
  const restored = await readEmbeddedNativeSources(JSON.parse(JSON.stringify(collected.embedded)));
  stageNativeSources(restored);
  try {
    expect(getSampleMapProvenance(getStagedNativeDatasets()[0]!.metadata)).toEqual(provenance);
  } finally {
    clearStagedNativeSources();
  }
});
