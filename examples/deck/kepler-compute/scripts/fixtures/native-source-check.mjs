// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import process from 'process';
globalThis.process ??= process;

function equal(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected))
    throw new Error(`Expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`);
}

try {
  // This isolated page needs the same parser registration as the app entry. The old fixture
  // omitted it too, so timestamp materialization threw 'timeToUnixMilli converter not installed'.
  await import('../../src/native-utils-host');
  const [{KeplerTable}, {initApplicationConfig}, arrow, {makeArrowSourceFromNativeDataset}] =
    await Promise.all([
      import('@kepler.gl/table'),
      import('@kepler.gl/utils'),
      import('apache-arrow'),
      import('../../src/native-dataset-source')
    ]);
  initApplicationConfig({maxArrowBatches: Number.MAX_SAFE_INTEGER});
  const limits = {maximumRows: 100, maximumCells: 1000};
  const rows = new KeplerTable({
    info: {id: 'native-csv', label: 'Native CSV'},
    color: [20, 40, 60]
  });
  await rows.importData({
    data: {
      fields: [
        {name: 'Revenue USD', type: 'real', format: ''},
        {name: 'Region', type: 'string', format: ''},
        {name: 'Observed', type: 'timestamp', format: 'X'}
      ],
      rows: [
        [1.25, 'north', 1700000000000],
        [2.5, 'south', null]
      ]
    }
  });
  const source = makeArrowSourceFromNativeDataset(rows, limits);
  equal(source.representation, 'materialized-native-rows');
  equal(
    source.table.schema.fields.map(field => field.name),
    ['Revenue USD', 'Region', 'Observed']
  );
  equal(Array.from(source.table.getChild('Revenue USD')), [1.25, 2.5]);
  equal(Array.from(source.table.getChild('Region')), ['north', 'south']);
  equal(source.table.getChild('Observed').get(0), 1700000000000);
  equal(source.table.getChild('Observed').get(1), null);
  equal(source.isCurrent(), true);
  rows.appendRows([[3.75, 'west', 1700000001000]]);
  equal(source.isCurrent(), false);
  let rejected = false;
  try {
    makeArrowSourceFromNativeDataset(rows, {maximumRows: 2, maximumCells: 100});
  } catch {
    rejected = true;
  }
  equal(rejected, true);
  const first = arrow.tableFromArrays({score: new Float32Array([1, 2])});
  const second = arrow.tableFromArrays({score: new Float32Array([3])});
  const original = first.concat(second);
  const nativeArrow = new KeplerTable({
    info: {id: 'native-arrow', label: 'Native Arrow'},
    color: [20, 40, 60]
  });
  await nativeArrow.importData({
    data: {
      fields: [{name: 'score', type: 'real', format: ''}],
      cols: [original.getChild('score')],
      arrowTable: original
    }
  });
  nativeArrow.dataContainer.valueAt = () => {
    throw new Error('Borrowed Arrow must not materialize CPU rows');
  };
  const borrowed = makeArrowSourceFromNativeDataset(nativeArrow, limits);
  equal(borrowed.representation, 'borrowed-arrow');
  equal(borrowed.table === nativeArrow.dataContainer.getTable(), true);
  equal(
    borrowed.table.batches.map(batch => batch.numRows),
    [2, 1]
  );
  window.nativeSourceResult = {
    passed: true,
    checks: [
      'native CSV rows',
      'original field names',
      'timestamp/null semantics',
      'source revisions',
      'capacity admission',
      'borrowed Arrow batches without row reads'
    ]
  };
  document.querySelector('#result').textContent = 'Passed';
} catch (error) {
  window.nativeSourceResult = {passed: false, error: error.stack ?? String(error)};
  document.querySelector('#result').textContent = String(error);
}
