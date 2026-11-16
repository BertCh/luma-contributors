// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {packGPUColor} from '@luma-contributors/analysis/dataframe-analysis';
import {
  KeplerComputeSession,
  KeplerResidentStyle,
  getKeplerBindingResultPresentations,
  getKeplerOperationDescriptor,
  getKeplerResultEncodingStyle,
  type CatalogOutputTable
} from '../src/index';
import {createNativeStylePlan} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import {
  prepareAnalysisResultPublication,
  getAnalysisResultDictionaries
} from '../../../examples/deck/kepler-compute/src/analysis-result-publication';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {getAnalysisLegendModel} from '../../../examples/deck/kepler-compute/src/analysis-legend';
import {
  getAnalysisHotspotColors,
  getAnalysisMoranColors
} from '../../../examples/deck/kepler-compute/src/analysis-result-colors';

/** JSON with object keys sorted, so property order does not decide equality. */
function stringifyCanonical(value: unknown): string {
  return JSON.stringify(value, (_key, item) =>
    item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(
          Object.entries(item).sort(([left], [right]) => left.localeCompare(right))
        )
      : item
  );
}

function compareResult(label: string, observed: unknown, expected: unknown) {
  if (stringifyCanonical(observed) !== stringifyCanonical(expected))
    throw new Error(
      `${label}\nObserved: ${JSON.stringify(observed)}\nExpected: ${JSON.stringify(expected)}`
    );
}

test('catalog hotspot, Moran and choropleth native plans render the exact AX legend colours', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('AX result publication requires WebGPU');
  const fixtures: {operation: string; table: CatalogOutputTable}[] = [
    {
      operation: 'hotspot-gi-star',
      table: {
        id: 'hotspots',
        label: 'Hot spots',
        rowCount: 8,
        columns: {
          classification: [
            'hot-99',
            'hot-95',
            'hot-90',
            'not-significant',
            'cold-90',
            'cold-95',
            'cold-99',
            'not-tested'
          ],
          neighbor_count: [1, 1, 1, 1, 1, 1, 1, 0]
        },
        columnTypes: {classification: 'string', neighbor_count: 'int32'}
      }
    },
    {
      operation: 'local-moran-statistics',
      table: {
        id: 'local-moran',
        label: 'Moran',
        rowCount: 6,
        columns: {cluster: ['HH', 'HL', 'LH', 'LL', 'not-significant', 'not-tested']},
        columnTypes: {cluster: 'string'}
      }
    },
    {
      operation: 'kernel-density-grid',
      table: {
        id: 'density-cells',
        label: 'Choropleth',
        rowCount: 8,
        columns: {density: [0, 1, 1, 1.000000001, 2, 3, null, 9]},
        columnTypes: {density: 'float64'}
      }
    }
  ];
  for (const {operation, table} of fixtures) {
    const descriptor = getKeplerOperationDescriptor(operation)!;
    table.id = descriptor.outputs[0].tableId;
    table.geometryColumn = 'geometry';
    table.columns['geometry'] = Array(table.rowCount).fill('{"type":"Point","coordinates":[0,0]}');
    table.columnTypes['geometry'] = 'string';
    const presentation = getKeplerBindingResultPresentations(
      {
        operationId: operation,
        engine: 'cpu-reference',
        parameters: {},
        tables: [table],
        diagnostics: [],
        summary: {},
        dispatchedContributors: []
      },
      descriptor
    )[0];
    const staged = prepareAnalysisResultPublication(table, presentation);
    const encoding = staged.analysisResult.encoding!;
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
    let style: KeplerResidentStyle | undefined;
    try {
      const dataset = session.importArrowDataset({
        id: table.id,
        table: staged.table,
        normalization: {
          columns: [encoding.classColumn],
          dictionaries: getAnalysisResultDictionaries(staged.analysisResult)
        }
      });
      const native = getKeplerResultEncodingStyle(encoding);
      const planned = createNativeStylePlan(
        {
          id: 'result',
          type: 'point',
          config: {
            dataId: dataset.id,
            color: [1, 2, 3],
            ...native.visualChannels,
            colorDomain: encoding.categories,
            visConfig: {filled: true, ...native.visConfig}
          }
        },
        {
          source: {datasetId: dataset.id, revision: dataset.revision},
          zoom: 10,
          columns: {[encoding.classColumn]: {format: 'uint32', dictionary: encoding.categories}}
        }
      );
      if (!planned.supported) throw new Error(`${operation}: ${planned.reasons.join('\n')}`);
      const channel = planned.plan.channels.fillColor;
      if (channel?.kind !== 'resident')
        throw new Error(`${operation}: missing native resident colour plan`);
      style = new KeplerResidentStyle({
        session,
        plan: channel.plan,
        workingMemoryBudgetBytes: 8 * 1024 * 1024,
        maximumPaletteCount: 16,
        maximumDomainCount: 16,
        maximumLegendBytes: 1024
      });
      const encoder = device.createCommandEncoder();
      style.encode(encoder);
      device.submit(encoder.finish());
      const observed: number[] = [];
      for (const chunk of style.colors!.data) {
        const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
        observed.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
      }
      const expected = encoding.values.map(value =>
        packGPUColor(...encoding.legend[encoding.categories.indexOf(value)].color)
      );
      compareResult(`${operation}: rendered colours`, observed, expected);
      compareResult(
        `${operation}: native legend labels`,
        (await style.readLegend())?.labels,
        encoding.categories
      );
    } finally {
      await style?.destroy(Promise.resolve());
      await session.destroy(Promise.resolve());
    }
  }
});

test('handwritten constant choropleth uses the produced class count and the middle AX colour', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('AX choropleth colours require WebGPU');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'constant',
    table: arrow.tableFromArrays({value: new Float32Array([7, 7, 7, 7])})
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const runtime = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    runtime.setMode('quantile');
    runtime.setClassCount(5);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    runtime.encode(encoder);
    device.submit(encoder.finish());
    await runtime.readSummary();
    if (runtime.error) throw new Error(runtime.error);
    compareResult('produced classes', runtime.summary?.classCount, 1);
    compareResult('active breaks', runtime.summary?.breaks, [7, 7]);
    compareResult('legend', getAnalysisLegendModel('quantile', runtime.summary)?.entries, [
      {label: '7 to 7', color: [66, 162, 142]}
    ]);
    const buffer = runtime.colors.data[0].buffer;
    if (!(buffer instanceof Buffer)) throw new Error('AX colour output must be a GPU buffer');
    compareResult(
      'constant map fill',
      Array.from(await buffer.readAsync(0, 16)),
      Array(4).fill([66, 162, 142, 255]).flat()
    );
  } finally {
    await runtime.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
  }
});

test('handwritten hotspots and Moran distinguish tested and excluded rows using AX tones', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('AX inference colours require WebGPU');
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'lattice',
    table: arrow.tableFromArrays({
      value: new Float32Array([0, 0, 0, 0, 1, 10, 0, 10, 10]),
      longitude: new Float32Array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
      latitude: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    })
  });
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const runtime = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    for (const mode of ['hotspot', 'moran'] as const) {
      if (mode === 'hotspot')
        runtime.configureHotspots({
          width: 3,
          height: 3,
          criterion: 'rook',
          radius: 1,
          minimumNeighbors: 4,
          selfWeight: 1,
          multipleTesting: 'none'
        });
      else
        runtime.configureLocalMoran({
          longitudeColumn: 'longitude',
          latitudeColumn: 'latitude',
          k: 1,
          minimumNeighbors: 1,
          significanceLevel: 0.05,
          multipleTesting: 'none',
          islands: 'exclude'
        });
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      runtime.encode(encoder);
      device.submit(encoder.finish());
      await runtime.readSummary();
      if (runtime.error) throw new Error(`${mode}: ${runtime.error}`);
      const columns = runtime.getTableColumns()!.columns;
      const readColumn = async (name: string, signed = false) => {
        const vector = columns[name];
        const values: number[] = [];
        for (const chunk of vector.data) {
          const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
          values.push(
            ...(signed
              ? new Int32Array(bytes.buffer, bytes.byteOffset, chunk.length)
              : name.endsWith('p_value')
                ? new Float32Array(bytes.buffer, bytes.byteOffset, chunk.length)
                : new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length))
          );
        }
        return values;
      };
      const names = Object.keys(columns);
      const probability = await readColumn(
        names.find(name => name.endsWith('_p_value') && !name.includes('adjusted'))!
      );
      const codes = await readColumn(
        names.find(name => name.endsWith(mode === 'hotspot' ? '_bin' : '_quadrant'))!,
        mode === 'hotspot'
      );
      const colors = mode === 'hotspot' ? getAnalysisHotspotColors() : getAnalysisMoranColors();
      const expected = codes.flatMap((code, row) => [
        ...(Number.isFinite(probability[row])
          ? colors[mode === 'hotspot' ? code + 3 : code]
          : [70, 70, 70]),
        255
      ]);
      const buffer = runtime.colors.data[0].buffer;
      if (!(buffer instanceof Buffer)) throw new Error('AX inference output must be a GPU buffer');
      compareResult(
        `${mode}: source-aligned fills`,
        Array.from(await buffer.readAsync(0, runtime.rowCount * 4)),
        expected
      );
      if (mode === 'hotspot' && !probability.some(value => !Number.isFinite(value)))
        throw new Error('Hotspot fixture must include excluded rows');
    }
  } finally {
    await runtime.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
  }
});
