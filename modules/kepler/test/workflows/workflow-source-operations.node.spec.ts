// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeWorkflowData} from '../../src/workflows/workflow-data';
import {
  getWorkflowSourceOperations,
  makeWorkflowDatasetRequest,
  WORKFLOW_SOURCE_DATASET_OPERATION_ID,
  WORKFLOW_SOURCE_DRAWING_OPERATION_ID,
  WORKFLOW_SOURCE_GEOJSON_LIMITS,
  WORKFLOW_SOURCE_GEOJSON_OPERATION_ID
} from '../../src/workflows/workflow-source-operations';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDatasetProfile,
  type WorkflowDatasetRequest,
  type WorkflowDrawing,
  type WorkflowHost,
  type WorkflowRunContext
} from '../../src/workflows/workflow-types';
import {makeFixtureHost, makeFixturePoints} from './workflow-fixtures';

const [datasetOperation, geojsonOperation, drawingOperation] = getWorkflowSourceOperations();
const DATASETS: readonly WorkflowDatasetProfile[] = [
  {
    id: 'points',
    label: 'Stations',
    rowCount: 2,
    resident: false,
    fields: [
      {name: 'geometry', type: 'geojson'},
      {name: 'longitude', type: 'real'},
      {name: 'latitude', type: 'real'}
    ],
    geometryFields: ['geometry'],
    pointColumns: {longitude: 'longitude', latitude: 'latitude'}
  },
  {
    id: 'other',
    label: 'Other stations',
    rowCount: 1,
    resident: true,
    fields: [{name: 'geometry', type: 'geojson'}],
    geometryFields: ['geometry']
  }
];
const POINT_GEOMETRY = {type: 'Point', coordinates: [-87.6, 41.8]};
const DATASET_PARAMETERS = {dataset: 'points', geometryField: 'geometry'};

function makeContext(
  host: WorkflowHost = makeFixtureHost(DATASETS, {
    points: makeFixturePoints(2),
    other: makeFixturePoints(1, 'other')
  }),
  signal = new AbortController().signal
): WorkflowRunContext {
  return {device: null, engine: 'auto', host, signal, workflowId: 'workflow', stepId: 'source'};
}

function makeDrawingHost() {
  const host = makeFixtureHost();
  const drawings: WorkflowDrawing[] = [
    {id: 'point', label: 'Marker', kind: 'point'},
    {id: 'line', label: 'Route', kind: 'line'},
    {id: 'polygon', label: 'Area', kind: 'polygon'},
    {id: 'rectangle', label: 'Bounds', kind: 'polygon'}
  ];
  const reads: Array<{drawingIds: readonly string[] | 'all'; signal: AbortSignal}> = [];
  const geometryByKind = {
    point: POINT_GEOMETRY,
    line: {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [1, 1]
      ]
    },
    polygon: {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
          [0, 0]
        ]
      ]
    }
  };
  host.listDrawings = () => drawings;
  host.getDrawingRevisionKey = () => `drawings@${host.revision}`;
  host.readDrawings = async (drawingIds, signal) => {
    reads.push({drawingIds, signal});
    const selected =
      drawingIds === 'all'
        ? drawings
        : drawingIds.flatMap(identity => drawings.filter(drawing => drawing.id === identity));
    return makeWorkflowData(
      {
        id: 'data',
        label: 'Drawn shapes',
        rowCount: selected.length,
        columns: {
          drawing_id: selected.map(drawing => drawing.id),
          label: selected.map(drawing => drawing.label),
          geometry: selected.map(drawing => JSON.stringify(geometryByKind[drawing.kind]))
        },
        columnTypes: {drawing_id: 'string', label: 'string', geometry: 'string'},
        geometryColumn: 'geometry'
      },
      {identityColumn: 'drawing_id'}
    );
  };
  return {host, drawings, reads};
}

function expectParameterRefusal(
  action: () => unknown,
  parameter: string,
  message: string,
  code = 'parameter-out-of-range'
): void {
  expect(action).toThrow(WorkflowError);
  expect(action).toThrow(expect.objectContaining({code, message, details: {parameter}}));
}

describe('workflow source declarations', () => {
  test('exposes the three bounded builtin host sources in a stable order', () => {
    expect(getWorkflowSourceOperations().map(operation => operation.id)).toEqual([
      WORKFLOW_SOURCE_DATASET_OPERATION_ID,
      WORKFLOW_SOURCE_GEOJSON_OPERATION_ID,
      WORKFLOW_SOURCE_DRAWING_OPERATION_ID
    ]);
    for (const operation of getWorkflowSourceOperations()) {
      expect(operation).toMatchObject({
        category: 'source',
        origin: 'builtin',
        engine: 'host',
        version: '1',
        inputs: [],
        outputs: [
          {
            name: 'data',
            label: 'Data',
            kind: 'features',
            identityColumns:
              operation.id === WORKFLOW_SOURCE_DRAWING_OPERATION_ID ? ['drawing_id'] : [],
            maximumRows:
              operation.id === WORKFLOW_SOURCE_DRAWING_OPERATION_ID
                ? 1000
                : WORKFLOW_LIMITS.maximumSourceRows
          }
        ]
      });
      expect(operation.summary.length).toBeGreaterThan(0);
      expect(operation.declaredBehavior.length).toBeGreaterThan(0);
      expect(operation.limitation.length).toBeGreaterThan(0);
    }
    expect(datasetOperation.title).toBe('Dataset');
    expect(geojsonOperation.title).toBe('GeoJSON');
    expect(drawingOperation.title).toBe('Drawn shapes');
    expect(geojsonOperation.getExternalRevisionKey).toBeUndefined();
  });

  test('declares ordered dataset controls and hides the replay label', () => {
    expect(datasetOperation.parameters.map(parameter => parameter.name)).toEqual([
      'dataset',
      'datasetLabel',
      'geometryMode',
      'geometryField',
      'longitudeField',
      'latitudeField',
      'population',
      'maximumRows'
    ]);
    expect(datasetOperation.parameters[1]).toEqual({
      name: 'datasetLabel',
      kind: 'string',
      label: 'Dataset label',
      maximumLength: 120,
      defaultValue: '',
      visibleWhen: {parameter: 'dataset', equals: []}
    });
    expect(datasetOperation.parameters.slice(3, 6)).toMatchObject([
      {columnsFrom: 'dataset', visibleWhen: {parameter: 'geometryMode', equals: ['geojson-field']}},
      {columnsFrom: 'dataset', visibleWhen: {parameter: 'geometryMode', equals: ['point-columns']}},
      {columnsFrom: 'dataset', visibleWhen: {parameter: 'geometryMode', equals: ['point-columns']}}
    ]);
    expect(geojsonOperation.parameters).toEqual([
      {
        name: 'text',
        kind: 'text',
        label: 'GeoJSON',
        maximumLength: WORKFLOW_LIMITS.maximumTextCharacters,
        defaultValue: ''
      },
      {name: 'name', kind: 'string', label: 'Name', defaultValue: ''}
    ]);
  });
});

describe('dataset parameters and requests', () => {
  test.each([
    [{}, 'dataset', 'Dataset: Dataset must be chosen.'],
    [{dataset: '  '}, 'dataset', 'Dataset: Dataset must be chosen.'],
    [{dataset: 'points'}, 'geometryField', 'Dataset: Geometry field must be chosen.'],
    [
      {dataset: 'points', geometryMode: 'point-columns'},
      'longitudeField',
      'Dataset: Longitude field must be chosen.'
    ],
    [
      {dataset: 'points', geometryMode: 'point-columns', longitudeField: 'longitude'},
      'latitudeField',
      'Dataset: Latitude field must be chosen.'
    ],
    [
      {
        dataset: 'points',
        geometryMode: 'point-columns',
        longitudeField: 'longitude',
        latitudeField: 'longitude'
      },
      'latitudeField',
      'Dataset: Latitude field must differ from Longitude field.'
    ],
    [
      {...DATASET_PARAMETERS, maximumRows: 0},
      'maximumRows',
      'Dataset: Maximum rows must be between 1 and 100000; got 0.'
    ],
    [
      {...DATASET_PARAMETERS, maximumRows: 100001},
      'maximumRows',
      'Dataset: Maximum rows must be between 1 and 100000; got 100001.'
    ],
    [
      {...DATASET_PARAMETERS, maximumRows: 1.5},
      'maximumRows',
      'Dataset: Maximum rows must be a whole number; got 1.5.'
    ],
    [
      {...DATASET_PARAMETERS, geometryMode: 'guess'},
      'geometryMode',
      'Dataset: Geometry must be one of geojson-field, point-columns, none; got "guess".'
    ],
    [
      {...DATASET_PARAMETERS, population: 'visible'},
      'population',
      'Dataset: Rows must be one of all, selected; got "visible".'
    ],
    [
      {...DATASET_PARAMETERS, datasetLabel: 'x'.repeat(121)},
      'datasetLabel',
      'Dataset: Dataset label must be at most 120 characters; got 121.'
    ]
  ])('refuses dataset configuration %#', (raw, parameter, message) => {
    expectParameterRefusal(() => datasetOperation.normalizeParameters(raw), parameter, message);
  });

  test('rejects unknown parameters', () => {
    expectParameterRefusal(
      () => datasetOperation.normalizeParameters({...DATASET_PARAMETERS, unknown: 1}),
      'unknown',
      'Dataset: Unknown parameter "unknown".',
      'unknown-parameter'
    );
  });

  test.each([
    [
      DATASET_PARAMETERS,
      {
        datasetId: 'points',
        geometry: {mode: 'geojson-field', field: 'geometry'},
        population: 'all',
        maximumRows: 20000
      }
    ],
    [
      {
        dataset: 'points',
        datasetLabel: 'Stations',
        geometryMode: 'point-columns',
        longitudeField: 'longitude',
        latitudeField: 'latitude',
        population: 'selected',
        maximumRows: '1'
      },
      {
        datasetId: 'points',
        datasetLabel: 'Stations',
        geometry: {mode: 'point-columns', longitude: 'longitude', latitude: 'latitude'},
        population: 'selected',
        maximumRows: 1
      }
    ],
    [
      {dataset: 'points', geometryMode: 'none', maximumRows: 100000},
      {
        datasetId: 'points',
        geometry: {mode: 'none'},
        population: 'all',
        maximumRows: 100000
      }
    ]
  ])('builds geometry request %# from normalized parameters', (raw, expected) => {
    expect(makeWorkflowDatasetRequest(datasetOperation.normalizeParameters(raw))).toEqual(expected);
  });

  test('keeps the original field names and accepts the dataset label length boundary', () => {
    const parameters = datasetOperation.normalizeParameters({
      ...DATASET_PARAMETERS,
      geometryField: 'geometry with spaces',
      datasetLabel: 'x'.repeat(120)
    });
    expect(parameters['datasetLabel']).toHaveLength(120);
    expect(makeWorkflowDatasetRequest(parameters).geometry).toEqual({
      mode: 'geojson-field',
      field: 'geometry with spaces'
    });
  });
});

describe('dataset source execution', () => {
  test('resolves by id before label and passes the request and signal to the host', async () => {
    const data = makeFixturePoints(2);
    const host = makeFixtureHost(DATASETS, {points: data});
    const context = makeContext(host);
    const requests: WorkflowDatasetRequest[] = [];
    host.readDataset = async (request, signal) => {
      requests.push(request);
      expect(signal).toBe(context.signal);
      return data;
    };
    const parameters = datasetOperation.normalizeParameters({
      ...DATASET_PARAMETERS,
      datasetLabel: 'Other stations'
    });
    const output = await datasetOperation.run(context, {}, parameters);
    expect(requests).toEqual([makeWorkflowDatasetRequest(parameters)]);
    expect(output).toEqual({
      outputs: {data},
      summary: {rows: 2, kind: 'point'},
      engine: 'host',
      diagnostics: [],
      dispatchedContributors: []
    });
    expect(output.outputs['data']).toBe(data);
  });

  test('finds a unique label after replay changes the dataset id', async () => {
    const data = makeFixturePoints(2);
    const host = makeFixtureHost([{...DATASETS[0], id: 'replayed-points'}, DATASETS[1]], {
      'replayed-points': data
    });
    const requests: WorkflowDatasetRequest[] = [];
    const readDataset = host.readDataset;
    host.readDataset = async (request, signal) => {
      requests.push(request);
      return readDataset(request, signal);
    };
    const parameters = datasetOperation.normalizeParameters({
      ...DATASET_PARAMETERS,
      datasetLabel: 'Stations'
    });
    expect((await datasetOperation.run(makeContext(host), {}, parameters)).outputs['data']).toBe(
      data
    );
    expect(requests).toEqual([
      {...makeWorkflowDatasetRequest(parameters), datasetId: 'replayed-points'}
    ]);
    expect(parameters['dataset']).toBe('points');
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('replayed-points@1');
  });

  test.each([
    [
      {dataset: 'missing', datasetLabel: 'Stations'},
      [
        {...DATASETS[0], id: 'first'},
        {...DATASETS[0], id: 'second'}
      ],
      'Dataset: Dataset "Stations" must match exactly one dataset; found 2.'
    ],
    [
      {dataset: 'missing', datasetLabel: 'Lost stations'},
      DATASETS,
      'Dataset: Dataset "Lost stations" was not found.'
    ],
    [{dataset: 'missing'}, DATASETS, 'Dataset: Dataset "missing" was not found.']
  ])('refuses unresolved dataset %# before reading', async (raw, profiles, message) => {
    const host = makeFixtureHost(profiles);
    host.readDataset = async () => {
      throw new Error('must not read an unresolved dataset');
    };
    const parameters = datasetOperation.normalizeParameters({...DATASET_PARAMETERS, ...raw});
    const result = datasetOperation.run(makeContext(host), {}, parameters);
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'dataset-not-found',
      message,
      details: {parameter: 'dataset'}
    });
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('missing');
  });

  test('keys follow host revision and never throw when host lookups fail', () => {
    const host = makeFixtureHost(DATASETS);
    const parameters = datasetOperation.normalizeParameters(DATASET_PARAMETERS);
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('points@1');
    host.revision = 2;
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('points@2');
    host.getDatasetRevisionKey = () => {
      throw new Error('revision unavailable');
    };
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('missing');
    host.listDatasets = () => {
      throw new Error('host unavailable');
    };
    expect(datasetOperation.getExternalRevisionKey?.(host, parameters)).toBe('missing');
  });

  test('reports the selected population', async () => {
    const output = await datasetOperation.run(
      makeContext(),
      {},
      datasetOperation.normalizeParameters({...DATASET_PARAMETERS, population: 'selected'})
    );
    expect(output.diagnostics).toEqual([
      {
        code: 'selected-population',
        severity: 'info',
        count: 2,
        message: 'Dataset: Rows follow the current host filters and selection.'
      }
    ]);
  });

  test('accepts the row cap exactly and refuses an oversized host result', async () => {
    const context = makeContext();
    expect(
      (
        await datasetOperation.run(
          context,
          {},
          datasetOperation.normalizeParameters({...DATASET_PARAMETERS, maximumRows: 2})
        )
      ).summary
    ).toEqual({rows: 2, kind: 'point'});
    await expect(
      datasetOperation.run(
        context,
        {},
        datasetOperation.normalizeParameters({...DATASET_PARAMETERS, maximumRows: 1})
      )
    ).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: 'Dataset: Maximum rows is 1; got 2 rows.',
      details: {parameter: 'maximumRows'}
    });
  });

  test.each(['read', 'list'] as const)('wraps a host %s failure with its cause', async failure => {
    const host = makeFixtureHost(DATASETS);
    const cause = new Error('Host storage unavailable');
    if (failure === 'read') {
      host.readDataset = async () => {
        throw cause;
      };
    } else {
      host.listDatasets = () => {
        throw cause;
      };
    }
    const result = datasetOperation.run(
      makeContext(host),
      {},
      datasetOperation.normalizeParameters(DATASET_PARAMETERS)
    );
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'dataset-read-failed',
      message: 'Dataset: Host storage unavailable',
      cause,
      details: {cause}
    });
  });

  test('preserves host WorkflowErrors', async () => {
    const host = makeFixtureHost(DATASETS);
    const cause = new WorkflowError(
      'geometry-required',
      'Dataset: Geometry field is unavailable.',
      {
        parameter: 'geometryField'
      }
    );
    host.readDataset = async () => {
      throw cause;
    };
    await expect(
      datasetOperation.run(
        makeContext(host),
        {},
        datasetOperation.normalizeParameters(DATASET_PARAMETERS)
      )
    ).rejects.toBe(cause);
  });

  test('cancels before reading or after a host read finishes or fails', async () => {
    for (const timing of ['before', 'after', 'failure']) {
      const controller = new AbortController();
      const host = makeFixtureHost(DATASETS);
      let readCount = 0;
      host.readDataset = async () => {
        readCount++;
        controller.abort();
        if (timing === 'failure') throw new Error('aborted by host');
        return makeFixturePoints(2);
      };
      if (timing === 'before') controller.abort();
      await expect(
        datasetOperation.run(
          makeContext(host, controller.signal),
          {},
          datasetOperation.normalizeParameters(DATASET_PARAMETERS)
        )
      ).rejects.toMatchObject({code: 'cancelled', message: 'Dataset: Reading was cancelled.'});
      expect(readCount).toBe(timing === 'before' ? 0 : 1);
    }
  });
});

describe('GeoJSON source', () => {
  test.each(['', ' \n\t'])('refuses blank text %j', text => {
    expectParameterRefusal(
      () => geojsonOperation.normalizeParameters({text}),
      'text',
      'GeoJSON: GeoJSON must not be blank.'
    );
  });

  test('enforces text length before parsing', () => {
    const maximumLength = WORKFLOW_LIMITS.maximumTextCharacters;
    expect(
      geojsonOperation.normalizeParameters({text: 'x'.repeat(maximumLength)})['text']
    ).toHaveLength(maximumLength);
    expectParameterRefusal(
      () => geojsonOperation.normalizeParameters({text: 'x'.repeat(maximumLength + 1)}),
      'text',
      `GeoJSON: GeoJSON must be at most ${maximumLength} characters; got ${maximumLength + 1}.`
    );
  });

  test.each([
    ['collection', {type: 'FeatureCollection', features: [{geometry: POINT_GEOMETRY}]}],
    ['feature', {type: 'Feature', id: 'station', geometry: POINT_GEOMETRY, properties: {count: 3}}],
    ['geometry', POINT_GEOMETRY]
  ])('reads a single-row %s', async (_label, value) => {
    const parameters = geojsonOperation.normalizeParameters({text: JSON.stringify(value)});
    const output = await geojsonOperation.run(makeContext(), {}, parameters);
    expect(output).toMatchObject({
      outputs: {
        data: {
          kind: 'point',
          table: {
            id: 'data',
            label: 'GeoJSON',
            rowCount: 1,
            columns: {geometry: [JSON.stringify(POINT_GEOMETRY)]},
            geometryColumn: 'geometry'
          }
        }
      },
      engine: 'host',
      summary: {rows: 1, kind: 'point'},
      diagnostics: [],
      dispatchedContributors: []
    });
    if ('id' in value) {
      expect(output.outputs['data'].identityColumn).toBe('feature_id');
      expect(output.outputs['data'].table.columns['feature_id']).toEqual(['station']);
      expect(output.outputs['data'].table.columns['count']).toEqual([3]);
    }
  });

  test('uses the configured name and accepts a geometry collection', async () => {
    const geometry = {type: 'GeometryCollection', geometries: [POINT_GEOMETRY]};
    const output = await geojsonOperation.run(
      makeContext(),
      {},
      geojsonOperation.normalizeParameters({text: JSON.stringify(geometry), name: 'Stations'})
    );
    expect(output.outputs['data'].table.label).toBe('Stations');
    expect(output.outputs['data'].table.columns['geometry']).toEqual([JSON.stringify(geometry)]);
  });

  test.each([
    '{',
    'null',
    '[]',
    '42',
    '{}',
    '{"type":"Unsupported","coordinates":[]}',
    '{"type":"FeatureCollection","features":{}}',
    '{"type":"Feature"}',
    '{"type":"FeatureCollection","features":[null]}',
    '{"type":"Point"}'
  ])('wraps invalid GeoJSON %s as a text parameter error', async text => {
    const result = geojsonOperation.run(
      makeContext(),
      {},
      geojsonOperation.normalizeParameters({text})
    );
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message: expect.stringMatching(/^GeoJSON: GeoJSON is not valid: .+\.$/),
      details: {parameter: 'text'}
    });
  });

  test('refuses an empty feature collection', async () => {
    await expect(
      geojsonOperation.run(
        makeContext(),
        {},
        geojsonOperation.normalizeParameters({text: '{"type":"FeatureCollection","features":[]}'})
      )
    ).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message: 'GeoJSON: GeoJSON must contain at least one feature; got 0.',
      details: {parameter: 'text'}
    });
  });

  test('defends the feature row cap even when called directly with oversized text', async () => {
    const feature = {type: 'Feature', geometry: null};
    const maximumRows = WORKFLOW_LIMITS.maximumSourceRows;
    const text = JSON.stringify({
      type: 'FeatureCollection',
      features: Array.from({length: maximumRows + 1}, () => feature)
    });
    // This exceeds the text parameter limit too; exercise the runtime row gate independently.
    await expect(geojsonOperation.run(makeContext(), {}, {text, name: ''})).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: `GeoJSON: GeoJSON must contain at most ${maximumRows} features; got ${maximumRows + 1}.`,
      details: {parameter: 'text'}
    });
  });

  test('cancels before parsing and checks again before returning data', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      geojsonOperation.run(makeContext(undefined, controller.signal), {}, {text: '{', name: ''})
    ).rejects.toMatchObject({code: 'cancelled', message: 'GeoJSON: Reading was cancelled.'});

    const duringConversion = new AbortController();
    const parameters = {
      text: JSON.stringify(POINT_GEOMETRY),
      get name() {
        duringConversion.abort();
        return 'Stations';
      }
    };
    await expect(
      geojsonOperation.run(makeContext(undefined, duringConversion.signal), {}, parameters)
    ).rejects.toMatchObject({code: 'cancelled', message: 'GeoJSON: Reading was cancelled.'});
  });
});

describe('drawing source', () => {
  test('declares drawing controls and refuses empty chosen ids or oversized text', () => {
    expect(drawingOperation.parameters).toEqual([
      {
        name: 'shapes',
        kind: 'enum',
        label: 'Shapes',
        options: ['all', 'chosen'],
        defaultValue: 'all'
      },
      {
        name: 'drawingIds',
        kind: 'string',
        label: 'Shape ids',
        maximumLength: 2000,
        defaultValue: '',
        visibleWhen: {parameter: 'shapes', equals: ['chosen']}
      }
    ]);
    expect(drawingOperation.normalizeParameters({})).toEqual({shapes: 'all', drawingIds: ''});
    for (const drawingIds of ['', ' , \t, ']) {
      expectParameterRefusal(
        () => drawingOperation.normalizeParameters({shapes: 'chosen', drawingIds}),
        'drawingIds',
        'Drawn shapes: Shape ids must contain at least one id when Shapes is chosen.'
      );
    }
    expectParameterRefusal(
      () => drawingOperation.normalizeParameters({drawingIds: 'x'.repeat(2001)}),
      'drawingIds',
      'Drawn shapes: Shape ids must be at most 2000 characters; got 2001.'
    );
  });

  test('reads all drawing kinds and keeps host identity and labels', async () => {
    const {host, reads} = makeDrawingHost();
    const context = makeContext(host);
    const output = await drawingOperation.run(
      context,
      {},
      drawingOperation.normalizeParameters({})
    );
    expect(reads).toEqual([{drawingIds: 'all', signal: context.signal}]);
    expect(output).toMatchObject({
      outputs: {
        data: {
          identityColumn: 'drawing_id',
          kind: 'features',
          table: {
            rowCount: 4,
            columns: {
              drawing_id: ['point', 'line', 'polygon', 'rectangle'],
              label: ['Marker', 'Route', 'Area', 'Bounds']
            }
          }
        }
      },
      engine: 'host',
      summary: {rows: 4, kind: 'features'},
      diagnostics: [],
      dispatchedContributors: []
    });
  });

  test('reads chosen ids after trimming, removing empty entries and deduplicating', async () => {
    const {host, reads} = makeDrawingHost();
    const context = makeContext(host);
    const output = await drawingOperation.run(
      context,
      {},
      drawingOperation.normalizeParameters({
        shapes: 'chosen',
        drawingIds: ' rectangle, point, ,rectangle '
      })
    );
    expect(reads).toEqual([{drawingIds: ['rectangle', 'point'], signal: context.signal}]);
    expect(output.outputs['data'].table.columns['drawing_id']).toEqual(['rectangle', 'point']);
    expect(output.summary).toEqual({rows: 2, kind: 'features'});
  });

  test('reports missing read and list capabilities', async () => {
    await expect(
      drawingOperation.run(makeContext(), {}, drawingOperation.normalizeParameters({}))
    ).rejects.toMatchObject({
      code: 'host-capability-missing',
      message: 'Drawn shapes: this application cannot read drawn shapes.'
    });
    const {host, reads} = makeDrawingHost();
    delete host.listDrawings;
    await expect(
      drawingOperation.run(
        makeContext(host),
        {},
        drawingOperation.normalizeParameters({shapes: 'chosen', drawingIds: 'point'})
      )
    ).rejects.toMatchObject({
      code: 'host-capability-missing',
      message: 'Drawn shapes: this application cannot list drawn shapes.'
    });
    expect(reads).toEqual([]);
  });

  test('refuses an empty drawing result with map tool guidance', async () => {
    const {host, drawings} = makeDrawingHost();
    drawings.length = 0;
    await expect(
      drawingOperation.run(makeContext(host), {}, drawingOperation.normalizeParameters({}))
    ).rejects.toMatchObject({
      code: 'input-required',
      message: 'Drawn shapes: draw a shape on the map first (polygon, rectangle or line tool).'
    });
  });

  test('refuses an unknown chosen id before reading', async () => {
    const {host, reads} = makeDrawingHost();
    await expect(
      drawingOperation.run(
        makeContext(host),
        {},
        drawingOperation.normalizeParameters({shapes: 'chosen', drawingIds: 'point, gone'})
      )
    ).rejects.toMatchObject({
      code: 'dataset-not-found',
      message: 'Drawn shapes: Shape id "gone" was not found.',
      details: {parameter: 'drawingIds'}
    });
    expect(reads).toEqual([]);
  });

  test('follows drawing revisions and uses a non-throwing unsupported key', () => {
    const {host} = makeDrawingHost();
    const parameters = drawingOperation.normalizeParameters({});
    expect(drawingOperation.getExternalRevisionKey?.(host, parameters)).toBe('drawings@1');
    host.revision = 2;
    expect(drawingOperation.getExternalRevisionKey?.(host, parameters)).toBe('drawings@2');
    delete host.getDrawingRevisionKey;
    expect(drawingOperation.getExternalRevisionKey?.(host, parameters)).toBe('unsupported');
    host.getDrawingRevisionKey = () => {
      throw new Error('Drawing state unavailable');
    };
    expect(drawingOperation.getExternalRevisionKey?.(host, parameters)).toBe('unsupported');
  });

  test('enforces the output row limit and handles cancellation', async () => {
    const {host} = makeDrawingHost();
    const parameters = drawingOperation.normalizeParameters({});
    host.readDrawings = async () => makeFixturePoints(1001);
    await expect(drawingOperation.run(makeContext(host), {}, parameters)).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: 'Drawn shapes: Data must contain at most 1000 rows; got 1001.',
      details: {port: 'data'}
    });
    const controller = new AbortController();
    host.readDrawings = async () => {
      controller.abort();
      return makeFixturePoints(1);
    };
    await expect(
      drawingOperation.run(makeContext(host, controller.signal), {}, parameters)
    ).rejects.toMatchObject({
      code: 'cancelled',
      message: 'Drawn shapes: Reading was cancelled.'
    });
  });
});

describe('source review regressions', () => {
  async function readGeoJSON(text: string) {
    return geojsonOperation.run(makeContext(), {}, geojsonOperation.normalizeParameters({text}));
  }

  test.each([
    '{"type":"Point","coordinates":["west",null]}',
    '{"type":"Point","coordinates":[1e400,0]}',
    '{"type":"MultiPoint","coordinates":[[0,1],[2]]}',
    '{"type":"LineString","coordinates":[[0,1],null]}',
    '{"type":"MultiLineString","coordinates":[[[0,1],[2,"north"]]]}',
    '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,"south"]]]}',
    '{"type":"MultiPolygon","coordinates":[[[[0,0],[1,0],[1,1],[0,null]]]]}',
    '{"type":"GeometryCollection","geometries":[{"type":"Unsupported"}]}',
    '{"type":"GeometryCollection","geometries":[null]}',
    '{"type":"GeometryCollection","geometries":[{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[0,null]}]}]}',
    '{"type":"Feature","id":{"x":1},"properties":{},"geometry":null}',
    '{"type":"Feature","id":[1],"properties":{},"geometry":null}',
    '{"type":"Feature","id":true,"properties":{},"geometry":null}',
    '{"type":"Feature","id":1e400,"properties":{},"geometry":null}'
  ])('F1 rejects malformed geometry or identity: %s', async text => {
    const result = readGeoJSON(text);
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message: text.includes('"id"')
        ? 'GeoJSON: GeoJSON is not valid: Feature id must be a string, a finite number or null.'
        : 'GeoJSON: GeoJSON is not valid: expected GeoJSON geometry with finite numeric coordinates and valid structure.',
      details: {parameter: 'text'}
    });
  });

  test.each([
    {type: 'Point', coordinates: [0, -1, 2]},
    {
      type: 'MultiPoint',
      coordinates: [
        [0, 1],
        [2, 3]
      ]
    },
    {
      type: 'LineString',
      coordinates: [
        [0, 1],
        [2, 3]
      ]
    },
    {
      type: 'MultiLineString',
      coordinates: [
        [
          [0, 1],
          [2, 3]
        ]
      ]
    },
    {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 0]
        ]
      ]
    },
    {
      type: 'MultiPolygon',
      coordinates: [
        [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0]
          ]
        ]
      ]
    },
    {
      type: 'GeometryCollection',
      geometries: [{type: 'GeometryCollection', geometries: [POINT_GEOMETRY]}]
    }
  ])('F1 preserves valid geometry structure: $type', async geometry => {
    const output = await readGeoJSON(JSON.stringify(geometry));
    expect(output.outputs['data'].table.columns['geometry']).toEqual([JSON.stringify(geometry)]);
    expect(output.outputs['data'].table.rowCount).toBe(1);
  });

  test.each([
    {type: 'Point', coordinates: []},
    {type: 'Point', coordinates: [[0, 1]]},
    {type: 'LineString', coordinates: [[0, 1]]},
    {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [0, 0]
        ]
      ]
    },
    {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1]
        ]
      ]
    }
  ])('F1 refuses invalid coordinate dimensions or ring structure: %j', async geometry => {
    await expect(readGeoJSON(JSON.stringify(geometry))).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message:
        'GeoJSON: GeoJSON is not valid: expected GeoJSON geometry with finite numeric coordinates and valid structure.',
      details: {parameter: 'text'}
    });
  });

  test.each([0, 17, 'station', null])('F1 preserves scalar feature identity %j', async identity => {
    const output = await readGeoJSON(
      JSON.stringify({type: 'Feature', id: identity, geometry: null})
    );
    expect(output.outputs['data'].identityColumn).toBe(
      identity === null ? undefined : 'feature_id'
    );
    expect(output.outputs['data'].table.columns['feature_id']).toEqual(
      identity === null ? undefined : [identity]
    );
  });

  test('F2 rejects deeply nested properties with a labelled nesting limit', async () => {
    const depth = 100000;
    const text =
      '{"type":"Feature","geometry":null,"properties":{"nested":' +
      '['.repeat(depth) +
      '0' +
      ']'.repeat(depth) +
      '}}';
    expect(text.length).toBe(200060);
    expect(text.length).toBeLessThan(WORKFLOW_LIMITS.maximumTextCharacters);
    const result = readGeoJSON(text);
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'parameter-out-of-range',
      message: 'GeoJSON: GeoJSON nesting depth must be at most 64; got 65.',
      details: {parameter: 'text'}
    });
  });

  test('F2 accepts the exact nesting limit and rejects one more level', async () => {
    for (const depth of [62, 63]) {
      const nested = '['.repeat(depth) + '0' + ']'.repeat(depth);
      const text = '{"type":"Feature","geometry":null,"properties":{"nested":' + nested + '}}';
      const result = readGeoJSON(text);
      if (depth === 62) {
        expect((await result).outputs['data'].table.columns['nested']).toEqual([nested]);
      } else {
        await expect(result).rejects.toMatchObject({
          code: 'parameter-out-of-range',
          message: 'GeoJSON: GeoJSON nesting depth must be at most 64; got 65.',
          details: {parameter: 'text'}
        });
      }
    }
  });

  test('F2 ignores brackets, escaped quotes and backslashes inside strings', async () => {
    const nested = '[{}]"\\'.repeat(1000);
    const text = JSON.stringify({type: 'Feature', geometry: null, properties: {nested}});
    expect((await readGeoJSON(text)).outputs['data'].table.columns['nested']).toEqual([nested]);
  });

  test('F2 and F3 publish immutable safety limits in operation metadata', () => {
    expect(WORKFLOW_SOURCE_GEOJSON_LIMITS).toEqual({
      maximumNestingDepth: 64,
      maximumColumns: 256,
      maximumTableCells: 1000000
    });
    expect(Object.isFrozen(WORKFLOW_SOURCE_GEOJSON_LIMITS)).toBe(true);
    expect(geojsonOperation.limitation).toContain('64 levels (root depth one)');
    expect(geojsonOperation.limitation).toContain('256 columns and 1000000 cells');
  });

  test.each([
    1000, 2000
  ])('F3 refuses sparse properties before dense conversion (%i rows)', async count => {
    const text = JSON.stringify({
      type: 'FeatureCollection',
      features: Array.from({length: count}, (_, index) => ({
        type: 'Feature',
        geometry: null,
        properties: {[`property${index}`]: index}
      }))
    });
    expect(text.length).toBeLessThan(WORKFLOW_LIMITS.maximumTextCharacters);
    const result = readGeoJSON(text);
    await expect(result).rejects.toBeInstanceOf(WorkflowError);
    await expect(result).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: `GeoJSON: GeoJSON table must contain at most 256 columns; got ${count + 1}.`,
      details: {parameter: 'text'}
    });
  });

  test('F3 counts geometry and identity columns in the column budget', async () => {
    const properties = {
      ...Object.fromEntries(Array.from({length: 253}, (_, index) => [`property${index}`, index])),
      geometry: 'geometry property',
      feature_id: 'identity property'
    };
    const feature = {type: 'Feature', geometry: null, properties};
    const output = await readGeoJSON(JSON.stringify(feature));
    expect(Object.keys(output.outputs['data'].table.columns)).toHaveLength(256);
    expect(output.outputs['data'].table.columns['geometry_property']).toEqual([
      'geometry property'
    ]);
    expect(output.outputs['data'].table.columns['feature_id_property']).toEqual([
      'identity property'
    ]);
    await expect(readGeoJSON(JSON.stringify({...feature, id: 0}))).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message: 'GeoJSON: GeoJSON table must contain at most 256 columns; got 257.',
      details: {parameter: 'text'}
    });
  });

  test('F3 accepts the exact cell budget and refuses one additional row', async () => {
    const properties = Object.fromEntries(
      Array.from({length: 249}, (_, index) => [`property${index}`, index])
    );
    for (const count of [4000, 4001]) {
      const text = JSON.stringify({
        type: 'FeatureCollection',
        features: Array.from({length: count}, (_, index) => ({
          type: 'Feature',
          geometry: null,
          ...(index === 0 ? {properties} : {})
        }))
      });
      expect(text.length).toBeLessThan(WORKFLOW_LIMITS.maximumTextCharacters);
      const result = readGeoJSON(text);
      if (count === 4000) {
        const output = await result;
        expect(output.summary).toEqual({rows: 4000, kind: 'table'});
        expect(Object.keys(output.outputs['data'].table.columns)).toHaveLength(250);
        expect(output.outputs['data'].table.columns['property1'][0]).toBe(1);
        expect(output.outputs['data'].table.columns['property1'][3999]).toBeNull();
      } else {
        await expect(result).rejects.toMatchObject({
          code: 'capacity-exceeded',
          message: 'GeoJSON: GeoJSON table must contain at most 1000000 cells; got 1000250.',
          details: {parameter: 'text'}
        });
      }
    }
  });

  test('F4 isolates exposed parameter definitions from normalization and later factories', () => {
    const [first] = getWorkflowSourceOperations();
    const originalMaximum = first.parameters[7].maximum;
    const originalOptions = first.parameters[2].options;
    const originalVisibility = first.parameters[3].visibleWhen;
    try {
      first.parameters[7].maximum = 200000;
      first.parameters[2].options = ['guess'];
      first.parameters[3].visibleWhen = {parameter: 'dataset', equals: ['points']};
      const [later] = getWorkflowSourceOperations();
      expect(later.parameters).not.toBe(first.parameters);
      expect(later.parameters[7].maximum).toBe(100000);
      for (const operation of [first, later]) {
        expectParameterRefusal(
          () =>
            operation.normalizeParameters({
              dataset: 'points',
              geometryMode: 'none',
              maximumRows: 100001
            }),
          'maximumRows',
          'Dataset: Maximum rows must be between 1 and 100000; got 100001.'
        );
      }
    } finally {
      first.parameters[7].maximum = originalMaximum;
      first.parameters[2].options = originalOptions;
      first.parameters[3].visibleWhen = originalVisibility;
    }
  });

  test('F4 isolates nested options, visibility metadata and GeoJSON definitions', () => {
    const [firstDataset, firstGeoJSON] = getWorkflowSourceOperations();
    const [laterDataset, laterGeoJSON] = getWorkflowSourceOperations();
    expect(firstDataset.parameters[2].options).not.toBe(laterDataset.parameters[2].options);
    expect(firstDataset.parameters[3].visibleWhen).not.toBe(laterDataset.parameters[3].visibleWhen);
    expect(firstDataset.parameters[3].visibleWhen?.equals).not.toBe(
      laterDataset.parameters[3].visibleWhen?.equals
    );
    (firstDataset.parameters[2].options as string[]).push('guess');
    const visibility = firstDataset.parameters[3].visibleWhen;
    if (!visibility) throw new Error('Missing fixture visibility');
    visibility.parameter = 'dataset';
    (visibility.equals as string[]).push('none');
    firstGeoJSON.parameters[0].maximumLength = 1;
    firstGeoJSON.parameters[0].defaultValue = 'changed';
    expect(laterDataset.parameters[2].options).toEqual(['geojson-field', 'point-columns', 'none']);
    expect(laterDataset.parameters[3].visibleWhen).toEqual({
      parameter: 'geometryMode',
      equals: ['geojson-field']
    });
    expect(laterGeoJSON.parameters[0]).toMatchObject({maximumLength: 262144, defaultValue: ''});
    expect(firstGeoJSON.normalizeParameters({text: JSON.stringify(POINT_GEOMETRY)})['text']).toBe(
      JSON.stringify(POINT_GEOMETRY)
    );
    expectParameterRefusal(
      () => firstDataset.normalizeParameters({...DATASET_PARAMETERS, geometryMode: 'guess'}),
      'geometryMode',
      'Dataset: Geometry must be one of geojson-field, point-columns, none; got "guess".'
    );
  });

  test.each([
    ['null prototype', () => Object.create(null) as unknown],
    [
      'throwing message getter',
      () =>
        Object.defineProperty(new Error(), 'message', {
          get() {
            throw new Error('cannot read message');
          }
        }) as unknown
    ],
    [
      'throwing prototype inspection',
      () =>
        new Proxy(
          {},
          {
            getPrototypeOf() {
              throw new Error('cannot read prototype');
            }
          }
        ) as unknown
    ],
    [
      'throwing coercion',
      () => ({
        toString() {
          throw new Error('cannot stringify');
        }
      })
    ]
  ])('F5 wraps a host rejection with %s', async (_label, makeCause) => {
    const cause = makeCause();
    const host = makeFixtureHost(DATASETS);
    host.readDataset = async () => {
      throw cause;
    };
    const result = datasetOperation.run(
      makeContext(host),
      {},
      datasetOperation.normalizeParameters(DATASET_PARAMETERS)
    );
    const error: unknown = await result.catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(WorkflowError);
    expect(error).toMatchObject({
      code: 'dataset-read-failed',
      message: 'Dataset: Unknown error'
    });
    expect((error as WorkflowError).cause).toBe(cause);
    expect((error as WorkflowError).details.cause).toBe(cause);
  });

  test.each([
    ['offline', 'offline'],
    [42, '42'],
    [null, 'null'],
    [undefined, 'undefined']
  ])('F5 wraps primitive host rejection %j', async (cause, message) => {
    const host = makeFixtureHost(DATASETS);
    host.readDataset = async () => {
      throw cause;
    };
    const error: unknown = await datasetOperation
      .run(makeContext(host), {}, datasetOperation.normalizeParameters(DATASET_PARAMETERS))
      .catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(WorkflowError);
    expect(error).toMatchObject({code: 'dataset-read-failed', message: `Dataset: ${message}`});
    expect((error as WorkflowError).details.cause).toBe(cause);
  });
});
