// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {
  LensController,
  createDefaultViewRegistry
} from '@luma-contributors/lenses/computation-views';
import {
  compareWorkflowPreflightMeasurement,
  createWorkflowPreflightMeasurementChains,
  estimateWorkflowPreflight
} from '../../src/workflows/workflow-preflight';
import {makeWorkflowComputationRecord} from '../../src/workflows/workflow-computation-record';
import {makeWorkflowRunReport} from '../../src/workflows/workflow-report';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {
  packCatalogLines,
  packCatalogPoints,
  packCatalogPolygons
} from '../../src/analysis/catalog-gpu-pack';
import {runGpuGeometryMeasures} from '../../src/analysis/catalog-gpu-geometry';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {CatalogFeatureCollection, CatalogGeometry} from '../../src/analysis/catalog-geojson';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';

describe('workflow preflight', () => {
  const registry = getWorkflowOperationRegistry();

  test('pins three cold-run chains to independent layout arithmetic, using max across branches', () => {
    const chains = createWorkflowPreflightMeasurementChains();
    expect(chains.map(chain => chain.document.id)).toEqual([
      'point-measures',
      'buffer-measures',
      'branched-buffer-measures'
    ]);
    // Four, 128 and 64 features × sixteen four-byte words; buffering has no transients.
    const expected = [
      {scratch: 256, nodes: 2},
      {scratch: 8192, nodes: 4},
      {scratch: 4096, nodes: 8}
    ];
    chains.forEach((chain, index) => {
      const document = parseWorkflowDocument(serializeWorkflowDocument(chain.document));
      const estimate = estimateWorkflowPreflight(document, registry, chain);
      expect(estimate.complete).toBe(true);
      expect(estimate.peakScratchBytes).toBe(expected[index].scratch);
      expect(estimate.commandNodeCount).toBe(expected[index].nodes);
      expect(estimate.boundaries).toHaveLength(
        document.steps.reduce((total, step) => total + Object.keys(step.inputs).length, 0)
      );
      expect(estimate.boundaries.every(boundary => boundary.materialization === 'cpu-table')).toBe(
        true
      );
    });
  });

  test('unknown operations and missing metadata remain unknown, including downstream steps', () => {
    const chain = createWorkflowPreflightMeasurementChains()[1];
    const estimate = estimateWorkflowPreflight(chain.document, registry);
    expect(estimate.complete).toBe(false);
    expect(estimate.peakScratchBytes).toBeNull();
    expect(estimate.commandNodeCount).toBeNull();
    expect(estimate.steps.find(step => step.stepId === 'buffer-0')?.reason).toContain('metadata');
    expect(estimate.steps.find(step => step.stepId === 'measures-0')?.scratchBytes).toBeNull();
    const fixture = estimateWorkflowPreflight(makeFixtureDocument(), makeFixtureRegistry());
    expect(fixture.steps.find(step => step.stepId === 'scale')?.status).toBe('unknown');
    expect(
      compareWorkflowPreflightMeasurement(fixture, {peakScratchBytes: 0, commandNodeCount: 0})
        .passed
    ).toBe(false);
  });

  test('reuses shared AX source profiles before execution and follows source edits', () => {
    const chain = createWorkflowPreflightMeasurementChains()[1];
    expect(
      estimateWorkflowPreflight(chain.document, registry, {profileSources: true}).peakScratchBytes
    ).toBe(8192);
    const source = chain.document.steps[0];
    const collection = JSON.parse(
      String(source.parameters['text'].from === 'static' && source.parameters['text'].value)
    );
    collection.features = collection.features.slice(0, 4);
    source.parameters['text'] = {from: 'static', value: JSON.stringify(collection)};
    const estimate = estimateWorkflowPreflight(chain.document, registry, {profileSources: true});
    expect(estimate.peakScratchBytes).toBe(256);
    expect(estimate.commandNodeCount).toBe(3);
  });

  test('a positive sort limit estimates selected single-point rows, matching actual execution', async () => {
    for (const limit of [0, 1, 4, 10]) {
      const chain = createWorkflowPreflightMeasurementChains()[0];
      chain.document.steps.splice(1, 0, {
        id: 'sort',
        operationId: 'table-sort-rows',
        position: {x: 100, y: 0},
        inputs: {rows: {stepId: 'source', output: 'data'}},
        parameters: {
          column: {from: 'static', value: 'feature_id'},
          limit: {from: 'static', value: limit}
        }
      });
      chain.document.steps[2].inputs['features'] = {stepId: 'sort', output: 'sorted'};
      const expectedRows = limit === 0 ? 4 : Math.min(limit, 4);
      const estimate = estimateWorkflowPreflight(chain.document, registry, {profileSources: true});
      expect(estimate.complete).toBe(true);
      expect(estimate.steps.find(step => step.stepId === 'sort')?.outputs['sorted']).toMatchObject({
        rowCount: expectedRows,
        packedFeatureCount: expectedRows,
        vertexCount: expectedRows
      });
      expect(estimate.peakScratchBytes).toBe(expectedRows * 64);
      const document = structuredClone(chain.document);
      document.settings.engine = 'cpu';
      const runner = new WorkflowRunner({registry, host: makeFixtureHost()});
      try {
        runner.setDocument(document);
        expect((await runner.run()).status).toBe('done');
        expect(runner.getState().steps['sort'].result?.outputs['sorted'].table.rowCount).toBe(
          expectedRows
        );
      } finally {
        runner.destroy();
      }
    }
  });

  test('truncating variable-sized geometry leaves the downstream shape unknown', () => {
    const chain = createWorkflowPreflightMeasurementChains()[1];
    chain.document.steps.splice(2, 0, {
      id: 'sort',
      operationId: 'table-sort-rows',
      position: {x: 150, y: 0},
      inputs: {rows: {stepId: 'buffer-0', output: 'buffers'}},
      parameters: {
        column: {from: 'static', value: 'source_index'},
        limit: {from: 'static', value: 1}
      }
    });
    chain.document.steps[3].inputs['features'] = {stepId: 'sort', output: 'sorted'};
    const estimate = estimateWorkflowPreflight(chain.document, registry, chain);
    expect(estimate.steps.find(step => step.stepId === 'sort')?.outputs).toEqual({});
    expect(estimate.steps.find(step => step.stepId === 'measures-0')?.scratchBytes).toBeNull();
    expect(estimate.complete).toBe(false);
  });

  test('measures use packed features for empty and partially empty point geometries', async () => {
    const cases: Array<{geometries: CatalogGeometry[]; packed: number; vertices: number}> = [
      {
        geometries: [
          {type: 'MultiPoint', coordinates: []},
          {type: 'MultiPoint', coordinates: []}
        ],
        packed: 0,
        vertices: 0
      },
      {
        geometries: [
          {type: 'Point', coordinates: [0, 0]},
          {type: 'MultiPoint', coordinates: []}
        ],
        packed: 1,
        vertices: 1
      },
      // Positions equal row count, but only one row is usable: totals alone are insufficient.
      {
        geometries: [
          {
            type: 'MultiPoint',
            coordinates: [
              [0, 0],
              [1, 1]
            ]
          },
          {type: 'MultiPoint', coordinates: []}
        ],
        packed: 1,
        vertices: 2
      }
    ];
    for (const fixture of cases) {
      const collection: CatalogFeatureCollection = {
        type: 'FeatureCollection',
        features: fixture.geometries.map(geometry => ({geometry, properties: {}}))
      };
      const packed = packCatalogPoints(collection.features);
      expect(packed.sourceIndices).toHaveLength(fixture.packed);
      expect(packed.vertexCount).toBe(fixture.vertices);
      const chain = createWorkflowPreflightMeasurementChains()[0];
      chain.document.steps[0].parameters['text'] = {
        from: 'static',
        value: JSON.stringify(collection)
      };
      const estimate = estimateWorkflowPreflight(chain.document, registry, {profileSources: true});
      const measures = estimate.steps.find(step => step.stepId === 'measures-0')!;
      expect(estimate.complete).toBe(true);
      expect(measures.scratchBytes).toBe(fixture.packed * 64);
      expect(measures.commandNodeCount).toBe(fixture.packed ? 2 : 0);
      expect(measures.outputs['measures']).toMatchObject({
        rowCount: 2,
        packedFeatureCount: fixture.packed
      });
      if (fixture.packed === 0) {
        const result = await runGpuGeometryMeasures(
          {
            get device(): never {
              throw new Error('Empty geometry must not access a GPU device');
            }
          },
          collection,
          'planar'
        );
        expect(result.tables[0].rowCount).toBe(2);
        expect(result.diagnostics).toContainEqual(
          expect.objectContaining({code: 'empty-geometry', count: 2})
        );
      }
    }
  });

  test('missing packed-feature metadata leaves GPU cost unknown, including legacy profiles', () => {
    const chain = createWorkflowPreflightMeasurementChains()[0];
    delete chain.sourceShapes['source']['data'].packedFeatureCount;
    const estimate = estimateWorkflowPreflight(chain.document, registry, chain);
    expect(estimate.steps.find(step => step.stepId === 'measures-0')).toMatchObject({
      status: 'unknown',
      scratchBytes: null,
      commandNodeCount: null
    });
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}]
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'legacy',
      label: 'Legacy profile'
    });
    delete profile.geometryPackedFeatureCounts;
    chain.document.steps[0].operationId = 'source-dataset';
    chain.document.steps[0].parameters = {
      dataset: {from: 'static', value: 'legacy'},
      geometryField: {from: 'static', value: 'geometry'}
    };
    expect(
      estimateWorkflowPreflight(chain.document, registry, {
        profileSources: true,
        host: {
          ...makeFixtureHost(),
          listDatasets: () => [
            {
              id: 'legacy',
              label: 'Legacy profile',
              rowCount: 1,
              resident: false,
              fields: [],
              geometryFields: ['geometry']
            }
          ],
          getAnalysisProfile: () => profile
        }
      }).complete
    ).toBe(false);
  });

  test('profile packed counts match every catalog geometry packer', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {geometry: {type: 'MultiPoint', coordinates: []}, properties: {}},
        {
          geometry: {
            type: 'MultiPoint',
            coordinates: [
              [0, 0],
              [1, 1]
            ]
          },
          properties: {}
        },
        {geometry: {type: 'MultiLineString', coordinates: []}, properties: {}},
        {
          geometry: {
            type: 'LineString',
            coordinates: [
              [0, 0],
              [1, 0]
            ]
          },
          properties: {}
        },
        {geometry: {type: 'MultiPolygon', coordinates: []}, properties: {}},
        {
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [0, 0],
                [1, 0],
                [0, 1],
                [0, 0]
              ]
            ]
          },
          properties: {}
        }
      ]
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'packed-families',
      label: 'Packed families'
    });
    expect(profile.geometryFeatureCounts).toEqual({
      point: 2,
      line: 2,
      polygon: 2,
      withoutGeometry: 0
    });
    expect(profile.geometryPackedFeatureCounts).toEqual({point: 1, line: 1, polygon: 1});
    expect(profile.geometryPackedFeatureCounts).toEqual({
      point: packCatalogPoints(collection.features).sourceIndices.length,
      line: packCatalogLines(collection.features).sourceIndices.length,
      polygon: packCatalogPolygons(collection.features, {closing: 'keep'}).sourceIndices.length
    });
  });

  test('resolves promoted parameters and CPU overrides, rejects invalid counts', () => {
    const chain = createWorkflowPreflightMeasurementChains()[1];
    chain.document.inputs = [{id: 'segments', label: 'Segments', kind: 'integer', value: 1}];
    chain.document.steps[1].parameters['segmentsPerQuarter'] = {from: 'input', inputId: 'segments'};
    expect(
      estimateWorkflowPreflight(chain.document, registry, chain).steps[1].outputs['buffers']
        .vertexCount
    ).toBe(640);
    chain.document.settings.engine = 'cpu';
    expect(estimateWorkflowPreflight(chain.document, registry, chain)).toMatchObject({
      peakScratchBytes: 0,
      commandNodeCount: 0
    });
    for (const invalid of [-1, 0.5, NaN, Infinity]) {
      expect(() =>
        estimateWorkflowPreflight(chain.document, registry, {
          sourceShapes: {source: {data: {rowCount: invalid}}}
        })
      ).toThrow(/WorkflowPreflight rowCount.*integer/);
    }
  });

  test('disabled/cyclic steps do not become executed work and raster boundaries are explicit', () => {
    const chain = createWorkflowPreflightMeasurementChains()[0];
    chain.document.steps[1].disabled = true;
    expect(estimateWorkflowPreflight(chain.document, registry, chain).steps[1]).toMatchObject({
      status: 'disabled',
      scratchBytes: 0,
      commandNodeCount: 0
    });
    chain.document.steps[1].disabled = false;
    const operations = registry.list().map(operation =>
      operation.id === 'source-geojson'
        ? {
            ...operation,
            outputs: operation.outputs.map(output => ({...output, kind: 'raster' as const}))
          }
        : operation
    );
    const rasterEstimate = estimateWorkflowPreflight(chain.document, {
      list: () => operations,
      get: id => operations.find(operation => operation.id === id)
    });
    expect(rasterEstimate.boundaries[0].materialization).toBe('cpu-raster');
    expect(rasterEstimate.boundaries[1].materialization).toBe('cpu-table');
    chain.document.steps[1].inputs['features'] = {stepId: 'calculate-0', output: 'calculated'};
    expect(estimateWorkflowPreflight(chain.document, registry, chain).complete).toBe(false);
  });

  test('qualification requires inclusive 25% scratch tolerance and exact command count', () => {
    const chain = createWorkflowPreflightMeasurementChains()[0];
    const estimate = estimateWorkflowPreflight(chain.document, registry, chain);
    expect(
      compareWorkflowPreflightMeasurement(
        {...estimate, peakScratchBytes: 250},
        {peakScratchBytes: 200, commandNodeCount: 2}
      ).passed
    ).toBe(true);
    expect(
      compareWorkflowPreflightMeasurement(
        {...estimate, peakScratchBytes: 251},
        {peakScratchBytes: 200, commandNodeCount: 2}
      ).passed
    ).toBe(false);
    expect(
      compareWorkflowPreflightMeasurement(estimate, {peakScratchBytes: 256, commandNodeCount: 3})
        .passed
    ).toBe(false);
    expect(
      compareWorkflowPreflightMeasurement(
        {...estimate, peakScratchBytes: 0},
        {peakScratchBytes: 0, commandNodeCount: 2}
      ).passed
    ).toBe(true);
    expect(
      compareWorkflowPreflightMeasurement(estimate, {peakScratchBytes: 0, commandNodeCount: 2})
        .passed
    ).toBe(false);
  });

  test('all three chains execute on CPU bindings and report materialisation boundaries', async () => {
    for (const chain of createWorkflowPreflightMeasurementChains()) {
      const document = structuredClone(chain.document);
      document.settings.engine = 'cpu';
      const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 1000});
      try {
        runner.setDocument(document);
        const record = await runner.run();
        expect(record.status, JSON.stringify(record)).toBe('done');
        const preflight = estimateWorkflowPreflight(chain.document, registry, chain);
        const report = makeWorkflowRunReport(
          document,
          registry,
          runner.getState(),
          record,
          preflight
        );
        expect(report).toContain('## Preflight');
        expect(report).toContain('### Materialisation boundaries');
        expect(report).toContain('CPU table');
        expect(report).toContain(`${preflight.peakScratchBytes} bytes`);
      } finally {
        runner.destroy();
      }
    }
  });
});

describe('workflow computation record bridge', () => {
  test('opens a real runner record in existing views without inventing GPU timings', async () => {
    const registry = makeFixtureRegistry();
    const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 1000});
    runner.setDocument(makeFixtureDocument());
    try {
      const run = await runner.run();
      const record = makeWorkflowComputationRecord(run, registry);
      expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
      expect(record.runId).toBe(run.runId);
      expect(record.createdAt).toBe('1970-01-01T00:00:01.000Z');
      expect(record.graphs[0].edges).toEqual([
        {from: 'source', to: 'scale', kind: 'dataflow'},
        {from: 'scale', to: 'count', kind: 'dataflow'}
      ]);
      expect(record.graphs[0].nodes[1].attrs).toMatchObject({
        engine: 'cpu-reference',
        cacheHit: false,
        timingScope: 'runner wall time, not GPU timestamps'
      });
      expect(record.workflow?.invocations[0].graph).toBe(record.graphs[0].id);
      expect(record).not.toHaveProperty('metrics');
      const controller = new LensController({
        record,
        registry: createDefaultViewRegistry(),
        rung: 'table',
        lensId: 'workflow-node-test'
      });
      try {
        expect(JSON.stringify(controller.getFrame().tree)).toContain('fixture-scale');
        expect(controller.getFrame().state.runId).toBe(run.runId);
        controller.dispatch({type: 'bind', axis: 'node', value: 'scale'});
        expect(JSON.stringify(controller.getFrame().tree)).toContain('durationMilliseconds');
      } finally {
        controller.destroy();
      }
      runner.setDocument({...makeFixtureDocument(), title: 'Edited later', steps: []});
      expect(makeWorkflowComputationRecord(run, registry)).toEqual(record);
    } finally {
      runner.destroy();
    }
  });

  test('retains cancellation, cache hits, errors and upstream producers outside a partial run', () => {
    const record = makeWorkflowComputationRecord(
      {
        schemaVersion: 1,
        runId: 'partial',
        workflowId: 'fixture',
        startedAt: 1000,
        durationMilliseconds: 12,
        status: 'cancelled',
        steps: [
          {
            stepId: 'scale',
            operationId: 'fixture-scale',
            key: 'current-key',
            status: 'cancelled',
            cacheHit: true,
            durationMilliseconds: 12,
            error: {
              severity: 'error',
              code: 'capacity-exceeded',
              message: 'Fixture scale factor must be at most 10'
            },
            inputs: [
              {stepId: 'source', output: 'data', port: 'points', key: 'upstream-key', rowCount: 4}
            ]
          }
        ]
      },
      makeFixtureRegistry()
    );
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(record.graphs[0].nodes[0].attrs['external']).toBe(true);
    expect(record.graphs[0].nodes[1].attrs).toMatchObject({
      status: 'cancelled',
      cacheHit: true,
      contentKey: 'current-key',
      errorCode: 'capacity-exceeded'
    });
    expect(record.graphs[0].nodes[1].attrs['inputs']).toContain('upstream-key');
    expect(record.graphs[0].edges).toEqual([{from: 'source', to: 'scale', kind: 'dataflow'}]);
  });
});
