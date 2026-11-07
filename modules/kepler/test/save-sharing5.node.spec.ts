// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {beforeEach, describe, expect, it} from 'vitest';
import {parseKeplerSandboxState} from '../src/recipes/sandbox-state';
import {
  clearCatalogInvocations,
  listCatalogInvocations,
  parseCatalogInvocationRecords,
  recordCatalogInvocation,
  restoreCatalogInvocations,
  MAXIMUM_CATALOG_INVOCATIONS
} from '../../../examples/deck/kepler-compute/src/catalog-invocation-store';
import {
  SANDBOX_EXTENSION_CATALOG,
  SANDBOX_EXTENSION_STREAMED_WEIGHTS,
  applySandboxExtensions,
  collectSandboxExtensions,
  validateSandboxExtensions
} from '../../../examples/deck/kepler-compute/src/sandbox-host-extensions';

const emptyState = {schemaVersion: 1, sources: [], nativeConfig: {}, selections: [], analyses: []};

describe('sandbox state extensions', () => {
  beforeEach(() => clearCatalogInvocations());

  it('round-trips opaque extension members through the strict state parser', () => {
    const parsed = parseKeplerSandboxState({
      ...emptyState,
      settings: {memoryBudgetBytes: 1024, extensions: {a: {b: [1, 'x', null]}}}
    });
    expect(parsed.settings).toEqual({
      memoryBudgetBytes: 1024,
      extensions: {a: {b: [1, 'x', null]}}
    });
    expect(() =>
      parseKeplerSandboxState({...emptyState, settings: {extensions: {a: () => 1}}})
    ).toThrow();
    expect(() => parseKeplerSandboxState({...emptyState, settings: {other: 1}})).toThrow();
  });

  it('collects catalog runs and chunked-weights policy, and omits defaults', () => {
    expect(collectSandboxExtensions([])).toBeUndefined();
    recordCatalogInvocation({
      operationId: 'op',
      invocation: '{"schemaVersion":1}',
      inputs: {a: {mode: 'native', datasetId: 'd1', field: 'geom'}},
      engine: 'auto'
    });
    const extensions = collectSandboxExtensions([
      {
        datasetId: 'd1',
        getStreamedWeightsOptions: () => ({forceStreamed: true, maximumSlotsPerChunk: 4096})
      },
      {datasetId: 'd2', getStreamedWeightsOptions: () => ({})}
    ])!;
    expect(Object.keys(extensions).sort()).toEqual(
      [SANDBOX_EXTENSION_CATALOG, SANDBOX_EXTENSION_STREAMED_WEIGHTS].sort()
    );
    const validated = validateSandboxExtensions(extensions);
    expect(validated.catalog).toHaveLength(1);
    expect(validated.streamedWeights?.datasets).toEqual({
      d1: {forceStreamed: true, maximumSlotsPerChunk: 4096}
    });
  });

  it('applies restored members and replaces (never merges) the catalog list', () => {
    recordCatalogInvocation({operationId: 'old', invocation: '{}', inputs: {}, engine: 'cpu'});
    const applied: [string, unknown][] = [];
    applySandboxExtensions(
      {
        [SANDBOX_EXTENSION_CATALOG]: {
          schemaVersion: 1,
          records: [
            {
              id: 'new#1',
              operationId: 'new',
              invocation: '{}',
              inputs: {s: {mode: 'sample'}},
              engine: 'auto'
            }
          ]
        },
        [SANDBOX_EXTENSION_STREAMED_WEIGHTS]: {
          schemaVersion: 1,
          datasets: {d1: {maximumSlotsPerChunk: 8}}
        }
      },
      (datasetId, options) => applied.push([datasetId, options])
    );
    expect(listCatalogInvocations().map(record => record.id)).toEqual(['new#1']);
    expect(applied).toEqual([['d1', {maximumSlotsPerChunk: 8}]]);
    applySandboxExtensions(undefined, () => undefined);
    expect(listCatalogInvocations()).toEqual([]);
  });

  it('rejects malformed members as a whole and ignores unknown members', () => {
    expect(() =>
      validateSandboxExtensions({
        [SANDBOX_EXTENSION_STREAMED_WEIGHTS]: {
          schemaVersion: 1,
          datasets: {d: {maximumSlotsPerChunk: 0}}
        }
      })
    ).toThrow();
    expect(() =>
      validateSandboxExtensions({
        [SANDBOX_EXTENSION_CATALOG]: {schemaVersion: 1, records: [{id: 1}]}
      })
    ).toThrow();
    expect(() =>
      parseCatalogInvocationRecords({
        schemaVersion: 1,
        records: [
          {id: 'a', operationId: 'x', invocation: '', inputs: {}, engine: 'cpu'},
          {id: 'a', operationId: 'x', invocation: '', inputs: {}, engine: 'cpu'}
        ]
      })
    ).toThrow();
    expect(validateSandboxExtensions({future: {x: 1}}).catalog).toEqual([]);
  });

  it('dedupes identical runs, bounds the list and truncates oversized pasted inputs', () => {
    const first = recordCatalogInvocation({
      operationId: 'op',
      invocation: 'a',
      inputs: {},
      engine: 'auto'
    });
    expect(
      recordCatalogInvocation({operationId: 'op', invocation: 'a', inputs: {}, engine: 'auto'})
    ).toBe(first);
    for (let index = 0; index < MAXIMUM_CATALOG_INVOCATIONS + 5; index++)
      recordCatalogInvocation({
        operationId: 'op',
        invocation: `i${index}`,
        inputs: {},
        engine: 'auto'
      });
    expect(listCatalogInvocations()).toHaveLength(MAXIMUM_CATALOG_INVOCATIONS);
    const big = recordCatalogInvocation({
      operationId: 'big',
      invocation: 'z',
      inputs: {a: {mode: 'text', text: 'x'.repeat(70000)}},
      engine: 'auto'
    });
    expect(big.inputs['a']).toEqual({mode: 'text', text: ''});
    restoreCatalogInvocations([]);
  });
});
