// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {RASTER_EMBEDDING_BINDINGS} from '../../src/analysis/catalog-raster-embeddings-bindings';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import {runKeplerLagrangianDiagnostics} from '../../src/analysis/lagrangian-diagnostics';
import {RASTER_EMBEDDING_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-raster-embedding';
import {
  getKeplerOperationDescriptorCoverage,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {
  evaluateKeplerOperationApplicability,
  suggestKeplerAnalyses
} from '../../src/descriptors/applicability';
import {describeKeplerCatalogResult} from '../../src/presentation/result-report';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

function getDescriptor(bindingId: string) {
  const descriptor = RASTER_EMBEDDING_OPERATION_DESCRIPTORS.find(
    candidate => candidate.bindingId === bindingId
  );
  if (!descriptor) throw new Error(`Missing raster demonstration descriptor ${bindingId}`);
  return descriptor;
}

describe('raster embedding and flow demonstration descriptors', () => {
  it('registers all 16 source bindings with validated hand-written descriptors', () => {
    expect(RASTER_EMBEDDING_OPERATION_DESCRIPTORS).toHaveLength(16);
    expect(
      new Set(RASTER_EMBEDDING_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).size
    ).toBe(16);
    const coverage = getKeplerOperationDescriptorCoverage(RASTER_EMBEDDING_BINDINGS);
    expect(coverage.derivedBindingIds).toEqual([]);
    expect(coverage.validationIssues).toEqual([]);
    expect(coverage.handWrittenBindingIds).toEqual(
      RASTER_EMBEDDING_BINDINGS.map(binding => binding.id)
    );
    for (const binding of RASTER_EMBEDDING_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      expect(resolveKeplerOperationDescriptor(binding)).toEqual({
        descriptor,
        source: 'hand-written'
      });
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      expect(descriptor.outputs.map(output => output.tableId)).toEqual(
        binding.outputs.map(output => output.tableId)
      );
    }
  });

  it('preserves fixture-only contracts without invented datasets, roles or settings', () => {
    for (const binding of RASTER_EMBEDDING_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      expect(binding.inputs, binding.id).toEqual([]);
      expect(descriptor.inputs, binding.id).toEqual([]);
      expect(descriptor.requirements, binding.id).toEqual([]);
      expect(
        descriptor.parameters.map(parameter => parameter.name),
        binding.id
      ).toEqual(binding.parameters.map(parameter => parameter.name));
      expect(descriptor.prominence, binding.id).toBe(3);
      const notes = descriptor.notes!.join(' ');
      expect(notes, binding.id).toContain('Generated fixture only');
      expect(notes, binding.id).toContain('accepts no loaded datasets');
      expect(notes, binding.id).toContain('without geometry or georeferencing');
      expect(
        descriptor.outputs.every(output => output.kind === 'summary'),
        binding.id
      ).toBe(true);
      expect(
        binding.outputs.every(output => output.geometry === 'none'),
        binding.id
      ).toBe(true);
      const resolution = resolveKeplerOperationParameters(binding, descriptor, {});
      expect(resolution.parameters, binding.id).toEqual(
        binding.id === 'raster-lagrangian-ftle' ? {timeStep: 0.1} : {}
      );
    }
  });

  it('tags the question answered by each demonstration rather than its raster domain', () => {
    const expectedQuestions = {
      kmeans: 'which-groups-or-regions',
      'nearest-prototype': 'which-groups-or-regions',
      'nearest-pixels': 'how-related',
      knn: 'which-groups-or-regions',
      'linear-probe': 'which-groups-or-regions',
      change: 'what-changed',
      covariance: 'how-related',
      pca: 'prepare-data',
      projection: 'prepare-data',
      dequantize: 'prepare-data',
      quantize: 'prepare-data',
      pyramid: 'prepare-data',
      similarity: 'how-related',
      'zone-mean': 'how-much-per-area',
      'usable-mask': 'prepare-data'
    };
    for (const [operation, question] of Object.entries(expectedQuestions)) {
      expect(getDescriptor(`raster-embedding-${operation}`).questions).toEqual([question]);
    }
    expect(getDescriptor('raster-lagrangian-ftle').questions).toEqual(['how-it-moves']);
  });

  it('keeps generated demonstrations out of suggestions for loaded data and for an empty host', () => {
    const profile = profileKeplerFeatureCollection(
      {type: 'FeatureCollection', features: [{geometry: {type: 'Point', coordinates: [0, 0]}}]},
      {datasetId: 'loaded', label: 'Loaded data', coordinateSystem: 'planar'}
    );
    for (const profiles of [[], [profile]]) {
      const suggestions = suggestKeplerAnalyses(RASTER_EMBEDDING_BINDINGS, profiles, {
        curated: [],
        executionEngine: 'gpu'
      });
      expect(suggestions.suggestions).toEqual([]);
      expect(suggestions.blocked).toEqual([]);
      expect(Object.values(suggestions.more).flat()).toEqual([]);
    }
  });

  it('provides table summaries and fixture cautions that survive result reporting', () => {
    for (const binding of RASTER_EMBEDDING_BINDINGS) {
      const descriptor = getDescriptor(binding.id);
      const output = descriptor.outputs[0];
      expect(output.interpretation, binding.id).toBeDefined();
      expect(output.next!.length, binding.id).toBeGreaterThanOrEqual(2);
      const column = output.presentation?.column ?? 'packed_word';
      const result: CatalogRunResult = {
        operationId: binding.id,
        engine: 'gpu',
        dispatchedContributors: binding.gpuContributorNames,
        parameters: binding.id === 'raster-lagrangian-ftle' ? {timeStep: 0.1} : {},
        tables: [
          {
            id: output.tableId,
            label: binding.outputs[0].label,
            rowCount: 2,
            columns: {[column]: [0, 1]},
            columnTypes: {[column]: 'float64'}
          }
        ],
        diagnostics: [],
        summary: {rowCount: 2}
      };
      const report = describeKeplerCatalogResult(result, binding, descriptor);
      expect(report.outputs, binding.id).toHaveLength(1);
      expect(report.outputs[0].usedTemplate, binding.id).toBe(output.interpretation!.summary);
      expect(report.outputs[0].sentence.length, binding.id).toBeGreaterThan(0);
      expect(report.notes, binding.id).toEqual(descriptor.notes);
    }
    // A codec word is not an ordered measurement; no numeric colour scale is assigned to it.
    expect(getDescriptor('raster-embedding-quantize').outputs[0].presentation).toBeUndefined();
    expect(getDescriptor('raster-embedding-projection').notes!.join(' ')).toContain('same');
    for (const operation of [
      'knn',
      'linear-probe',
      'nearest-prototype',
      'nearest-pixels',
      'similarity'
    ]) {
      expect(getDescriptor(`raster-embedding-${operation}`).notes!.join(' ')).toContain(
        'not learned land-cover classes'
      );
    }
  });

  it('preserves the constant signed FTLE step and documents its nonzero duration constraint', async () => {
    const binding = RASTER_EMBEDDING_BINDINGS.find(
      candidate => candidate.id === 'raster-lagrangian-ftle'
    )!;
    const descriptor = getDescriptor(binding.id);
    expect(descriptor.parameters).toEqual([
      {name: 'timeStep', defaultPolicy: {from: 'constant'}, level: 'basic', apply: 'live'}
    ]);
    const negative = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {},
      {overrides: {timeStep: -0.1}}
    );
    expect(negative.parameters).toEqual({timeStep: -0.1});
    // The dataset applicability checker deliberately blocks zero-input demonstrations from
    // suggestions, directing the user to their separate catalog execution path.
    const applicability = evaluateKeplerOperationApplicability(binding, descriptor, [], {
      executionEngine: 'gpu'
    });
    expect(applicability.status).toBe('blocked');
    expect(applicability.missing).toEqual([
      {
        what: 'This operation reads no layer, so there is nothing to suggest it for',
        remedy: 'Open it from the catalog'
      }
    ]);
    expect(descriptor.notes!.join(' ')).toContain('zero duration is refused before GPU allocation');
    await expect(
      runKeplerLagrangianDiagnostics(
        {} as never,
        {
          width: 8,
          height: 8,
          extent: [-4, -4, 1, 1],
          velocitiesStart: new Float32Array(128)
        },
        {timeStep: 0, stepCount: 10}
      )
    ).rejects.toThrow('finite nonzero time step');
  });
});
