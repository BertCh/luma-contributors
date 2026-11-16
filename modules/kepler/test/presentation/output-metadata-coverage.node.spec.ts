// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_ANALYSIS_CAUTIONS,
  GPU_ANALYSIS_SUMMARY_TEMPLATES
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  getKeplerOperationDescriptor,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {describeKeplerCatalogResult} from '../../src/presentation/result-report';

// The audit's 23 omissions. Keep these named so an accidental removal of a descriptor cannot
// make the all-registry coverage check pass by falling back to a derived descriptor.
const AUDITED_OUTPUTS = [
  ['geometry-measures', 'measures'],
  ['geometry-point-buffer', 'buffers'],
  ['geometry-convex-clip', 'clipped'],
  ['line-smooth', 'smoothed-lines'],
  ['line-precision-reduction', 'reduced-lines'],
  ['geodesic-destination', 'geodesic-destinations'],
  ['geometry-orientation', 'oriented-polygons'],
  ['grid-generator', 'grid-cells'],
  ['random-points-in-polygon', 'random-points'],
  ['random-points-on-line', 'random-points'],
  ['geometry-line-buffer', 'buffers'],
  ['geometry-polygon-buffer', 'buffers'],
  ['geometry-overlay-intersection', 'overlay'],
  ['geometry-overlay-union', 'overlay'],
  ['geometry-overlay-difference', 'overlay'],
  ['geometry-dot-density', 'dots'],
  ['geometry-local-frame-projection', 'local-frame-features'],
  ['network-map-matching', 'matched-route'],
  ['cell-rollup', 'rollup'],
  ['dggs-point-index', 'point-cells'],
  ['dggs-cell-rollup', 'rollup'],
  ['s2-cell-hierarchy', 's2-hierarchy'],
  ['s2-cap-covering', 's2-cap-cover']
] as const;

const POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [{geometry: {type: 'Point', coordinates: [5, 5]}, properties: {}}]
};
const LINES: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [5, 3],
          [10, 0]
        ]
      },
      properties: {}
    }
  ]
};
const POLYGONS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0]
          ]
        ]
      },
      properties: {}
    }
  ]
};

describe('output metadata coverage', () => {
  it('gives every hand-written geometric output a column plan or a documented uniform-style exception', () => {
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
      if (source === 'derived') continue; // Explicit derived allowance is enforced by the registry spec.
      expect(validateKeplerOperationDescriptor(descriptor, binding), binding.id).toEqual([]);
      for (const outputSpec of binding.outputs.filter(output => output.geometry !== 'none')) {
        const output = descriptor.outputs.find(
          candidate => candidate.tableId === outputSpec.tableId
        )!;
        const identifier = `${binding.id}.${outputSpec.tableId}`;
        expect(
          Boolean(output.presentation) !== Boolean(output.presentationException),
          identifier
        ).toBe(true);
        expect(output.interpretation, identifier).toBeDefined();
        expect(GPU_ANALYSIS_SUMMARY_TEMPLATES, identifier).toContain(
          output.interpretation!.summary
        );
        for (const caution of output.interpretation!.cautions)
          expect(GPU_ANALYSIS_CAUTIONS, identifier).toContain(caution);
        if (output.presentation)
          expect(output.presentation.column.trim().length, identifier).toBeGreaterThan(0);
        if (output.presentationException) {
          expect(output.presentationException.type, identifier).toBe('uniform-style');
          expect(output.presentationException.reason.length, identifier).toBeGreaterThan(40);
        }
      }
    }
    expect(AUDITED_OUTPUTS).toHaveLength(23);
    for (const [bindingId, tableId] of AUDITED_OUTPUTS) {
      const output = getKeplerOperationDescriptor(bindingId)?.outputs.find(
        candidate => candidate.tableId === tableId
      );
      expect(
        Boolean(output?.presentation) !== Boolean(output?.presentationException),
        `${bindingId}.${tableId}`
      ).toBe(true);
    }
  });

  for (const bindingId of [
    'geometry-point-buffer',
    'geometry-convex-clip',
    'line-smooth',
    'line-precision-reduction'
  ]) {
    it(`${bindingId}: newly declared columns exist and produce plans from CPU output`, async () => {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getKeplerOperationDescriptor(bindingId)!;
      const result = await runCatalogOperationAsync(
        binding,
        Object.fromEntries(
          Object.entries({coordinateSystem: 'planar', radius: 1}).filter(([name]) =>
            binding.parameters.some(parameter => parameter.name === name)
          )
        ),
        {points: POINTS, lines: LINES, features: POLYGONS, clip: POLYGONS},
        {engine: 'cpu'}
      );
      for (const output of descriptor.outputs) {
        const table = result.tables.find(candidate => candidate.id === output.tableId)!;
        expect(table.rowCount).toBeGreaterThan(0);
        expect(table.columns).toHaveProperty(output.presentation!.column);
        const plan = planKeplerResultPresentation(table, output);
        expect(plan?.column).toBe(output.presentation!.column);
        expect(plan!.ranges.length + plan!.legend.length).toBeGreaterThan(0);
        expect(
          [...plan!.ranges, ...plan!.legend].reduce((total, entry) => total + entry.count, 0)
        ).toBe(table.rowCount);
      }
    });
  }

  it('keeps geometry measures uniform for points, lines and polygons and explains the exception in reports', async () => {
    const binding = getCatalogBinding('geometry-measures')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const output = descriptor.outputs[0];
    for (const features of [POINTS, LINES, POLYGONS]) {
      const result = await runCatalogOperationAsync(
        binding,
        {coordinateSystem: 'planar'},
        {features},
        {engine: 'cpu'}
      );
      expect(planKeplerResultPresentation(result.tables[0], output)).toBeUndefined();
      expect(describeKeplerCatalogResult(result, binding, descriptor).notes).toContain(
        `${output.tableId}: ${output.presentationException!.reason}`
      );
    }
  });

  it('rejects ambiguous plans and undocumented exceptions', () => {
    const binding = getCatalogBinding('geometry-measures')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const invalid = {
      ...descriptor,
      outputs: [
        {
          ...descriptor.outputs[0],
          presentation: {column: 'area', scale: 'sequential' as const},
          presentationException: {type: 'uniform-style' as const, reason: ''}
        }
      ]
    };
    expect(validateKeplerOperationDescriptor(invalid, binding)).toEqual([
      `${binding.id}: output measures declares both a presentation and a uniform-style exception`,
      `${binding.id}: output measures uniform-style exception needs a reason`
    ]);
  });
});
