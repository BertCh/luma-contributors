// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {getCatalogSampleInputs} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {POINT_CLOUD_BINDINGS} from '../../src/analysis/catalog-pointcloud-bindings';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {POINT_CLOUD_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-pointcloud';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

// These expected columns are an independent contract, checked against real GPU publication.
const EXPECTED_COLUMNS: Readonly<Record<string, readonly string[]>> = {
  ground: ['classification', 'ground_mask', 'ground_status'],
  smrf: ['classification', 'ground_mask', 'ground_status'],
  surfaces: ['terrain', 'surface', 'canopy', 'coverage'],
  height: ['height_above_ground', 'height_status'],
  neighborhoods: ['neighbor_count', 'neighborhood_status', 'neighbor_1_row', 'neighbor_1_distance'],
  features: ['planarity', 'linearity', 'normal_z', 'feature_status'],
  'statistical-filter': ['inlier', 'mean_distance', 'neighborhood_status'],
  'normal-scale': ['normal_scale', 'normal_variance', 'normal_status'],
  icp: ['inlier', 'registration_status', 'original_altitude'],
  change: ['change', 'lower_bound', 'upper_bound', 'detection_limit', 'change_status'],
  'voxel-merge': ['voxel_id', 'count', 'representative', 'merge_status']
};

for (const binding of POINT_CLOUD_BINDINGS) {
  test(`${binding.id}: GPU output schema supplies descriptor presentation columns`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const samples = getCatalogSampleInputs(binding.id);
    const inputs = Object.fromEntries(
      binding.inputs.map(input => [
        input.name,
        JSON.parse(samples[input.name]) as CatalogFeatureCollection
      ])
    );
    const descriptor = POINT_CLOUD_OPERATION_DESCRIPTORS.find(
      candidate => candidate.bindingId === binding.id
    )!;
    const result = await runCatalogOperationAsync(
      binding,
      {
        coordinateSystem: binding.id === 'pointcloud-voxel-merge' ? 'planar' : 'EPSG:4326',
        ...(binding.id === 'pointcloud-change' ? {alreadyAligned: true} : {}),
        ...(binding.id === 'pointcloud-voxel-merge' ? {disjointOwnedRows: true} : {})
      },
      inputs,
      {device}
    );
    const report = JSON.stringify({
      bindingId: binding.id,
      diagnostics: result.diagnostics,
      tables: result.tables.map(table => ({
        id: table.id,
        rowCount: table.rowCount,
        columns: Object.keys(table.columns)
      }))
    });
    const points = result.tables.find(table => table.id === 'points')!;
    for (const column of EXPECTED_COLUMNS[binding.id.slice('pointcloud-'.length)]) {
      expect(points.columns[column], report).toBeDefined();
      expect(points.columns[column].length, report).toBe(points.rowCount);
    }
    for (const output of descriptor.outputs) {
      const table = result.tables.find(candidate => candidate.id === output.tableId);
      expect(table, report).toBeDefined();
      if (!table) throw new Error(`Missing descriptor output: ${report}`);
      if (output.presentation) {
        expect(table.columns[output.presentation.column], report).toBeDefined();
        const plan = planKeplerResultPresentation(table, output);
        expect(plan?.column, report).toBe(output.presentation.column);
        if (binding.id === 'pointcloud-change') {
          // This sample moves the comparison upward by 0.2: every usable change is positive.
          // The descriptor supports signed change; the planner adapts one-sided results to a
          // sequential ramp. Node regressions separately cover both gain and loss together.
          expect(output.presentation.scale, report).toBe('diverging');
          expect(output.presentation.center, report).toBe(0);
          const changes = table.columns['change'].filter(
            (value): value is number => typeof value === 'number' && Number.isFinite(value)
          );
          expect(changes.length, report).toBeGreaterThan(0);
          expect(
            changes.every(value => value > 0),
            report
          ).toBe(true);
          expect(plan?.scale, report).toBe('sequential');
          expect(plan?.notes.join(' '), report).toContain('lies on one side of 0');
        } else {
          expect(plan?.scale, report).toBe(output.presentation.scale);
        }
      } else {
        expect(output.kind, report).toBe('summary');
        expect(table.columns['root_mean_square_error'], report).toBeDefined();
        expect(table.columns['registration_status'], report).toBeDefined();
      }
    }
  });
}
