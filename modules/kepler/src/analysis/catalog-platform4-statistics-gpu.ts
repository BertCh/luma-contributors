// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  runEmpiricalBayesRates,
  runMultipleTestingAdjustment,
  type AdjustmentMethod
} from './catalog-platform3-b-statistics';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogGpuRunner} from './catalog-gpu-bindings';
import {
  GPUEmpiricalBayesRates,
  GPUMultipleTestingAdjustment,
  GPU_EMPIRICAL_BAYES_SUMMARY
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {addCatalogDiagnostic, type CatalogCoordinateSystem} from './catalog-geojson';
import {finishGrid, requireValueProperty} from './catalog-platform3-b-bindings';
import {
  computeKrigingFieldGpu,
  computeLineDensityGpu,
  makeGridRunOutput,
  makeGridTable,
  prepareKrigingGrid,
  resolveKrigingSettings,
  prepareLineDensity
} from './catalog-platform3-b-grids';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {readNumericProperty, type Inputs, type Values} from './catalog-platform3-b-common';

const getCoordinateSystem = (values: Values) =>
  values['coordinateSystem'] as CatalogCoordinateSystem;

async function runGpuLineDensity(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const cellSize = values['cellSize'] as number;
  const prepared = prepareLineDensity(inputs['lines'], getCoordinateSystem(values), cellSize);
  const result = await computeLineDensityGpu(context, prepared, cellSize);
  const table = makeGridTable(result.frame, result.grid, result.field, {
    tableId: 'line-density-cells',
    label: 'Line density cells',
    valueColumn: 'density',
    includeEmptyCells: values['includeEmptyCells'] as boolean,
    isEmpty: index => !(result.lengths[index] > 0)
  });
  return {
    ...makeGridRunOutput(table, result.diagnostics, result.grid, result.lengths.length),
    dispatchedContributors: ['GPULineDensity']
  };
}

async function runGpuKriging(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const prepared = prepareKrigingGrid(
    inputs['points'],
    requireValueProperty(values),
    getCoordinateSystem(values),
    values['cellSize'] as number
  );
  const settings = resolveKrigingSettings(prepared.sample, values);
  const field = await computeKrigingFieldGpu(context, prepared.sample, prepared.grid, settings);
  if (field.singularCount)
    addCatalogDiagnostic(
      prepared.diagnostics,
      'singular-kriging-system',
      'warning',
      'Cells with a singular or ill-conditioned kriging system are nodata.',
      field.singularCount
    );
  return finishGrid(
    prepared,
    field,
    'kriging-cells',
    'Kriged cells',
    'value',
    index => !Number.isFinite(field.values[index]),
    values['includeEmptyCells'] as boolean,
    ['GPUKriging']
  );
}

/**
 * GPUMultipleTestingAdjustment over the valid p-values. The contributor has no Sidak method, and
 * float32 p-values can flip a rejection that sits within rounding of alpha.
 */
async function runGpuMultipleTesting(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const method = values['method'] as AdjustmentMethod;
  // GPUMultipleTestingAdjustment implements none, bonferroni, holm, BH and BY.
  if (method === 'sidak') throw new CatalogGpuUnsupportedError('The Sidak method has no GPU path');
  const parameters = {
    pValueProperty: requireValueProperty(values, 'pValueProperty'),
    method,
    alpha: values['alpha'] as number
  };
  const pValues: number[] = [];
  for (const feature of inputs['features'].features) {
    const p = readNumericProperty(feature, parameters.pValueProperty);
    if (p !== null && p >= 0 && p <= 1) pValues.push(p);
  }
  // No valid p-values; the CPU reference throws the declared error.
  if (!pValues.length) throw new CatalogGpuUnsupportedError('No valid p-values');
  const adjusted = await withCatalogGpuSession(
    context.device,
    'catalog-multiple-testing',
    async session => {
      const output = session.output('adjusted', 'float32', pValues.length);
      session.graph.add(
        new GPUMultipleTestingAdjustment({
          id: 'catalog-multiple-testing',
          pValues: session.input('p-values', Float32Array.from(pValues), 'float32'),
          method,
          output: {adjustedPValues: output.view}
        })
      );
      session.run();
      return session.readFloat32(output);
    }
  );
  return {
    ...runMultipleTestingAdjustment(inputs['features'], parameters, adjusted),
    dispatchedContributors: ['GPUMultipleTestingAdjustment']
  };
}

/**
 * GPUEmpiricalBayesRates: the pooled moments, raw rates and clamped smoothed rates come from the
 * contributor in float32; the per-row shrinkage weight column is derived from its summary on the CPU.
 */
async function runGpuEmpiricalBayes(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const parameters = {
    numeratorProperty: requireValueProperty(values, 'numeratorProperty'),
    denominatorProperty: requireValueProperty(values, 'denominatorProperty')
  };
  runEmpiricalBayesRates(inputs['features'], parameters);
  const events: number[] = [];
  const populations: number[] = [];
  for (const feature of inputs['features'].features) {
    const eventCount = readNumericProperty(feature, parameters.numeratorProperty);
    const population = readNumericProperty(feature, parameters.denominatorProperty);
    if (eventCount === null || population === null || population <= 0 || eventCount < 0) continue;
    events.push(eventCount);
    populations.push(population);
  }
  // Fewer than two usable areas; the CPU reference throws the declared error.
  if (events.length < 2) throw new CatalogGpuUnsupportedError('Too few usable areas');
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-empirical-bayes',
    async session => {
      const raw = session.output('raw-rates', 'float32', events.length);
      const smoothed = session.output('smoothed-rates', 'float32', events.length);
      const summary = session.output('summary', 'float32', GPU_EMPIRICAL_BAYES_SUMMARY.length);
      session.graph.add(
        new GPUEmpiricalBayesRates({
          id: 'catalog-empirical-bayes',
          events: session.input('events', Float32Array.from(events), 'float32'),
          populations: session.input('populations', Float32Array.from(populations), 'float32'),
          output: {rawRates: raw.view, smoothedRates: smoothed.view, summary: summary.view}
        })
      );
      session.run();
      return {
        rawRates: await session.readFloat32(raw),
        smoothedRates: await session.readFloat32(smoothed),
        summary: await session.readFloat32(summary)
      };
    }
  );
  return {
    ...runEmpiricalBayesRates(inputs['features'], parameters, {
      globalRate: result.summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate],
      priorVariance: result.summary[GPU_EMPIRICAL_BAYES_SUMMARY.clampedPriorVariance],
      rawRates: result.rawRates,
      smoothedRates: result.smoothedRates
    }),
    dispatchedContributors: ['GPUEmpiricalBayesRates']
  };
}

/** PLATFORM4 GPU runners for grid, statistics and clustering bindings that were CPU-reference only, keyed by binding id. */
export const PLATFORM4_STATISTICS_GPU_RUNNERS: Readonly<Record<string, CatalogGpuRunner>> = {
  'line-density-grid': {contributorNames: ['GPULineDensity'], run: runGpuLineDensity},
  'kriging-grid': {contributorNames: ['GPUKriging'], run: runGpuKriging},
  'empirical-bayes-rates': {
    contributorNames: ['GPUEmpiricalBayesRates'],
    run: runGpuEmpiricalBayes
  },
  'multiple-testing-adjustment': {
    contributorNames: ['GPUMultipleTestingAdjustment'],
    run: runGpuMultipleTesting
  }
};
