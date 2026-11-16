// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {CURATED_EMERGING_HOT_SPOTS_BINDING} from '../../src/curated/bindings/index';
import {createEmergingHotSpotsSample} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-emerging-hot-spots';
import {describeKeplerCatalogResult} from '../../src/presentation/result-report';
import {hasKeplerResultInference} from '../../src/presentation/result-inference';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {
  computeSpaceTimeGiStar,
  createSeededRandom
} from '../../../analysis/test/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spots-oracle';
import {checkEmergingHotSpotsProbabilitiesAndTrends} from './curated-emerging-hot-spots-oracle';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';

/** Independent recipe-fixture histogram; consume the same random draws as its jitter and dates. */
function recipeCounts(seed: number) {
  const random = createSeededRandom(seed);
  const counts = new Uint32Array(30 * 12);
  for (let cell = 0; cell < 30; cell++) {
    for (let month = 0; month < 12; month++) {
      const column = cell % 6,
        row = Math.floor(cell / 6);
      let count = 8 + Math.floor(random() * 5);
      if (column >= 1 && column <= 3 && row >= 1 && row <= 2) count += 6 * month;
      else if (column >= 4 && row >= 3) count = Math.floor(random() * 2);
      counts[cell * 12 + month] = count;
      for (let event = 0; event < count; event++) {
        random();
        random();
        random();
      }
    }
  }
  return counts;
}

for (const seedOffset of [0, 101]) {
  it(`emerging hot spots dispatches the recipe and matches its fixture oracle (seed offset ${seedOffset})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points = createEmergingHotSpotsSample(seedOffset);
    const analysis = getKeplerCuratedAnalysis('emerging-hot-spots')!;
    if (seedOffset)
      for (const feature of points.features)
        feature.properties = {
          recorded_at: new Date(Number(feature.properties!['observed'])).toISOString()
        };
    const profile = profileKeplerFeatureCollection(points, {
      datasetId: 'points',
      label: 'Events',
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'events'}
    });
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'points', role: 'time', columns: [seedOffset ? 'recorded_at' : 'observed']},
      analysis,
      [profile]
    );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {coordinateSystem: 'planar', cellSize: 1}
    });
    const result = await runCatalogOperationAsync(
      CURATED_EMERGING_HOT_SPOTS_BINDING,
      request.parameters,
      {points},
      {device}
    );
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual([
      'GPUCalendarBuckets',
      'GPUPeriodIndex',
      'GPUEmergingHotSpots'
    ]);
    const report = describeKeplerCatalogResult(
      result,
      CURATED_EMERGING_HOT_SPOTS_BINDING,
      CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR
    );
    if (!hasKeplerResultInference(report) || !report.outputs.every(output => output.inference)) {
      throw new Error(`emerging hot spots lost inference semantics: ${JSON.stringify(report)}`);
    }
    const bins = result.tables.find(table => table.id === 'cube')!;
    const cells = result.tables.find(table => table.id === 'cells')!;
    const counts = recipeCounts(31 + seedOffset);
    const shape = {gridWidth: 6, gridHeight: 5, sliceCount: 12, values: counts};
    const packed = getGPUEmergingHotSpotsParameterValues({radius: 1.5, temporalWindow: 1});
    const expectedScores = computeSpaceTimeGiStar(shape, packed, 4);
    const {scores, oracle, mismatches} = checkEmergingHotSpotsProbabilitiesAndTrends(
      bins,
      cells,
      shape,
      packed
    );
    for (let bin = 0; bin < counts.length; bin++) {
      if (bins.columns['event_count'][bin] !== counts[bin])
        mismatches.push(`count ${bin}: ${bins.columns['event_count'][bin]} != ${counts[bin]}`);
      if (
        !Number.isFinite(scores[bin]) ||
        Math.abs(scores[bin] - expectedScores[bin]) > 2e-3 + Math.abs(expectedScores[bin]) * 1e-3
      )
        mismatches.push(`Gi* ${bin}: ${scores[bin]} != ${expectedScores[bin]}`);
      if (
        bins.columns['cell_index'][bin] !== Math.floor(bin / 12) ||
        bins.columns['slice_index'][bin] !== bin % 12
      )
        mismatches.push(`bin identity ${bin}`);
    }
    let compared = 0;
    const names = Object.fromEntries(
      Object.entries(GPU_EMERGING_HOT_SPOTS_CATEGORIES).map(([name, code]) => [code, name])
    );
    for (let cell = 0; cell < 30; cell++) {
      // Match the existing recipe fixture's exclusion around the trend-significance boundary.
      if (Math.abs(oracle.trendP[cell] - packed[3]) > 0.02 * packed[3]) {
        compared++;
        if (cells.columns['category'][cell] !== oracle.category[cell])
          mismatches.push(
            `category ${cell}: ${cells.columns['category'][cell]} != ${oracle.category[cell]}`
          );
      }
      for (const [column, expected] of [
        ['hot_slice_count', oracle.hotSliceCount[cell]],
        ['cold_slice_count', oracle.coldSliceCount[cell]],
        ['trend_s', oracle.trendS[cell]]
      ] as const)
        if (cells.columns[column][cell] !== expected)
          mismatches.push(`${column} ${cell}: ${cells.columns[column][cell]} != ${expected}`);
      if (
        cells.columns['cell_index'][cell] !== cell ||
        cells.columns['category_name'][cell] !== names[Number(cells.columns['category'][cell])]
      )
        mismatches.push(`cell identity/name ${cell}`);
      const geometry = JSON.parse(String(cells.columns['geometry'][cell]));
      if (geometry.type !== 'Polygon' || geometry.coordinates[0].length !== 5)
        mismatches.push(`publication polygon ${cell}`);
    }
    if (compared <= 15) mismatches.push(`only ${compared} classifications compared`);
    if (
      result.summary['countedEventCount'] !== points.features.length ||
      result.summary['skippedEventCount'] !== 0
    )
      mismatches.push(`status ${JSON.stringify(result.summary)}`);
    expect(cells.rowCount).toBe(30);
    expect(bins.rowCount).toBe(360);
    expect(result.summary['multipleTesting']).toBe('benjamini-hochberg');
    if (seedOffset === 0) {
      expect(
        cells.columns['category'].filter(code => Number(code) >= 1 && Number(code) <= 8).length
      ).toBeGreaterThanOrEqual(3);
      expect(cells.columns['category'][24]).toBe(GPU_EMERGING_HOT_SPOTS_CATEGORIES.NO_PATTERN);
    }
    if (mismatches.length)
      throw new Error(
        `Emerging hot spots recipe parity seed ${seedOffset}:\n${mismatches.join('\n')}`
      );
    // Explicitly narrowed calendar range preserves and reports excluded events.
    const narrow = await runCatalogOperationAsync(
      CURATED_EMERGING_HOT_SPOTS_BINDING,
      {...request.parameters, sliceCount: 6},
      {points},
      {device}
    );
    const expectedSkipped = counts.reduce(
      (total, count, index) => total + (index % 12 >= 6 ? count : 0),
      0
    );
    expect(narrow.summary['skippedEventCount']).toBe(expectedSkipped);
    expect(narrow.summary['countedEventCount']).toBe(points.features.length - expectedSkipped);
    const narrowCells = narrow.tables.find(table => table.id === 'cells')!;
    expect(narrowCells.columns['trend_z']).toEqual(Array(30).fill(null));
    expect(narrowCells.columns['trend_p']).toEqual(Array(30).fill(null));
    expect(narrowCells.columns['last_valid_slice']).toEqual(Array(30).fill(5));
    expect(narrow.diagnostics.find(diagnostic => diagnostic.code === 'slice-range')?.count).toBe(
      expectedSkipped
    );
  }, 120000);
}
