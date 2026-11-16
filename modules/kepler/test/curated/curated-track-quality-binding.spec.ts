// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeArrowTableFromCatalogTable} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-table';
import {previewArrowSource} from '../../../../examples/deck/kepler-compute/src/source-preview';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {CURATED_TRACK_QUALITY_BINDING} from '../../src/curated/bindings/index';
import {createTrackQualitySample} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {CURATED_TRACK_QUALITY_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-track-quality';
import {profileKeplerNativeDataset} from '../../src/profile/native-dataset-profile';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

/** Independent 3-4-5 steps and a 60-second gap crossing the Unix epoch, with renamed ISO columns. */
function independentTracks(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [
      ...[
        [0, 0],
        [3, 4],
        [6, 8],
        [126, 8]
      ].map((coordinates, row) => ({
        geometry: {type: 'Point' as const, coordinates},
        properties: {
          vehicle: 'triangle',
          recorded_at: new Date([-2000, 0, 2000, 62000][row]).toISOString()
        }
      })),
      {
        geometry: {type: 'Point', coordinates: [10, 10]},
        properties: {vehicle: 'single', recorded_at: new Date(-1).toISOString()}
      }
    ]
  };
}

for (const independent of [false, true]) {
  it(`track quality dispatches its recipe on ${independent ? 'mapped independent ISO data' : 'the bundled recipe-review sample'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const collection = independent ? independentTracks() : createTrackQualitySample();
    const trackColumn = independent ? 'vehicle' : 'track_id';
    const timeColumn = independent ? 'recorded_at' : 'timestamp';
    const profile = profileKeplerNativeDataset(
      {
        rowCount: collection.features.length,
        geometryColumn: 'geometry',
        columns: [
          {
            name: 'geometry',
            type: 'string',
            values: collection.features.map(feature => JSON.stringify(feature.geometry))
          },
          ...[trackColumn, timeColumn].map(name => ({
            name,
            values: collection.features.map(feature => feature.properties![name])
          }))
        ]
      },
      {
        datasetId: independent ? 'other-data' : 'tracks',
        label: 'Declared track fixes',
        coordinateSystem: 'planar'
      },
      {
        kind: 'trajectories',
        trackColumn,
        timeColumn,
        timestampEncoding: independent ? 'iso8601' : 'unix-millisecond'
      }
    );
    const analysis = getKeplerCuratedAnalysis('track-quality')!;
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    for (const [role, column] of [
      ['track', trackColumn],
      ['time', timeColumn]
    ])
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'tracks', role, columns: [column]},
        analysis,
        [profile]
      );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {
        coordinateSystem: 'planar',
        maximumSpeed: independent ? 2.25 : 50,
        maximumGap: independent ? 30 : 60
      }
    });
    const resolved = resolveKeplerOperationParameters(
      CURATED_TRACK_QUALITY_BINDING,
      CURATED_TRACK_QUALITY_DESCRIPTOR,
      {tracks: profile},
      {overrides: request.parameters}
    );
    const expectedSpeeds = independent ? [0, 2.5, 2.5, 2, 0] : [0, 1, 501, 499, 1, 0, 1 / 120, 0];
    const expectedGaps = independent ? [null, 2, 2, 60, null] : [null, 1, 1, 1, 1, null, 120, null];
    const expectedSourceRows = independent ? [0, 1, 2, 3, 4] : [2, 7, 4, 0, 6, 5, 1, 3];
    const mismatches: string[] = [];
    for (const speedOutlierMode of ['step', 'point-centred'] as const) {
      const result = await runCatalogOperationAsync(
        CURATED_TRACK_QUALITY_BINDING,
        {...resolved.parameters, speedOutlierMode},
        {tracks: collection},
        {device}
      );
      const fixes = result.tables.find(table => table.id === 'trajectory-fixes')!;
      const tracks = result.tables.find(table => table.id === 'track-quality-summary')!;
      const expectedFlags = independent
        ? speedOutlierMode === 'step'
          ? [0, 4, 4, 8, 0]
          : [0, 4, 0, 8, 0]
        : speedOutlierMode === 'step'
          ? [0, 0, 4, 4, 0, 0, 8, 0]
          : [0, 0, 4, 0, 0, 0, 8, 0];
      const close = (name: string, values: readonly unknown[], expected: readonly number[]) => {
        for (let row = 0; row < expected.length; row++)
          if (
            typeof values[row] !== 'number' ||
            !Number.isFinite(values[row]) ||
            Math.abs(Number(values[row]) - expected[row]) > 1e-4 + Math.abs(expected[row]) * 1e-5
          )
            mismatches.push(
              `${speedOutlierMode} ${name}[${row}]: ${values[row]} != ${expected[row]}`
            );
      };
      close('step_speed', fixes.columns['step_speed'], expectedSpeeds);
      close('track_length', tracks.columns['track_length'], independent ? [130, 0] : [1002, 1, 0]);
      close(
        'duration_seconds',
        tracks.columns['duration_seconds'],
        independent ? [64, 0] : [4, 120, 0]
      );
      close(
        'maximum_speed',
        tracks.columns['maximum_speed'],
        independent ? [2.5, 0] : [501, 1 / 120, 0]
      );
      for (const [name, actual, expected] of [
        ['flags', fixes.columns['quality_flags'], expectedFlags],
        ['gaps', fixes.columns['time_gap_seconds'], expectedGaps],
        ['identity', fixes.columns['source_index'], expectedSourceRows],
        [
          'flagged counts',
          tracks.columns['flagged_count'],
          independent
            ? [speedOutlierMode === 'step' ? 3 : 2, 0]
            : [speedOutlierMode === 'step' ? 2 : 1, 1, 0]
        ]
      ] as const)
        if (JSON.stringify(actual) !== JSON.stringify(expected))
          mismatches.push(
            `${speedOutlierMode} ${name}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`
          );
      for (let row = 0; row < fixes.rowCount; row++) {
        const source = collection.features[expectedSourceRows[row]];
        if (
          JSON.stringify(JSON.parse(String(fixes.columns['geometry'][row]))) !==
          JSON.stringify(source.geometry)
        )
          mismatches.push(`publication geometry row ${row}`);
        if (
          fixes.columns['timestamp'][row] !==
          (independent ? [-2000, 0, 2000, 62000, -1][row] : source.properties![timeColumn])
        )
          mismatches.push(`epoch row ${row}`);
        if (fixes.columns['flagged'][row] !== (expectedFlags[row] !== 0))
          mismatches.push(`flagged row ${row}`);
        if (fixes.columns['excessive_speed'][row] !== Boolean(expectedFlags[row] & 4))
          mismatches.push(`speed bit row ${row}`);
        if (fixes.columns['time_gap'][row] !== Boolean(expectedFlags[row] & 8))
          mismatches.push(`gap bit row ${row}`);
      }
      if (
        result.engine !== 'gpu' ||
        JSON.stringify(result.dispatchedContributors) !== JSON.stringify(['GPUTrajectoryMetrics'])
      )
        mismatches.push(`dispatch ${JSON.stringify(result.dispatchedContributors)}`);
      if (
        result.summary['unknownCount'] !== 0 ||
        result.summary['flaggedCount'] !== expectedFlags.filter(flags => flags !== 0).length
      )
        mismatches.push(`status ${JSON.stringify(result.summary)}`);
      if (result.diagnostics.some(diagnostic => diagnostic.severity === 'error'))
        mismatches.push(`diagnostics ${JSON.stringify(result.diagnostics)}`);
      expect(fixes.rowCount).toBe(expectedFlags.length);
      const arrowTable = makeArrowTableFromCatalogTable(fixes);
      const publication = previewArrowSource(
        arrowTable,
        {
          columns: [
            'source_index',
            'vertex_index',
            'fix_index',
            'track_id',
            'quality_flags',
            'flagged'
          ]
        },
        ['geometry']
      );
      if (!publication.canUpload)
        mismatches.push(`native Arrow publication: ${publication.reason}`);
      for (let row = 0; row < fixes.rowCount; row++)
        if (arrowTable.getChild('quality_flags')!.get(row) !== expectedFlags[row])
          mismatches.push(`Arrow flag row ${row}`);
      expect(
        planKeplerResultPresentation(fixes, CURATED_TRACK_QUALITY_DESCRIPTOR.outputs[0])
      ).toMatchObject({column: 'flagged', scale: 'categorical', rowCount: fixes.rowCount});
    }
    if (mismatches.length)
      throw new Error(
        `Track quality ${independent ? 'independent ISO dataset' : 'recipe-review sample'} parity:\n${mismatches.join('\n')}`
      );
  }, 120000);
}
