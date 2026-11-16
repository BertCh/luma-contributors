// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics,
  decodeGPUAnalysisContiguityStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {runExample as runSpacingExample} from './analysis-spacing.example';
import {runExample as runContiguityExample} from './analysis-contiguity-statistics.example';
import {WEIGHTS_OPERATORS} from '../../../gpu-spatial-analysis/computation-pack/operators/weights';
import {
  createSpatialRecordingGraph,
  uploadSpatialView,
  runSpatialScenario,
  type SpatialCaptureTag
} from './spatial-scenario-utils';

/** Records the actual contributors on small, licence-clean fixtures with checkable answers. */
export async function runAnalysisProfileExample(
  device: Device,
  element: 'GPUAnalysisSpacing' | 'GPUAnalysisContiguityStatistics'
) {
  const spacing = element === 'GPUAnalysisSpacing';
  const id = `gpu-spatial-analysis.${spacing ? 'analysis-spacing' : 'analysis-contiguity-statistics'}`;
  const graph = createSpatialRecordingGraph(device, id);
  const uploads: {destroy(): void}[] = [];
  const upload = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const owned = uploadSpatialView(graph, name, values, format);
    uploads.push(owned);
    return owned.view;
  };
  const integer = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  const scalar = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const views: Record<string, GraphDataView> = {};
  const tags: Record<string, SpatialCaptureTag> = {};
  try {
    if (spacing) {
      const positions = upload(
        'positions',
        new Float32Array([0, 0, 0, 0, 3, 0, 0, 4, 3, 4]),
        'float32x2'
      );
      const summary = integer('summary', GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
      const {nearestNeighborMaximum} = await runSpacingExample(device, {graph, positions, summary});
      Object.assign(views, {positions, summary, nearestNeighborMaximum});
      tags['positions'] = {role: 'input', keySpace: 'point'};
      tags['summary'] = {
        role: 'intermediate',
        keySpace: 'summary-word',
        meaning: 'Packed mixed uint32 and float32 summary; decode with the public spacing decoder'
      };
      tags['nearestNeighborMaximum'] = {role: 'output', keySpace: 'population'};
    } else {
      const positions = upload(
        'positions',
        new Float32Array([
          0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 10, 0, 11, 0, 11, 1, 10, 1
        ]),
        'float32x2'
      );
      const ringOffsets = upload('ringOffsets', new Uint32Array([0, 4, 8, 12]), 'uint32');
      const polygonOffsets = upload('polygonOffsets', new Uint32Array([0, 1, 2, 3]), 'uint32');
      const featureOffsets = upload('featureOffsets', new Uint32Array([0, 1, 2, 3]), 'uint32');
      const output = {
        weights: {
          offsets: integer('weightsRowOffsets', 4),
          neighbors: integer('weightsNeighbors', 12),
          weights: scalar('weightsValues', 12),
          status: {
            overflow: integer('weightsOverflow', 1),
            candidateOverflow: integer('candidateOverflow', 1),
            requiredCount: integer('requiredCount', 1)
          }
        },
        cardinality: integer('cardinality', 3),
        counts: integer('counts', 6),
        median: scalar('median', 1),
        validity: integer('validity', 1)
      };
      await runContiguityExample(device, {
        graph,
        polygons: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
        output
      });
      Object.assign(views, {
        positions,
        ringOffsets,
        polygonOffsets,
        featureOffsets,
        cardinality: output.cardinality,
        counts: output.counts,
        median: output.median,
        validity: output.validity,
        weightsOverflow: output.weights.status.overflow,
        candidateOverflow: output.weights.status.candidateOverflow
      });
      for (const name of Object.keys(views)) {
        const input = ['positions', 'ringOffsets', 'polygonOffsets', 'featureOffsets'].includes(
          name
        );
        tags[name] = {
          role: input ? 'input' : 'output',
          keySpace: name === 'cardinality' ? 'region' : name
        };
      }
    }
    const result = await runSpatialScenario({
      id,
      graph,
      encodeOptions: {parameters: undefined},
      views,
      entry: WEIGHTS_OPERATORS[element],
      choices: spacing ? {neighborRank: 3} : {criterion: 'queen'},
      datasetVersion: 'synthetic-analysis-profile/1',
      seed: 0,
      axes: {
        keySpaces: Object.entries(tags).map(([name, tag]) => ({
          id: tag.keySpace,
          label: tag.keySpace,
          size: views[name].length,
          indexSpace: '1d' as const
        })),
        axes: [
          {
            id: 'entity',
            label: spacing ? 'Point' : 'Square',
            kind: 'data',
            keySpace: spacing ? 'point' : 'region',
            size: spacing ? 5 : 3
          }
        ]
      },
      tags,
      metadata: spacing ? {coordinateModel: 'planar'} : {weightsStandardisation: 'binary'}
    });
    const values = (name: string) => {
      const capture = result.captures?.snapshots.find(candidate => candidate.name === name);
      if (!capture) throw new Error(`${element}: missing ${name} capture`);
      return capture.values;
    };
    if (spacing) {
      const statistics = decodeGPUAnalysisSpacingStatistics(Uint32Array.from(values('summary')));
      if (statistics?.pointCount !== 5 || statistics.nearestNeighbor?.maximum !== 3) {
        throw new Error(
          `${element}: expected 5 points and nearest maximum 3; got ${JSON.stringify(statistics)}`
        );
      }
    } else {
      const statistics = decodeGPUAnalysisContiguityStatistics({
        counts: values('counts'),
        median: values('median'),
        validity: values('validity'),
        overflow: values('weightsOverflow'),
        candidateOverflow: values('candidateOverflow')
      });
      if (
        !statistics.exact ||
        statistics.islands !== 1 ||
        statistics.median !== 1 ||
        Array.from(values('cardinality')).join(',') !== '1,1,0'
      ) {
        throw new Error(
          `${element}: expected exact cardinalities [1,1,0], 1 island and median 1; got ${JSON.stringify(statistics)}`
        );
      }
    }
    return result;
  } finally {
    for (const owned of uploads) owned.destroy();
  }
}
