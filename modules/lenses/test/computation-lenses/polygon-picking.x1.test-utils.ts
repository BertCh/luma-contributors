// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationRecordWithSidecars} from '@luma-contributors/analysis/computation-record';
import {tagSpatialCaptures} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';

/** Replay the live geometry producer over archived numeric evidence. */
export function createLocalMoranPickingRecord(record: GPUComputationRecordWithSidecars) {
  const snapshots = record.captures!.snapshots.filter(
    capture => !capture.name.endsWith('.geometry')
  );
  return {
    ...record,
    captures: {
      ...record.captures!,
      snapshots: tagSpatialCaptures(snapshots, {
        vertices: {role: 'intermediate', keySpace: 'vertex'},
        ringOffsets: {role: 'intermediate', keySpace: 'boundary'},
        values: {
          role: 'input',
          keySpace: 'tract',
          polygons: {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'}
        },
        localI: {
          role: 'output',
          keySpace: 'tract',
          polygons: {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'}
        }
      })
    }
  };
}
