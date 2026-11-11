// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ModalSplitOperationProps} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation';
import type {GPUModalSplitMetric} from '../../../src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split';

/** Tripleg lengths 3,12,20,6,40; durations 3,2,0,2,2; two populated tracks. */
export function getModalFixture() {
  return {
    positions: new Float32Array([
      0, 0, 1, 0, 3, 0, 0, 0, 6, 0, 12, 0, 0, 0, 10, 0, 20, 0, 0, 0, 3, 0, 6, 0, 0, 0, 20, 0, 40, 0
    ]),
    timestamps: new Float32Array([0, 1, 3, 4, 5, 6, 7, 7, 7, 0, 1, 2, 3, 4, 5]),
    trackOffsets: new Uint32Array([0, 9, 15]),
    triplegs: {
      tracks: new Uint32Array([0, 0, 0, 1, 1, 99]),
      startRows: new Uint32Array([0, 3, 6, 9, 12, 99]),
      endRows: new Uint32Array([3, 6, 9, 12, 15, 99]),
      count: Uint32Array.of(5)
    },
    modes: new Uint32Array([9, 0xffffffff, 0, 1, 1, 99])
  };
}

export function getModalProps(
  metric: GPUModalSplitMetric,
  prediction: boolean
): ModalSplitOperationProps {
  return {
    id: 'modal-test',
    positions: 'positions',
    timestamps: 'timestamps',
    trackOffsets: 'trackOffsets',
    rowCount: 15,
    trackCount: 2,
    triplegCapacity: 6,
    modeCount: 3,
    metric,
    triplegs: {
      tracks: 'triplegTracks',
      startRows: 'triplegStartRows',
      endRows: 'triplegEndRows',
      status: {count: 'triplegCount'}
    },
    ...(prediction ? {modeSpeedBreaks: [2, 10, Infinity]} : {modes: 'modes'}),
    output: {modalSplit: 'modalSplit', ...(prediction ? {predictedModes: 'predictedModes'} : {})}
  };
}
