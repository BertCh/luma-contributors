// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {
  GPURegionSelection,
  GPURegionMaskOutput,
  GPURegionStatisticsOutput
} from './region-statistics-types';

/** Capture a selection's declared fields without cloning per-frame views. @internal */
export function snapshotRegionSelection<Region extends GPURegionSelection>(region: Region): Region {
  const selection: GPURegionSelection = region;
  let result: GPURegionSelection;
  switch (selection.kind) {
    case 'rectangle':
      result = {
        ...snapshotConstructionFields(selection, ['bounds', 'screenTransform']),
        kind: 'rectangle'
      };
      break;
    case 'polygon':
      result = {
        ...snapshotConstructionFields(selection, ['vertices', 'vertexCount', 'screenTransform']),
        kind: 'polygon'
      };
      break;
    case 'radius':
      result = {...snapshotConstructionFields(selection, ['circle']), kind: 'radius'};
      break;
    case 'mask': {
      const captured = snapshotConstructionFields(selection, ['mask', 'summary', 'status']);
      result = {
        ...captured,
        kind: 'mask',
        ...(captured.status
          ? {status: snapshotConstructionFields(captured.status, ['overflow'])}
          : {})
      };
      break;
    }
    case 'pick-region':
      result = {
        ...snapshotConstructionFields(selection, ['picks', 'batchIndex']),
        kind: 'pick-region'
      };
      break;
  }
  return Object.freeze(result) as Region;
}

/** Capture mask/status port identities, keeping GPU contents live. @internal */
export function snapshotRegionMaskOutput(output: GPURegionMaskOutput): GPURegionMaskOutput {
  const result = snapshotConstructionFields(output, ['mask', 'status']);
  return Object.freeze({
    ...result,
    status: snapshotConstructionFields(result.status, [
      'overflow',
      'antimeridianUncertainCount',
      'poleUncertainCount'
    ])
  });
}

/** Capture summary, compact-output and status maps. @internal */
export function snapshotRegionStatisticsOutput(
  output: GPURegionStatisticsOutput
): GPURegionStatisticsOutput {
  const result = snapshotConstructionFields(output, [
    'summary',
    'mask',
    'selectedRows',
    'drawInstanceCount',
    'status'
  ]);
  const selected =
    result.selectedRows && snapshotConstructionFields(result.selectedRows, ['ids', 'status']);
  return Object.freeze({
    ...result,
    ...(result.status
      ? {
          status: snapshotConstructionFields(result.status, [
            'overflow',
            'candidateOverflow',
            'antimeridianUncertainCount',
            'poleUncertainCount'
          ])
        }
      : {}),
    ...(selected
      ? {
          selectedRows: Object.freeze({
            ...selected,
            status: snapshotConstructionFields(selected.status, [
              'count',
              'requiredCount',
              'overflow',
              'candidateOverflow'
            ])
          })
        }
      : {})
  });
}
