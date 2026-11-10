// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

const BINDING_BYTES = 134217728;

/** Expected refusals of the staypoint, mobility and cleaning cases. */
export const MOVEMENT_SEMANTICS_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUTrajectoryStaypoints',
    name: 'great-circle words, one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-staypoints positions needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUTrajectoryLocations',
    name: 'one staypoint above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-locations centroids needs 134217736 bytes in one binding \(limit 134217728\); use fewer staypoints per graph$/
  },
  {
    exportName: 'GPUTrajectorySegments',
    name: 'triplegs and trips, one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-segments positions needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUMobilityIndicators',
    name: 'all indicators, one staypoint above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: mobility-indicators centroids needs 134217736 bytes in one binding \(limit 134217728\); use fewer staypoints per graph$/
  },
  {
    exportName: 'GPUModalSplit',
    name: 'count metric, one track above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: modal-split output\.modalSplit needs 134217744 bytes in one binding \(limit 134217728\); reduce modeCount or the track count$/
  },
  {
    exportName: 'GPUTrajectoryOutlierFlags',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-outlier-flags positions needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUTrajectoryHampel',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-hampel values needs 134217732 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUTrajectorySplit',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-split positions needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUTrajectoryThinning',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-thinning positions needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  },
  {
    exportName: 'GPUTrajectoryTrackRelativeTimes',
    name: 'one row above the binding limit',
    kind: 'binding',
    limit: BINDING_BYTES,
    rejects:
      /^Error: trajectory-track-relative-times timestamps needs 134217736 bytes in one binding \(limit 134217728\); split the tracks over several graphs$/
  }
];
