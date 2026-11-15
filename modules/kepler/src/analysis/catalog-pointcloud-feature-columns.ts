// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE,
  GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED
} from '@luma-contributors/analysis/pointcloud-analysis';

type PointCloudFeatureKind = 'normal' | 'shape' | 'eigenvalue' | 'covariance';

/** Apply each contributor port's validity contract at publication, preserving finite moments. */
export function readPointCloudFeatureValue(
  value: number,
  kind: PointCloudFeatureKind,
  status: number,
  retainedSupport: number
): number | null {
  // Below three retained neighbors the contributor returns before computing any moments.
  if (retainedSupport < 3 || !Number.isFinite(value)) return null;
  // Covariance is written before the eigensolve; its validity does not depend on rank/convergence.
  if (kind === 'covariance') return value;
  if (status & GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED) return null;
  if (kind !== 'eigenvalue' && status & GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE) return null;
  return value;
}
