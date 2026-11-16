// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createCuratedWorkflowAnalysisTemplate,
  type WorkflowAnalysisTemplate
} from '../../workflows/workflow-analysis-template';
import {CURATED_DRIVE_TIME_BANDS_ANALYSIS} from './drive-time-bands-data';

export * from './drive-time-bands-data';

/** Shared AX-30 template; sample-specific values are supplied by the sample registry at launch. */
export function createDriveTimeBandsTemplate(): WorkflowAnalysisTemplate {
  return createCuratedWorkflowAnalysisTemplate(CURATED_DRIVE_TIME_BANDS_ANALYSIS);
}
