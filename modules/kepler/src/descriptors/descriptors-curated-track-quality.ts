// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CURATED_TRACK_QUALITY_BINDING} from '../curated/bindings/curated-track-quality-binding';
import {TRAJECTORY_OPERATION_DESCRIPTORS} from './descriptors-trajectory';
import type {KeplerOperationDescriptor} from './operation-descriptor-types';

// Reuse established coordinate requirements, time encoding and field roles.
const trajectory = TRAJECTORY_OPERATION_DESCRIPTORS.find(
  descriptor => descriptor.bindingId === 'trajectory-outlier-flags'
)!;
export const CURATED_TRACK_QUALITY_DESCRIPTOR: KeplerOperationDescriptor = {
  ...trajectory,
  bindingId: CURATED_TRACK_QUALITY_BINDING.id,
  action: 'Check observation quality',
  inputs: trajectory.inputs.map(input => ({
    ...input,
    fields: input.fields.map(field => ({
      ...field,
      required: field.role === 'track' || field.role === 'time'
    }))
  })),
  parameters: [
    ...trajectory.parameters.filter(parameter =>
      ['coordinateSystem', 'timestampEncoding'].includes(parameter.name)
    ),
    ...['maximumSpeed', 'maximumGap', 'speedOutlierMode'].map(name => ({
      name,
      apply: 'rebuild' as const,
      defaultPolicy: {from: 'constant' as const},
      level: 'basic' as const
    }))
  ],
  outputs: [
    {
      tableId: 'trajectory-fixes',
      kind: 'points',
      presentation: {column: 'flagged', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['how-it-moves']
    },
    {
      tableId: 'track-quality-summary',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: [
    ...(trajectory.notes ?? []).slice(0, 4),
    'Track and time columns are explicit curated roles. Direct timed-line execution also accepts embedded times and feature IDs. Preparation rejects invalid observations and duplicate times before GPU execution.',
    'Speed and gaps are compared strictly above the limits. Zero is a literal threshold. Step mode can flag both an outlier and the observation after it; point-centred mode applies the recipe adjacent-step rule.',
    'The map colors flagged observations, with unknown quality excluded. Inspect quality_flags for the independent speed and gap bits, step_speed in length/second and time_gap_seconds. First fixes have no elapsed step.',
    'Summary metrics describe all raw observations, including flagged ones. Flags are plausibility checks, not statistical significance. Turning angles and acceleration are not computed.'
  ]
};
