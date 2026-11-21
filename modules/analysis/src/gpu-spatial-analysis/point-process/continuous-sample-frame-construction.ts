// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

type ContinuousSampleFrame = {positions?: unknown; legacySites?: unknown} & (
  | {type: 'line'}
  | {type: 'area'; bounds: readonly [number, number, number, number]}
);

/** @internal Capture a variant once, copying bounds while retaining resource identities. */
export function snapshotContinuousSampleFrame<Frame extends ContinuousSampleFrame>(
  id: string,
  callerFrame: Frame
): Frame {
  const frame: ContinuousSampleFrame = callerFrame;
  const frameType = frame.type;
  validateRequirement(
    id,
    frameType !== 'line' || !('bounds' in frame),
    'frame.bounds is only supported for area frames; omit bounds for a line frame'
  );
  const fields = snapshotConstructionFields<
    Pick<ContinuousSampleFrame, 'positions' | 'legacySites'>
  >(frame, ['positions', 'legacySites']);
  const captured =
    frameType === 'area'
      ? {
          ...fields,
          type: frameType,
          bounds: Object.freeze([...frame.bounds]) as readonly [number, number, number, number]
        }
      : {...fields, type: frameType};
  return Object.freeze(captured) as Frame;
}
