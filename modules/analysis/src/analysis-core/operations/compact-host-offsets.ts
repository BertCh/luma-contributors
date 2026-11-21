// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../kernels/gpu-contributor-utils';

/** Compact arrays cover every child exactly; resident storage retains GPU capacity clamping. @internal */
export type CPUHostRowsOptions = {storage?: 'compact' | 'resident-capacity'};

/** Checks a parent-to-child offsets level; a single zero denotes zero parents. @internal */
export function validateCPUHostOffsets(
  id: string,
  name: string,
  offsets: Uint32Array,
  childCount: number,
  options: CPUHostRowsOptions = {}
): void {
  validateRequirement(
    id,
    options.storage === undefined || ['compact', 'resident-capacity'].includes(options.storage),
    `storage must be compact or resident-capacity; received ${options.storage}`
  );
  validateRequirement(
    id,
    offsets.length >= 1,
    `${name} length ${offsets.length} must include a terminal offset`
  );
  const resident = options.storage === 'resident-capacity';
  if (!resident) {
    const initialOffset = offsets[0];
    validateRequirement(
      id,
      initialOffset === 0,
      `${name} initial offset must be 0; received ${initialOffset}`
    );
  }
  let previous = 0;
  for (let index = 0; index < offsets.length; index++) {
    const sourceOffset = offsets[index];
    const offset = resident ? Math.min(sourceOffset, childCount) : sourceOffset;
    validateRequirement(
      id,
      offset >= previous && offset <= childCount,
      `${name}[${index}] must be nondecreasing and <= child count ${childCount}; received ${sourceOffset}`
    );
    previous = offset;
  }
  if (!resident) {
    validateRequirement(
      id,
      previous === childCount,
      `${name} terminal offset must equal child count ${childCount}; received ${previous}`
    );
  }
}
