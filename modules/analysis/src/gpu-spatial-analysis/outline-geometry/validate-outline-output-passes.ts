// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

/** Checks the small fixed set of output fields, independently of the number of geometry rows. */
export function validateOutlineOutputPasses(
  id: string,
  outputs: Record<string, GraphDataView | undefined>,
  passes: readonly (readonly string[])[]
): void {
  const fields = Object.entries(outputs).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1]?.length)
  );
  for (let first = 0; first < fields.length; first++) {
    const [firstName, firstView] = fields[first];
    const firstBuffer = firstView.buffer.defaultBuffer
      ? getCoreDefaultBuffer(firstView.buffer.defaultBuffer)
      : firstView.buffer;
    for (let second = first + 1; second < fields.length; second++) {
      const [secondName, secondView] = fields[second];
      const secondBuffer = secondView.buffer.defaultBuffer
        ? getCoreDefaultBuffer(secondView.buffer.defaultBuffer)
        : secondView.buffer;
      validateRequirement(
        id,
        firstBuffer !== secondBuffer ||
          firstView.byteOffset + firstView.length * firstView.rowByteLength <=
            secondView.byteOffset ||
          secondView.byteOffset + secondView.length * secondView.rowByteLength <=
            firstView.byteOffset,
        `${firstName} and ${secondName} logical byte ranges must not overlap`
      );
    }
  }
  for (const pass of passes)
    validateDisjointWritableViews(id, Object.fromEntries(pass.map(name => [name, outputs[name]])));
}
