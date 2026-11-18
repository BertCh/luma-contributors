// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Packed regression outputs may share an aligned prefix only across separate dispatches.
 * Logical ranges must always be disjoint. Each group includes every caller output bound in that
 * dispatch, including output fields subsequently read by another output pass.
 */
export function validateRegressionOutputDispatches(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>,
  dispatches: readonly (readonly string[] | {read: readonly string[]; write: readonly string[]})[]
): void {
  const fields = Object.entries(outputs).filter(
    (entry): entry is [string, GraphDataView] => entry[1] !== undefined && entry[1].length > 0
  );
  for (let firstIndex = 0; firstIndex < fields.length; firstIndex++) {
    const [firstName, first] = fields[firstIndex];
    const firstBuffer = first.buffer.defaultBuffer
      ? getCoreDefaultBuffer(first.buffer.defaultBuffer)
      : first.buffer;
    for (const [secondName, second] of fields.slice(firstIndex + 1)) {
      const secondBuffer = second.buffer.defaultBuffer
        ? getCoreDefaultBuffer(second.buffer.defaultBuffer)
        : second.buffer;
      if (firstBuffer !== secondBuffer) continue;
      const firstEnd =
        first.byteOffset + (first.length - 1) * first.byteStride + first.rowByteLength;
      const secondEnd =
        second.byteOffset + (second.length - 1) * second.byteStride + second.rowByteLength;
      validateRequirement(
        id,
        firstEnd <= second.byteOffset || secondEnd <= first.byteOffset,
        `${firstName} and ${secondName} logical writable byte ranges must not overlap`
      );
    }
  }
  for (const dispatch of dispatches) {
    const names = 'write' in dispatch ? dispatch.write : dispatch;
    validateDisjointWritableViews(id, Object.fromEntries(names.map(name => [name, outputs[name]])));
    if ('write' in dispatch) {
      for (const readName of dispatch.read) {
        const read = outputs[readName];
        if (!read) continue;
        const readBuffer = read.buffer.defaultBuffer
          ? getCoreDefaultBuffer(read.buffer.defaultBuffer)
          : read.buffer;
        for (const writeName of dispatch.write) {
          const write = outputs[writeName];
          if (!write) continue;
          const writeBuffer = write.buffer.defaultBuffer
            ? getCoreDefaultBuffer(write.buffer.defaultBuffer)
            : write.buffer;
          validateRequirement(
            id,
            readBuffer !== writeBuffer,
            `${readName} and ${writeName} must not bind one physical buffer read-only and writable in the same dispatch`
          );
        }
      }
    }
  }
}
