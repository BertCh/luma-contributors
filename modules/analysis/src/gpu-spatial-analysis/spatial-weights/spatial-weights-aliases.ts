// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Logical preservation checks are grouped by physical buffer; separately allocated chunks compare once. @internal */
export function validateWeightsLogicalAliases(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>,
  inputs: Readonly<Record<string, GraphDataView | undefined>> = {},
  allowExactAlias?: (output: string, input: string) => boolean
): void {
  type Range = {
    name: string;
    view: GraphDataView;
    start: number;
    end: number;
    writable: boolean;
    handle: GraphDataView['buffer'];
  };
  const identities = new Map<GraphDataView['buffer'], object>();
  const groups = new Map<object, Range[]>();
  for (const [fields, writable] of [
    [outputs, true],
    [inputs, false]
  ] as const)
    for (const [name, view] of Object.entries(fields))
      if (view) {
        const handle = view.buffer;
        let identity = identities.get(handle);
        if (!identity) {
          identity = handle.defaultBuffer ? getCoreDefaultBuffer(handle.defaultBuffer) : handle;
          identities.set(handle, identity);
        }
        const range = {
          name,
          view,
          start: view.byteOffset,
          end: view.byteOffset + view.length * view.rowByteLength,
          writable,
          handle
        };
        const group = groups.get(identity);
        if (group) group.push(range);
        else groups.set(identity, [range]);
      }
  for (const ranges of groups.values()) {
    const writable = ranges
      .filter(range => range.writable)
      .sort((first, second) => first.start - second.start);
    if (!writable.length) continue;
    validateRequirement(
      id,
      new Set(ranges.map(range => range.handle)).size === 1,
      `${writable[0].name} and other ports share a physical buffer; import it once and use disjoint offset views`
    );
    const requireDisjoint = (first: Range, second: Range) =>
      validateRequirement(
        id,
        first.end <= second.start || second.end <= first.start,
        `${first.name} and ${second.name} logical byte ranges must not overlap; use disjoint views`
      );
    for (let index = 1; index < writable.length; index++)
      requireDisjoint(writable[index - 1], writable[index]);
    let firstWritable = 0;
    for (const input of ranges
      .filter(range => !range.writable)
      .sort((first, second) => first.start - second.start)) {
      while (firstWritable < writable.length && writable[firstWritable].end <= input.start)
        firstWritable++;
      for (
        let index = firstWritable;
        index < writable.length && writable[index].start < input.end;
        index++
      ) {
        const output = writable[index];
        if (
          allowExactAlias?.(output.name, input.name) &&
          output.start === input.start &&
          output.end === input.end
        )
          continue;
        requireDisjoint(output, input);
      }
    }
  }
}
