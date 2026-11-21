// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ColumnQuantilesOperationProps} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';

/** Count reads of inherited or non-enumerable declarations without weakening their types. */
export function createQuantilesSnapshotFixture(inherited: boolean) {
  const names = {
    values: 'values',
    parameters: 'parameters',
    quantiles: 'quantiles',
    validCount: 'validCount',
    filterBounds: 'initial-bounds'
  };
  let rowCount = 2;
  const reads: Record<string, number> = {};
  function readName(name: keyof typeof names): string {
    reads[name] = (reads[name] ?? 0) + 1;
    return names[name];
  }
  const output: ColumnQuantilesOperationProps['output'] = {
    get quantiles() {
      return readName('quantiles');
    },
    get validCount() {
      return readName('validCount');
    },
    get filterBounds() {
      return readName('filterBounds');
    }
  };
  const props: ColumnQuantilesOperationProps = {
    get values() {
      return readName('values');
    },
    get parameters() {
      return readName('parameters');
    },
    get rowCount() {
      reads['rowCount'] = (reads['rowCount'] ?? 0) + 1;
      return rowCount;
    },
    quantileCount: 1,
    output
  };
  for (const object of [output, props]) {
    const descriptors = Object.getOwnPropertyDescriptors(object);
    for (const descriptor of Object.values(descriptors)) descriptor.enumerable = false;
    if (inherited) {
      Object.setPrototypeOf(object, Object.defineProperties({}, descriptors));
      for (const name of Object.keys(descriptors)) Reflect.deleteProperty(object, name);
    } else {
      Object.defineProperties(object, descriptors);
    }
  }
  return {
    props,
    reads,
    mutate() {
      for (const name of Object.keys(names) as (keyof typeof names)[])
        names[name] = `later-${name}`;
      rowCount = 99;
    }
  };
}
