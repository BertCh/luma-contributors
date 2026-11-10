// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Which block answers which need, a real operator that uses it, and how often it appears in the
// records and the source. Counts come from generate-data.ts.

import {Link} from '@tanstack/react-router';
import {getEntry} from '../../../../catalog';
import {formatCount} from '../../../../lens/format';
import {RECORDED} from './toy-data';

type BlockRow = {
  need: string;
  block: string;
  /** Key into the generated counts. */
  key: 'reduction' | 'scan' | 'compaction' | 'sort' | 'gridIndex' | 'loadBalanced';
  /** Catalog id of an operator that uses the block (verified in its source). */
  example: string;
};

const ROWS: readonly BlockRow[] = [
  // gpu-variable-standardization.ts constructs GPUReduction.
  {
    need: 'A total, minimum or maximum',
    block: 'GPUReduction',
    key: 'reduction',
    example: 'variable-standardization'
  },
  // gpu-isolines.ts:343 scans per-cell counts into offsets.
  {need: 'Where each row writes', block: 'GPUScan', key: 'scan', example: 'isolines'},
  {
    need: 'Keep some rows, packed',
    block: 'GPUCompaction',
    key: 'compaction',
    example: 'time-window-filter'
  },
  // cell-table.ts:217 sorts keys with GPUSort.
  {need: 'Group or order rows', block: 'GPUSort', key: 'sort', example: 'cell-aggregation'},
  {need: 'Nearby candidates', block: 'GPUGridIndex', key: 'gridIndex', example: 'kernel-density'},
  // gpu-spatial-lag.ts imports load-balanced-for-each.
  {
    need: 'Rows of uneven length',
    block: 'merge-path loop',
    key: 'loadBalanced',
    example: 'spatial-lag'
  }
];

export function BlocksTable() {
  const {records, sourceFiles, recordCount} = RECORDED.blocks;
  return (
    <div className="meta-scroll">
      <table className="building-blocks-table">
        <thead>
          <tr>
            <th scope="col">Need</th>
            <th scope="col">Block</th>
            <th scope="col">For example</th>
            <th scope="col" className="building-blocks-table__number">
              Records
            </th>
            <th scope="col" className="building-blocks-table__number">
              Source files
            </th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(row => {
            const recorded = row.key === 'loadBalanced' ? undefined : records[row.key];
            return (
              <tr key={row.key}>
                <td>{row.need}</td>
                <td>
                  <code>{row.block}</code>
                </td>
                <td>
                  <Link to="/fn/$id" params={{id: row.example}}>
                    {getEntry(row.example)?.title ?? row.example}
                  </Link>
                </td>
                <td className="building-blocks-table__number">
                  {recorded === undefined
                    ? '–'
                    : `${formatCount(recorded)} of ${formatCount(recordCount)}`}
                </td>
                <td className="building-blocks-table__number">
                  {formatCount(sourceFiles[row.key])}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
