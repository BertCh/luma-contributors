// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {formatCount} from '../../../../lens/format';
import {
  getAutomaticWorkgroupSize,
  getDispatchShape,
  WEBGPU_DEFAULT_WORKGROUPS_PER_DIMENSION
} from '../../kernel-source';
import type {RecordedDispatch} from './dispatch-numbers';

type SizingRow = {
  key: string;
  name: string;
  invocations: number;
  workgroupSize: number;
  how: string;
  /** Workgroup count in the record, when the row comes from one. */
  recordedWorkgroups?: number;
};

function SizingCells({row}: {row: SizingRow}) {
  const shape = getDispatchShape(row.invocations, row.workgroupSize);
  const slots = shape.workgroupCount * row.workgroupSize;
  return (
    <>
      <td>{formatCount(row.invocations)}</td>
      <td>
        {row.workgroupSize} <small>{row.how}</small>
      </td>
      <td>
        {formatCount(shape.workgroupCount)}
        {row.recordedWorkgroups !== undefined ? (
          <small> recorded {formatCount(row.recordedWorkgroups)}</small>
        ) : null}
      </td>
      <td>
        {formatCount(shape.x)} x {formatCount(shape.y)} x {formatCount(shape.z)}
      </td>
      <td>{formatCount(slots - row.invocations)}</td>
    </>
  );
}

/** What size and dispatch shape a count of invocations gets, for two recorded kernels and any count. */
export function SizingTable({
  pointWeights,
  gather
}: {
  pointWeights: RecordedDispatch | null;
  gather: RecordedDispatch | null;
}) {
  const [exponent, setExponent] = useState(30);
  const custom = Math.round(2 ** exponent);
  const rows: SizingRow[] = [];
  if (pointWeights) {
    rows.push({
      key: 'point-weights',
      name: 'point-weights',
      invocations: pointWeights.invocations,
      workgroupSize: getAutomaticWorkgroupSize(pointWeights.invocations),
      how: 'automatic',
      recordedWorkgroups: pointWeights.workgroups
    });
  }
  if (gather?.workgroupSize) {
    rows.push({
      key: 'gather',
      name: 'gather',
      invocations: gather.invocations,
      workgroupSize: gather.workgroupSize,
      how: 'set by the kernel',
      recordedWorkgroups: gather.workgroups
    });
  }
  const customRow: SizingRow = {
    key: 'custom',
    name: 'your count',
    invocations: custom,
    workgroupSize: getAutomaticWorkgroupSize(custom),
    how: 'automatic'
  };
  return (
    <div className="meta-panel kernels-sizing">
      <div className="meta-scroll">
        <table className="kernels-sizing__table" style={{minWidth: 560}}>
          <thead>
            <tr>
              <th scope="col">Kernel</th>
              <th scope="col">Invocations</th>
              <th scope="col">Workgroup size</th>
              <th scope="col">Workgroups</th>
              <th scope="col">Dispatch x y z</th>
              <th scope="col">Guarded lanes</th>
            </tr>
          </thead>
          <tbody>
            {[...rows, customRow].map(row => (
              <tr key={row.key} data-custom={row.key === 'custom' ? 'true' : undefined}>
                <th scope="row">{row.name}</th>
                <SizingCells row={row} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <label className="meta-slider kernels-sizing__slider">
        <span>
          your count <b>{formatCount(custom)}</b>
        </span>
        <input
          type="range"
          min={0}
          max={32}
          step={0.5}
          value={exponent}
          onChange={event => setExponent(Number(event.target.value))}
          aria-valuetext={`${formatCount(custom)} invocations`}
        />
      </label>
      <p className="kernels-size-note kernels-sizing__note">
        Each dimension holds at most {formatCount(WEBGPU_DEFAULT_WORKGROUPS_PER_DIMENSION)}{' '}
        workgroups here, the WebGPU default; the library reads the device limit. Past it, the grid
        grows in y, then z.
      </p>
    </div>
  );
}
