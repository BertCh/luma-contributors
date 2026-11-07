// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatCount} from '../../lens/format';
import type {LensViewModel} from '../../lens/types';
import './meta.css';

type HazardSpec = {
  key: 'raw' | 'war' | 'waw';
  title: string;
  text: string;
  first: string;
  second: string;
};

const HAZARDS: readonly HazardSpec[] = [
  {
    key: 'raw',
    title: 'read after write',
    text: 'The reader waits for the writer.',
    first: 'write',
    second: 'read'
  },
  {
    key: 'war',
    title: 'write after read',
    text: 'The writer waits until the reader is done.',
    first: 'read',
    second: 'write'
  },
  {
    key: 'waw',
    title: 'write after write',
    text: 'The second write lands second.',
    first: 'write',
    second: 'write'
  }
];

/** Three inferred dependency kinds with the counts of one recorded run. */
export function HazardDiagram({model}: {model: LensViewModel}) {
  return (
    <ul className="hazard-row" aria-label="Dependency kinds the compiler infers">
      {HAZARDS.map(hazard => (
        <li className="hazard" key={hazard.key}>
          <svg viewBox="0 0 260 74" aria-hidden="true">
            <rect className="hazard-node" x={4} y={14} width={78} height={34} rx={7} />
            <text className="hazard-text" x={43} y={35} textAnchor="middle">
              {hazard.first}
            </text>
            <rect className="hazard-resource" x={104} y={20} width={52} height={22} rx={5} />
            <text className="hazard-text" x={130} y={35} textAnchor="middle">
              buffer
            </text>
            <rect className="hazard-node" x={178} y={14} width={78} height={34} rx={7} />
            <text className="hazard-text" x={217} y={35} textAnchor="middle">
              {hazard.second}
            </text>
            <path className="hazard-line" d="M82 31 H104 M156 31 H178" />
            <path className="hazard-flow" d="M43 52 C43 70 217 70 217 52" />
          </svg>
          <strong>{hazard.title}</strong>
          <span>{hazard.text}</span>
          <span>
            <span className="hazard__count">{formatCount(model.hazards[hazard.key])}</span> in this
            run
          </span>
        </li>
      ))}
    </ul>
  );
}
