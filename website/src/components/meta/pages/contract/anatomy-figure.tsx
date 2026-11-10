// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The shape of a contributor: a recorded construction of GPUNetworkServiceAreas (the code that
// produced the recorded run) beside a diagram of what each part of it does.

import {useState} from 'react';
import {CodeWalk, type CodeWalkStep} from '../../page-kit';
import {findExampleLines} from '../../examples';
import type {TopicExample} from '../../topic-types';
import './contract.css';

type PartId = 'props' | 'construct' | 'outputs' | 'nodes';

const BOXES = [
  {id: 'props', x: 8, y: 20, width: 150, title: 'props', note: 'inputs, options', kind: 'setup'},
  {
    id: 'construct',
    x: 200,
    y: 20,
    width: 190,
    title: 'constructor',
    note: 'validates once, may throw',
    kind: 'setup'
  },
  {
    id: 'nodes',
    x: 432,
    y: 20,
    width: 230,
    title: 'getCommandNodes(graph)',
    note: 'returns the kernels',
    kind: 'kernel'
  },
  {
    id: 'outputs',
    x: 432,
    y: 120,
    width: 230,
    title: 'outputs + status',
    note: 'caller-owned buffers',
    kind: 'status'
  }
] as const satisfies readonly {
  id: PartId;
  x: number;
  y: number;
  width: number;
  title: string;
  note: string;
  kind: string;
}[];

export function AnatomyFigure({example}: {example: TopicExample}) {
  const [active, setActive] = useState<PartId>('construct');
  const code = example.code;
  const steps: (CodeWalkStep & {id: PartId})[] = [
    {
      id: 'props',
      title: 'Props in',
      text: 'The caller hands over views and options. A prop that sets a bound, here the rounds, is fixed at construction.',
      lines: findExampleLines(code, 'topology:', 'maximumIterations')
    },
    {
      id: 'construct',
      title: 'Constructor checks',
      text: 'Every prop is validated once, here, with no GPU involved. A bad value throws before anything is built.',
      lines: findExampleLines(code, 'new GPUNetworkServiceAreas(')
    },
    {
      id: 'outputs',
      title: 'Outputs and status',
      text: 'The caller owns the output buffers and the named status ports the kernels will write.',
      lines: findExampleLines(code, 'output: {', 'status:')
    },
    {
      id: 'nodes',
      title: 'Nodes out',
      text: 'The graph calls getCommandNodes and gets kernels back. It does not change the graph itself.',
      lines: findExampleLines(code, 'graph.add(')
    }
  ];
  return (
    <>
      <CodeWalk
        example={example}
        steps={steps}
        active={active}
        onActive={id => setActive(id as PartId)}
        label="GPUNetworkServiceAreas, as constructed for the recorded run"
      />
      <div className="meta-panel meta-scroll contract-anatomy">
        <svg
          viewBox="0 0 680 200"
          style={{minWidth: 560, width: '100%', display: 'block'}}
          role="img"
          aria-label="Props go into the constructor, which validates and may throw. getCommandNodes then returns the kernels, which write the caller's outputs and status."
        >
          {BOXES.map(box => (
            <g key={box.id}>
              <rect
                className="contract-anatomy__box"
                data-kind={box.kind}
                data-active={box.id === active ? 'true' : undefined}
                x={box.x}
                y={box.y}
                width={box.width}
                height={64}
                rx={8}
              />
              <text className="contract-anatomy__title" x={box.x + 14} y={box.y + 27}>
                {box.title}
              </text>
              <text className="contract-anatomy__note" x={box.x + 14} y={box.y + 47}>
                {box.note}
              </text>
            </g>
          ))}
          <path className="contract-anatomy__arrow" d="M158 52 H198 M390 52 H430" />
          <path className="contract-anatomy__arrow" d="M547 84 V118" />
          <text className="contract-anatomy__note" x={8} y={190}>
            muted: CPU setup. teal: GPU kernels. amber: status.
          </text>
        </svg>
      </div>
    </>
  );
}
