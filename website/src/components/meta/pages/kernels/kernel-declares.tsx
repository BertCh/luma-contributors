// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatBytes} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';

/** The buffers one recorded kernel reads and writes: its whole declaration to the command graph. */
export function KernelDeclares({model, label}: {model: LensViewModel; label: string}) {
  const node = model.nodes.find(candidate => candidate.label === label);
  if (!node?.reads || !node.writes) return null;
  const writes = new Set(node.writes);
  const reads = new Set(node.reads);
  const bytes = (id: string) => model.bufferBytes?.[id];
  const rowHeight = 30;
  const rowCount = Math.max(node.reads.length, node.writes.length);
  const height = rowCount * rowHeight + 16;
  const kernelX = 420;
  const kernelWidth = 150;
  const width = 920;
  return (
    <div className="meta-panel meta-scroll kernels-declares">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{minWidth: 760, width: '100%', display: 'block'}}
        role="img"
        aria-label={`The kernel ${label} reads ${node.reads.length} buffers and writes ${node.writes.length}`}
      >
        <rect
          x={kernelX}
          y={8}
          width={kernelWidth}
          height={height - 16}
          rx={8}
          fill="var(--color-accent-soft)"
          stroke="var(--color-accent)"
          strokeWidth={1.5}
        />
        <text
          x={kernelX + kernelWidth / 2}
          y={height / 2 + 4}
          textAnchor="middle"
          fill="var(--color-text)"
          fontSize={13}
        >
          {label}
        </text>
        {node.reads.map((id, index) => {
          const y = 8 + index * rowHeight + rowHeight / 2;
          return (
            <g key={`read-${id}`}>
              <text
                x={kernelX - 40}
                y={y + 4}
                textAnchor="end"
                fill="var(--color-text)"
                fontSize={12}
              >
                {id}
                <tspan fill="var(--color-muted)">
                  {writes.has(id) ? '  read_write' : '  read'}
                  {bytes(id) !== undefined ? `  ${formatBytes(bytes(id) ?? 0)}` : ''}
                </tspan>
              </text>
              <line
                x1={kernelX - 32}
                x2={kernelX - 4}
                y1={y}
                y2={y}
                stroke="var(--color-accent)"
                strokeWidth={1.5}
              />
            </g>
          );
        })}
        {node.writes.map((id, index) => {
          const y = 8 + index * rowHeight + rowHeight / 2;
          return (
            <g key={`write-${id}`}>
              <line
                x1={kernelX + kernelWidth + 4}
                x2={kernelX + kernelWidth + 32}
                y1={y}
                y2={y}
                stroke="var(--color-accent)"
                strokeWidth={1.5}
              />
              <text x={kernelX + kernelWidth + 40} y={y + 4} fill="var(--color-text)" fontSize={12}>
                {id}
                <tspan fill="var(--color-muted)">
                  {reads.has(id) ? '  read_write' : '  write'}
                </tspan>
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
