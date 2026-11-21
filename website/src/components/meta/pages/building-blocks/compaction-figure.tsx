// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Flags, scan, scatter on twenty recorded rows of the time-window filter, beside the nine recorded
// kernels that do the same on all 1,500 rows.

import {type ReactNode, useState} from 'react';
import {formatCount} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';
import {ALL_STAGES, isLit, KernelStrip, Stepper, type StripKernel} from './stepper';
import {COMPACTION_KEPT, COMPACTION_ROWS, RECORDED} from './toy-data';

const SECONDS_PER_DAY = 86400;
const ROW = 22;
const TOP = 34;
const STAGES = [
  {id: 'classify', title: 'classify'},
  {id: 'scan', title: 'scan'},
  {id: 'scatter', title: 'scatter'},
  {id: 'publish', title: 'publish'}
] as const;

function formatDay(seconds: number): string {
  return (seconds / SECONDS_PER_DAY).toFixed(1);
}

function getStage(nodeId: string): string[] {
  if (nodeId.endsWith('-classify')) return ['classify'];
  if (/compact-(normalize|scan)/.test(nodeId)) return ['scan'];
  if (nodeId.endsWith('-scatter')) return ['scatter'];
  if (nodeId.endsWith('-publish')) return ['publish'];
  return [];
}

function getShortLabel(nodeId: string): string {
  return nodeId
    .replace(/^gpu-dataframe-time-window-filter-/, '')
    .replace(/^time-window-filter-/, '')
    .replace(/^visibility-compact-/, '')
    .replace(/^visibility-/, '');
}

export function getTimeWindowKernels(model: LensViewModel): StripKernel[] {
  return model.nodes.map(node => ({
    id: node.id,
    label: getShortLabel(node.id),
    stages: getStage(node.id),
    kind: node.id.endsWith('snapshot-tap') ? 'readback' : 'kernel',
    detail: node.invocations
      ? formatCount(node.invocations)
      : node.id.endsWith('snapshot-tap')
        ? 'read back'
        : 'setup'
  }));
}

function getParameter(model: LensViewModel, name: string): number {
  return Number(model.parameters.find(parameter => parameter.name === name)?.value ?? Number.NaN);
}

export function CompactionFigure({model}: {model: LensViewModel}) {
  const [active, setActive] = useState<string>(ALL_STAGES);
  const start = getParameter(model, 'start');
  const end = getParameter(model, 'end');
  const kernels = getTimeWindowKernels(model);
  const classify = model.nodes.find(node => node.id.endsWith('-classify'));
  const firstScan = model.nodes.find(node => node.id.endsWith('scan-level-0-scan'));
  const blockSumsKey = Object.keys(model.bufferBytes ?? {}).find(key =>
    key.endsWith('scan-level-0-block-sums')
  );
  const blockCount = blockSumsKey ? (model.bufferBytes?.[blockSumsKey] ?? 0) / 4 : 0;
  const example = COMPACTION_ROWS.find(row => row.recordedSlot > 0);
  const {rowCount, keptCount} = RECORDED.timeWindow;

  const rowsHeight = TOP + COMPACTION_ROWS.length * ROW + 6;
  const slotY = (slot: number) => TOP + slot * ROW;
  const columns = {
    row: 34,
    day: 92,
    flag: 102,
    scan: 150,
    linesFrom: 186,
    linesTo: 252,
    slot: 258,
    output: 280
  };
  const lit = {
    classify: isLit(active, 'classify'),
    scan: isLit(active, 'scan'),
    scatter: isLit(active, 'scatter'),
    publish: isLit(active, 'publish')
  };
  const strong = (stage: string) => (active === stage ? 'true' : undefined);

  const readouts: Record<string, ReactNode> = {
    [ALL_STAGES]: (
      <>
        {COMPACTION_ROWS.length} of the run's {formatCount(rowCount)} rows (every{' '}
        {RECORDED.timeWindow.stride}th). The flags and the packed ids are recorded; the scan column
        is the same prefix sum, computed here.
      </>
    ),
    classify: (
      <>
        One invocation per row tests its timestamp against the window, day {formatDay(start)} to{' '}
        {formatDay(end)}, and writes a 0 or 1 to <code>outputMask</code>. Recorded:{' '}
        <b>{formatCount(classify?.invocations ?? 0)}</b> invocations.
      </>
    ),
    scan: (
      <>
        An exclusive prefix sum of the flags: each row learns how many kept rows come before it,
        which is its slot. Recorded over <b>{formatCount(firstScan?.invocations ?? 0)}</b> lanes in
        two levels: {blockCount} block sums, a scan of those, then the block offsets added back.
      </>
    ),
    scatter: (
      <>
        Each kept row writes its id at its slot. Slots are distinct, so no atomics are needed and
        row order survives.
      </>
    ),
    publish: (
      <>
        One invocation writes the total to <code>count</code> and sets <code>overflow</code> if it
        exceeds capacity. Full run: <b>{formatCount(keptCount)}</b> of {formatCount(rowCount)} kept
        {example ? (
          <>
            ; row {example.row} sits at <code>ids[{example.recordedSlot}]</code>
          </>
        ) : null}
        .
      </>
    )
  };

  return (
    <div className="meta-panel">
      <Stepper label="Compaction stages" stages={STAGES} active={active} onActive={setActive} />
      <div className="building-blocks-compaction">
        <div className="building-blocks-figure building-blocks-figure--narrow">
          <svg
            viewBox={`0 0 340 ${rowsHeight}`}
            className="building-blocks-svg"
            role="img"
            aria-label={`Twenty recorded rows: their day, a kept flag, the exclusive scan of the flags, and lines from the ${COMPACTION_KEPT} kept rows to consecutive output slots.`}
          >
            <text
              x={columns.row}
              y={16}
              textAnchor="end"
              className="building-blocks-text building-blocks-text--muted"
            >
              row
            </text>
            <text
              x={columns.day}
              y={16}
              textAnchor="end"
              className="building-blocks-text building-blocks-text--muted"
              data-dim={!lit.classify ? 'true' : undefined}
            >
              day
            </text>
            <text
              x={columns.flag + 14}
              y={16}
              textAnchor="middle"
              className="building-blocks-text building-blocks-text--muted"
              data-dim={!lit.classify ? 'true' : undefined}
            >
              kept
            </text>
            <text
              x={columns.scan + 14}
              y={16}
              textAnchor="middle"
              className="building-blocks-text building-blocks-text--muted"
              data-dim={!lit.scan ? 'true' : undefined}
            >
              scan
            </text>
            <text
              x={columns.output + 22}
              y={16}
              textAnchor="middle"
              className="building-blocks-text building-blocks-text--muted"
              data-dim={!lit.scatter ? 'true' : undefined}
            >
              ids
            </text>
            {COMPACTION_ROWS.map((row, index) => {
              const y = TOP + index * ROW;
              return (
                <g key={row.row}>
                  <text
                    x={columns.row}
                    y={y + 4}
                    textAnchor="end"
                    className="building-blocks-text building-blocks-text--muted"
                  >
                    {row.row}
                  </text>
                  <g
                    data-dim={!lit.classify ? 'true' : undefined}
                    className="building-blocks-node-group"
                  >
                    <text
                      x={columns.day}
                      y={y + 4}
                      textAnchor="end"
                      className="building-blocks-text"
                    >
                      {formatDay(row.timestamp)}
                    </text>
                    <rect
                      x={columns.flag}
                      y={y - 9}
                      width={28}
                      height={18}
                      rx={3}
                      className="building-blocks-cell"
                      data-on={row.flag ? 'true' : undefined}
                      data-strong={strong('classify')}
                    />
                    <text
                      x={columns.flag + 14}
                      y={y + 4}
                      textAnchor="middle"
                      className="building-blocks-text"
                    >
                      {row.flag}
                    </text>
                  </g>
                  <g
                    data-dim={!lit.scan ? 'true' : undefined}
                    className="building-blocks-node-group"
                  >
                    <rect
                      x={columns.scan}
                      y={y - 9}
                      width={28}
                      height={18}
                      rx={3}
                      className="building-blocks-cell"
                      data-on={active === 'scan' ? 'true' : undefined}
                    />
                    <text
                      x={columns.scan + 14}
                      y={y + 4}
                      textAnchor="middle"
                      className="building-blocks-text"
                    >
                      {row.position}
                    </text>
                  </g>
                  {row.flag ? (
                    <path
                      d={`M${columns.linesFrom} ${y} C${columns.linesFrom + 36} ${y} ${columns.linesTo - 36} ${slotY(row.position)} ${columns.linesTo} ${slotY(row.position)}`}
                      className="building-blocks-edge"
                      data-lit={active === 'scatter' ? 'true' : undefined}
                      data-dim={!lit.scatter ? 'true' : undefined}
                    />
                  ) : null}
                </g>
              );
            })}
            {COMPACTION_ROWS.filter(row => row.flag).map(row => {
              const y = slotY(row.position);
              return (
                <g
                  key={`slot-${row.row}`}
                  data-dim={!lit.scatter ? 'true' : undefined}
                  className="building-blocks-node-group"
                >
                  <text
                    x={columns.slot}
                    y={y + 4}
                    className="building-blocks-text building-blocks-text--muted"
                  >
                    {row.position}
                  </text>
                  <rect
                    x={columns.output}
                    y={y - 9}
                    width={44}
                    height={18}
                    rx={3}
                    className="building-blocks-cell"
                    data-on={active === 'scatter' ? 'true' : undefined}
                  />
                  <text
                    x={columns.output + 22}
                    y={y + 4}
                    textAnchor="middle"
                    className="building-blocks-text"
                  >
                    {row.row}
                  </text>
                </g>
              );
            })}
            <g data-dim={!lit.publish ? 'true' : undefined} className="building-blocks-node-group">
              <text
                x={columns.output + 22}
                y={slotY(COMPACTION_KEPT) + 14}
                textAnchor="middle"
                className="building-blocks-text building-blocks-text--muted"
              >
                count
              </text>
              <rect
                x={columns.output}
                y={slotY(COMPACTION_KEPT) + 20}
                width={44}
                height={20}
                rx={3}
                className="building-blocks-cell"
                data-root="true"
                data-on={active === 'publish' ? 'true' : undefined}
              />
              <text
                x={columns.output + 22}
                y={slotY(COMPACTION_KEPT) + 34}
                textAnchor="middle"
                className="building-blocks-text building-blocks-text--strong"
              >
                {COMPACTION_KEPT}
              </text>
            </g>
          </svg>
        </div>
        <div className="building-blocks-compaction__side">
          <KernelStrip
            title={
              <>
                Recorded run: <b>{model.recordedNodeCount}</b> graph nodes over{' '}
                {formatCount(rowCount)} rows
              </>
            }
            kernels={kernels}
            active={active}
          />
          <p className="meta-readout building-blocks-readout" aria-live="polite">
            {readouts[active]}
          </p>
        </div>
      </div>
    </div>
  );
}
