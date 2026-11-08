// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Static thumbnails for the how-it-works index: one tiny diagram per topic, drawn with tokens.

import type {TopicSlug} from './topics';

const STROKE = {fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 1.6} as const;

export function TopicThumb({slug}: {slug: TopicSlug}) {
  return (
    <svg viewBox="0 0 320 200" role="img" aria-label="" aria-hidden="true">
      {slug === 'anatomy' ? (
        <g>
          {Array.from({length: 5}, (_, index) => (
            <rect
              key={index}
              className={index % 2 === 0 ? 'thumb-lane' : 'thumb-lane thumb-lane--step'}
              x={18 + index * 60}
              y={78}
              width={44}
              height={44}
              rx={8}
              {...STROKE}
              stroke={index > 2 ? 'var(--color-signal)' : 'var(--color-accent)'}
            />
          ))}
          {Array.from({length: 4}, (_, index) => (
            <path
              key={index}
              className="thumb-link"
              d={`M${62 + index * 60} 100 h16`}
              {...STROKE}
              strokeDasharray="3 3"
            />
          ))}
          <circle cx={170} cy={100} r={5} fill="var(--color-accent)" />
        </g>
      ) : null}
      {slug === 'command-graph' ? (
        <g>
          {[
            [30, 100],
            [110, 50],
            [110, 150],
            [200, 100],
            [280, 100]
          ].map(([x, y], index) => (
            <rect
              key={index}
              className={index === 4 ? 'thumb-sink' : undefined}
              x={x - 22}
              y={y - 14}
              width={44}
              height={28}
              rx={7}
              {...STROKE}
              stroke={index === 4 ? 'var(--color-signal)' : 'var(--color-accent)'}
            />
          ))}
          <path
            d="M52 100 C80 100 80 50 88 50 M52 100 C80 100 80 150 88 150 M132 50 C165 50 165 100 178 100 M132 150 C165 150 165 100 178 100 M222 100 H258"
            {...STROKE}
            strokeOpacity={0.7}
          />
        </g>
      ) : null}
      {slug === 'data-on-the-gpu' ? (
        <g>
          {[0, 1, 2, 3].map(column =>
            [0, 1, 2, 3, 4].map(row => (
              <rect
                key={`${column}-${row}`}
                className={`thumb-column thumb-column--${column}`}
                x={40 + column * 62}
                y={26 + row * 28 + (row > 2 ? 10 : 0)}
                width={48}
                height={20}
                rx={3}
                fill="var(--color-accent)"
                fillOpacity={0.18 + ((column * 5 + row * 3) % 7) * 0.1}
              />
            ))
          )}
          {[0, 1, 2, 3].map(column => (
            <rect
              key={column}
              className={`thumb-column thumb-column--${column}`}
              x={36 + column * 62}
              y={22}
              width={56}
              height={86}
              rx={5}
              {...STROKE}
              strokeDasharray="4 3"
              strokeOpacity={0.6}
            />
          ))}
          {[0, 1, 2, 3].map(column => (
            <rect
              key={column}
              className={`thumb-column thumb-column--${column}`}
              x={36 + column * 62}
              y={118}
              width={56}
              height={58}
              rx={5}
              {...STROKE}
              strokeDasharray="4 3"
              strokeOpacity={0.6}
            />
          ))}
        </g>
      ) : null}
      {slug === 'lenses' ? (
        <g>
          <rect x={24} y={24} width={272} height={152} rx={8} {...STROKE} strokeOpacity={0.5} />
          {Array.from({length: 11}, (_, column) =>
            Array.from({length: 6}, (_, row) => {
              const distance = Math.hypot(column - 5, row - 2.6);
              return (
                <rect
                  key={`${column}-${row}`}
                  x={34 + column * 24}
                  y={34 + row * 23}
                  width={20}
                  height={19}
                  rx={2}
                  fill="var(--color-accent)"
                  fillOpacity={Math.max(0.06, 0.9 - distance * 0.2)}
                />
              );
            })
          )}
          <circle className="thumb-pulse" cx={172} cy={96} r={5} fill="var(--color-signal)" />
        </g>
      ) : null}
      {slug === 'scheduling' ? (
        <g>
          {[
            [30, 14, 90],
            [60, 40, 100],
            [100, 66, 120],
            [140, 92, 80],
            [190, 118, 90],
            [220, 144, 70]
          ].map(([x, y, width], index) => (
            <rect
              key={index}
              className={index === 3 ? 'thumb-bar thumb-bar--active' : 'thumb-bar'}
              x={x}
              y={y}
              width={width}
              height={18}
              rx={5}
              fill="var(--color-accent)"
              fillOpacity={index === 3 ? 0.9 : 0.22 + index * 0.05}
              {...(index === 3 ? {} : {stroke: 'var(--color-accent)', strokeWidth: 1.2})}
            />
          ))}
          <path
            className="thumb-cursor"
            d="M162 8 V176"
            stroke="var(--color-signal)"
            strokeWidth={1.6}
            strokeDasharray="4 4"
          />
        </g>
      ) : null}
      {slug === 'kernels' ? (
        <g>
          {Array.from({length: 12}, (_, column) =>
            Array.from({length: 8}, (_, row) => {
              const index = row * 12 + column;
              const workgroup = Math.floor(index / 16);
              return (
                <rect
                  key={`${column}-${row}`}
                  x={28 + column * 22}
                  y={22 + row * 20}
                  width={17}
                  height={15}
                  rx={3}
                  fill="var(--color-accent)"
                  fillOpacity={workgroup % 2 === 0 ? 0.85 : 0.35}
                />
              );
            })
          )}
          <path
            className="thumb-scan"
            d="M28 18 H292"
            stroke="var(--color-signal)"
            strokeWidth={1.6}
          />
        </g>
      ) : null}
      {slug === 'this-page' ? (
        <g>
          <rect x={24} y={20} width={272} height={110} rx={8} {...STROKE} strokeOpacity={0.6} />
          {Array.from({length: 8}, (_, column) =>
            Array.from({length: 3}, (_, row) => (
              <rect
                key={`${column}-${row}`}
                x={36 + column * 32}
                y={32 + row * 32}
                width={24}
                height={24}
                rx={4}
                fill="var(--color-accent)"
                fillOpacity={0.15 + ((column + row * 3) % 5) * 0.16}
              />
            ))
          )}
          {[0, 1, 2, 3].map(index => (
            <rect
              key={index}
              className={`thumb-step thumb-step--${index}`}
              x={24 + index * 70}
              y={142}
              width={62}
              height={30}
              rx={4}
              fill="none"
              stroke="var(--color-signal)"
              strokeWidth={1.4}
              strokeOpacity={0.8}
            />
          ))}
        </g>
      ) : null}
      {slug === 'upstream-first' ? (
        <g>
          {[0, 1, 2].map(index => (
            <rect
              key={index}
              className={`thumb-layer thumb-layer--${index}`}
              x={50 + index * 14}
              y={26 + index * 24}
              width={150}
              height={26}
              rx={7}
              {...STROKE}
            />
          ))}
          {[3, 4, 5].map(index => (
            <rect
              key={index}
              className={`thumb-layer thumb-layer--${index}`}
              x={50 + index * 14}
              y={26 + index * 24}
              width={150}
              height={26}
              rx={7}
              fill="none"
              stroke="var(--color-muted)"
              strokeWidth={1.6}
            />
          ))}
          <path
            d="M222 38 V160"
            stroke="var(--color-accent)"
            strokeWidth={1.6}
            strokeDasharray="3 4"
          />
          <path className="thumb-arrow" d="M214 150 l8 10 l8 -10" {...STROKE} />
        </g>
      ) : null}
    </svg>
  );
}
