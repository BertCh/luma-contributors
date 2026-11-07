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
            <path key={index} d={`M${62 + index * 60} 100 h16`} {...STROKE} strokeDasharray="3 3" />
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
          <circle cx={172} cy={96} r={5} fill="var(--color-signal)" />
        </g>
      ) : null}
      {slug === 'upstream-first' ? (
        <g>
          {[0, 1, 2].map(index => (
            <rect
              key={index}
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
          <path d="M214 150 l8 10 l8 -10" {...STROKE} />
        </g>
      ) : null}
    </svg>
  );
}
