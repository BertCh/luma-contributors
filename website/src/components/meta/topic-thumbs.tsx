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
          {/* an offsets array marks where each row's slice of a flat array starts */}
          {[0, 3, 7, 9].map((start, index) => (
            <g key={`offset-${start}`} className={`thumb-column thumb-column--${index}`}>
              <rect x={62 + index * 50} y={32} width={40} height={26} rx={5} {...STROKE} />
              <path
                className="thumb-link"
                d={`M${82 + index * 50} 58 C${82 + index * 50} 96 ${32 + start * 22} 96 ${32 + start * 22} 128`}
                {...STROKE}
                strokeOpacity={0.6}
                strokeDasharray="3 3"
              />
            </g>
          ))}
          {Array.from({length: 12}, (_, cell) => (
            <rect
              key={`cell-${cell}`}
              x={24 + cell * 22}
              y={132}
              width={18}
              height={30}
              rx={3}
              fill="var(--color-accent)"
              fillOpacity={cell < 3 ? 0.25 : cell < 7 ? 0.85 : cell < 9 ? 0.45 : 0.65}
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
      {slug === 'building-blocks' ? (
        <g>
          {/* flags, their running total, and the kept rows packed to the front */}
          {[1, 0, 1, 1, 0, 0, 1, 0].map((flag, index) => (
            <rect
              key={`flag-${index}`}
              x={34 + index * 32}
              y={30}
              width={24}
              height={24}
              rx={4}
              fill="var(--color-accent)"
              fillOpacity={flag ? 0.85 : 0.12}
              stroke="var(--color-accent)"
              strokeOpacity={0.5}
            />
          ))}
          {[0, 1, 1, 2, 3, 3, 3, 4].map((offset, index) => (
            <text
              key={`scan-${index}`}
              x={46 + index * 32}
              y={98}
              textAnchor="middle"
              fontSize={13}
              fill="var(--color-muted)"
              fontFamily="var(--font-code)"
            >
              {offset}
            </text>
          ))}
          {[0, 2, 3, 6].map((source, slot) => (
            <g key={`kept-${source}`}>
              <path
                className="thumb-link"
                d={`M${46 + source * 32} 106 L${46 + slot * 32} 142`}
                {...STROKE}
                strokeOpacity={0.5}
                strokeDasharray="3 3"
              />
              <rect
                className={`thumb-step thumb-step--${slot}`}
                x={34 + slot * 32}
                y={146}
                width={24}
                height={24}
                rx={4}
                fill="var(--color-accent)"
                fillOpacity={0.85}
              />
            </g>
          ))}
        </g>
      ) : null}
      {slug === 'iteration' ? (
        <g>
          {/* rounds laid out in advance: the frontier rises and falls, the rest run nothing */}
          {[2, 5, 9, 14, 18, 15, 10, 6, 3, 1, 0, 0, 0, 0, 0, 0].map((size, index) => (
            <g key={index}>
              <rect
                className={size > 0 ? 'thumb-bar thumb-bar--active' : 'thumb-bar'}
                x={24 + index * 17.5}
                y={128 - size * 5}
                width={11}
                height={Math.max(2, size * 5)}
                rx={2}
                fill="var(--color-accent)"
                fillOpacity={size > 0 ? 0.8 : 0}
              />
              <rect
                x={24 + index * 17.5}
                y={146}
                width={11}
                height={22}
                rx={3}
                fill={size > 0 ? 'var(--color-accent)' : 'none'}
                fillOpacity={0.3}
                stroke={size > 0 ? 'var(--color-accent)' : 'var(--color-muted)'}
                strokeOpacity={size > 0 ? 0.9 : 0.5}
                strokeWidth={1.2}
              />
            </g>
          ))}
          <path
            className="thumb-cursor"
            d="M112 24 V172"
            stroke="var(--color-signal)"
            strokeWidth={1.4}
            strokeDasharray="4 4"
          />
        </g>
      ) : null}
      {slug === 'contract' ? (
        <g>
          {/* props in, checks, kernels out, status beside */}
          <rect
            x={26}
            y={76}
            width={66}
            height={48}
            rx={8}
            fill="none"
            stroke="var(--color-muted)"
            strokeWidth={1.6}
          />
          <rect x={126} y={60} width={78} height={80} rx={8} {...STROKE} />
          {[0, 1, 2].map(index => (
            <path
              key={index}
              className="thumb-link"
              d={`M140 ${82 + index * 18} l6 6 l12 -12`}
              {...STROKE}
            />
          ))}
          {[0, 1, 2].map(index => (
            <rect
              key={`out-${index}`}
              className={`thumb-layer thumb-layer--${index + 1}`}
              x={238}
              y={52 + index * 26}
              width={56}
              height={18}
              rx={5}
              fill="var(--color-accent)"
              fillOpacity={0.3 + index * 0.25}
            />
          ))}
          <rect
            className="thumb-sink"
            x={238}
            y={136}
            width={56}
            height={18}
            rx={5}
            fill="none"
            stroke="var(--color-signal)"
            strokeWidth={1.6}
          />
          <path
            d="M92 100 H126 M204 88 H238 M204 122 C222 122 222 145 238 145"
            {...STROKE}
            strokeOpacity={0.7}
          />
        </g>
      ) : null}
      {slug === 'workflows' ? (
        <g>
          {/* a chain of operations, each with its own picture */}
          {[0, 1, 2, 3].map(index => (
            <g
              key={index}
              className={index % 2 === 0 ? 'thumb-lane' : 'thumb-lane thumb-lane--step'}
            >
              <rect
                x={20 + index * 74}
                y={62}
                width={58}
                height={76}
                rx={8}
                {...STROKE}
                stroke={index === 3 ? 'var(--color-signal)' : 'var(--color-accent)'}
              />
              <rect
                x={27 + index * 74}
                y={69}
                width={44}
                height={38}
                rx={4}
                fill={index === 3 ? 'var(--color-signal)' : 'var(--color-accent)'}
                fillOpacity={0.18 + index * 0.14}
              />
              <path
                d={`M28 ${118} h30 M28 ${127} h18`}
                transform={`translate(${index * 74} 0)`}
                stroke="var(--color-muted)"
                strokeWidth={1.6}
                strokeOpacity={0.7}
              />
            </g>
          ))}
          {[0, 1, 2].map(index => (
            <path
              key={`link-${index}`}
              className="thumb-link"
              d={`M${78 + index * 74} 100 h16`}
              {...STROKE}
              strokeDasharray="3 3"
            />
          ))}
        </g>
      ) : null}
      {slug === 'glossary' ? (
        <g>
          {[0, 1, 2, 3, 4].map(index => (
            <g key={index} className={`thumb-layer thumb-layer--${index}`}>
              <rect
                x={36}
                y={32 + index * 28}
                width={[64, 88, 52, 76, 60][index]}
                height={14}
                rx={4}
                fill="var(--color-accent)"
                fillOpacity={0.75}
              />
              <path
                d={`M${[116, 140, 104, 128, 112][index]} ${39 + index * 28} H284`}
                stroke="var(--color-muted)"
                strokeWidth={1.6}
                strokeOpacity={0.55}
                strokeDasharray="2 5"
              />
            </g>
          ))}
        </g>
      ) : null}
    </svg>
  );
}
