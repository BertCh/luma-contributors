// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Gridlight mark: a 3x3 GPU tile grid whose lit cells trace a coastline-like path, with one
 * signal-colored hotspot. Uses currentColor for the lit cells and --color-signal for the hotspot.
 */
export function LogoMark({size = 26}: {size?: number}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role="img"
      aria-label="Gridlight"
      fill="none"
    >
      <rect
        x="2.5"
        y="2.5"
        width="27"
        height="27"
        rx="5"
        stroke="currentColor"
        strokeOpacity="0.45"
      />
      <path d="M12 3v26M21 3v26M3 12h26M3 21h26" stroke="currentColor" strokeOpacity="0.22" />
      <path
        d="M3 21c4-1 5-6 9-6s5 5 9 3 4-6 8-6"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <rect
        x="12.8"
        y="3.8"
        width="7.4"
        height="7.4"
        rx="1.5"
        fill="currentColor"
        fillOpacity="0.9"
      />
      <circle cx="24.5" cy="24.5" r="2.4" fill="var(--color-signal)" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="brand-link__text">
      grid<span style={{color: 'var(--color-accent)'}}>light</span>
    </span>
  );
}
