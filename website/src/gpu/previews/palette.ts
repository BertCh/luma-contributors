// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Preview colors. Previews are always a dark map panel, in both themes. The brand may override the
// panel and data colors with --preview-background, --preview-primary and --preview-signal.

import type {RampName, Role} from './types.ts';

export type Rgb = [number, number, number];

export const PREVIEW_BACKGROUND = '#070b14';

export const ROLE_COLORS: Record<Role, string> = {
  primary: '#3de0c5',
  warm: '#ffb454',
  hot: '#ff5c8f',
  cool: '#4da3ff',
  violet: '#9e7bff',
  ink: '#dbe6f5',
  dim: '#6c7e99'
};

/** CSS variable that can override each role, when the brand defines it. */
const ROLE_VARIABLES: Partial<Record<Role, string>> = {
  primary: '--preview-primary',
  warm: '--preview-signal'
};

export const RAMP_STOPS: Record<RampName, string[]> = {
  heat: ['#0a1020', '#3a2a8c', '#c2307f', '#ff7a45', '#ffe9a8'],
  ice: ['#08111f', '#0f3a5e', '#1d86b8', '#3de0c5', '#eafffa'],
  terrain: ['#0e2a3a', '#1d5a58', '#4f8a4a', '#b9a35a', '#f2efe6'],
  spectral: ['#2b2f8f', '#2d8fc9', '#3de0c5', '#f2e066', '#ff6a5a']
};

export function parseHex(hex: string): Rgb {
  const value = Number.parseInt(hex.slice(1), 16);
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}

export function formatHex(color: Rgb): string {
  const toByte = (channel: number) =>
    Math.round(Math.min(1, Math.max(0, channel)) * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${toByte(color[0])}${toByte(color[1])}${toByte(color[2])}`;
}

/** Piecewise linear color along a ramp, t in 0..1. */
export function getRampColor(ramp: RampName, t: number): Rgb {
  const stops = RAMP_STOPS[ramp].map(parseHex);
  const scaled = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const amount = scaled - index;
  return [0, 1, 2].map(
    channel => stops[index][channel] + (stops[index + 1][channel] - stops[index][channel]) * amount
  ) as Rgb;
}

/** Role color as a CSS value for the static SVG fallback. */
export function getRoleCss(role: Role): string {
  const variable = ROLE_VARIABLES[role];
  return variable ? `var(${variable}, ${ROLE_COLORS[role]})` : ROLE_COLORS[role];
}

export type PreviewPalette = {
  background: Rgb;
  roles: Record<Role, Rgb>;
};

function readCssColor(text: string, fallback: Rgb): Rgb {
  const value = text.trim();
  if (!value) return fallback;
  try {
    const context = document.createElement('canvas').getContext('2d');
    if (!context) return fallback;
    context.clearRect(0, 0, 1, 1);
    context.fillStyle = value;
    context.fillRect(0, 0, 1, 1);
    const data = context.getImageData(0, 0, 1, 1).data;
    return [data[0] / 255, data[1] / 255, data[2] / 255];
  } catch {
    return fallback;
  }
}

/** Reads the palette in the browser, honoring brand overrides. */
export function readPalette(): PreviewPalette {
  const style = getComputedStyle(document.documentElement);
  const roles = {} as Record<Role, Rgb>;
  for (const role of Object.keys(ROLE_COLORS) as Role[]) {
    const variable = ROLE_VARIABLES[role];
    roles[role] = readCssColor(
      variable ? style.getPropertyValue(variable) : '',
      parseHex(ROLE_COLORS[role])
    );
  }
  return {
    background: readCssColor(
      style.getPropertyValue('--preview-background'),
      parseHex(PREVIEW_BACKGROUND)
    ),
    roles
  };
}
