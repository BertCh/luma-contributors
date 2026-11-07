// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared helpers of the how-it-works scenes: brand colors per theme, blend states and a uniform
// buffer. Scenes are WGSL-only and read their data from storage buffers, like src/gpu/demos/kit.ts.

import {Buffer, type Device} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import type {GpuTheme} from '../../../gpu/types';
import {VALUE_RAMP} from '../../../lens/ramp';

export type Color4 = [number, number, number, number];

export type SceneColors = {
  /** Panel surface; the canvas clears to it so the page and the canvas agree. */
  background: Color4;
  accent: Color4;
  signal: Color4;
  ink: Color4;
  muted: Color4;
};

function parseHex(value: string, alpha = 1): Color4 {
  const number = Number.parseInt(value.slice(1), 16);
  return [((number >> 16) & 255) / 255, ((number >> 8) & 255) / 255, (number & 255) / 255, alpha];
}

// Mirrors --color-surface, --color-accent, --color-signal, --color-text and --color-muted in
// src/styles/tokens.css.
export function getSceneColors(theme: GpuTheme): SceneColors {
  return theme === 'dark'
    ? {
        background: parseHex('#0d121b'),
        accent: parseHex('#3de0c5'),
        signal: parseHex('#ffb454'),
        ink: parseHex('#e8eef6'),
        muted: parseHex('#93a1b4')
      }
    : {
        background: parseHex('#ffffff'),
        accent: parseHex('#07745f'),
        signal: parseHex('#b4570a'),
        ink: parseHex('#0b1220'),
        muted: parseHex('#4d5a6b')
      };
}

/** Glow on dark surfaces, ordinary alpha on light ones. */
export function getBlendParameters(theme: GpuTheme) {
  return {
    blend: true,
    blendColorOperation: 'add',
    blendColorSrcFactor: 'src-alpha',
    blendColorDstFactor: theme === 'dark' ? 'one' : 'one-minus-src-alpha',
    blendAlphaOperation: 'add',
    blendAlphaSrcFactor: 'one',
    blendAlphaDstFactor: 'one-minus-src-alpha'
  } as const;
}

/** WGSL for the shared ramp, generated from VALUE_RAMP so it matches the static PNG fallbacks. */
export const RAMP_WGSL = /* wgsl */ `
fn sampleRamp(position: f32) -> vec3f {
  var stops = array<vec3f, ${VALUE_RAMP.length}>(
    ${VALUE_RAMP.map(([red, green, blue]) => `vec3f(${red}, ${green}, ${blue})`).join(',\n    ')}
  );
  let scaled = clamp(position, 0.0, 1.0) * ${VALUE_RAMP.length - 1}.0;
  let index = min(u32(floor(scaled)), ${VALUE_RAMP.length - 2}u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index));
}
`;

export function createUniformBuffer(device: Device, floatCount: number): Buffer {
  return device.createBuffer({byteLength: floatCount * 4, usage: Buffer.UNIFORM | Buffer.COPY_DST});
}

export function createStorageBuffer(device: Device, values: Float32Array): Buffer {
  return device.createBuffer({
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
}

export type SceneModel = {model: Model; destroy: () => void};

/** Draws `count` instanced quads (6 vertices each) or a full-screen triangle (`count` of 1). */
export function createQuadModel(
  device: Device,
  options: {
    id: string;
    source: string;
    bindings: Record<string, Buffer>;
    instanceCount: number;
    theme: GpuTheme;
    /** `glow` adds light on dark surfaces; `opaque` replaces what is below (backgrounds, rasters). */
    blendMode?: 'glow' | 'opaque';
  }
): Model {
  return new Model(device, {
    id: options.id,
    source: options.source,
    topology: 'triangle-list',
    vertexCount: 6,
    instanceCount: Math.max(1, options.instanceCount),
    bindings: options.bindings,
    parameters:
      options.blendMode === 'opaque' ? {blend: false} : {...getBlendParameters(options.theme)}
  });
}

/** Storage buffers need WebGPU; on WebGL2 the scene throws and the page keeps its static fallback. */
export function assertWebGpu(device: Device): void {
  if (device.type !== 'webgpu') {
    // These scenes read storage buffers, which WebGL2 does not have.
    throw new Error('WebGPU required');
  }
}

/** Reduced motion renders one still frame at this time (milliseconds). */
export const STILL_FRAME_MILLISECONDS = 2600;

/** Saturating pulse in 0..1 around `center`; a small helper kept on the CPU for tests and layout. */
export function getPulse(position: number, center: number, width: number): number {
  const distance = (position - center) / width;
  return Math.exp(-distance * distance);
}
