// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Static SVG rendering of a preview. Same seeded data as the live scene, so the canvas fades in
// over a picture that already matches. Server-renderable: no DOM access, no effects.

import {useId, type ReactElement} from 'react';
import type {PreviewKind} from '../../catalog/schema.ts';
import {buildPreview} from './builders.ts';
import {createFieldContours, createFieldImageUri} from './field-image.ts';
import {PREVIEW_BACKGROUND, formatHex, getRoleCss, ROLE_COLORS} from './palette.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FieldLayer,
  type FillLayer,
  type Point,
  type PreviewLayer,
  type Role,
  type SpriteLayer,
  type StrokeLayer
} from './types.ts';

const BULK_SPRITE_COUNT = 300;

function formatNumber(value: number): string {
  return String(Math.round(value * 10) / 10);
}

function formatPoints(points: Point[]): string {
  return points.map(([x, y]) => `${formatNumber(x)},${formatNumber(y)}`).join(' ');
}

function projectTurntable(layer: SpriteLayer, x: number, y: number, z: number) {
  const turntable = layer.turntable;
  if (!turntable) return {x, y, scale: 1};
  const localX = x - SCENE_WIDTH / 2;
  const localY = y - SCENE_HEIGHT / 2;
  const cosine = Math.cos(turntable.angle);
  const sine = Math.sin(turntable.angle);
  const depth = localX * sine + z * cosine;
  const scale = 1 / (1 + depth * turntable.perspective);
  return {
    x: SCENE_WIDTH / 2 + (localX * cosine - z * sine) * scale,
    y: SCENE_HEIGHT / 2 + localY * scale,
    scale
  };
}

function renderField(layer: FieldLayer, key: number, gradientPrefix: string): ReactElement {
  const contours =
    layer.style === 'contours' || layer.style === 'expand'
      ? createFieldContours(layer, SCENE_WIDTH, SCENE_HEIGHT)
      : [];
  return (
    <g key={key}>
      <defs>
        <filter id={`${gradientPrefix}-soften`} x="0" y="0" width="100%" height="100%">
          <feGaussianBlur
            stdDeviation={layer.filter === 'nearest' ? 0 : 1.1}
            edgeMode="duplicate"
          />
        </filter>
      </defs>
      <image
        filter={`url(#${gradientPrefix}-soften)`}
        href={createFieldImageUri(layer)}
        x={0}
        y={0}
        width={SCENE_WIDTH}
        height={SCENE_HEIGHT}
        preserveAspectRatio="none"
        style={{imageRendering: layer.filter === 'nearest' ? 'pixelated' : 'auto'}}
      />
      {contours.map((contour, index) => (
        <path
          key={index}
          d={contour.path}
          fill="none"
          stroke={formatHex(contour.color)}
          strokeWidth={contour.width}
          strokeLinecap="round"
          opacity={layer.style === 'expand' ? 0.55 : 0.9}
        />
      ))}
    </g>
  );
}

function renderSprites(layer: SpriteLayer, key: number, gradientPrefix: string): ReactElement {
  if (layer.items.length > BULK_SPRITE_COUNT) {
    const pathsByRole = new Map<Role, string>();
    const ordered = layer.items
      .map(item => ({item, projected: projectTurntable(layer, item.x, item.y, item.z ?? 0)}))
      .sort((first, second) => first.projected.scale - second.projected.scale);
    for (const {item, projected} of ordered) {
      pathsByRole.set(
        item.role,
        `${pathsByRole.get(item.role) ?? ''}M${formatNumber(projected.x)} ${formatNumber(projected.y)}h0`
      );
    }
    return (
      <g key={key} style={{mixBlendMode: 'screen'}}>
        {[...pathsByRole].map(([role, path]) => (
          <path
            key={role}
            d={path}
            stroke={getRoleCss(role)}
            strokeWidth={1.5}
            strokeLinecap="round"
            opacity={0.8}
          />
        ))}
      </g>
    );
  }
  return (
    <g key={key} style={{mixBlendMode: 'screen'}}>
      {layer.items.map((item, index) => {
        const projected = projectTurntable(layer, item.x, item.y, item.z ?? 0);
        return (
          <circle
            key={index}
            cx={formatNumber(projected.x)}
            cy={formatNumber(projected.y)}
            r={formatNumber(item.radius * 2.5 * projected.scale)}
            fill={`url(#${gradientPrefix}-${item.role})`}
            opacity={item.alpha ?? 0.9}
          />
        );
      })}
    </g>
  );
}

function renderStrokes(layer: StrokeLayer, key: number): ReactElement {
  const flowing = layer.animation === 'flow';
  return (
    <g
      key={key}
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={layer.blend === 'add' ? {mixBlendMode: 'screen'} : undefined}
    >
      {layer.items.map((item, index) => (
        <polyline
          key={index}
          points={formatPoints(item.points)}
          stroke={getRoleCss(item.role)}
          strokeWidth={item.width}
          opacity={(item.alpha ?? 0.9) * (flowing ? 0.65 : 1)}
        />
      ))}
    </g>
  );
}

function renderFills(layer: FillLayer, key: number): ReactElement {
  return (
    <g key={key} style={layer.blend === 'add' ? {mixBlendMode: 'screen'} : undefined}>
      {layer.items.map((item, index) => (
        <polygon
          key={index}
          points={formatPoints(item.ring)}
          fill={getRoleCss(item.role)}
          fillOpacity={(item.alpha ?? 0.5) * (layer.animation === 'pulse' ? 0.85 : 1)}
        />
      ))}
    </g>
  );
}

function renderLayer(layer: PreviewLayer, key: number, gradientPrefix: string): ReactElement {
  switch (layer.type) {
    case 'field':
      return renderField(layer, key, gradientPrefix);
    case 'sprites':
      return renderSprites(layer, key, gradientPrefix);
    case 'strokes':
      return renderStrokes(layer, key);
    case 'fills':
      return renderFills(layer, key);
  }
}

export type PreviewFallbackProps = {
  kind: PreviewKind;
  seed?: number;
  /** Catalog entry id; draws the entry's function-specific preview when it has one. */
  entryId?: string;
};

/** Static SVG picture of a preview kind; fills its container. */
export function PreviewFallback({kind, seed = 1, entryId}: PreviewFallbackProps): ReactElement {
  const gradientPrefix = `glow${useId().replace(/:/g, '')}`;
  const data = buildPreview(kind, seed, {}, null, entryId);
  const roles = Object.keys(ROLE_COLORS) as Role[];
  return (
    <svg
      viewBox={`0 0 ${SCENE_WIDTH} ${SCENE_HEIGHT}`}
      preserveAspectRatio="xMidYMid slice"
      width="100%"
      height="100%"
      aria-hidden="true"
      focusable="false"
      style={{display: 'block', background: `var(--preview-background, ${PREVIEW_BACKGROUND})`}}
    >
      <defs>
        {roles.map(role => (
          <radialGradient key={role} id={`${gradientPrefix}-${role}`}>
            {[
              [0, 1],
              [0.28, 0.85],
              [0.4, 0.5],
              [0.6, 0.2],
              [0.8, 0.07],
              [1, 0]
            ].map(([offset, opacity]) => (
              <stop
                key={offset}
                offset={offset}
                style={{stopColor: getRoleCss(role), stopOpacity: opacity}}
              />
            ))}
          </radialGradient>
        ))}
      </defs>
      {data.layers.map((layer, index) => renderLayer(layer, index, gradientPrefix))}
    </svg>
  );
}
