// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import {normalizeProjectionLabel} from './resident-spatial-join-crs';

export type SpatialJoinCrsRecord = Readonly<{
  id: string;
  label: string;
  leftCrs: string;
  rightCrs: string;
}>;

export type SpatialJoinCrsChange = Partial<{leftCrs: string; rightCrs: string}>;

type SpatialJoinOperationLike = {
  id: string;
  implementation?: string;
  resultLabel?: string;
  leftCrs?: string;
  rightCrs?: string;
};

/** Pending spatial joins with their declared CRS per side (EPSG:4326 when undeclared). */
export function listSpatialJoinCrsRecords(
  joins: readonly SpatialJoinOperationLike[] | undefined
): SpatialJoinCrsRecord[] {
  return (joins ?? [])
    .filter(operation => operation.implementation === 'spatial')
    .map(operation => ({
      id: operation.id,
      label: operation.resultLabel || operation.id,
      leftCrs: operation.leftCrs ?? 'EPSG:4326',
      rightCrs: operation.rightCrs ?? 'EPSG:4326'
    }));
}

const PRESETS = [
  ['EPSG:4326', 'EPSG:4326 longitude/latitude'],
  ['EPSG:3857', 'EPSG:3857 Web Mercator']
] as const;

export type SpatialJoinCrsPanelProps = Readonly<{
  joins: readonly SpatialJoinCrsRecord[];
  setCrs(id: string, change: SpatialJoinCrsChange): void;
}>;

function CrsField({
  title,
  value,
  testId,
  onCommit
}: {
  title: string;
  value: string;
  testId: string;
  onCommit(value: string): void;
}) {
  const [text, setText] = useState(value);
  const normalized = normalizeProjectionLabel(text);
  const isPreset = PRESETS.some(([code]) => code === value);
  return (
    <div>
      <label>
        {title}{' '}
        <select
          value={isPreset ? value : 'custom'}
          data-testid={`${testId}-select`}
          onChange={event => {
            if (event.target.value === 'custom') return;
            setText(event.target.value);
            onCommit(event.target.value);
          }}
        >
          {PRESETS.map(([code, label]) => (
            <option key={code} value={code}>
              {label}
            </option>
          ))}
          <option value="custom">EPSG code or proj4 parameters</option>
        </select>
      </label>{' '}
      <input
        type="text"
        value={text}
        size={40}
        aria-label={`${title} EPSG code or proj4 parameters`}
        data-testid={`${testId}-input`}
        placeholder="EPSG:32633 or +proj=utm +zone=33 +datum=WGS84"
        onChange={event => setText(event.target.value)}
        onBlur={() => {
          if (normalized && text !== value) onCommit(text);
        }}
      />
      {normalized ? (
        <span className="hint" data-testid={`${testId}-normalized`}>
          {' '}
          {normalized === 'EPSG:4326' || normalized === 'EPSG:3857'
            ? normalized
            : 'transverse Mercator on the GPU'}
        </span>
      ) : (
        <span role="alert" data-testid={`${testId}-error`}>
          {' '}
          Unsupported CRS: use EPSG:4326, EPSG:3857, a UTM EPSG code (326NN, 327NN, 258NN, 269NN) or
          proj4 longlat/utm/tmerc parameters on a WGS84 or GRS80 ellipsoid.
        </span>
      )}
    </div>
  );
}

/**
 * Companion to Kepler's native Spatial Join panel, which has no CRS setting: declare the
 * coordinate system of the point side (left) and the polygon side (right). Points are moved into
 * the polygon CRS on the GPU; polygon vertices are never rewritten. Stored on the join operation, so
 * the native Run button executes it. Renders nothing without a spatial join.
 */
export function SpatialJoinCrsPanel({joins, setCrs}: SpatialJoinCrsPanelProps) {
  if (joins.length === 0) return null;
  return (
    <section aria-label="Spatial join coordinate systems" data-testid="spatial-join-crs-panel">
      <h2>Spatial join CRS</h2>
      {joins.map(join => (
        <fieldset key={join.id} data-testid={`spatial-join-crs-${join.id}`}>
          <legend>{join.label}</legend>
          <CrsField
            title="Point CRS (left)"
            value={join.leftCrs}
            testId="spatial-join-left-crs"
            onCommit={value => setCrs(join.id, {leftCrs: value})}
          />
          <CrsField
            title="Polygon CRS (right)"
            value={join.rightCrs}
            testId="spatial-join-right-crs"
            onCommit={value => setCrs(join.id, {rightCrs: value})}
          />
        </fieldset>
      ))}
    </section>
  );
}
