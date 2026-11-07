// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  makePointCoordinateDraft,
  getPointCoordinateDraftSettings,
  type PointCoordinateSettings
} from './analysis-point-coordinate-settings';

/** UI drafts retain both coordinate choices while the parent owns validation and submission. */
export type PointCoordinateDraft = {
  coordinateSpace: 'longitude-latitude' | 'planar';
  longitudeColumn: string;
  latitudeColumn: string;
  xColumn: string;
  yColumn: string;
  crs: string;
  minX: string;
  minY: string;
  maxX: string;
  maxY: string;
};

const PROJECTED_BOUNDS = [
  {key: 'minX', label: 'minimum X'},
  {key: 'minY', label: 'minimum Y'},
  {key: 'maxX', label: 'maximum X'},
  {key: 'maxY', label: 'maximum Y'}
] as const;

/** Restore all scalar recipe fields without resetting drafts on unrelated runtime updates. */
export function usePointCoordinates(settings?: PointCoordinateSettings, resetKey?: unknown) {
  const saved = makePointCoordinateDraft(settings);
  const [draft, setDraft] = useState(() => makePointCoordinateDraft(settings));
  useEffect(() => {
    setDraft(makePointCoordinateDraft(settings));
  }, [
    resetKey,
    saved.coordinateSpace,
    saved.longitudeColumn,
    saved.latitudeColumn,
    saved.xColumn,
    saved.yColumn,
    saved.crs,
    saved.minX,
    saved.minY,
    saved.maxX,
    saved.maxY
  ]);
  return {
    draft,
    setDraft,
    getSettings: () => getPointCoordinateDraftSettings(draft)
  };
}

/** Analysis coordinates are independent of the geographic geometry used to display the map. */
export function PointCoordinateControls({
  value,
  onChange,
  columns,
  labelPrefix,
  idPrefix,
  longitudeId = `${idPrefix}-longitude`,
  latitudeId = `${idPrefix}-latitude`
}: {
  value: PointCoordinateDraft;
  onChange(value: PointCoordinateDraft): void;
  columns: readonly string[];
  labelPrefix: string;
  idPrefix: string;
  longitudeId?: string;
  latitudeId?: string;
}) {
  const isProjected = value.coordinateSpace === 'planar';
  const columnField = (
    field: 'longitudeColumn' | 'latitudeColumn' | 'xColumn' | 'yColumn',
    label: string,
    id: string
  ) => (
    <>
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        aria-label={label}
        required
        value={value[field]}
        onChange={event => onChange({...value, [field]: event.target.value})}
      >
        <option value="">Choose a prepared Float32 field</option>
        {columns.map(column => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </>
  );
  return (
    <fieldset className="analysis-point-coordinate-settings">
      <legend>{labelPrefix} analysis coordinates</legend>
      <label htmlFor={`${idPrefix}-coordinate-space`}>{labelPrefix} coordinate system</label>
      <select
        id={`${idPrefix}-coordinate-space`}
        value={value.coordinateSpace}
        onChange={event =>
          onChange({
            ...value,
            coordinateSpace: event.target.value === 'planar' ? 'planar' : 'longitude-latitude'
          })
        }
      >
        <option value="longitude-latitude">Geographic · longitude / latitude</option>
        <option value="planar">Already projected · meters</option>
      </select>
      {isProjected ? (
        <>
          {columnField('xColumn', `${labelPrefix} projected X field`, `${idPrefix}-x`)}
          {columnField('yColumn', `${labelPrefix} projected Y field`, `${idPrefix}-y`)}
          <label htmlFor={`${idPrefix}-declared-crs`}>{labelPrefix} declared projected CRS</label>
          <input
            id={`${idPrefix}-declared-crs`}
            type="text"
            required
            placeholder="For example, EPSG:32618"
            value={value.crs}
            onChange={event => onChange({...value, crs: event.target.value})}
          />
          {PROJECTED_BOUNDS.map(field => (
            <label key={field.key} htmlFor={`${idPrefix}-${field.key}`}>
              {labelPrefix} projected {field.label} in meters
              <input
                id={`${idPrefix}-${field.key}`}
                type="number"
                required
                step="any"
                value={value[field.key]}
                onChange={event => onChange({...value, [field.key]: event.target.value})}
              />
            </label>
          ))}
          <p className="hint">
            X and Y must already use the declared projected CRS with meter units. Distances are
            planar Euclidean distances in these coordinates. Declare finite bounds with maximum X
            greater than or equal to minimum X and maximum Y greater than or equal to minimum Y,
            covering the source coordinates. The CRS declaration records their meaning; it does not
            transform values or verify the source projection.
          </p>
          <p className="hint">
            Projected X/Y are analysis-only fields. Keep separate geographic coordinates or geometry
            for map display. Choosing these fields does not change map geometry, project longitude
            and latitude, or convert feet or other units to meters.
          </p>
          <p className="hint">
            Float32 storage and distance calculations limit precision, especially for large
            projected coordinate magnitudes. Nearby positions can round to the same value, and
            neighbors near a distance boundary can differ from higher-precision calculations.
            Confirm that the prepared coordinates retain the precision needed for this analysis.
          </p>
        </>
      ) : (
        <>
          {columnField('longitudeColumn', `${labelPrefix} longitude field`, longitudeId)}
          {columnField('latitudeColumn', `${labelPrefix} latitude field`, latitudeId)}
          <p className="hint">
            Longitude and latitude are degrees, within ±180° and ±90° respectively. Analysis uses
            great-circle distances on a sphere, reported in meters. Float32 coordinates and
            calculations limit distance precision, including near neighbor boundaries.
          </p>
        </>
      )}
    </fieldset>
  );
}
