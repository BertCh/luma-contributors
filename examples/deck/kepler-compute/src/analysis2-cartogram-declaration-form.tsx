// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type {AnalysisRuntime} from './analysis-runtime';

/**
 * Declares how a projected (non EPSG:3857) cartogram CRS sits on the map: the size of a CRS unit
 * in meters, the plane coordinates of a geographic anchor, and the anchor itself. The overlay is a
 * local tangent plane (display only, 200 km half extent); nothing is reprojected or verified.
 */
export function CartogramDeclarationForm({analysis}: {analysis: AnalysisRuntime}) {
  const current = analysis.cartogramDeclaration;
  const getDraft = () => ({
    metersPerUnit: String(current?.metersPerUnit ?? 1),
    planeX: String(current?.planeOrigin[0] ?? ''),
    planeY: String(current?.planeOrigin[1] ?? ''),
    longitude: String(current?.anchor[0] ?? ''),
    latitude: String(current?.anchor[1] ?? '')
  });
  const [draft, setDraft] = useState(getDraft);
  // A restored recipe or bundle replaces the declaration; show what is now in effect.
  useEffect(() => setDraft(getDraft()), [current]);
  const fields = [
    ['metersPerUnit', 'Meters per CRS unit'],
    ['planeX', 'Anchor plane X (CRS units)'],
    ['planeY', 'Anchor plane Y (CRS units)'],
    ['longitude', 'Anchor longitude'],
    ['latitude', 'Anchor latitude']
  ] as const;
  return (
    <fieldset className="analysis-cartogram-declaration">
      <legend>Map overlay for a declared projected CRS</legend>
      {fields.map(([key, label]) => (
        <label key={key} htmlFor={`analysis-cartogram-declared-${key}`}>
          {label}
          <input
            id={`analysis-cartogram-declared-${key}`}
            type="number"
            step="any"
            value={draft[key]}
            onChange={event => setDraft({...draft, [key]: event.target.value})}
          />
        </label>
      ))}
      <button
        type="button"
        onClick={() => {
          const numbers = Object.values(draft).map(value => (value.trim() ? Number(value) : NaN));
          analysis.setCartogramDeclaration(
            numbers.every(Number.isFinite)
              ? {
                  metersPerUnit: numbers[0],
                  planeOrigin: [numbers[1], numbers[2]],
                  anchor: [numbers[3], numbers[4]]
                }
              : undefined
          );
        }}
      >
        Apply overlay declaration
      </button>
      <p className="hint">
        Display only: a local tangent plane at the anchor, valid up to 200 km from it. Coordinates
        are not reprojected and the declaration is not verified against the source.
      </p>
    </fieldset>
  );
}
