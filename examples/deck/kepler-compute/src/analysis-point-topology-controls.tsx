// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {
  MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY,
  MAXIMUM_KEPLER_DISTANCE_BAND_ROWS
} from '@luma-contributors/kepler';
import type {PointWeightTopology} from './analysis-point-weights-settings';

import {clamp} from '@math.gl/core';

/** Scalar dependencies restore saved recipes without resetting unsaved controls on every render. */
export function usePointTopology(
  settings: PointWeightTopology | undefined,
  rowCount: number,
  resetKey?: unknown
) {
  const savedKind = settings?.kind === 'distance-band' ? 'distance-band' : 'knn';
  const savedNeighbors = settings && settings.kind !== 'distance-band' ? settings.k : 8;
  const savedRadius = settings?.kind === 'distance-band' ? settings.radiusMeters : 1000;
  const savedCapacity =
    settings?.kind === 'distance-band'
      ? settings.neighborCapacity
      : clamp(rowCount * 32, 1, MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY);
  const savedDegree = settings?.kind === 'distance-band' ? (settings.permutationDegree ?? 64) : 64;
  const [permutationDegree, setPermutationDegree] = useState(String(savedDegree));
  const [kind, setKind] = useState<'knn' | 'distance-band'>(savedKind);
  const [neighborCount, setNeighborCount] = useState(String(savedNeighbors));
  const [radiusMeters, setRadiusMeters] = useState(String(savedRadius));
  const [neighborCapacity, setNeighborCapacity] = useState(String(savedCapacity));
  useEffect(() => {
    setKind(savedKind);
    setNeighborCount(String(savedNeighbors));
    setRadiusMeters(String(savedRadius));
    setNeighborCapacity(String(savedCapacity));
    setPermutationDegree(String(savedDegree));
  }, [savedKind, savedNeighbors, savedRadius, savedCapacity, savedDegree, resetKey]);
  return {
    rowCount,
    kind,
    setKind,
    neighborCount,
    setNeighborCount,
    radiusMeters,
    setRadiusMeters,
    neighborCapacity,
    setNeighborCapacity,
    permutationDegree,
    setPermutationDegree,
    getSettings(): PointWeightTopology {
      return kind === 'distance-band'
        ? {
            kind,
            radiusMeters: Number(radiusMeters),
            neighborCapacity: Number(neighborCapacity),
            ...(Number(permutationDegree) === 64
              ? {}
              : {permutationDegree: Number(permutationDegree)})
          }
        : {k: Number(neighborCount)};
    }
  };
}

/** One set of explicit topology controls shared by every point-weight consumer. */
export function PointTopologyControls({
  topology,
  idPrefix,
  labelPrefix,
  nearestLabel,
  nearestId = `${idPrefix}-neighbors`,
  coordinateSpace = 'longitude-latitude',
  permutationMaximumNeighbors
}: {
  topology: ReturnType<typeof usePointTopology>;
  idPrefix: string;
  labelPrefix: string;
  nearestLabel: string;
  nearestId?: string;
  coordinateSpace?: 'longitude-latitude' | 'planar';
  permutationMaximumNeighbors?: number;
}) {
  return (
    <>
      <label htmlFor={`${idPrefix}-topology`}>{labelPrefix} topology</label>
      <select
        id={`${idPrefix}-topology`}
        value={topology.kind}
        onChange={event =>
          topology.setKind(event.target.value === 'distance-band' ? 'distance-band' : 'knn')
        }
      >
        <option value="knn">k nearest neighbors</option>
        <option
          value="distance-band"
          disabled={topology.rowCount > MAXIMUM_KEPLER_DISTANCE_BAND_ROWS}
        >
          Distance band · all neighbors within radius
        </option>
      </select>
      {topology.kind === 'knn' ? (
        <>
          <label htmlFor={nearestId}>{nearestLabel}</label>
          <input
            id={nearestId}
            type="number"
            min="1"
            max="32"
            step="1"
            required
            value={topology.neighborCount}
            onChange={event => topology.setNeighborCount(event.target.value)}
          />
        </>
      ) : (
        <>
          <label htmlFor={`${idPrefix}-radius`}>{labelPrefix} radius in meters</label>
          <input
            id={`${idPrefix}-radius`}
            type="number"
            min="0"
            step="any"
            required
            value={topology.radiusMeters}
            onChange={event => topology.setRadiusMeters(event.target.value)}
          />
          <label htmlFor={`${idPrefix}-capacity`}>{labelPrefix} directed neighbor capacity</label>
          <input
            id={`${idPrefix}-capacity`}
            type="number"
            min="1"
            max={MAXIMUM_KEPLER_DISTANCE_BAND_CAPACITY}
            step="1"
            required
            value={topology.neighborCapacity}
            onChange={event => topology.setNeighborCapacity(event.target.value)}
          />
          <p className="hint">
            {coordinateSpace === 'planar'
              ? 'A distance band includes other coordinate-valid observations at planar Euclidean distance less than or equal to the radius in meters.'
              : 'A distance band includes other coordinate-valid observations at great-circle distance less than or equal to the radius, including across the antimeridian.'}{' '}
            Coordinates and distance calculations use Float32; points close to the radius boundary
            can be included or excluded differently from higher-precision calculations.
          </p>
          <p className="hint">
            Distance bands support at most {MAXIMUM_KEPLER_DISTANCE_BAND_ROWS.toLocaleString()}{' '}
            source rows. Capacity is the total directed neighbor slots for the entire source graph.
            A symmetric pair uses two slots. If the graph exceeds capacity, results are unavailable
            and the required slot count is reported. Increase capacity and apply again; neighbors
            are not silently truncated.
          </p>
          {permutationMaximumNeighbors !== undefined && (
            <>
              <label htmlFor={`${idPrefix}-permutation-degree`}>
                {labelPrefix} permutation neighbor slots per row
              </label>
              <input
                id={`${idPrefix}-permutation-degree`}
                type="number"
                min="64"
                max="512"
                step="1"
                required
                value={topology.permutationDegree}
                onChange={event => topology.setPermutationDegree(event.target.value)}
              />
            </>
          )}
          {permutationMaximumNeighbors !== undefined && (
            <p className="hint">
              Permutation inference supports at most {topology.permutationDegree} active neighbors
              per row (raise the slots above, up to 512; cost grows quadratically). Exceeding that
              limit makes inference unavailable even when total storage capacity is sufficient.
              Reduce the radius or change the analyzed population.
            </p>
          )}
        </>
      )}
      <p className="hint">
        Topology is built from all source rows with valid coordinates. Selected-population analysis
        filters this fixed graph without adding replacement neighbors.
      </p>
    </>
  );
}
