// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The library's own validators, called at render with a value the reader controls. They are pure
// functions of their arguments, so the first error message is already in the prerendered page.

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {useState} from 'react';
import {formatCount} from '../../../../lens/format';
import './contract.css';

/** The message a validator throws for these arguments, or null when it accepts them. */
function getMessage(check: () => void): string | null {
  try {
    check();
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

// GPUNetworkServiceAreas: validateCapacity(id, 'maximumIterations', value, 1,
// GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS) in gpu-network-service-areas.ts:272-278; the
// maximum is 1024 at gpu-network-reachability.ts:53.
const MAXIMUM_ITERATIONS_LIMIT = 1024;
// GPUPointDensity: validateMaximumCount(id, 'gridSize cells', cellCount, maximum) in
// gpu-point-density.ts:302; the maximum is floor(2^27 / 12), 12 bytes per cell in one 128 MiB
// binding (gpu-point-density.ts:83-89).
const POINT_DENSITY_CELL_LIMIT = Math.floor(2 ** 27 / 12);
// GPUTerrainTiledFlow: validateRequirement(id, isSatisfied, requirement) in
// gpu-terrain-tiled-flow.ts:79-91, reduced here to one dimension.
const TILE_WIDTH = 256;

function Result({message}: {message: string | null}) {
  return (
    <p className="contract-case__result" data-state={message ? 'error' : 'ok'} aria-live="polite">
      <span className="contract-case__label">{message ? 'throws' : 'result'}</span>
      {message ?? 'accepted'}
    </p>
  );
}

export function ValidatorDemo() {
  const [iterations, setIterations] = useState(2000);
  const [columns, setColumns] = useState(4000);
  const [ownedWidth, setOwnedWidth] = useState(300);
  const rows = 3000;
  const cellCount = columns * rows;
  const iterationsMessage = getMessage(() =>
    validateCapacity(
      'network-service-areas',
      'maximumIterations',
      iterations,
      1,
      MAXIMUM_ITERATIONS_LIMIT
    )
  );
  const cellsMessage = getMessage(() =>
    validateMaximumCount('point-density', 'gridSize cells', cellCount, POINT_DENSITY_CELL_LIMIT)
  );
  const requirementMessage = getMessage(() =>
    validateRequirement(
      'terrain-tiled-flow',
      ownedWidth > 0 && ownedWidth <= TILE_WIDTH,
      'tiles[0] owned rectangle must be non-empty and inside the tile'
    )
  );
  return (
    <div className="meta-panel contract-validators">
      <div className="contract-case">
        <p className="contract-case__call">
          <b>validateCapacity</b>(id, name, value, minimum = 1, maximum = 0xffffffff)
        </p>
        <label className="contract-case__input">
          <span>maximumIterations</span>
          <input
            type="number"
            value={iterations}
            step={1}
            onChange={event => setIterations(Number(event.target.value))}
            aria-label="maximumIterations"
          />
          <input
            type="range"
            min={0}
            max={2048}
            step={1}
            value={Math.min(Math.max(iterations, 0), 2048)}
            onChange={event => setIterations(Number(event.target.value))}
            aria-label="maximumIterations slider"
          />
        </label>
        <p className="contract-case__call">
          validateCapacity('network-service-areas', 'maximumIterations', {iterations}, 1,{' '}
          {MAXIMUM_ITERATIONS_LIMIT})
        </p>
        <Result message={iterationsMessage} />
      </div>
      <div className="contract-case">
        <p className="contract-case__call">
          <b>validateMaximumCount</b>(id, name, value, maximum)
        </p>
        <label className="contract-case__input">
          <span>gridSize columns, 3,000 rows</span>
          <input
            type="number"
            value={columns}
            step={100}
            onChange={event => setColumns(Number(event.target.value))}
            aria-label="gridSize columns"
          />
          <input
            type="range"
            min={100}
            max={6000}
            step={100}
            value={Math.min(Math.max(columns, 100), 6000)}
            onChange={event => setColumns(Number(event.target.value))}
            aria-label="gridSize columns slider"
          />
        </label>
        <p className="contract-case__call">
          validateMaximumCount('point-density', 'gridSize cells', {formatCount(cellCount)},{' '}
          {formatCount(POINT_DENSITY_CELL_LIMIT)})
        </p>
        <Result message={cellsMessage} />
      </div>
      <div className="contract-case">
        <p className="contract-case__call">
          <b>validateRequirement</b>(id, isSatisfied, requirement)
        </p>
        <label className="contract-case__input">
          <span>owned width in a {TILE_WIDTH}-wide tile</span>
          <input
            type="number"
            value={ownedWidth}
            step={1}
            onChange={event => setOwnedWidth(Number(event.target.value))}
            aria-label="owned width"
          />
          <input
            type="range"
            min={0}
            max={400}
            step={1}
            value={Math.min(Math.max(ownedWidth, 0), 400)}
            onChange={event => setOwnedWidth(Number(event.target.value))}
            aria-label="owned width slider"
          />
        </label>
        <Result message={requirementMessage} />
      </div>
    </div>
  );
}
