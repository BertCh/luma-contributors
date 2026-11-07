// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A recorded snapshot drawn by luma.gl (live) with the same data as a static PNG and SVG (fallback).

import {Suspense, useMemo} from 'react';
import {GpuCanvas} from '../../gpu/gpu-canvas';
import {sampleValueRamp} from '../../lens/ramp';
import type {LensMap, LensViewModel} from '../../lens/types';
import {useLensViewModel} from '../../lens/use-lens-view-model';
import {createMapSceneLoader} from './scenes/map-scene';
import './meta.css';

export function SnapshotMap({map, label}: {map: LensMap; label: string}) {
  const loader = useMemo(() => createMapSceneLoader(map), [map]);
  return (
    <div className="snapshot-card__view">
      <GpuCanvas
        scene={loader}
        fallback={<SnapshotMapFallback map={map} label={label} />}
        seed={1}
        label={label}
        aspectRatio={map.aspectRatio}
      />
    </div>
  );
}

/** The same raster and points as plain markup: what prerendering and GPU-less browsers show. */
export function SnapshotMapFallback({map, label}: {map: LensMap; label: string}) {
  const width = Math.round(map.aspectRatio * 1000);
  return (
    <div
      style={{position: 'relative', aspectRatio: `${map.aspectRatio}`}}
      role="img"
      aria-label={label}
    >
      {map.raster ? (
        <img
          src={map.raster.png}
          alt=""
          style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}}
        />
      ) : null}
      <svg
        viewBox={`0 0 ${width} 1000`}
        style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}}
        aria-hidden="true"
      >
        {map.layers.map((layer, layerIndex) => {
          const count = layer.positions.length / 2;
          const radius = Math.min(11, Math.max(3, 260 / Math.sqrt(Math.max(1, count))));
          const circles = [];
          for (let index = 0; index < count; index++) {
            const value = layer.values?.[index];
            const [red, green, blue] =
              value === undefined ? [0.5, 0.55, 0.6] : sampleValueRamp(value);
            const highlighted = value === undefined && layerIndex > 0 && count <= 64;
            circles.push(
              <circle
                key={index}
                cx={layer.positions[index * 2] * width}
                cy={layer.positions[index * 2 + 1] * 1000}
                r={highlighted ? radius * 1.6 : radius}
                fill={
                  highlighted
                    ? 'var(--color-signal)'
                    : `rgb(${Math.round(red * 255)} ${Math.round(green * 255)} ${Math.round(blue * 255)})`
                }
                fillOpacity={value === undefined && !highlighted ? 0.45 : 0.95}
              />
            );
          }
          return <g key={layer.name}>{circles}</g>;
        })}
      </svg>
    </div>
  );
}

/** Loads a scenario and shows its map; renders nothing when the record has no geometry. */
export function ScenarioSnapshotMap({scenarioId}: {scenarioId: string}) {
  return (
    <Suspense fallback={<div className="meta-skeleton">loading</div>}>
      <LoadedScenarioMap scenarioId={scenarioId} />
    </Suspense>
  );
}

function LoadedScenarioMap({scenarioId}: {scenarioId: string}) {
  const model: LensViewModel | undefined = useLensViewModel(scenarioId);
  return model?.map ? <SnapshotMap map={model.map} label={`Snapshot of ${model.title}`} /> : null;
}
