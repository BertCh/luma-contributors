// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One record's parts, counted from a real view model, and what no record holds.

import {
  formatBytes,
  formatCount,
  formatResourceBytes,
  formatResourceCount
} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';
import {Term} from '../../page-kit';

function shortName(id: string): string {
  return id.replace(/^kernel-density-/, '');
}

export function RecordParts({model}: {model: LensViewModel}) {
  const compute = model.nodes.filter(node => node.kind === 'compute');
  const kernelCount = compute.reduce((total, node) => total + node.count, 0);
  const copyCount = model.nodes.filter(node => node.kind === 'copy').length;
  const hazardTotal = model.hazards.raw + model.hazards.war + model.hazards.waw;
  const busiest = [...compute]
    .filter(node => node.reads && node.writes)
    .sort(
      (left, right) => right.readBytes + right.writeBytes - (left.readBytes + left.writeBytes)
    )[0];
  const raster = model.map?.raster ? 1 : 0;
  const layers = model.map?.layers.length ?? 0;
  return (
    <div className="lenses-parts">
      <ul className="lenses-parts__held" aria-label={`What the ${model.title} record holds`}>
        <li className="lenses-part">
          <span className="lenses-part__name">Graph</span>
          <span className="lenses-part__value">
            {kernelCount} kernels, {copyCount} copy
          </span>
          <p className="lenses-part__text">
            Each kernel lists the buffers it reads and writes, its bytes moved and its variant.
          </p>
          {busiest ? (
            <p className="lenses-part__kernel">
              <b>{busiest.label}</b> reads {busiest.reads?.map(shortName).join(', ')}; writes{' '}
              {busiest.writes?.map(shortName).join(', ')}
            </p>
          ) : null}
        </li>
        <li className="lenses-part">
          <span className="lenses-part__name">
            <Term id="hazard">Dependencies</Term>
          </span>
          <span className="lenses-part__value">{formatCount(hazardTotal)}</span>
          <p className="lenses-part__text">
            One per buffer, by hazard kind: read-after-write {model.hazards.raw}, write-after-read{' '}
            {model.hazards.war}, write-after-write {model.hazards.waw}.
          </p>
        </li>
        <li className="lenses-part">
          <span className="lenses-part__name">Buffers</span>
          <span className="lenses-part__value">
            {formatResourceCount(model.resources)},{' '}
            {formatResourceBytes(model.resources.totalBytes, model.resources.totalBytesKind)}
          </span>
          <p className="lenses-part__text">
            Recorded resource ids and sizes, with ownership and size precision when available.
          </p>
        </li>
        <li className="lenses-part" data-tone="signal">
          <span className="lenses-part__name">Captured values</span>
          <span className="lenses-part__value">
            {raster} raster, {layers} point layer
          </span>
          <p className="lenses-part__text">
            Final values, copied once by the tap at the end of the run. Not one per step.
          </p>
        </li>
        <li className="lenses-part">
          <span className="lenses-part__name">Formulas</span>
          <span className="lenses-part__value">{model.formulas.length}</span>
          <p className="lenses-part__text">The math the kernels implement, as text.</p>
        </li>
        <li className="lenses-part">
          <span className="lenses-part__name">Choices</span>
          <span className="lenses-part__value">{model.parameters.length}</span>
          <p className="lenses-part__text">
            {model.parameters.map(parameter => parameter.label).join(', ')}.
          </p>
        </li>
      </ul>
      <div className="lenses-absent">
        <span className="lenses-absent__title">In no record</span>
        <ul>
          <li>GPU timings</li>
          <li>WGSL source</li>
          <li>Transient buffer aliasing</li>
        </ul>
      </div>
    </div>
  );
}

export type KernelSpread = {
  bins: readonly {label: string; count: number}[];
  median: {id: string; kernels: number};
  smallest: {id: string; kernels: number};
  largest: {id: string; kernels: number};
};

export function KernelSpreadChart({spread}: {spread: KernelSpread}) {
  const tallest = Math.max(...spread.bins.map(bin => bin.count));
  return (
    <div className="lenses-spread">
      <ol
        className="lenses-spread__bars"
        style={{'--bins': spread.bins.length} as React.CSSProperties}
        aria-label="Records by number of kernels"
      >
        {spread.bins.map(bin => (
          <li key={bin.label}>
            <span
              className="lenses-spread__bar"
              style={{height: `${(bin.count / tallest) * 100}%`}}
            />
            <span className="lenses-spread__count">{bin.count}</span>
            <span>{bin.label}</span>
          </li>
        ))}
      </ol>
      <dl className="lenses-spread__facts">
        {[
          ['Smallest', spread.smallest],
          ['Median', spread.median],
          ['Largest', spread.largest]
        ].map(([label, entry]) => {
          const item = entry as {id: string; kernels: number};
          return (
            <div key={label as string}>
              <dt>{label as string}: </dt>
              <dd>
                {item.id} ({item.kernels})
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}

export type SizeBar = {label: string; note: string; bytes: number; shipped: boolean};

/** Bars on a log scale: the sizes differ by more than three orders of magnitude. */
export function SizeBars({bars}: {bars: readonly SizeBar[]}) {
  const exponents = bars.map(bar => Math.log10(bar.bytes));
  const low = Math.floor(Math.min(...exponents));
  const high = Math.ceil(Math.max(...exponents));
  return (
    <div className="lenses-sizes-wrap">
      <ul className="lenses-sizes" aria-label="Sizes on a log scale">
        {bars.map(bar => (
          <li key={bar.label} data-shipped={bar.shipped}>
            <span className="lenses-sizes__name">
              {bar.label}
              <small>{bar.note}</small>
            </span>
            <span className="lenses-sizes__track">
              <span
                className="lenses-sizes__fill"
                style={{width: `${((Math.log10(bar.bytes) - low) / (high - low)) * 100}%`}}
              />
            </span>
            <span className="lenses-sizes__value">{formatBytes(bar.bytes)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
