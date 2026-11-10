// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Figures of "Data on the GPU": buffers and views, polygon offsets, CSR rows, capacity and count,
// mask and validity, and float32 precision. Inline SVG and HTML cells, so the finished picture
// prerenders and needs no GPU. Toy data is fixed in toy-data.ts.

import {useState} from 'react';
import {formatCount} from '../../../../lens/format';
import type {LensViewModel} from '../../../../lens/types';
import {MissingRun} from '../../page-kit';
import {TOY_FEATURES, TOY_GRAPH, TOY_POLYGON_LAYER} from './toy-data';
import './data-on-the-gpu.css';

// Buffers and views

type BufferSpec = {
  id: string;
  format: 'float32x2' | 'float32';
  /** What one row holds, one label per float32. */
  components: readonly string[];
  rowName: string;
};

// Row sizes follow node_modules/@luma.gl/graph/dist/gpu-data/gpu-vector-format.d.ts:
// float32 is 4 bytes, float32x2 is 8. Buffer sizes come from the recorded run.
const BUFFER_SPECS: readonly BufferSpec[] = [
  {id: 'observations', format: 'float32x2', components: ['x', 'y'], rowName: 'points'},
  {id: 'density', format: 'float32', components: ['v'], rowName: 'cells'},
  {id: 'kde-parameters', format: 'float32', components: ['p'], rowName: 'values'}
];
const FLOAT32_BYTES = 4;
const SHOWN_ROWS = 4;

export function BufferViewsFigure({model}: {model: LensViewModel | undefined}) {
  const sizes = model?.bufferBytes;
  if (!model || !sizes) return <MissingRun scenarioId="gpu-spatial-analysis.kernel-density" />;
  return (
    <div className="meta-panel dotg-buffers">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">Three buffers of one recorded run</span>
      </div>
      <div className="meta-scroll">
        <ul className="dotg-buffers__list" style={{minWidth: 520}}>
          {BUFFER_SPECS.map(spec => {
            const byteLength = sizes[spec.id];
            if (byteLength === undefined) return null;
            const rowBytes = spec.components.length * FLOAT32_BYTES;
            const rowCount = byteLength / rowBytes;
            const shown = Math.min(rowCount, SHOWN_ROWS);
            return (
              <li key={spec.id} className="dotg-buffer">
                <div className="dotg-buffer__head">
                  <b>{spec.id}</b>
                  <span>{spec.format}</span>
                </div>
                <div className="dotg-buffer__strip" aria-hidden="true">
                  {Array.from({length: shown}, (_, row) => (
                    <span key={row} className="dotg-buffer__row">
                      <span className="dotg-buffer__cells">
                        {spec.components.map(component => (
                          <span key={component} className="dotg-buffer__cell">
                            {component}
                          </span>
                        ))}
                      </span>
                      <span className="dotg-buffer__offset">{row * rowBytes}</span>
                    </span>
                  ))}
                  {rowCount > shown ? <span className="dotg-buffer__more">...</span> : null}
                </div>
                <div className="dotg-buffer__sum">
                  {formatCount(rowCount)} {spec.rowName} x {rowBytes} B ={' '}
                  <b>{formatCount(byteLength)} B</b>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

// Offset arrays

type OffsetArrayProps = {
  name: string;
  type: string;
  values: readonly string[];
  /** The cells of the selected row's slice: start inclusive, end exclusive. */
  start: number;
  end: number;
  /** Whether the entry at `end` is the closing bound of the slice. */
  closed?: boolean;
  note: string;
};

function OffsetArray({name, type, values, start, end, closed = true, note}: OffsetArrayProps) {
  return (
    <div className="dotg-array">
      <div className="dotg-array__label">
        <b>{name}</b>
        <span>
          {type} x {values.length}
        </span>
        <span className="dotg-array__note">{note}</span>
      </div>
      <div className="dotg-array__cells">
        {values.map((value, index) => {
          const state =
            index >= start && index < end ? 'slice' : closed && index === end ? 'bound' : 'none';
          return (
            <span key={index} className="dotg-cell" data-state={state}>
              <span className="dotg-cell__index">{index}</span>
              <span className="dotg-cell__value">{value}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

function SelectorBar({
  label,
  options,
  selected,
  onSelect
}: {
  label: string;
  options: readonly string[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="meta-controls">
      <span>{label}</span>
      <div className="meta-toggle" role="group" aria-label={label}>
        {options.map((option, index) => (
          <button
            key={option}
            type="button"
            aria-pressed={index === selected}
            onClick={() => onSelect(index)}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}

// Polygons

const MAP_SCALE_X = 13;
const MAP_SCALE_Y = 15;
const MAP_LEFT = 12;
const MAP_BOTTOM = 160;

function toMapX(x: number): number {
  return MAP_LEFT + x * MAP_SCALE_X;
}
function toMapY(y: number): number {
  return MAP_BOTTOM - y * MAP_SCALE_Y;
}

export function PolygonOffsetsFigure() {
  const [selected, setSelected] = useState(1);
  const layer = TOY_POLYGON_LAYER;
  const firstPolygon = layer.featureOffsets[selected] ?? 0;
  const endPolygon = layer.featureOffsets[selected + 1] ?? 0;
  const firstRing = layer.polygonOffsets[firstPolygon] ?? 0;
  const endRing = layer.polygonOffsets[endPolygon] ?? 0;
  const firstVertex = layer.ringOffsets[firstRing] ?? 0;
  const endVertex = layer.ringOffsets[endRing] ?? 0;
  const range = (from: number, to: number) => `[${from}, ${to})`;
  const polygonRings = new Map<number, number[]>();
  for (const ring of layer.rings) {
    polygonRings.set(ring.polygon, [...(polygonRings.get(ring.polygon) ?? []), ring.ring]);
  }
  return (
    <div className="meta-panel dotg-panel">
      <SelectorBar
        label="Feature"
        options={TOY_FEATURES.map((feature, index) => `${index} ${feature.label}`)}
        selected={selected}
        onSelect={setSelected}
      />
      <div className="meta-scroll">
        <svg
          className="dotg-map"
          viewBox="0 0 520 190"
          role="img"
          aria-label={`Three toy features: a simple polygon, a polygon with a hole and a multipolygon of two triangles. Feature ${selected} is selected: ${endPolygon - firstPolygon} polygon${endPolygon - firstPolygon === 1 ? '' : 's'}, ${endRing - firstRing} ring${endRing - firstRing === 1 ? '' : 's'}, ${endVertex - firstVertex} vertices.`}
          style={{minWidth: 460}}
        >
          {TOY_FEATURES.map((feature, featureIndex) => {
            const isSelected = featureIndex === selected;
            const firstFeaturePolygon = layer.featureOffsets[featureIndex] ?? 0;
            const centerX =
              feature.polygons.flat(2).reduce((sum, point) => sum + point[0], 0) /
              feature.polygons.flat(2).length;
            return (
              <g key={feature.label} className="dotg-feature" data-selected={isSelected}>
                {feature.polygons.map((polygon, polygonIndex) => {
                  const fillPath = polygon
                    .map(
                      ring =>
                        `${ring.map((point, i) => `${i === 0 ? 'M' : 'L'}${toMapX(point[0])} ${toMapY(point[1])}`).join(' ')} Z`
                    )
                    .join(' ');
                  return (
                    <g key={polygonIndex}>
                      <path d={fillPath} fillRule="evenodd" className="dotg-feature__fill" />
                      {(polygonRings.get(firstFeaturePolygon + polygonIndex) ?? []).map(
                        (ringIndex, ringPosition) => {
                          const ring = polygon[ringPosition];
                          if (!ring) return null;
                          return (
                            <path
                              key={ringIndex}
                              d={`${ring.map((point, i) => `${i === 0 ? 'M' : 'L'}${toMapX(point[0])} ${toMapY(point[1])}`).join(' ')} Z`}
                              className="dotg-feature__ring"
                              data-hole={ringPosition > 0}
                            />
                          );
                        }
                      )}
                    </g>
                  );
                })}
                <text
                  x={toMapX(centerX)}
                  y={182}
                  textAnchor="middle"
                  className="dotg-feature__label"
                >
                  feature {featureIndex}
                </text>
              </g>
            );
          })}
          {layer.rings.map(ring => {
            if (ring.feature !== selected) return null;
            const points = layer.positions.slice(ring.start, ring.end);
            const centerX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
            const centerY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
            return points.map((point, offset) => {
              const dx = point[0] - centerX;
              const dy = point[1] - centerY;
              const length = Math.hypot(dx, dy) || 1;
              const x = toMapX(point[0]);
              const y = toMapY(point[1]);
              return (
                <g key={ring.start + offset}>
                  <circle cx={x} cy={y} r={3.5} className="dotg-vertex" />
                  <text
                    x={x + (dx / length) * 13}
                    y={y - (dy / length) * 13 + 4}
                    textAnchor="middle"
                    className="dotg-vertex__label"
                  >
                    {ring.start + offset}
                  </text>
                </g>
              );
            });
          })}
        </svg>
      </div>
      <div className="meta-scroll dotg-arrays">
        <div className="dotg-arrays__inner" style={{minWidth: 800}}>
          <OffsetArray
            name="featureOffsets"
            type="uint32"
            values={layer.featureOffsets.map(String)}
            start={selected}
            end={selected + 1}
            note={`feature ${selected}: polygons ${range(firstPolygon, endPolygon)}`}
          />
          <OffsetArray
            name="polygonOffsets"
            type="uint32"
            values={layer.polygonOffsets.map(String)}
            start={firstPolygon}
            end={endPolygon}
            note={`polygons ${range(firstPolygon, endPolygon)}: rings ${range(firstRing, endRing)}`}
          />
          <OffsetArray
            name="ringOffsets"
            type="uint32"
            values={layer.ringOffsets.map(String)}
            start={firstRing}
            end={endRing}
            note={`rings ${range(firstRing, endRing)}: vertices ${range(firstVertex, endVertex)}`}
          />
          <OffsetArray
            name="positions"
            type="float32x2"
            values={layer.positions.map(point => `${point[0]},${point[1]}`)}
            start={firstVertex}
            end={endVertex}
            closed={false}
            note={`${endVertex - firstVertex} points`}
          />
        </div>
      </div>
    </div>
  );
}

// CSR rows

export function CsrFigure() {
  const [selected, setSelected] = useState(2);
  const graph = TOY_GRAPH;
  const start = graph.offsets[selected] ?? 0;
  const end = graph.offsets[selected + 1] ?? 0;
  const neighborSet = new Set(graph.neighbors.slice(start, end));
  return (
    <div className="meta-panel dotg-panel">
      <SelectorBar
        label="Node"
        options={graph.nodes.map((_, index) => String(index))}
        selected={selected}
        onSelect={setSelected}
      />
      <svg
        className="dotg-graph"
        viewBox="0 0 290 110"
        role="img"
        aria-label={`A graph of six nodes. Node ${selected} is selected and has ${end - start} neighbours.`}
      >
        {graph.edges.map(([from, to]) => {
          const a = graph.nodes[from];
          const b = graph.nodes[to];
          if (!a || !b) return null;
          const lit =
            (from === selected && neighborSet.has(to)) ||
            (to === selected && neighborSet.has(from));
          return (
            <line
              key={`${from}-${to}`}
              x1={a[0]}
              y1={a[1]}
              x2={b[0]}
              y2={b[1]}
              className="dotg-edge"
              data-lit={lit}
            />
          );
        })}
        {graph.nodes.map((node, index) => (
          <g
            key={index}
            className="dotg-node"
            data-state={index === selected ? 'self' : neighborSet.has(index) ? 'neighbor' : 'none'}
          >
            <circle cx={node[0]} cy={node[1]} r={14} />
            <text x={node[0]} y={node[1] + 4.5} textAnchor="middle">
              {index}
            </text>
          </g>
        ))}
      </svg>
      <div className="meta-scroll dotg-arrays">
        <div className="dotg-arrays__inner" style={{minWidth: 800}}>
          <OffsetArray
            name="offsets"
            type="uint32"
            values={graph.offsets.map(String)}
            start={selected}
            end={selected + 1}
            note={`node ${selected}: neighbors ${`[${start}, ${end})`}`}
          />
          <OffsetArray
            name="neighbors"
            type="uint32"
            values={graph.neighbors.map(String)}
            start={start}
            end={end}
            closed={false}
            note={`${end - start} neighbours`}
          />
          <OffsetArray
            name="weights"
            type="float32"
            values={graph.weights.map(weight => weight.toFixed(2))}
            start={start}
            end={end}
            closed={false}
            note="aligned with neighbors"
          />
        </div>
      </div>
    </div>
  );
}

// Capacity

const SLOT_COUNT = 24;
const INITIAL_CAPACITY = 12;
const INITIAL_FOUND = 17;

export function CapacityFigure() {
  const [found, setFound] = useState(INITIAL_FOUND);
  const [capacity, setCapacity] = useState(INITIAL_CAPACITY);
  const count = Math.min(found, capacity);
  const overflow = found > capacity ? 1 : 0;
  return (
    <div className="meta-panel dotg-panel dotg-capacity" data-overflow={overflow}>
      <div className="meta-controls">
        <label className="meta-slider meta-slider--wide">
          <span>Rows the operation finds</span>
          <input
            type="range"
            min={0}
            max={SLOT_COUNT}
            value={found}
            onChange={event => setFound(Number(event.target.value))}
          />
          <b>{found}</b>
        </label>
        <button
          type="button"
          className="meta-play"
          disabled={overflow === 0}
          onClick={() => setCapacity(Math.max(1, found))}
        >
          Rebuild with capacity = requiredCount
        </button>
        <button type="button" className="meta-play" onClick={() => setCapacity(INITIAL_CAPACITY)}>
          Reset capacity
        </button>
      </div>
      <div className="meta-scroll">
        <div
          className="dotg-slots"
          style={{minWidth: 560}}
          role="img"
          aria-label={`A buffer of ${capacity} slots. The operation finds ${found} rows: count is ${count}, requiredCount is ${found}, overflow is ${overflow}.`}
        >
          {Array.from({length: SLOT_COUNT}, (_, index) => {
            const state =
              index < count
                ? 'written'
                : index < capacity
                  ? 'free'
                  : index < found
                    ? 'dropped'
                    : 'outside';
            return (
              <span
                key={index}
                className="dotg-slot"
                data-state={state}
                data-edge={index === capacity ? 'true' : undefined}
              />
            );
          })}
        </div>
        <div className="dotg-slots__legend" style={{minWidth: 560}}>
          <span>ids buffer: {capacity} rows, fixed when the graph is built</span>
          {overflow ? (
            <span className="dotg-slots__dropped">dropped rows: {found - capacity}</span>
          ) : null}
        </div>
      </div>
      <dl className="dotg-status" aria-live="polite">
        <div>
          <dt>status.count</dt>
          <dd>{count}</dd>
        </div>
        <div>
          <dt>status.requiredCount</dt>
          <dd>{found}</dd>
        </div>
        <div data-flag={overflow}>
          <dt>status.overflow</dt>
          <dd>{overflow}</dd>
        </div>
      </dl>
    </div>
  );
}

// Mask and validity

const MASK_VALUES = [4.2, 7.9, 1.3, 9.1, 5.5, 2.8, 6.4, 3.7];
const INITIAL_MASK = [1, 1, 0, 1, 1, 0, 1, 1];

export function MaskValidityFigure() {
  const [mask, setMask] = useState(INITIAL_MASK);
  return (
    <div className="meta-panel dotg-panel dotg-mask">
      <div className="meta-scroll">
        <div className="dotg-mask__grid" style={{minWidth: 560}}>
          <MaskRow label="values" detail="float32">
            {MASK_VALUES.map((value, index) => (
              <span key={index} className="dotg-mask__cell">
                {value.toFixed(1)}
              </span>
            ))}
          </MaskRow>
          <MaskRow label="mask" detail="uint32, in">
            {mask.map((flag, index) => (
              <button
                key={index}
                type="button"
                className="dotg-mask__cell dotg-mask__toggle"
                aria-pressed={flag !== 0}
                aria-label={`Row ${index} mask`}
                data-flag={flag}
                onClick={() => setMask(mask.map((other, i) => (i === index ? 1 - other : other)))}
              >
                {flag}
              </button>
            ))}
          </MaskRow>
          <div className="dotg-mask__kernel">kernel: reads rows where mask is nonzero</div>
          <MaskRow label="validity" detail="uint32, out">
            {mask.map((flag, index) => (
              <span key={index} className="dotg-mask__cell" data-flag={flag}>
                {flag}
              </span>
            ))}
          </MaskRow>
          <MaskRow label="result" detail="float32">
            {mask.map((flag, index) => (
              <span key={index} className="dotg-mask__cell" data-valid={flag !== 0}>
                {flag !== 0 ? ((MASK_VALUES[index] ?? 0) * 2).toFixed(1) : '-'}
              </span>
            ))}
          </MaskRow>
        </div>
      </div>
    </div>
  );
}

function MaskRow({
  label,
  detail,
  children
}: {
  label: string;
  detail: string;
  children: React.ReactNode;
}) {
  return (
    <div className="dotg-mask__row">
      <div className="dotg-mask__label">
        <b>{label}</b>
        <span>{detail}</span>
      </div>
      <div className="dotg-mask__cells">{children}</div>
    </div>
  );
}

// Precision

// Mean Earth radius: modules/analysis/src/analysis-core/contracts/earth.ts:9
// (GPU_EARTH_MEAN_RADIUS_METERS), the radius docs/api/gpu-spatial-analysis/concepts.md uses.
const EARTH_RADIUS_METERS = 6371008.8;
// Worst-case float32 longitude spacing at |longitude| >= 128 degrees:
// modules/analysis/src/analysis-core/contracts/spatial-context.ts:448 (getGPULongitudeLatitudeQuantum).
const WORST_CASE_STEP_DEGREES = 2 ** -16;
// modules/analysis/src/analysis-core/contracts/spatial-context.ts:464.
const SHORT_DISTANCE_FACTOR = 10;
// Combined east-west and north-south quanta, docs/api/gpu-spatial-analysis/concepts.md:172-179.
const PLACE_QUANTA: readonly {place: string; meters: number}[] = [
  {place: 'London', meters: 0.42},
  {place: 'Chicago', meters: 0.76},
  {place: 'NYC', meters: 0.77},
  {place: 'San Francisco', meters: 0.79},
  {place: 'Sydney', meters: 1.47}
];
// A toy local-frame coordinate, in metres.
const LOCAL_FRAME_EXAMPLE_METERS = 500;

/** Spacing of float32 values in the binade of `value`. */
function getFloat32Spacing(value: number): number {
  return 2 ** (Math.floor(Math.log2(Math.abs(value))) - 23);
}

export function PrecisionFigure() {
  const worstCaseMeters = ((WORST_CASE_STEP_DEGREES * Math.PI) / 180) * EARTH_RADIUS_METERS;
  const warnMeters = SHORT_DISTANCE_FACTOR * worstCaseMeters;
  const localStepMeters = getFloat32Spacing(LOCAL_FRAME_EXAMPLE_METERS);
  const places = [...PLACE_QUANTA, {place: 'Equator, near 180 degrees', meters: worstCaseMeters}];
  const maximum = Math.max(...places.map(place => place.meters));
  const rulerWidth = 480;
  const metersToPixels = rulerWidth / warnMeters;
  return (
    <div className="meta-panel dotg-panel dotg-precision">
      <div className="meta-scroll">
        <div className="dotg-precision__body" style={{minWidth: 560}}>
          <div className="dotg-precision__part">
            <h3 className="dotg-precision__title">Smallest step of a float32 degree</h3>
            <ul className="dotg-bars">
              {places.map(place => (
                <li key={place.place}>
                  <span className="dotg-bars__name">{place.place}</span>
                  <span className="dotg-bars__track">
                    <span
                      className="dotg-bars__fill"
                      style={{width: `${(place.meters / maximum) * 100}%`}}
                    />
                  </span>
                  <span className="dotg-bars__value">{place.meters.toFixed(2)} m</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="dotg-precision__part">
            <h3 className="dotg-precision__title">The same {warnMeters.toFixed(2)} m, two ways</h3>
            <svg
              viewBox={`0 0 ${rulerWidth + 24} 112`}
              role="img"
              aria-label={`A ${warnMeters.toFixed(2)} metre span. In float32 degrees it holds ${SHORT_DISTANCE_FACTOR} steps of ${worstCaseMeters.toFixed(2)} metres. In a local frame near ${LOCAL_FRAME_EXAMPLE_METERS} metres the step is ${(localStepMeters * 1000).toFixed(2)} millimetres.`}
              className="dotg-ruler"
            >
              <text x={12} y={14} className="dotg-ruler__label">
                float32 degrees: positions snap to these ticks
              </text>
              <rect x={12} y={28} width={rulerWidth} height={18} className="dotg-ruler__bar" />
              {Array.from({length: SHORT_DISTANCE_FACTOR + 1}, (_, index) => (
                <line
                  key={index}
                  x1={12 + index * worstCaseMeters * metersToPixels}
                  x2={12 + index * worstCaseMeters * metersToPixels}
                  y1={22}
                  y2={52}
                  className="dotg-ruler__tick"
                />
              ))}
              <text x={12} y={68} className="dotg-ruler__value">
                {SHORT_DISTANCE_FACTOR} steps of {worstCaseMeters.toFixed(2)} m
              </text>
              <text x={12} y={90} className="dotg-ruler__label">
                local frame, metres near {LOCAL_FRAME_EXAMPLE_METERS}: a step is{' '}
                {(localStepMeters * 1000).toFixed(2)} mm
              </text>
              <rect x={12} y={96} width={rulerWidth} height={10} className="dotg-ruler__fine" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
