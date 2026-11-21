// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  isUnknownCaptureValue,
  getInterpretedCaptureValue,
  isUncertifiedRasterJoinCapture
} from './capture-missing-values';
import {resolveCaptureCompanion, resolvePointCaptureWindow} from './capture-scope';
import {renderRasterImage, getRasterSamplingNote} from '../encoding/raster-image';
import type {
  GPUComputationCapture,
  GPUComputationCaptureGeometry
} from '@luma-contributors/analysis/computation-record';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import {h, findViewNodes, viewNodeToDom, type ViewElementNode} from '../view-node';
import {groupTableColumns} from './table-column-groups';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {crispnessDefs} from '../encoding/crispness';
import {getSelectionInk} from '../encoding/selection-ink';
import {
  CARD_FIGURE,
  CARD_COLORS,
  getCardMarkAttributes,
  getCardPaletteColors,
  getCardPaletteForVariable,
  type CardRole,
  type CardMarkShape,
  type CardPalette
} from '../encoding/card-tokens';
import type {ResolvedScale} from '../scales/scale-registry';
import {
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureRowCount,
  getCaptureFirstKey,
  getCaptureValue,
  getPinnedEntity,
  getDatumKeySpace
} from './record-queries';
import {getFinalStateCapture, resolveValueScale} from './value-strip';
import {createGeometryProjection, type GeometryProjection} from './geometry-projection';

type SampledGeometryFeature = {row: number; vertices: [number, number][]};
type DrawableGeometryFeature = SampledGeometryFeature & {minimumVertices: number};
type GeometryFeatureSample = {features: SampledGeometryFeature[]; note: string; fallback?: string};

type ResultPanel = {
  capture: GPUComputationCapture;
  /** Row domain for rendered features, which may differ from coordinate vertex keys. */
  keySpace?: string;
  geometry?: GPUComputationCaptureGeometry;
  role: 'input' | 'output' | 'context';
  points: [number, number][];
  offsets?: GPUComputationCapture;
  targets?: GPUComputationCapture;
  values?: GPUComputationCapture;
  count: number | 'unknown';
  noun: 'points' | 'lines' | 'polygons' | 'cells' | 'nodes' | 'values' | 'rows';
  blockSize: number;
  notes: string[];
  unknown?: string;
  palette: CardPalette;
  scale?: ResolvedScale;
  categories?: number[];
  featureSample?: GeometryFeatureSample;
};

/** Final captures, drawable feature samples and their single shared native-space projection. */
export type ResultViewModel = {
  panels: ResultPanel[];
  output: ResultPanel;
  projection: GeometryProjection;
  layout: 'side-by-side' | 'overlay';
  thumbnail: boolean;
  width: number;
  height: number;
  grammar: 'geometry' | 'strips' | 'table';
  frames: ResultPanel[];
};

function getResultCaptures(record: LensRecord): GPUComputationCapture[] {
  const names = [...new Set(getAllCaptures(record).map(capture => capture.name))];
  return names
    .map(name => getFinalStateCapture(record, name)!)
    .filter(capture => ['input', 'output', 'context'].includes(capture.role ?? ''));
}

/** Points per capture, so re-renders (scrubbing, highlight passes) do not rematerialise them. */
const pointCache = new WeakMap<
  GPUComputationCapture,
  {values: GPUComputationCapture['values']; components: number; points: [number, number][]}
>();

function getPoints(capture: GPUComputationCapture | undefined): [number, number][] {
  if (!capture) return [];
  const components = capture.components ?? 2;
  const cached = pointCache.get(capture);
  if (cached && cached.values === capture.values && cached.components === components) {
    return cached.points;
  }
  const points = Array.from(
    {length: Math.floor(capture.values.length / components)},
    (_, index): [number, number] => [
      Number(capture.values[index * components]),
      Number(capture.values[index * components + 1])
    ]
  );
  pointCache.set(capture, {values: capture.values, components, points});
  return points;
}

function getRole(capture: GPUComputationCapture): ResultPanel['role'] {
  return capture.role === 'context' ? 'context' : capture.role === 'input' ? 'input' : 'output';
}

function buildPanel(capture: GPUComputationCapture, context: ViewContext): ResultPanel {
  // SVG raster placement currently supports square axis-aligned cells. Preserve the captured
  // affine in the record, but state pixel space rather than inventing square ground footprints.
  let geometry = capture.geometry;
  let pixelSpace = false;
  if (geometry?.kind === 'raster' && geometry.affine) {
    const [columnX, rowX, originX, columnY, rowY, originY] = geometry.affine;
    if (columnX > 0 && rowX === 0 && columnY === 0 && Math.abs(rowY) === columnX) {
      const shift = geometry.pixelInterpretation === 'point' ? -0.5 : 0;
      geometry = {
        ...geometry,
        cellSize: columnX,
        rowDirection: rowY > 0 ? 'north' : 'south',
        origin: [originX + shift * columnX, originY + shift * rowY]
      };
    } else {
      pixelSpace = true;
      geometry = {
        ...geometry,
        coordinateModel: 'projected',
        unit: 'pixels',
        origin: undefined,
        cellSize: 1,
        rowDirection: 'south'
      };
    }
  }
  const resolve = (reference?: string) =>
    resolveCaptureCompanion(context.record, capture, reference);
  const offsets =
    geometry && 'offsetsCapture' in geometry ? resolve(geometry.offsetsCapture) : undefined;
  const targets = geometry?.kind === 'graph' ? resolve(geometry.targetsCapture) : undefined;
  const resolvedValues =
    !geometry || geometry.kind === 'raster'
      ? capture
      : geometry.valuesCapture !== undefined
        ? resolve(geometry.valuesCapture)
        : geometry.kind === 'graph' && capture.values.length
          ? capture
          : undefined;
  const uncertifiedJoin =
    resolvedValues && isUncertifiedRasterJoinCapture(context.record, resolvedValues, capture);
  const values = uncertifiedJoin ? undefined : resolvedValues;
  let points =
    geometry?.kind === 'graph'
      ? getPoints(resolve(geometry.layoutCapture))
      : geometry && geometry.kind !== 'raster'
        ? getPoints(capture)
        : [];
  let pointCount: number | 'unknown' = points.length;
  if (geometry?.kind === 'points' && geometry.countCapture !== undefined) {
    const window = resolvePointCaptureWindow(context.record, capture);
    if (window) {
      pointCount = window.count;
      points = points.slice(0, Math.max(0, window.end - window.first));
    } else {
      pointCount = 'unknown';
      points = [];
    }
  }
  const palette = getCardPaletteForVariable(
    context,
    values?.name ?? capture.name,
    values?.meaning ? values : capture
  );
  const panel: ResultPanel = {
    capture,
    keySpace:
      geometry?.kind === 'lines' || geometry?.kind === 'polygons'
        ? geometry.featureKeySpace
        : capture.keySpace,
    geometry,
    role: getRole(capture),
    points,
    offsets,
    targets,
    values,
    palette,
    scale: values ? resolveValueScale(context, values.name, values)?.scale : undefined,
    count: !geometry
      ? getCaptureRowCount(capture)
      : geometry.kind === 'raster'
        ? geometry.shape[0] * geometry.shape[1]
        : geometry.kind === 'points'
          ? pointCount
          : offsets
            ? Math.max(0, offsets.values.length - 1)
            : 'unknown',
    noun: !geometry
      ? (capture.components ?? 1) > 1
        ? 'rows'
        : 'values'
      : geometry.kind === 'raster'
        ? 'cells'
        : geometry.kind === 'graph'
          ? 'nodes'
          : geometry.kind,
    blockSize: 1,
    notes: pixelSpace
      ? ['Ground affine recorded; rectangular or rotated placement is shown in pixel space.']
      : []
  };
  if (pointCount === 'unknown') panel.unknown = 'active point count capture missing or invalid';
  if (uncertifiedJoin)
    panel.notes.push('unknown — raster join assignments are uncertified for this graph and frame');
  if (palette === 'nom' && values) {
    const frequencies = new Map<number, number>();
    const finalValues =
      geometry?.kind === 'points' && geometry.countCapture !== undefined
        ? points.map(
            (_, index) => getCaptureValue(values, getCaptureFirstKey(capture) + index) ?? NaN
          )
        : (getFinalStateCapture(context.record, values.name)?.values ?? values.values);
    for (const value of Array.from(finalValues)) {
      if (Number.isFinite(value)) frequencies.set(value, (frequencies.get(value) ?? 0) + 1);
    }
    panel.categories = [...frequencies.keys()];
    if (panel.categories.length > 5) {
      panel.notes.push(`${panel.categories.length} categories; 5 largest coloured`);
      panel.categories.sort((left, right) => frequencies.get(right)! - frequencies.get(left)!);
      panel.categories = panel.categories.slice(0, 5);
    }
  }
  if (points.length && (geometry?.kind === 'points' || geometry?.kind === 'graph'))
    panel.notes.push(
      `${points.length} source vertices/points; stride decimation to the figure budget (400 vertices, 64 in thumbnails, shared across panels and frames)`
    );
  if (!geometry) {
    const keySpace = context.record.axes.keySpaces.find(space => space.id === capture.keySpace);
    if (
      keySpace?.size !== undefined &&
      getCaptureFirstKey(capture) + getCaptureRowCount(capture) > keySpace.size
    ) {
      panel.unknown = `capture row range exceeds key space ${keySpace.id} (${keySpace.size} rows); component layout must be recorded, values withheld`;
      panel.count = 'unknown';
    }
    if (getCaptureRowCount(capture) > CARD_FIGURE.maxStripCells)
      panel.notes.push(`${getCaptureRowCount(capture)} ${panel.noun}, 16 shown`);
    return panel;
  }
  if (geometry.kind === 'raster') {
    const limit = context.options?.['size'] === 'thumbnail' ? 64 : 160;
    panel.blockSize = Math.max(1, Math.ceil(Math.max(...geometry.shape) / limit));
    if (panel.blockSize > 1)
      panel.notes.push(getRasterSamplingNote(...geometry.shape, limit, panel.palette === 'nom'));
    panel.notes.push(
      geometry.cellSize !== undefined
        ? `cell size ${geometry.cellSize} ${geometry.unit ?? '(unit unknown)'}`
        : 'cell size unknown'
    );
    if (!geometry.origin) panel.notes.push('Raster pixel grid; origin not recorded');
  }
  if ((geometry.kind === 'lines' || geometry.kind === 'polygons') && !offsets)
    panel.unknown = 'offsets capture missing';
  if ('valuesCapture' in geometry && geometry.valuesCapture && !values)
    panel.notes.push('unknown — values capture missing');
  if (geometry.kind === 'graph') {
    if (!points.length)
      panel.notes.push('no layout recorded: adjacency table instead of a node-link drawing');
    if (!offsets || !targets) panel.unknown = 'CSR offsets or targets capture missing';
    if (points.length && targets && targets.values.length > 2000)
      panel.notes.push(`${targets.values.length} edges; stride decimation for the figure budget`);
  }
  return panel;
}

/** Picks up to `limit` evenly spaced values, keeping the first and last; a single sample is the last value. */ export function sampleEvenly<
  Value
>(values: Value[], limit: number): Value[] {
  if (values.length <= limit) return values;
  // Prefer the final state when only one sample is requested.
  if (limit < 2) return values.slice(values.length - Math.max(limit, 0));
  return Array.from(
    {length: limit},
    (_, index) => values[Math.round((index * (values.length - 1)) / (limit - 1))]
  );
}

/** Builds the final input → output comparison, retaining recorded geometry and feature order. */
export function buildResultViewModel(context: ViewContext): ResultViewModel {
  const captures = getResultCaptures(context.record);
  const primaryOutputCaptureId = context.options?.['primaryOutputCaptureId'];
  const outputCapture =
    primaryOutputCaptureId === undefined
      ? captures.filter(capture => capture.role === 'output').at(-1)
      : captures.find(
          capture => capture.role === 'output' && capture.id === primaryOutputCaptureId
        );
  if (!outputCapture)
    throw new Error('result-view: primaryOutputCaptureId must identify a final output capture');
  const panels = ['context', 'input', 'output'].flatMap(role =>
    captures
      .filter(capture => getRole(capture) === role)
      .map(capture => buildPanel(capture, context))
  );
  const output = panels.find(panel => panel.capture === outputCapture)!;
  const rolePanels = panels.filter(panel => panel.role !== 'context');
  const arrays = rolePanels.filter(panel => !panel.geometry);
  const loneAnimatedArray =
    arrays.length === 1 &&
    getAllCaptures(context.record).filter(
      capture => capture.name === arrays[0].capture.name && capture.iteration !== undefined
    ).length > 1;
  const table =
    !loneAnimatedArray &&
    arrays.some(panel => panel.capture.keySpace || panel.capture.meaning?.startsWith('column'));
  const grammar = output.geometry ? 'geometry' : table ? 'table' : 'strips';
  if (grammar === 'table' || grammar === 'geometry')
    for (const panel of arrays) {
      panel.noun = 'rows';
      panel.notes = panel.notes.filter(note => !/^(\d+) (values|rows), 16 shown$/.test(note));
    }
  const thumbnail = context.options?.['size'] === 'thumbnail';
  const sameSupport = rolePanels.some(
    panel =>
      panel.role === 'input' &&
      panel.geometry?.kind === output.geometry?.kind &&
      panel.count !== 'unknown' &&
      panel.count === output.count
  );
  const contextRaster = panels.some(
    panel => panel.role === 'context' && panel.geometry?.kind === 'raster'
  );
  const defaultLayout =
    output.geometry &&
    output.values &&
    sameSupport &&
    !(output.geometry.kind === 'raster' && contextRaster)
      ? 'side-by-side'
      : 'overlay';
  const requestedLayout = context.options?.['layout'];
  const layout =
    requestedLayout === 'overlay' || requestedLayout === 'side-by-side'
      ? requestedLayout
      : defaultLayout;
  const spatialRoleCount = rolePanels.filter(panel => panel.geometry).length;
  const width = thumbnail
    ? CARD_FIGURE.thumbnailWidth
    : layout === 'side-by-side' && grammar === 'geometry'
      ? (CARD_FIGURE.width - CARD_FIGURE.panelGap * (spatialRoleCount - 1)) / spatialRoleCount
      : CARD_FIGURE.width;
  const height = thumbnail
    ? CARD_FIGURE.thumbnailHeight
    : grammar === 'strips'
      ? arrays.length * CARD_FIGURE.stripRowHeight +
        (arrays.length - 1) * CARD_FIGURE.stripRowGap +
        2 * CARD_FIGURE.stripPadding
      : CARD_FIGURE.height;
  const iterations = new Map<number, GPUComputationCapture>();
  if (
    !thumbnail &&
    grammar !== 'table' &&
    (layout === 'overlay' || output.geometry?.kind === 'raster')
  ) {
    for (const capture of context.record.captures?.snapshots ?? []) {
      if (capture.name === outputCapture.name && capture.iteration !== undefined)
        iterations.set(capture.iteration, capture);
    }
  }
  const snapshots = [...iterations.values()].sort(
    (left, right) => left.iteration! - right.iteration!
  );
  const candidates = snapshots.map(capture => buildPanel(capture, context));
  if (snapshots.length >= 2 && snapshots.at(-1)?.id !== outputCapture.id) candidates.push(output);
  const signature = (panel: ResultPanel) => {
    if (panel.geometry?.kind === 'lines' || panel.geometry?.kind === 'polygons') {
      if (panel.unknown) return ['unknown', panel.unknown];
      return getDrawablePanelFeatures(panel).flatMap(feature => [
        feature.row,
        feature.vertices.length,
        ...feature.vertices.flat(),
        panel.values ? getCaptureValue(panel.values, feature.row) : undefined
      ]);
    }
    return [
      panel.count,
      ...(panel.geometry?.kind === 'points'
        ? [getCaptureFirstKey(panel.capture), panel.points.length, ...panel.points.flat()]
        : []),
      ...Array.from(panel.values?.values ?? panel.capture.values)
    ];
  };
  const firstValues = candidates[0] && signature(candidates[0]);
  const changed =
    firstValues &&
    candidates.some(panel => {
      const values = signature(panel);
      return (
        values.length !== firstValues.length ||
        values.some((value, index) => !Object.is(value, firstValues[index]))
      );
    });
  const frames =
    changed && snapshots.length >= 2 ? sampleEvenly(candidates, CARD_FIGURE.maxFrames) : [];
  if (frames.length && output.geometry?.kind === 'raster') {
    const note = getRasterSamplingNote(...output.geometry.shape, 96, output.palette === 'nom');
    if (note) output.notes.push(`Animation: ${note}`);
  }
  alignPixelRasterCompanions([...new Set([...panels, ...frames])], output);
  const spatialPanels = (
    thumbnail
      ? [...panels.filter(panel => panel.role === 'context'), output]
      : panels.flatMap(panel => (panel === output && frames.length ? frames : [panel]))
  ).filter(panel => panel.geometry && !panel.unknown);
  const models = spatialPanels.map(panel =>
    panel.geometry!.kind === 'raster' && !panel.geometry!.origin
      ? ('projected' as const)
      : (panel.geometry!.coordinateModel ?? 'unknown')
  );
  const points = spatialPanels.flatMap(panel => {
    const geometry = panel.geometry!;
    if (geometry.kind === 'lines' || geometry.kind === 'polygons')
      return getDrawablePanelFeatures(panel).flatMap(feature => feature.vertices);
    if (geometry.kind !== 'raster') return panel.points;
    const {columns, rows} = getRasterPlacement(geometry);
    return [
      [columns(0), rows(0)],
      [columns(geometry.shape[0]), rows(geometry.shape[1])]
    ] as [number, number][];
  });
  const units = spatialPanels.map(panel =>
    'unit' in panel.geometry! ? panel.geometry!.unit : undefined
  );
  const projection = createGeometryProjection({
    coordinateModel: models.every(model => model === models[0])
      ? (models[0] ?? 'unknown')
      : 'unknown',
    points,
    width,
    height,
    padding: thumbnail ? CARD_FIGURE.thumbnailPadding : CARD_FIGURE.padding,
    unit: units.every(unit => unit === units[0]) ? units[0] : undefined
  });
  if (!models.every(model => model === models[0]))
    projection.statement += ' Captures have differing coordinate models; alignment is unknown.';
  const model: ResultViewModel = {
    panels,
    output,
    projection,
    layout,
    thumbnail,
    width,
    height,
    grammar,
    frames
  };
  for (const panel of new Set([...panels, ...frames])) {
    if (
      !panel.unknown &&
      panel.offsets &&
      (panel.geometry?.kind === 'lines' || panel.geometry?.kind === 'polygons')
    ) {
      panel.featureSample = samplePanelFeatures(panel, model, context);
      panel.notes.push(panel.featureSample.note);
    }
  }
  return model;
}

/** A ground-space companion must use the same displayed pixel frame as an affine raster. */
function alignPixelRasterCompanions(panels: ResultPanel[], output: ResultPanel): void {
  const isPixelRaster = (panel: ResultPanel) =>
    panel.geometry?.kind === 'raster' &&
    panel.geometry.unit === 'pixels' &&
    panel.geometry.affine !== undefined;
  const raster = isPixelRaster(output) ? output : panels.find(isPixelRaster);
  const source = raster?.capture.geometry;
  if (!raster || source?.kind !== 'raster' || !source.affine) return;
  const [columnX, rowX, originX, columnY, rowY, originY] = source.affine;
  const determinant = columnX * rowY - rowX * columnY;
  const shift = source.pixelInterpretation === 'point' ? 0.5 : 0;
  for (const panel of panels) {
    const geometry = panel.geometry;
    if (!geometry || panel === raster || panel.unknown) continue;
    if (geometry.kind === 'raster') {
      if (
        JSON.stringify(geometry.affine) === JSON.stringify(source.affine) &&
        geometry.pixelInterpretation === source.pixelInterpretation &&
        geometry.crs === source.crs &&
        geometry.frameId === source.frameId
      )
        continue;
    } else if (
      geometry.kind !== 'graph' &&
      geometry.coordinateModel === source.coordinateModel &&
      geometry.unit === source.unit &&
      geometry.unit !== undefined &&
      Number.isFinite(determinant) &&
      determinant !== 0
    ) {
      panel.points = panel.points.map(([horizontal, vertical]): [number, number] => {
        const deltaX = horizontal - originX;
        const deltaY = vertical - originY;
        return [
          (rowY * deltaX - rowX * deltaY) / determinant + shift,
          source.shape[1] - ((columnX * deltaY - columnY * deltaX) / determinant + shift)
        ];
      });
      panel.geometry = {...geometry, coordinateModel: 'projected', unit: 'pixels'};
      panel.notes.push('Ground coordinates transformed into the raster pixel frame.');
      continue;
    }
    panel.unknown = 'cannot align this ground placement with the raster pixel frame';
    panel.notes.push(panel.unknown);
  }
}

/** Base geometry shares a vertex budget; selection and highlight overlays reuse its samples. */
function getPanelVertexBudget(model: ResultViewModel): number {
  return Math.max(
    3,
    Math.floor(
      (model.thumbnail ? 64 : 400) /
        (Math.max(1, model.panels.length) * Math.max(1, model.frames.length))
    )
  );
}

function getPointStride(panel: ResultPanel, model: ResultViewModel): number {
  return Math.max(1, Math.ceil(panel.points.length / Math.min(2000, getPanelVertexBudget(model))));
}

/** Only offset-delimited, finite, drawable spans contribute to the projection or sampling. */
function getDrawablePanelFeatures(panel: ResultPanel): DrawableGeometryFeature[] {
  const polygons = panel.geometry?.kind === 'polygons';
  const candidates: DrawableGeometryFeature[] = [];
  if (panel.unknown || !panel.offsets) return [];
  const offsets = panel.offsets.values;
  const featureCount = Math.max(0, offsets.length - 1);
  for (let row = 0; row < featureCount; row++) {
    const vertexOrigin = getCaptureFirstKey(panel.capture);
    const first = Number(offsets[row]) - vertexOrigin;
    const end = Number(offsets[row + 1]) - vertexOrigin;
    if (
      !Number.isInteger(first) ||
      !Number.isInteger(end) ||
      first < 0 ||
      end > panel.points.length ||
      end <= first
    )
      continue;
    const vertices = panel.points.slice(first, end);
    if (vertices.some(point => !point.every(Number.isFinite))) continue;
    const closed =
      vertices.length > 1 &&
      vertices[0][0] === vertices.at(-1)![0] &&
      vertices[0][1] === vertices.at(-1)![1];
    // SVG closes polygons with Z. Sampling a repeated closing vertex would waste a corner.
    if (polygons && closed) vertices.pop();
    const minimumVertices = polygons || closed ? 3 : 2;
    if (vertices.length < minimumVertices) continue;
    if (
      polygons
        ? !findPolygonCorners(vertices)
        : !vertices.some(point => point[0] !== vertices[0][0] || point[1] !== vertices[0][1])
    )
      continue;
    candidates.push({row: row + getCaptureFirstKey(panel.offsets!), vertices, minimumVertices});
  }
  return candidates;
}

function getPolygonArea(vertices: [number, number][]): number {
  if (vertices.length < 3) return 0;
  const [originX, originY] = vertices[0];
  return vertices.reduce((area, point, index) => {
    const next = vertices[(index + 1) % vertices.length];
    return (
      area + (point[0] - originX) * (next[1] - originY) - (next[0] - originX) * (point[1] - originY)
    );
  }, 0);
}

/** Picks source-order corners with a nonzero triangle area in the displayed coordinate space. */
function findPolygonCorners(
  vertices: [number, number][],
  project: (point: [number, number]) => [number, number] = point => point
): [number, number][] | undefined {
  const origin = project(vertices[0]);
  let second = 0;
  let maximumDistance = 0;
  for (let index = 1; index < vertices.length; index++) {
    const point = project(vertices[index]);
    const distance = Math.hypot(point[0] - origin[0], point[1] - origin[1]);
    if (distance > maximumDistance) {
      maximumDistance = distance;
      second = index;
    }
  }
  const endpoint = project(vertices[second]);
  let third = 0;
  let maximumArea = 0;
  for (let index = 1; index < vertices.length; index++) {
    const point = project(vertices[index]);
    const area = Math.abs(
      (endpoint[0] - origin[0]) * (point[1] - origin[1]) -
        (endpoint[1] - origin[1]) * (point[0] - origin[0])
    );
    if (Number.isFinite(area) && area > maximumArea) {
      maximumArea = area;
      third = index;
    }
  }
  return maximumArea
    ? [0, second, third].sort((first, last) => first - last).map(index => vertices[index])
    : undefined;
}

/** Resolves only the selection image in this panel's declared key space. */
function getPanelSelection(panel: ResultPanel, context: ViewContext) {
  return panel.keySpace
    ? (context.selections?.get(panel.keySpace) ??
        (panel.keySpace === getDatumKeySpace(context) ? context.selection : undefined))
    : panel.geometry?.kind === 'lines' || panel.geometry?.kind === 'polygons'
      ? undefined
      : context.selection;
}

/** Selects complete drawable features before allocating their vertex samples. */
function samplePanelFeatures(
  panel: ResultPanel,
  model: ResultViewModel,
  context: ViewContext
): GeometryFeatureSample {
  const polygons = panel.geometry?.kind === 'polygons';
  const candidates = getDrawablePanelFeatures(panel);
  const featureCount = Math.max(0, panel.offsets!.values.length - 1);
  const reservedRows = new Set([
    panel.keySpace === getDatumKeySpace(context) ? getPinnedEntity(context) : undefined,
    getHighlight(panel)?.key
  ]);
  const authored = candidates.filter(feature => reservedRows.has(feature.row));
  const minimumVertices = candidates.some(feature => feature.minimumVertices === 3) ? 3 : 2;
  // A panel always retains its pinned/highlighted drawable features, even at the minimum budget.
  const vertexBudget = Math.max(getPanelVertexBudget(model), authored.length * minimumVertices);
  const featureCapacity = Math.floor(vertexBudget / minimumVertices);
  const liveRows = new Set([
    ...(getPanelSelection(panel, context)?.highlighted?.keys() ?? []),
    ...(getPanelSelection(panel, context)?.hovered ?? [])
  ]);
  const reserved = [
    ...authored,
    ...candidates.filter(feature => !reservedRows.has(feature.row) && liveRows.has(feature.row))
  ].slice(0, featureCapacity);
  for (const feature of reserved) reservedRows.add(feature.row);
  const selected = [
    ...reserved,
    ...sampleEvenly(
      candidates.filter(feature => !reservedRows.has(feature.row)),
      Math.max(0, featureCapacity - reserved.length)
    )
  ].sort((first, second) => first.row - second.row);
  const verticesPerFeature = selected.length ? Math.floor(vertexBudget / selected.length) : 0;
  const remainder = selected.length ? vertexBudget % selected.length : 0;
  const project = (point: [number, number]): [number, number] =>
    model.projection.project(...point).map(value => Math.round(value * 100) / 100) as [
      number,
      number
    ];
  const features = selected.flatMap((feature, index) => {
    let vertices = sampleEvenly(feature.vertices, verticesPerFeature + (index < remainder ? 1 : 0));
    if (polygons && !getPolygonArea(vertices.map(project))) {
      const corners = findPolygonCorners(feature.vertices, project);
      if (!corners) return [];
      vertices = corners;
    }
    if (!polygons) {
      const projected = vertices.map(project);
      if (projected.every(point => point[0] === projected[0][0] && point[1] === projected[0][1]))
        return [];
    }
    return [{row: feature.row, vertices}];
  });
  const fallback = !featureCount
    ? 'empty captured geometry'
    : !features.length
      ? 'no drawable captured features'
      : undefined;
  const shownRows = new Set(features.map(feature => feature.row));
  const omittedLiveRows = candidates.filter(
    feature => liveRows.has(feature.row) && !shownRows.has(feature.row)
  ).length;
  return {
    features,
    note:
      `${panel.capture.name}: ${features.length} of ${featureCount} ${panel.noun} shown; ` +
      (fallback ??
        `evenly sampled features and vertices${polygons ? '' : '; line endpoints retained'}`) +
      (candidates.length < featureCount
        ? `; ${featureCount - candidates.length} invalid or uncaptured features omitted`
        : '') +
      (features.length < selected.length
        ? `; ${selected.length - features.length} features below figure resolution omitted`
        : '') +
      (omittedLiveRows ? `; ${omittedLiveRows} selected or hovered features omitted` : ''),
    fallback
  };
}

function getMarkStyle(
  panel: ResultPanel,
  shape: CardMarkShape,
  value: number | undefined,
  model: ResultViewModel,
  role: CardRole = panel.role
) {
  const valued =
    Boolean(panel.values) &&
    role !== 'context' &&
    role !== 'highlight' &&
    (Boolean(panel.geometry) || role === 'output');
  const noData = valued && !Number.isFinite(value);
  const classIndex =
    !valued || noData
      ? undefined
      : panel.categories
        ? panel.categories.indexOf(value!)
        : panel.scale?.classFor(value!);
  const other = valued && panel.categories && classIndex === -1;
  return getCardMarkAttributes(other ? 'context' : role, shape, {
    palette: other ? undefined : panel.palette,
    classIndex: other ? undefined : classIndex,
    noData,
    scale: model.thumbnail ? CARD_FIGURE.thumbnailMarkScale : 1
  });
}

function drawMark(
  tag: string,
  attributes: ViewElementNode['attributes'],
  panel: ResultPanel,
  key: number,
  shape: CardMarkShape,
  model: ResultViewModel,
  context: ViewContext,
  value = panel.values ? getCaptureValue(panel.values, key) : undefined,
  role: CardRole = panel.role,
  selected = (!panel.keySpace || panel.keySpace === getDatumKeySpace(context)) &&
    getPinnedEntity(context) === key
): ViewElementNode {
  const interpreted =
    value === undefined
      ? value
      : getInterpretedCaptureValue(context.record, panel.values ?? panel.capture, value);
  const style = getMarkStyle(panel, shape, interpreted, model, role);
  const provenance = provenanceMarkAttributes(
    getCaptureProvenance(panel.values ?? panel.capture),
    style['fill'],
    context.idPrefix
  );
  const interactive =
    style['data-role'] !== 'context' &&
    role !== 'highlight' &&
    (!(panel.geometry?.kind === 'lines' || panel.geometry?.kind === 'polygons') ||
      Boolean(panel.keySpace));
  const mark = h(tag, {
    ...style,
    ...attributes,
    'data-provenance': provenance['data-provenance'],
    ...(interactive
      ? {
          'data-action': 'pick',
          'data-axis': 'entity',
          'data-key': key,
          'data-key-space': panel.keySpace
        }
      : {})
  });
  const overlays: ViewElementNode[] = [];
  if (getCaptureProvenance(panel.values ?? panel.capture) === 'replay')
    overlays.push(
      h(tag, {
        ...attributes,
        ...(shape === 'point' ? {r: style['r']} : {}),
        fill: shape === 'line' ? 'none' : provenance['fill'],
        stroke: shape === 'line' ? provenance['fill'] : 'none',
        'stroke-width': style['stroke-width'],
        opacity: 0.55,
        'pointer-events': 'none',
        'data-provenance': 'replay'
      })
    );
  if (selected && interactive) {
    const ink = getSelectionInk(context.ground);
    for (const [state, color, width] of [
      ['selection-halo', ink.halo, 4],
      ['selection', ink.core, 2]
    ] as const)
      overlays.push(
        h(tag, {
          ...attributes,
          ...(shape === 'point' ? {r: style['r']} : {}),
          fill: 'none',
          stroke: color,
          'stroke-width': width,
          'data-state': state,
          'pointer-events': 'none'
        })
      );
  }
  return overlays.length ? h('g', null, mark, overlays) : mark;
}

function drawPanel(
  panel: ResultPanel,
  model: ResultViewModel,
  context: ViewContext,
  highlightOnly = false
): ViewElementNode[] {
  const renderFallback = (reason: string) => [
    h(
      'text',
      {
        class: 'cv-label',
        x: 8,
        y: 16,
        'data-geometry-fallback': panel.role,
        'data-capture-id': panel.capture.id
      },
      reason
    )
  ];
  if (panel.unknown)
    return !highlightOnly &&
      (panel.geometry?.kind === 'lines' || panel.geometry?.kind === 'polygons')
      ? renderFallback(`unknown — ${panel.unknown}`)
      : [];
  const project = (horizontal: number, vertical: number) =>
    model.projection.project(horizontal, vertical).map(value => Math.round(value * 100) / 100);
  const panelBudget = getPanelVertexBudget(model);
  const marks: ViewElementNode[] = [];
  const geometry = panel.geometry;
  if (!geometry) return marks;
  const highlight = getHighlight(panel);
  if (highlightOnly && !highlight) return marks;
  const role = highlightOnly ? 'highlight' : panel.role;
  const addMark = (
    tag: string,
    attributes: ViewElementNode['attributes'],
    key: number,
    shape: CardMarkShape,
    value?: number,
    selected?: boolean
  ) => {
    if (highlightOnly && key !== highlight!.key) return;
    marks.push(
      drawMark(
        tag,
        attributes,
        panel,
        key,
        shape,
        model,
        context,
        value ?? (panel.values ? getCaptureValue(panel.values, key) : undefined),
        role,
        selected
      )
    );
    if (highlightOnly && !model.thumbnail) {
      const point =
        panel.points[
          geometry.kind === 'lines' || geometry.kind === 'polygons'
            ? Number(panel.offsets?.values[key - getCaptureFirstKey(panel.offsets!)]) -
              getCaptureFirstKey(panel.capture)
            : key - (geometry.kind === 'points' ? getCaptureFirstKey(panel.capture) : 0)
        ];
      const position = point
        ? project(...point)
        : [Number(attributes['x']) + Number(attributes['width'] ?? 0) / 2, Number(attributes['y'])];
      marks.push(
        h(
          'text',
          {
            class: 'cv-label cv-label--subject',
            x: position[0],
            y: position[1] - 8,
            'text-anchor': 'middle'
          },
          highlight!.label
        )
      );
    }
  };
  if (geometry.kind === 'raster') {
    const [columns, rows] = geometry.shape;
    const placement = getRasterPlacement(geometry);
    const [left, firstEdge] = project(placement.columns(0), placement.rows(0));
    const [right, lastEdge] = project(placement.columns(columns), placement.rows(rows));
    if (!highlightOnly) {
      marks.push(
        renderRasterImage({
          columns,
          rows,
          limit: model.thumbnail ? 64 : model.frames.length ? 96 : 160,
          valueAt: key =>
            getInterpretedCaptureValue(
              context.record,
              panel.capture,
              getCaptureValue(panel.capture, key) ?? NaN
            ),
          categorical: panel.palette === 'nom',
          flipRows: firstEdge > lastEdge,
          colorFor: value => String(getMarkStyle(panel, 'cell', value, model)['fill']),
          attributes: {
            ...getMarkStyle(panel, 'cell', 0, model),
            x: left,
            y: Math.min(firstEdge, lastEdge),
            width: right - left,
            height: Math.abs(lastEdge - firstEdge),
            'data-provenance': getCaptureProvenance(panel.capture)
          }
        })
      );
    }
    const selectionMatches =
      !panel.capture.keySpace || panel.capture.keySpace === getDatumKeySpace(context);
    const selectedKeys = highlightOnly
      ? [highlight?.key]
      : [
          ...new Set([
            selectionMatches ? getPinnedEntity(context) : undefined,
            ...(getPanelSelection(panel, context)?.highlighted?.keys() ?? [])
          ])
        ].slice(0, 64);
    for (const selectedKey of selectedKeys) {
      if (selectedKey === undefined || selectedKey < 0 || selectedKey >= columns * rows) continue;
      const column = selectedKey % columns;
      const row = Math.floor(selectedKey / columns);
      const [startX, startY] = project(placement.columns(column), placement.rows(row));
      const [endX, endY] = project(placement.columns(column + 1), placement.rows(row + 1));
      addMark(
        'rect',
        {
          x: startX,
          y: Math.min(startY, endY),
          width: endX - startX,
          height: Math.abs(endY - startY),
          fill: 'none'
        },
        selectedKey,
        'cell'
      );
    }
  } else if (geometry.kind === 'points' || geometry.kind === 'graph') {
    if (
      !highlightOnly &&
      geometry.kind === 'graph' &&
      panel.offsets &&
      panel.targets &&
      panel.points.length
    ) {
      let edgeCount = 0;
      const edgeStride = Math.max(
        1,
        Math.ceil(panel.targets.values.length / Math.floor(panelBudget / 2))
      );
      for (let node = 0; node < panel.offsets.values.length - 1; node++) {
        for (
          let edge = Number(panel.offsets.values[node]);
          edge < Number(panel.offsets.values[node + 1]);
          edge++
        ) {
          const from = panel.points[node];
          const to = panel.points[Number(panel.targets.values[edge])];
          if (
            edgeCount % edgeStride === 0 &&
            from?.every(Number.isFinite) &&
            to?.every(Number.isFinite)
          ) {
            const [startX, startY] = project(...from);
            const [endX, endY] = project(...to);
            marks.push(
              drawMark(
                'line',
                {x1: startX, y1: startY, x2: endX, y2: endY},
                panel,
                node,
                'line',
                model,
                context,
                undefined,
                panel.role === 'output' ? 'output' : 'context',
                false
              )
            );
          }
          edgeCount++;
        }
      }
    }
    const pointStride = getPointStride(panel, model);
    panel.points.forEach((point, row) => {
      if ((!highlightOnly && row % pointStride !== 0) || !point.every(Number.isFinite)) return;
      const key = row + (geometry.kind === 'points' ? getCaptureFirstKey(panel.capture) : 0);
      const [horizontal, vertical] = project(...point);
      addMark('circle', {cx: horizontal, cy: vertical}, key, 'point');
    });
  } else if (panel.offsets) {
    const sample = panel.featureSample ?? samplePanelFeatures(panel, model, context);
    if (sample.fallback && !highlightOnly) return renderFallback(sample.fallback);
    for (const {row, vertices} of sample.features) {
      const path =
        vertices
          .map((point, index) => `${index ? 'L' : 'M'}${project(...point).join(',')}`)
          .join(' ') + (geometry.kind === 'polygons' ? ' Z' : '');
      addMark(
        'path',
        {
          d: path,
          'fill-rule': 'evenodd',
          'data-feature-row': row,
          'data-capture-id': panel.capture.id
        },
        row,
        geometry.kind === 'lines' ? 'line' : 'area'
      );
    }
  }
  return marks;
}

function renderAdjacency(panel: ResultPanel, model: ResultViewModel): ViewElementNode | null {
  if (
    panel.geometry?.kind !== 'graph' ||
    panel.points.length ||
    panel.unknown ||
    !panel.offsets ||
    !panel.targets
  )
    return null;
  const count = Number(panel.count);
  const limit = Math.min(count, model.thumbnail ? 8 : 32);
  return h(
    'div',
    null,
    h(
      'table',
      {class: 'lens-adjacency'},
      h('thead', null, h('tr', null, h('th', null, 'node'), h('th', null, 'neighbours'))),
      h(
        'tbody',
        null,
        Array.from({length: limit}, (_, node) => {
          const first = Number(panel.offsets!.values[node]);
          const last = Number(panel.offsets!.values[node + 1]);
          const neighbours = Array.from({length: last - first}, (_, index) =>
            Number(panel.targets!.values[first + index])
          );
          return h(
            'tr',
            {
              'data-action': 'pick',
              'data-axis': 'entity',
              'data-key': node,
              'data-provenance': getCaptureProvenance(panel.capture)
            },
            h('td', null, String(node)),
            h('td', null, neighbours.join(', ') || '—')
          );
        })
      )
    ),
    count > limit ? h('p', null, `${count - limit} more rows`) : null
  );
}

function getHighlight(panel: ResultPanel): {key: number; label: string} | undefined {
  if (panel.role !== 'output' || panel.palette === 'div') return undefined;
  return panel.capture.highlight;
}

/**
 * World coordinate of a raster's column and row edges. Without a recorded `origin` the raster
 * sits in its own cell space with row 0 at the top (`rows × cellSize`), as before placement existed.
 */
function getRasterPlacement(geometry: Extract<GPUComputationCaptureGeometry, {kind: 'raster'}>): {
  columns: (edge: number) => number;
  rows: (edge: number) => number;
} {
  const cellSize = geometry.cellSize ?? 1;
  const north = geometry.rowDirection === 'north';
  const [originX, originY] = geometry.origin ?? [0, north ? 0 : geometry.shape[1] * cellSize];
  return {
    columns: edge => originX + edge * cellSize,
    rows: edge => (north ? originY + edge * cellSize : originY - edge * cellSize)
  };
}

function renderLegend(model: ResultViewModel, figure: ViewElementNode): ViewElementNode {
  const valuedMarks = findViewNodes(figure, node =>
    String(node.attributes['class'] ?? '')
      .split(' ')
      .includes('cv-v')
  );
  const panel = model.output.values
    ? model.output
    : model.panels.find(candidate => candidate.values && candidate.role !== 'context');
  if (!panel || !valuedMarks.length) {
    const inputs = model.panels
      .filter(candidate => candidate.role === 'input')
      .map(candidate => candidate.capture.name);
    const outputs = model.panels
      .filter(candidate => candidate.role === 'output')
      .map(candidate => candidate.capture.name);
    return h(
      'p',
      {class: 'cv-legend'},
      [
        inputs.length ? `grey: input ${inputs.join(', ')}` : '',
        outputs.length ? `blue: output ${outputs.join(', ')}` : ''
      ]
        .filter(Boolean)
        .join(' · ')
    );
  }
  const labels =
    panel.categories?.map(String) ??
    panel.scale
      ?.legend()
      .classes.filter(entry => !entry.beyondRange)
      .slice(0, 5)
      .map(entry => entry.label) ??
    [];
  const colors = getCardPaletteColors(panel.palette, 'light');
  const noData = valuedMarks.some(mark => String(mark.attributes['class']).includes('cv-no-data'));
  const swatch = (className: string, fill: string, label: string) =>
    h(
      'span',
      null,
      h(
        'svg',
        {class: 'cv-legend__swatch', width: 12, height: 12, 'aria-hidden': true},
        h('rect', {class: `cv-v ${className}`, width: 12, height: 12, fill})
      ),
      ` ${label}`
    );
  return h(
    'p',
    {class: 'cv-legend'},
    `${panel.values!.name}: `,
    labels
      .slice(0, 5)
      .map((label, index) => swatch(`cv-${panel.palette}-${index + 1}`, colors[index], label)),
    noData ? swatch('cv-no-data', CARD_COLORS.light['no-data'], 'No data') : null
  );
}

function renderSvg(
  model: ResultViewModel,
  context: ViewContext,
  children: ViewElementNode[],
  panel = false,
  displayedPanel = model.output
): ViewElementNode {
  const patterns = provenancePatternDefs(context.idPrefix, context.ground);
  for (const pattern of findViewNodes(patterns, node => node.tag === 'pattern')) {
    pattern.attributes['width'] = 3;
    pattern.attributes['height'] = 3;
  }
  return h(
    'svg',
    {
      class: `cv-figure${panel ? ' cv-panel' : ''}`,
      width: model.width,
      height: model.height,
      viewBox: `0 0 ${model.width} ${model.height}`,
      preserveAspectRatio: 'xMidYMid meet',
      'data-extent': JSON.stringify(model.projection.extent),
      role: 'img',
      'aria-label': `Result: ${displayedPanel.capture.name}`
    },
    patterns,
    crispnessDefs(context.idPrefix),
    displayedPanel.featureSample ? h('desc', null, displayedPanel.featureSample.note) : null,
    children
  );
}

function renderScaleTick(panel: ResultPanel, model: ResultViewModel): ViewElementNode[] {
  const geometry = panel.geometry;
  if (model.thumbnail || geometry?.kind !== 'raster' || geometry.cellSize === undefined) return [];
  const start = model.projection.project(0, 0);
  const end = model.projection.project(geometry.cellSize, 0);
  const cellWidth = Math.abs(end[0] - start[0]);
  if (!cellWidth) return [];
  const multiple = Math.max(1, Math.ceil(12 / cellWidth));
  const left = CARD_FIGURE.padding;
  const bottom = model.height - 12;
  return [
    h('line', {
      ...getCardMarkAttributes('context', 'line'),
      x1: left,
      y1: bottom,
      x2: left + cellWidth * multiple,
      y2: bottom
    }),
    h(
      'text',
      {class: 'cv-label', x: left, y: bottom - 4},
      `${Number((geometry.cellSize * multiple).toPrecision(3))} ${geometry.unit ?? '(unit unknown)'}`
    )
  ];
}

function renderSpatialGroups(
  panels: ResultPanel[],
  model: ResultViewModel,
  context: ViewContext,
  output = model.output
): ViewElementNode[] {
  const groups = panels.map(panel =>
    h('g', {'data-role': panel.role}, drawPanel(panel, model, context))
  );
  groups.push(h('g', {'data-role': 'highlight'}, drawPanel(output, model, context, true)));
  return groups;
}

function renderSpatialFigure(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const contexts = model.panels.filter(panel => panel.role === 'context');
  const roles = model.panels.filter(panel => panel.role !== 'context' && panel.geometry);
  if (model.thumbnail) {
    const output = model.output;
    if (
      output.geometry?.kind === 'graph' &&
      !output.points.length &&
      output.offsets &&
      output.targets &&
      !output.unknown
    ) {
      const count = Math.min(8, Number(output.count));
      const size = (model.height - 12) / Math.max(1, count);
      const left = (model.width - count * size) / 2;
      const marks: ViewElementNode[] = [];
      for (let row = 0; row < count; row++) {
        const neighbours = new Set<number>();
        for (
          let edge = Number(output.offsets.values[row]);
          edge < Number(output.offsets.values[row + 1]);
          edge++
        )
          neighbours.add(Number(output.targets.values[edge]));
        for (let column = 0; column < count; column++)
          marks.push(
            drawMark(
              'rect',
              {x: left + column * size, y: 6 + row * size, width: size, height: size},
              output,
              row,
              'cell',
              model,
              context,
              undefined,
              neighbours.has(column) ? 'output' : 'context',
              false
            )
          );
      }
      return renderSvg(model, context, marks);
    }
    return renderSvg(model, context, renderSpatialGroups([...contexts, output], model, context));
  }
  const renderOutput = (panel: ResultPanel, panelContext: ViewContext) =>
    model.frames.length && panel === model.output
      ? model.frames.map((frame, index) =>
          h(
            'g',
            {class: 'cv-frame', 'data-cv-frame': index, hidden: index !== model.frames.length - 1},
            frame.featureSample ? h('desc', null, frame.featureSample.note) : null,
            renderSpatialGroups([frame], model, panelContext, frame)
          )
        )
      : panel === model.output
        ? renderSpatialGroups([panel], model, panelContext, panel)
        : [h('g', {'data-role': panel.role}, drawPanel(panel, model, panelContext))];
  if (model.layout === 'side-by-side')
    return h(
      'div',
      {class: 'cv-panels'},
      roles.map((panel, index) => {
        const panelContext = {...context, idPrefix: `${context.idPrefix}-panel-${index}`};
        return h(
          'figure',
          {class: 'lens-result-panel', 'data-role': panel.role},
          renderSvg(
            model,
            panelContext,
            [
              ...contexts.map(background =>
                h('g', {'data-role': 'context'}, drawPanel(background, model, panelContext))
              ),
              ...renderOutput(panel, panelContext),
              ...renderScaleTick(panel, model)
            ],
            true,
            panel
          ),
          renderAdjacency(panel, model),
          h(
            'figcaption',
            {class: 'cv-panel-caption'},
            `${panel.role}: ${panel.capture.name} `,
            provenanceBadge(getCaptureProvenance(panel.values ?? panel.capture))
          )
        );
      })
    );
  return h(
    'div',
    null,
    renderSvg(model, context, [
      ...model.panels
        .filter(panel => panel !== model.output)
        .map(panel => h('g', {'data-role': panel.role}, drawPanel(panel, model, context))),
      ...renderOutput(model.output, context),
      ...renderScaleTick(model.output, model)
    ]),
    roles.map(panel => renderAdjacency(panel, model))
  );
}

function formatValue(value: number | undefined): string {
  if (value === undefined) return '—';
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : '−∞';
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(3)));
}

function hasCaptureKey(capture: GPUComputationCapture, key: number): boolean {
  const index = key - getCaptureFirstKey(capture);
  return index >= 0 && index < getCaptureRowCount(capture);
}

function formatCaptureEntity(
  capture: GPUComputationCapture,
  key: number,
  context: ViewContext
): string {
  if (!hasCaptureKey(capture, key)) return formatValue(undefined);
  const components = capture.components ?? 1;
  const index = (key - getCaptureFirstKey(capture)) * components;
  return Array.from({length: components}, (_, component) =>
    isUnknownCaptureValue(context.record, capture, Number(capture.values[index + component]))
      ? 'unknown'
      : formatValue(Number(capture.values[index + component]))
  ).join(', ');
}

function renderStrips(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const availableRows = model.panels.filter(panel => !panel.geometry && panel.role !== 'context');
  const selectedRows = new Set(
    [
      model.output,
      ...availableRows.filter(panel => panel.role === 'input').slice(0, 1),
      ...availableRows
    ].slice(0, 4)
  );
  const rows = model.thumbnail
    ? availableRows.filter(panel => selectedRows.has(panel))
    : availableRows;
  const padding = model.thumbnail ? CARD_FIGURE.thumbnailPadding : CARD_FIGURE.stripPadding;
  const gutter = model.thumbnail ? 0 : 72;
  const rowHeight = model.thumbnail
    ? Math.min(28, (model.height - padding * 2) / rows.length - 4)
    : CARD_FIGURE.stripRowHeight;
  const renderRow = (panel: ResultPanel, row: number): ViewElementNode => {
    const count = panel.unknown ? 0 : getCaptureRowCount(panel.capture);
    const first = getCaptureFirstKey(panel.capture);
    const keys: (number | null)[] = model.thumbnail
      ? Array.from({length: Math.min(8, count)}, (_, index) => index)
      : count <= 16
        ? Array.from({length: count}, (_, index) => index)
        : [
            ...Array.from({length: 8}, (_, index) => index),
            null,
            ...Array.from({length: 8}, (_, index) => count - 8 + index)
          ];
    const cellWidth =
      (model.width - padding * 2 - gutter) /
      Math.max(
        1,
        ...rows.map(candidate =>
          Math.min(
            model.thumbnail ? 8 : getCaptureRowCount(candidate.capture) > 16 ? 17 : 16,
            getCaptureRowCount(candidate.capture)
          )
        )
      );
    const top = padding + row * (rowHeight + (model.thumbnail ? 4 : CARD_FIGURE.stripRowGap));
    const marks: ViewElementNode[] = [];
    if (!model.thumbnail)
      marks.push(
        h(
          'text',
          {
            class: 'cv-label',
            x: padding,
            y: top + rowHeight / 2,
            'font-size': 11,
            'dominant-baseline': 'central'
          },
          panel.capture.name
        )
      );
    keys.forEach((key, index) => {
      const left = padding + gutter + index * cellWidth;
      if (key === null) {
        marks.push(
          h('rect', {
            ...getCardMarkAttributes('context', 'cell'),
            x: left,
            y: top,
            width: cellWidth,
            height: rowHeight
          })
        );
        if (!model.thumbnail)
          marks.push(
            h(
              'text',
              {
                class: 'cv-cell-text',
                x: left + cellWidth / 2,
                y: top + rowHeight / 2,
                'font-size': 7
              },
              `… ${count - 16} more …`
            )
          );
        return;
      }
      key += first;
      marks.push(
        drawMark(
          'rect',
          {x: left, y: top, width: cellWidth, height: rowHeight},
          panel,
          key,
          'cell',
          model,
          context
        )
      );
      const highlight =
        panel.capture.name === model.output.capture.name ? getHighlight(panel) : undefined;
      if (highlight?.key === key) {
        marks.push(
          drawMark(
            'rect',
            {x: left, y: top, width: cellWidth, height: rowHeight},
            panel,
            key,
            'cell',
            model,
            context,
            undefined,
            'highlight',
            false
          )
        );
        if (!model.thumbnail)
          marks.push(
            h(
              'text',
              {
                class: 'cv-label cv-label--subject',
                x: left + cellWidth / 2,
                y: top - 4,
                'text-anchor': 'middle'
              },
              highlight.label
            )
          );
      }
      if (!model.thumbnail)
        marks.push(
          h(
            'text',
            {class: 'cv-cell-text', x: left + cellWidth / 2, y: top + rowHeight / 2},
            formatCaptureEntity(panel.capture, key, context)
          )
        );
    });
    return h('g', {'data-role': panel.role}, marks);
  };
  const groups = rows.map((panel, row) =>
    panel === model.output && model.frames.length
      ? h(
          'g',
          {'data-role': panel.role},
          model.frames.map((frame, index) =>
            h(
              'g',
              {
                class: 'cv-frame',
                'data-cv-frame': index,
                hidden: index !== model.frames.length - 1
              },
              renderRow(frame, row)
            )
          )
        )
      : renderRow(panel, row)
  );
  return renderSvg(model, context, groups);
}

/** Union readable windows without enumerating large or uncaptured gaps. */
function getTableWindowSummary(columns: ResultPanel[]): {keys: number[]; rowCount: number} {
  const windows = columns
    .filter(panel => !panel.unknown && getCaptureRowCount(panel.capture) > 0)
    .map(panel => ({
      first: getCaptureFirstKey(panel.capture),
      end: getCaptureFirstKey(panel.capture) + getCaptureRowCount(panel.capture)
    }))
    .sort((left, right) => left.first - right.first);
  const merged: {first: number; end: number}[] = [];
  for (const window of windows) {
    const previous = merged.at(-1);
    if (previous && window.first <= previous.end) previous.end = Math.max(previous.end, window.end);
    else merged.push({...window});
  }
  const keys = merged
    .flatMap(window =>
      Array.from({length: Math.min(6, window.end - window.first)}, (_, row) => window.first + row)
    )
    .slice(0, 6);
  return {keys, rowCount: merged.reduce((count, window) => count + window.end - window.first, 0)};
}

function renderTable(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const columns = model.panels.filter(panel => !panel.geometry && panel.role !== 'context');
  const groups = groupTableColumns(columns);
  if (!model.thumbnail)
    return h(
      'div',
      {class: 'cv-result-tables'},
      groups.map(group => renderTableGroup(group, context))
    );
  // A picture of one domain must not imply a positional join with another domain.
  const primaryGroup = groups.find(group => group.includes(model.output)) ?? [model.output];
  const primaryColumns = model.output.capture.keySpace
    ? groups.filter(group => group[0].capture.keySpace === model.output.capture.keySpace).flat()
    : primaryGroup;
  const {keys} = getTableWindowSummary(primaryColumns);
  return renderSvg(
    model,
    context,
    keys.map((key, row) =>
      h(
        'g',
        null,
        primaryColumns.map((panel, column) =>
          !panel.unknown && hasCaptureKey(panel.capture, key)
            ? h('rect', {
                ...getCardMarkAttributes(panel.role, 'cell', {
                  scale: CARD_FIGURE.thumbnailMarkScale
                }),
                x: 6 + (column * 148) / primaryColumns.length,
                y: 6 + row * 14,
                width: 148 / primaryColumns.length,
                height: 14,
                'data-provenance': getCaptureProvenance(panel.capture)
              })
            : null
        )
      )
    )
  );
}

function renderTableGroup(columns: ResultPanel[], context: ViewContext): ViewElementNode {
  const {keys, rowCount} = getTableWindowSummary(columns);
  const count = keys.length;
  const keySpace = columns[0].capture.keySpace;
  const label = context.record.axes.keySpaces.find(space => space.id === keySpace)?.label;
  const columnAttributes = (panel: ResultPanel) => ({
    class: panel.role === 'output' ? 'cv-col--output' : undefined
  });
  return h(
    'table',
    {class: 'cv-table', 'data-key-space': keySpace},
    h(
      'caption',
      null,
      `${keySpace ? (label ? `${label} (${keySpace})` : keySpace) : `${columns[0].capture.name}: row domain unknown`} · ${rowCount} rows`
    ),
    h(
      'thead',
      null,
      h(
        'tr',
        null,
        columns.map(panel =>
          h('th', {...columnAttributes(panel), scope: 'col'}, panel.capture.name)
        )
      )
    ),
    h(
      'tbody',
      null,
      keys.map(key =>
        h(
          'tr',
          null,
          columns.map(panel =>
            h(
              'td',
              {...columnAttributes(panel), 'data-provenance': getCaptureProvenance(panel.capture)},
              panel.unknown
                ? formatValue(undefined)
                : formatCaptureEntity(panel.capture, key, context)
            )
          )
        )
      ),
      rowCount > count
        ? h('tr', null, h('td', {colspan: columns.length}, `… ${rowCount - count} more rows`))
        : null
    )
  );
}

function renderMotion(model: ResultViewModel, context: ViewContext): ViewElementNode[] {
  if (!model.frames.length || model.thumbnail) return [];
  const smallModel = {
    ...model,
    thumbnail: true,
    width: 96,
    height: 58,
    projection: createGeometryProjection({
      coordinateModel: model.projection.model,
      points: model.projection.extent,
      width: 96,
      height: 58,
      padding: 6
    })
  };
  return [
    h(
      'button',
      {type: 'button', class: 'cv-play', 'data-cv-play': true},
      context.reducedMotion ? 'Step ▸' : '▶ Play'
    ),
    h('span', {class: 'cv-play-status', 'aria-live': 'polite'}, 'final'),
    h(
      'div',
      {class: 'cv-key-frames'},
      sampleEvenly(model.frames, 6).map((frame, index, frames) => {
        const frameContext = {
          ...context,
          bindings: {},
          idPrefix: `${context.idPrefix}-key-frame-${index}`
        };
        const smallFrame: ResultPanel = {...frame, featureSample: undefined};
        if (
          !smallFrame.unknown &&
          smallFrame.offsets &&
          (smallFrame.geometry?.kind === 'lines' || smallFrame.geometry?.kind === 'polygons')
        )
          smallFrame.featureSample = samplePanelFeatures(smallFrame, smallModel, frameContext);
        return h(
          'figure',
          null,
          model.grammar === 'strips'
            ? renderStrips(
                {...smallModel, panels: [frame], output: frame, frames: []},
                frameContext
              )
            : renderSvg(
                smallModel,
                frameContext,
                [h('g', null, drawPanel(smallFrame, smallModel, frameContext))],
                false,
                smallFrame
              ),
          h(
            'figcaption',
            null,
            index === frames.length - 1 ? 'final' : `round ${frame.capture.iteration}`
          )
        );
      })
    )
  ];
}

/** Renders role-styled native geometry, strips or tables from recorded evidence only. */
export function renderResultView(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const figure =
    model.grammar === 'geometry'
      ? renderSpatialFigure(model, context)
      : model.grammar === 'table'
        ? renderTable(model, context)
        : renderStrips(model, context);
  const notes = model.panels.flatMap(panel => [
    ...(panel.unknown ? [`unknown — ${panel.unknown}`] : []),
    ...panel.notes
  ]);
  if (model.grammar === 'geometry') notes.unshift(model.projection.statement);
  return h(
    'section',
    {class: 'lens-view lens-result-view'},
    figure,
    model.grammar === 'geometry' &&
      !model.thumbnail &&
      model.panels.some(panel => !panel.geometry && panel.role !== 'context')
      ? renderTable(model, context)
      : null,
    model.thumbnail
      ? null
      : [
          renderMotion(model, context),
          renderLegend(model, figure),
          // The adapter line belongs to the host caption (twin provenance, lens footer).
          notes.length ? h('p', {class: 'cv-note'}, notes.join(' · ')) : null
        ]
  );
}

/** Public native-geometry result rung for tagged input and output captures. */
export const RESULT_VIEW: ViewRegistration<ResultViewModel> = {
  id: 'result-view',
  label: 'Result',
  rung: 'result',
  recordType: 'record',
  order: 50,
  signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
  descriptor: {audience: 'public', task: 'present', interaction: 'low'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context =>
    context.options?.['primaryOutputCaptureId'] !== undefined &&
    !getResultCaptures(context.record).some(
      capture =>
        capture.role === 'output' && capture.id === context.options?.['primaryOutputCaptureId']
    )
      ? {ok: false, reason: 'primaryOutputCaptureId does not identify a final output capture'}
      : getResultCaptures(context.record).some(capture => capture.role === 'output')
        ? {ok: true}
        : {
            ok: false,
            reason:
              'no captures tagged role: output (set capture.role, and capture.geometry for spatial results, when recording)'
          },
  build: buildResultViewModel,
  render: renderResultView,
  describe: (model, context) => ({
    subject: `Result of ${context.record.id}: ${
      model.panels
        .filter(panel => panel.role === 'input')
        .map(panel => panel.capture.name)
        .join(', ') || 'no recorded inputs'
    } → ${model.output.capture.name}`,
    keyNumber: {label: model.output.noun, value: model.output.count},
    facts: [
      ...model.panels
        .filter(panel => panel.role === 'input')
        .map(panel => `input ${panel.capture.name}: ${panel.count} ${panel.noun}`),
      model.projection.statement,
      ...model.panels.flatMap(panel => panel.notes)
    ],
    display: describePanelDisplays(model),
    provenance: getCaptureProvenance(model.output.capture),
    adapterLabel: getAdapterLabel(context.record)
  })
};

/** Concise forms identify suppressed inputs while still describing the headline output. */
function describePanelDisplays(model: ResultViewModel): string | undefined {
  const suppressedInputs = model.panels
    .filter(panel => panel.role === 'input')
    .map(panel => describeOutputDisplay(panel, model))
    .filter((statement): statement is string => statement !== undefined);
  const output = describeOutputDisplay(model.output, model);
  if (!suppressedInputs.length) return output;
  return [output ?? `${model.output.capture.name} drawn`, ...suppressedInputs].join('; ');
}

/** States when the headline output is not drawn, so concise text never implies a figure (LENS-103). */
function describeOutputDisplay(output: ResultPanel, model: ResultViewModel): string | undefined {
  if (output.unknown) return `${output.capture.name} not drawn: unknown — ${output.unknown}`;
  const fallback =
    output.featureSample?.fallback ??
    (output.geometry?.kind === 'points' &&
    !output.points.some(
      (point, row) =>
        point.every(Number.isFinite) &&
        (row % getPointStride(output, model) === 0 ||
          getHighlight(output)?.key === row + getCaptureFirstKey(output.capture))
    )
      ? output.points.length
        ? 'no drawable captured features'
        : 'empty captured geometry'
      : undefined);
  if (!fallback) return undefined;
  const total = output.count === 'unknown' ? 'unknown' : output.count.toLocaleString('en-US');
  return `${output.capture.name} not drawn: ${fallback} (0 of ${total} ${output.noun} shown)`;
}

let mountCounter = 0;

/** Mounts a host-owned result view, with replacement updates and explicit teardown. */
export function mountResultView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-result-view-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = RESULT_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderResultView(buildResultViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
