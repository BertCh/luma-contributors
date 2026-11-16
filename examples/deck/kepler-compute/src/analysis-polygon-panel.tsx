// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_POLYGON_ROWS,
  MAXIMUM_KEPLER_POLYGON_VERTICES,
  MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY,
  MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY,
  MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS,
  parseKeplerPolygonWeightsSettings,
  type KeplerPolygonAnalysisOperation,
  type KeplerPolygonWeightsSettings,
  type KeplerMultipleTestingMethod,
  type CatalogParameterValue
} from '@luma-contributors/kepler';
import {AnalysisWeightsPanel} from './analysis-weights-panel';
import {useAnalysisFormInitialization} from './analysis-form-initialization';
import type {AnalysisRuntime} from './analysis-runtime';
import {normalizePolygonAnalysisOperation} from './analysis-recipe-settings';
import {ResidentAnalysisSheetForm} from './analysis-sheet';
import {
  createResidentSheetFields,
  getResidentSheetChecks,
  readResidentSheetNumber
} from './analysis-sheet-model';
import {
  usePointAnalysisSheet,
  makePointSheetColumn,
  POINT_SHEET_CORRECTIONS,
  type PointSheetValues,
  type PointSheetSpecification
} from './analysis-point-weights-panel';

export function makePolygonSheetDefaults(
  saved: KeplerPolygonAnalysisOperation | undefined,
  geometryColumn: string,
  populationColumn: string,
  rowCount: number
): Record<string, CatalogParameterValue> {
  const analytic =
    saved?.kind === 'local-moran-polygons' || saved?.kind === 'hot-spots-polygons'
      ? saved
      : undefined;
  const moran = saved?.kind === 'local-moran-polygons' ? saved : undefined;
  const quantile = saved?.kind === 'quantile-lisa-polygons' ? saved : undefined;
  const rate = saved?.kind === 'spatial-rate-polygons' ? saved : undefined;
  return {
    kind: saved?.kind ?? 'local-moran-polygons',
    geometryColumn: saved?.weights.geometryColumn ?? geometryColumn,
    criterion: saved?.weights.criterion ?? 'queen',
    neighborCapacity:
      saved?.weights.neighborCapacity ??
      Math.max(1, Math.min(rowCount * 8, MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY)),
    pairCapacity:
      saved?.weights.pairCapacity ??
      Math.max(1, Math.min(rowCount * 32, MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY)),
    populationColumn: rate?.populations ?? populationColumn,
    scale: rate?.scale ?? 1,
    islandPolicy: rate?.islandPolicy ?? 'undefined',
    minimumNeighbors: analytic?.minimumNeighbors ?? 4,
    selfWeight: saved?.kind === 'hot-spots-polygons' ? saved.selfWeight : 1,
    multipleTesting: analytic?.multipleTesting ?? 'benjamini-hochberg',
    islands: analytic?.islands ?? 'exclude',
    significanceLevel: moran?.significanceLevel ?? 0.05,
    inference: moran?.inference ?? 'analytic-conditional-normal-two-sided',
    permutationCount:
      moran?.inference === 'permutation-conditional-two-sided'
        ? (moran.permutationCount ?? 999)
        : 999,
    seed: moran?.inference === 'permutation-conditional-two-sided' ? (moran.seed ?? 0) : 0,
    quantileCount: quantile?.quantileCount ?? 2,
    selectedQuantile: quantile?.selectedQuantile ?? 2,
    quantileSeed: quantile?.seed ?? 314159,
    quantilePermutations: quantile?.permutationCount ?? 199,
    quantileSignificance: quantile?.significanceLevel ?? 0.05,
    editedFields: '[]'
  };
}

export function getPolygonSheetWeights(values: PointSheetValues): KeplerPolygonWeightsSettings {
  return parseKeplerPolygonWeightsSettings({
    kind: 'contiguity',
    geometryColumn: values['geometryColumn'],
    criterion: values['criterion'],
    equality: 'exact-source-xy',
    neighborCapacity: readResidentSheetNumber(values, 'neighborCapacity'),
    pairCapacity: readResidentSheetNumber(values, 'pairCapacity')
  });
}

export function getPolygonSheetOperation(
  values: PointSheetValues,
  column: string,
  rowCount: number
): KeplerPolygonAnalysisOperation {
  const weights = getPolygonSheetWeights(values);
  const number = (name: string) => readResidentSheetNumber(values, name);
  const kind = values['kind'] as KeplerPolygonAnalysisOperation['kind'];
  if (!Number.isSafeInteger(rowCount) || rowCount < 1 || rowCount > MAXIMUM_KEPLER_POLYGON_ROWS)
    throw new Error(`Polygon analysis supports 1 to ${MAXIMUM_KEPLER_POLYGON_ROWS} source rows`);
  let operation: KeplerPolygonAnalysisOperation;
  if (kind === 'spatial-rate-polygons')
    operation = {
      kind,
      events: column,
      populations: String(values['populationColumn']),
      weights,
      scale: number('scale'),
      islandPolicy: values['islandPolicy'] as 'undefined' | 'own-rate'
    };
  else if (kind === 'hot-spots-polygons')
    operation = {
      kind,
      column,
      weights,
      minimumNeighbors: number('minimumNeighbors'),
      selfWeight: number('selfWeight'),
      multipleTesting: values['multipleTesting'] as KeplerMultipleTestingMethod,
      islands: values['islands'] as 'exclude' | 'keep',
      inference: 'analytic-two-sided'
    };
  else if (kind === 'quantile-lisa-polygons') {
    if (rowCount < 3 || number('quantileCount') >= rowCount)
      throw new Error(
        'Quantile LISA requires at least 3 source rows and fewer quantiles than rows'
      );
    operation = {
      kind,
      column,
      weights,
      quantileCount: number('quantileCount'),
      selectedQuantile: number('selectedQuantile'),
      seed: number('quantileSeed'),
      permutationCount: number('quantilePermutations'),
      significanceLevel: number('quantileSignificance'),
      inference: 'legacy-directed-pseudo',
      multipleTesting: 'none',
      islands: 'exclude',
      quantileMethod: 'type-5'
    };
  } else if (kind === 'local-moran-polygons')
    operation = {
      kind,
      column,
      weights,
      minimumNeighbors:
        values['inference'] === 'permutation-conditional-two-sided'
          ? 1
          : number('minimumNeighbors'),
      multipleTesting: values['multipleTesting'] as KeplerMultipleTestingMethod,
      islands: values['islands'] as 'exclude' | 'keep',
      significanceLevel: number('significanceLevel'),
      ...(values['inference'] === 'permutation-conditional-two-sided'
        ? {
            inference: 'permutation-conditional-two-sided',
            permutationCount: number('permutationCount'),
            seed: number('seed')
          }
        : {inference: values['inference'] as 'analytic-conditional-normal-two-sided'})
    };
  else throw new Error('Choose a supported polygon statistic');
  return normalizePolygonAnalysisOperation(operation);
}

export function createPolygonSheetSpecifications(
  values: PointSheetValues,
  rowCount: number
): PointSheetSpecification[] {
  const number = (
    name: string,
    label: string,
    minimum: number,
    maximum?: number,
    kind: 'integer' | 'number' = 'integer',
    level: 'basic' | 'advanced' = 'advanced'
  ): PointSheetSpecification => ({
    name,
    label,
    controlId: `analysis-polygon-${name}`,
    kind,
    minimum,
    maximum,
    level
  });
  const enumeration = (
    name: string,
    label: string,
    controlId: string,
    options: readonly string[],
    level: 'basic' | 'advanced' = 'basic'
  ): PointSheetSpecification => ({name, label, controlId, kind: 'enum', options, level});
  const quantile = values['kind'] === 'quantile-lisa-polygons';
  const moran = values['kind'] === 'local-moran-polygons';
  const permutation = moran && values['inference'] === 'permutation-conditional-two-sided';
  return [
    enumeration('kind', 'Polygon statistic', 'analysis-polygon-statistic', [
      'spatial-rate-polygons',
      'hot-spots-polygons',
      'local-moran-polygons',
      'quantile-lisa-polygons'
    ]),
    {
      ...makePointSheetColumn(
        'geometryColumn',
        'Polygon geometry field',
        'analysis-polygon-geometry'
      ),
      columnsFrom: 'polygons',
      fieldRole: {
        parameter: 'geometryColumn',
        role: 'geometry',
        label: 'Polygon geometry field',
        accepts: ['category'],
        required: true
      }
    },
    enumeration('criterion', 'Polygon contiguity rule', 'analysis-polygon-criterion', [
      'queen',
      'rook'
    ]),
    number(
      'neighborCapacity',
      'Polygon directed neighbor capacity',
      1,
      MAXIMUM_KEPLER_POLYGON_NEIGHBOR_CAPACITY
    ),
    number(
      'pairCapacity',
      'Polygon candidate pair capacity',
      1,
      MAXIMUM_KEPLER_POLYGON_PAIR_CAPACITY
    ),
    ...(values['kind'] === 'spatial-rate-polygons'
      ? [
          makePointSheetColumn(
            'populationColumn',
            'Polygon rate population field',
            'analysis-polygon-population'
          ),
          number('scale', 'Polygon rate scale', 0, undefined, 'number', 'basic'),
          enumeration(
            'islandPolicy',
            'Polygon rate rows without neighbors',
            'analysis-polygon-island-policy',
            ['undefined', 'own-rate'],
            'advanced'
          )
        ]
      : quantile
        ? [
            {
              ...number(
                'quantileCount',
                'Polygon Quantile LISA quantile count',
                2,
                rowCount - 1,
                'integer',
                'basic'
              ),
              controlId: 'analysis-polygon-quantile-quantileCount'
            },
            {
              ...number(
                'selectedQuantile',
                'Polygon Quantile LISA selected quantile',
                1,
                Number(values['quantileCount']),
                'integer',
                'basic'
              ),
              controlId: 'analysis-polygon-quantile-selectedQuantile'
            },
            {
              ...number('quantileSeed', 'Polygon Quantile LISA seed', 0, Number.MAX_SAFE_INTEGER),
              controlId: 'analysis-polygon-quantile-seed'
            },
            {
              ...number(
                'quantilePermutations',
                'Polygon Quantile LISA permutations',
                1,
                MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS
              ),
              controlId: 'analysis-polygon-quantile-permutationCount'
            },
            {
              ...number(
                'quantileSignificance',
                'Polygon Quantile LISA significance level',
                0,
                1,
                'number'
              ),
              controlId: 'analysis-polygon-quantile-significanceLevel'
            }
          ]
        : [
            ...(moran
              ? [
                  enumeration(
                    'inference',
                    'Polygon Local Moran inference',
                    'analysis-polygon-inference',
                    ['analytic-conditional-normal-two-sided', 'permutation-conditional-two-sided']
                  ),
                  number(
                    'significanceLevel',
                    'Polygon Local Moran significance level',
                    0,
                    1,
                    'number'
                  )
                ]
              : [number('selfWeight', 'Polygon hot-spot self weight', 0, undefined, 'number')]),
            ...(!permutation
              ? [number('minimumNeighbors', 'Polygon minimum neighbors', 1)]
              : [
                  number('permutationCount', 'Polygon Local Moran permutations', 1, 9999),
                  number('seed', 'Polygon Local Moran permutation seed', 0, 0xffffffff)
                ]),
            enumeration(
              'islands',
              'Polygon isolated rows in global moments',
              'analysis-polygon-islands',
              ['exclude', 'keep'],
              'advanced'
            ),
            enumeration(
              'multipleTesting',
              'Polygon multiple-testing correction',
              'analysis-polygon-correction',
              POINT_SHEET_CORRECTIONS,
              'advanced'
            )
          ])
  ];
}

/** Retain the binary source-wide contiguity method and its four recipe contracts. */
export function PolygonPanel({analysis}: {analysis: AnalysisRuntime}) {
  const initialization = useAnalysisFormInitialization(analysis);
  const saved = analysis.settings.polygon;
  const geometry = analysis.availablePolygonColumns.includes(initialization.geometryColumn)
    ? initialization.geometryColumn
    : (analysis.availablePolygonColumns[0] ?? '');
  const sheet = usePointAnalysisSheet(
    makePolygonSheetDefaults(
      saved,
      geometry,
      analysis.settings.denominatorColumn,
      analysis.rowCount
    ),
    saved,
    analysis.coordinateRestorationKey ?? analysis.datasetId
  );
  if (!analysis.availablePolygonColumns.length) return null;
  const values = sheet.values;
  const fields = createResidentSheetFields(
    values,
    createPolygonSheetSpecifications(values, analysis.rowCount),
    name => {
      if (sheet.source(name) === 'User override') return 'User override';
      const savedField =
        saved &&
        (['kind', 'geometryColumn', 'criterion', 'neighborCapacity', 'pairCapacity'].includes(
          name
        ) ||
          (saved.kind === 'spatial-rate-polygons' &&
            ['populationColumn', 'scale', 'islandPolicy'].includes(name)) ||
          ((saved.kind === 'local-moran-polygons' || saved.kind === 'hot-spots-polygons') &&
            ['minimumNeighbors', 'multipleTesting', 'islands'].includes(name)) ||
          (saved.kind === 'hot-spots-polygons' && name === 'selfWeight') ||
          (saved.kind === 'local-moran-polygons' &&
            ['inference', 'significanceLevel'].includes(name)) ||
          (saved.kind === 'local-moran-polygons' &&
            saved.inference === 'permutation-conditional-two-sided' &&
            ['seed', 'permutationCount'].includes(name)) ||
          (saved.kind === 'quantile-lisa-polygons' &&
            [
              'quantileCount',
              'selectedQuantile',
              'quantileSeed',
              'quantilePermutations',
              'quantileSignificance'
            ].includes(name)));
      if (savedField) return 'Saved resident settings';
      if (name === 'geometryColumn')
        return 'Publication polygon role or first available polygon field; check the mapping';
      if (name === 'populationColumn') return 'Numeric population field above';
      if (name === 'neighborCapacity')
        return '8 directed slots per source row, clamped to the polygon neighbor limit';
      if (name === 'pairCapacity')
        return '32 candidate pairs per source row, clamped to the polygon pair limit';
      return `Declared resident default: ${String(values[name])}`;
    },
    name =>
      name === 'geometryColumn' ? analysis.availablePolygonColumns : analysis.availableColumns
  );
  const checks = getResidentSheetChecks(fields, () =>
    getPolygonSheetOperation(values, analysis.settings.column, analysis.rowCount)
  );
  if (!analysis.availableColumns.includes(analysis.settings.column))
    checks.push('Choose an available numeric value field above.');
  return (
    <details className="analysis-polygon-settings">
      <summary>Configure polygon contiguity analysis</summary>
      <ResidentAnalysisSheetForm
        key={JSON.stringify([
          analysis.coordinateRestorationKey,
          saved,
          values['kind'],
          values['inference']
        ])}
        fields={fields}
        checks={checks}
        method={`Resident polygon analysis uses the numeric field above (${analysis.settings.column}) and binary contiguity weights. Catalog Local Moran uses row-standardized weights, 999 permutations and Benjamini–Hochberg correction by default. Resident Local Moran defaults to analytic conditional-normal two-sided inference; these are different methods. Each source row retains its polygon parts, holes and GeometryCollections. Queen requires an identical shared vertex; rook requires identical endpoints of a complete edge. Matching uses exact ORIGINAL source X/Y equality tokens with Z ignored. No snapping, edge splitting, partial-overlap repair or antimeridian normalization is applied.`}
        submitLabel="Apply contiguity and run polygon analysis"
        onDraftChange={sheet.markEdited}
        onChange={sheet.change}
        onApply={current =>
          analysis.configurePolygon(
            getPolygonSheetOperation(
              {...values, ...current},
              analysis.settings.column,
              analysis.rowCount
            )
          )
        }
      >
        {({disabled, readValues}) =>
          analysis.weightsStore && (
            <AnalysisWeightsPanel
              analysis={analysis}
              family="polygon"
              draftDisabled={disabled}
              getSettings={() => getPolygonSheetWeights({...values, ...readValues()})}
            />
          )
        }
      </ResidentAnalysisSheetForm>
      <p className="hint">
        Directed neighbor capacity stores two slots per symmetric pair. Candidate pair capacity
        includes repeated contacts before deduplication. Overflow prevents publication; after
        candidate overflow the required neighbor count is only a lower bound. Increase capacity and
        apply again. At most {MAXIMUM_KEPLER_POLYGON_ROWS.toLocaleString()} source rows and{' '}
        {MAXIMUM_KEPLER_POLYGON_VERTICES.toLocaleString()} polygon vertices; vertex capacity is
        checked when applying. Empty, null and nonpolygon rows retain identity and are excluded.
        Selection filters the fixed source-wide graph without rebuilding topology.
      </p>
      {values['kind'] === 'spatial-rate-polygons' ? (
        <p className="hint">
          Rates pool each row's own events and population with eligible neighbors, then apply the
          scale.
        </p>
      ) : values['kind'] === 'quantile-lisa-polygons' ? (
        <p className="hint">
          Quantile LISA uses type-5 quantiles, including the lower boundary and excluding the upper.
          Ties stay together and may leave empty quantiles. Missing values and nonpolygon rows are
          omitted. Legacy directed pseudo p-values use (min(upper-tail count, permutations −
          upper-tail count) + 1) / (permutations + 1). This anti-conservative exploratory method has
          no two-sided error-control guarantee and no multiple-testing correction. Tied or constant
          memberships can appear significant. Only selected-quantile members with positive local
          joins are tested; islands are excluded from inference but can remain in the quantile
          distribution. More than 64 active neighbors makes inference unavailable. Gray includes
          nonsignificant and untested rows, red marks legacy significance, orange marks islands. The
          table retains membership, join count, pseudo p-value and tail count. A fixed seed
          reproduces this implementation, without GeoDa random-stream parity.
        </p>
      ) : (
        <p className="hint">
          {values['inference'] === 'permutation-conditional-two-sided' &&
          values['kind'] === 'local-moran-polygons'
            ? 'Conditional permutations fix the focus value and draw distinct other observations. Minimum neighbors is fixed at 1. The raw two-sided p-value floor is 2 / (P + 1); all ties give p = 1. More than 64 active neighbors makes inference unavailable. The z-score remains a separate analytic diagnostic. Seeded draws reproduce this implementation without GeoDa random-stream parity.'
            : values['kind'] === 'local-moran-polygons'
              ? 'Local Moran uses a two-sided conditional-normal approximation; corrected p-values gate association quadrants. Small neighborhoods and skewed values can make analytic approximations poor.'
              : 'Hot spots use analytic two-sided inference. Self weight 1 gives Gi*; 0 gives Gi. Corrected p-values set significance bins. Small neighborhoods and skewed values can make analytic approximations poor.'}
        </p>
      )}
    </details>
  );
}
