// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Bounded planning for the native Kepler Suitability operation (pinned 3.3.0-alpha.18). Native
 * semantics: numeric columns with weights in [0, 1]; each weighted column is standardized
 * (`range`, `zScore` or `raw`); weights are normalized by their sum (or kept `raw`); a row scores
 * the weighted sum and is null when any weighted value is missing. There are no constraints and no
 * rasters in the native operation, so there is nothing raster-shaped to execute or reject here.
 *
 * The plan targets GPU standardization (`min-max` or `z-score` scaler) of each resident column
 * followed by a weighted sum. Where the kernels cannot reproduce native results exactly the plan
 * rejects instead of approximating.
 */
export const MAXIMUM_RESIDENT_SUITABILITY_INDICATORS = 128;

export type ResidentSuitabilityOperation = {
  id: string;
  dataId: string;
  weights: Readonly<Record<string, number>>;
  weightStandardization: 'normalize' | 'raw';
  dataStandardization: 'range' | 'zScore' | 'raw';
  outputFieldName: string;
  columns?: readonly string[];
  /**
   * Optional reclassification of categorical (string dictionary) weighted fields: label to score.
   * Without it a categorical field scores each label by its dictionary position (0, 1, 2, ...).
   * Labels missing from a supplied map are undefined, so their rows score null.
   */
  categoryScores?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  /**
   * Rows missing in only some weighted fields. `refuse` (default) stops the run; `exclude` scores
   * such rows null; `renormalize` scores them over their defined fields with the defined weights
   * rescaled to the full weight total. Rows missing in every field are always null.
   */
  missingValues?: ResidentSuitabilityMissingValuePolicy;
  resultId: string;
  resultLabel: string;
};
export type ResidentSuitabilityMissingValuePolicy = 'refuse' | 'exclude' | 'renormalize';
export type ResidentSuitabilityField = {name: string; type: string};
export type ResidentSuitabilityPlan = {
  operationId: string;
  datasetId: string;
  indicators: readonly {column: string; weight: number}[];
  /** `none` is native raw data standardization: values are weighted unchanged. */
  scaler: 'min-max' | 'z-score' | 'none';
  aggregation: 'weighted-sum';
  /** Raw weights keep their magnitude (the total); normalized weights divide by it (scale 1). */
  scoreScale: number;
  /** Native range scaling leaves a constant column unchanged; the kernels yield 0, so reject it. */
  rejectConstantColumns: true;
  /** Partially missing rows: refuse the run, null the score (exclude) or renormalize the weights. */
  missingValuePolicy: ResidentSuitabilityMissingValuePolicy;
  includedColumns: readonly string[];
  scoreColumn: string;
  resultId: string;
  resultLabel: string;
  /** Divisor applied to every weight so the largest is 1 (1 when all weights already lie in [0, 1]). */
  weightRescale: number;
  /** Visible policy notes: rescaled weights, reclassified categorical fields. */
  notes: string[];
};

// Strings are admitted here and routed through reclassification by the result plan.
const SCORABLE_TYPES = new Set(['integer', 'real', 'string']);

export function createResidentSuitabilityPlan(
  operation: ResidentSuitabilityOperation,
  fields: readonly ResidentSuitabilityField[]
): ResidentSuitabilityPlan {
  const entries = Object.entries(operation.weights);
  if (!entries.length) throw new Error('Select at least one field to score');
  if (entries.length > MAXIMUM_RESIDENT_SUITABILITY_INDICATORS)
    throw new Error(
      `Resident suitability scores at most ${MAXIMUM_RESIDENT_SUITABILITY_INDICATORS} fields`
    );
  if (!['range', 'zScore', 'raw'].includes(operation.dataStandardization))
    throw new Error('Unknown suitability data standardization');
  if (!['normalize', 'raw'].includes(operation.weightStandardization))
    throw new Error('Unknown suitability weight standardization');
  const missingValuePolicy = operation.missingValues ?? 'refuse';
  if (!['refuse', 'exclude', 'renormalize'].includes(missingValuePolicy))
    throw new Error('Unknown suitability missing-value policy');
  const indicators = entries.map(([column, weight]) => {
    const field = fields.find(candidate => candidate.name === column);
    if (!field) throw new Error(`Suitability field "${column}" was not found`);
    if (!SCORABLE_TYPES.has(field.type))
      throw new Error(`Suitability field "${column}" is not numeric or categorical`);
    // Negative weights would change the normalization meaning; weights above 1 are rescaled below.
    if (!Number.isFinite(weight) || weight < 0)
      throw new Error(`Suitability weight for "${column}" must be a nonnegative number`);
    return {column, weight};
  });
  const notes: string[] = [];
  const largestWeight = Math.max(...indicators.map(indicator => indicator.weight));
  const weightRescale = largestWeight > 1 ? largestWeight : 1;
  if (weightRescale !== 1) {
    for (const indicator of indicators) indicator.weight /= weightRescale;
    notes.push(
      `Weights above 1 were rescaled by 1/${weightRescale} so the largest is 1` +
        (operation.weightStandardization === 'raw'
          ? '; raw scores shrink by the same factor.'
          : '; normalized scores are unchanged.')
    );
  }
  if (missingValuePolicy === 'exclude')
    notes.push('Rows missing in some weighted fields score null (excluded from scoring).');
  if (missingValuePolicy === 'renormalize')
    notes.push(
      'Rows missing in some weighted fields are scored over their defined fields with the defined weights rescaled to the full weight total.'
    );
  const total = indicators.reduce((sum, indicator) => sum + indicator.weight, 0);
  // Native leaves an all-zero normalized weight vector unchanged, scoring 0; the kernel yields NaN.
  if (total === 0) throw new Error('At least one suitability weight must be positive');
  const includedColumns = operation.columns
    ? fields.map(field => field.name).filter(name => operation.columns!.includes(name))
    : fields.map(field => field.name);
  const usedNames = new Set(includedColumns);
  // Native uniqueColumnName: the bare name, then name_1, name_2, ...
  const scoreBase = operation.outputFieldName || 'score';
  let scoreColumn = scoreBase;
  for (let suffix = 1; usedNames.has(scoreColumn); suffix++) scoreColumn = `${scoreBase}_${suffix}`;
  return {
    operationId: operation.id,
    datasetId: operation.dataId,
    indicators,
    scaler:
      operation.dataStandardization === 'range'
        ? 'min-max'
        : operation.dataStandardization === 'raw'
          ? 'none'
          : 'z-score',
    aggregation: 'weighted-sum',
    scoreScale: operation.weightStandardization === 'raw' ? total : 1,
    rejectConstantColumns: true,
    missingValuePolicy,
    includedColumns,
    scoreColumn,
    resultId: operation.resultId,
    resultLabel: operation.resultLabel,
    weightRescale,
    notes
  };
}

//
// Result planning: carried columns, output schema, bounds and lineage for the resident executor.
//

/**
 * Row capacity of the GPU standardization and reduction kernels (one uint32 count per column).
 * This is a kernel capacity, not a policy cap: memory admission is decided by the session's
 * reservations, and the source's chunk layout is preserved.
 */
export const RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT = 2 ** 24;
/** Carried columns plus the score. */
export const MAXIMUM_RESIDENT_SUITABILITY_COLUMNS = 32;

export type ResidentSuitabilityColumnFormat = 'float32' | 'sint32' | 'uint32';
export type ResidentSuitabilityResultField = {name: string; type: string; analyzerType: string};
export type ResidentSuitabilityColumn = {
  sourceColumn: string;
  outputColumn: string;
  type: string;
  analyzerType: string;
  format: ResidentSuitabilityColumnFormat;
  /** Dictionary labels are shared with the source; only codes are copied. */
  dictionary: boolean;
};
export type ResidentSuitabilityProvenance = {
  schemaVersion: 1;
  kind: 'suitability';
  source: {datasetId: string; revision: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  indicators: readonly {column: string; weight: number}[];
  weightStandardization: 'normalize' | 'raw';
  dataStandardization: 'range' | 'zScore' | 'raw';
  scoreColumn: string;
  columns: readonly {column: string; name: string}[];
  fields: readonly ResidentSuitabilityResultField[];
  scorePolicy:
    | 'float32-composite-score-aligned-missing-values-null'
    | 'float32-composite-score-renormalized-defined-weights';
  /** Absent means `refuse`, the original policy. */
  missingValues?: ResidentSuitabilityMissingValuePolicy;
  rowOrder: 'source-order';
  /** Label scores of reclassified categorical weighted fields (null scores are undefined rows). */
  reclassifications?: readonly {column: string; scores: readonly (number | null)[]}[];
};
export type ResidentSuitabilitySourceMetadata = {
  datasetId: string;
  label: string;
  revision: number;
  rowCount: number;
  fields: readonly {name: string; type: string; analyzerType?: string}[];
  columns: Readonly<Record<string, {format?: string}>>;
  dictionaries: Readonly<Record<string, {values: readonly string[]; ordered: boolean}>>;
  nullCounts?: Readonly<Record<string, readonly number[]>>;
};
export type ResidentSuitabilityResultPlan = {
  source: {datasetId: string; revision: number; rowCount: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  scoring: ResidentSuitabilityPlan;
  /** Format of each weighted column, in indicator order. */
  indicatorFormats: readonly ResidentSuitabilityColumnFormat[];
  /**
   * Per indicator: null for numeric fields; for categorical fields the score of each dictionary
   * code, with one extra trailing entry for the null code (always undefined).
   */
  indicatorReclassifications: readonly ((number | null)[] | null)[];
  columns: readonly ResidentSuitabilityColumn[];
  provenance: ResidentSuitabilityProvenance;
  derivedDataset: {type: 'suitability'; sourceDataIds: [string]; operationId: string};
};

const GUIDANCE =
  ' Narrow the weighted fields or Columns, or prepare a smaller numeric table with SQL; the resident scorer never falls back to retained rows.';

/** Admit metadata before reading any indicator or allocating output storage. */
export function createResidentSuitabilityResultPlan(
  operation: ResidentSuitabilityOperation,
  source: ResidentSuitabilitySourceMetadata
): ResidentSuitabilityResultPlan {
  const fail = (message: string): never => {
    throw new Error(`${message}${GUIDANCE}`);
  };
  if (source.datasetId !== operation.dataId) fail('The Suitability source has changed.');
  if (!operation.id.trim() || !operation.resultId.trim() || !operation.resultLabel.trim())
    fail('Invalid Suitability operation identity.');
  if (operation.resultId === source.datasetId)
    fail('Suitability must create a separate result dataset.');
  if (
    !Number.isSafeInteger(source.rowCount) ||
    source.rowCount < 1 ||
    source.rowCount > RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT
  )
    fail(
      `Suitability input must have 1 to ${RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT} rows (found ${source.rowCount}).`
    );
  let scoring: ResidentSuitabilityPlan;
  try {
    scoring = createResidentSuitabilityPlan(operation, source.fields);
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
  const supportedFormat = (name: string): ResidentSuitabilityColumnFormat | undefined => {
    const field = source.fields.find(candidate => candidate.name === name);
    const format = source.columns[name]?.format;
    const dictionary = Object.hasOwn(source.dictionaries, name)
      ? source.dictionaries[name]
      : undefined;
    if (!field) return undefined;
    if (format === 'float32' && !dictionary && ['integer', 'real'].includes(field.type))
      return format;
    if (format === 'sint32' && !dictionary && field.type === 'integer') return format;
    if (
      format === 'uint32' &&
      (dictionary ? field.type === 'string' && !dictionary.ordered : field.type === 'integer')
    )
      return format;
    return undefined;
  };
  const indicatorReclassifications = scoring.indicators.map(({column}) => {
    const dictionary = Object.hasOwn(source.dictionaries, column)
      ? source.dictionaries[column]
      : undefined;
    if (!dictionary) return null;
    const field = source.fields.find(candidate => candidate.name === column);
    if (
      field?.type !== 'string' ||
      dictionary.ordered ||
      source.columns[column]?.format !== 'uint32'
    )
      return fail(
        `Suitability categorical field "${column}" needs an unordered prepared string dictionary.`
      );
    const supplied = operation.categoryScores?.[column];
    if (supplied)
      for (const score of Object.values(supplied))
        if (!Number.isFinite(score))
          return fail(`Suitability category scores for "${column}" must be finite numbers.`);
    const scores: (number | null)[] = dictionary.values.map((label, index) =>
      supplied ? (Object.hasOwn(supplied, label) ? supplied[label] : null) : index
    );
    scores.push(null);
    scoring.notes.push(
      supplied
        ? `Categorical field "${column}" was reclassified by the supplied category scores.`
        : `Categorical field "${column}" was reclassified by dictionary position (0 to ${dictionary.values.length - 1}); supply category scores to control the order.`
    );
    return scores;
  });
  const indicatorFormats = scoring.indicators.map(({column}, index) => {
    const format = supportedFormat(column);
    if (indicatorReclassifications[index]) return 'uint32' as const;
    if (!format || Object.hasOwn(source.dictionaries, column))
      return fail(
        `Suitability field "${column}" needs prepared Float32, Int32 or Uint32 numeric storage.`
      );
    return format;
  });
  const usedNames = new Set<string>();
  const columns: ResidentSuitabilityColumn[] = scoring.includedColumns.map(name => {
    const field = source.fields.find(candidate => candidate.name === name)!;
    const format = supportedFormat(name);
    if (!format)
      return fail(
        `Resident Suitability cannot carry column ${name} (${field.type}); deselect it in Columns to include.`
      );
    let outputColumn = name;
    for (let suffix = 1; usedNames.has(outputColumn); suffix++) outputColumn = `${name}_${suffix}`;
    usedNames.add(outputColumn);
    return {
      sourceColumn: name,
      outputColumn,
      type: field.type,
      analyzerType: field.analyzerType || field.type,
      format,
      dictionary: Object.hasOwn(source.dictionaries, name)
    };
  });
  // Re-derive the score name against the carried output names, as native does.
  let scoreColumn = scoring.scoreColumn;
  for (let suffix = 1; usedNames.has(scoreColumn); suffix++)
    scoreColumn = `${scoring.scoreColumn}_${suffix}`;
  if (columns.length + 1 > MAXIMUM_RESIDENT_SUITABILITY_COLUMNS)
    fail(`Suitability supports at most ${MAXIMUM_RESIDENT_SUITABILITY_COLUMNS} output columns.`);
  const fields = [
    ...columns.map(column => ({
      name: column.outputColumn,
      type: column.type,
      analyzerType: column.analyzerType
    })),
    {name: scoreColumn, type: 'real', analyzerType: 'real'}
  ];
  const provenance: ResidentSuitabilityProvenance = {
    schemaVersion: 1,
    kind: 'suitability',
    source: {datasetId: source.datasetId, revision: source.revision},
    operationId: scoring.operationId,
    resultId: operation.resultId,
    resultLabel: operation.resultLabel,
    indicators: scoring.indicators.map(indicator => ({...indicator})),
    weightStandardization: operation.weightStandardization,
    dataStandardization: operation.dataStandardization,
    scoreColumn,
    columns: columns.map(column => ({column: column.sourceColumn, name: column.outputColumn})),
    fields,
    scorePolicy:
      scoring.missingValuePolicy === 'renormalize'
        ? 'float32-composite-score-renormalized-defined-weights'
        : 'float32-composite-score-aligned-missing-values-null',
    ...(scoring.missingValuePolicy === 'refuse' ? {} : {missingValues: scoring.missingValuePolicy}),
    rowOrder: 'source-order',
    ...(indicatorReclassifications.some(Boolean)
      ? {
          reclassifications: scoring.indicators.flatMap(({column}, index) =>
            indicatorReclassifications[index]
              ? [{column, scores: indicatorReclassifications[index]}]
              : []
          )
        }
      : {})
  };
  return {
    source: {datasetId: source.datasetId, revision: source.revision, rowCount: source.rowCount},
    operationId: scoring.operationId,
    resultId: operation.resultId,
    resultLabel: operation.resultLabel,
    scoring: {...scoring, scoreColumn},
    indicatorFormats,
    indicatorReclassifications,
    columns,
    provenance,
    derivedDataset: {
      type: 'suitability',
      sourceDataIds: [source.datasetId],
      operationId: scoring.operationId
    }
  };
}

/**
 * Independent reference for the pinned native operation: the acceptance oracle for GPU results.
 * Not a runtime fallback. Values are numbers, bigints or null.
 */
export function computeNativeSuitabilityScores(
  columns: readonly (readonly (number | bigint | null | undefined)[])[],
  weights: readonly number[],
  options: {
    dataStandardization: 'range' | 'zScore' | 'raw';
    weightStandardization: 'normalize' | 'raw';
    /** `renormalize` scores partial rows over their defined columns; default is null (native). */
    missingValues?: ResidentSuitabilityMissingValuePolicy;
  }
): (number | null)[] {
  const toNumber = (value: number | bigint | null | undefined): number | null =>
    typeof value === 'number'
      ? Number.isFinite(value)
        ? value
        : null
      : typeof value === 'bigint'
        ? Number(value)
        : null;
  const standardized = columns.map(column => {
    const values = column.map(toNumber);
    if (options.dataStandardization === 'range') {
      let minimum = Infinity;
      let maximum = -Infinity;
      for (const value of values) {
        if (value === null) continue;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
      const span = maximum - minimum;
      return !Number.isFinite(span) || span === 0
        ? values
        : values.map(value => (value === null ? null : (value - minimum) / span));
    }
    if (options.dataStandardization === 'zScore') {
      const defined = values.filter((value): value is number => value !== null);
      if (!defined.length) return values;
      const mean = defined.reduce((sum, value) => sum + value, 0) / defined.length;
      const deviation = Math.sqrt(
        defined.reduce((sum, value) => sum + (value - mean) ** 2, 0) / defined.length
      );
      return deviation === 0
        ? values
        : values.map(value => (value === null ? null : (value - mean) / deviation));
    }
    return values;
  });
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const effective =
    options.weightStandardization === 'normalize' && total !== 0
      ? weights.map(weight => weight / total)
      : weights;
  return Array.from({length: columns[0]?.length ?? 0}, (_, row) => {
    let score = 0;
    let definedWeight = 0;
    let definedCount = 0;
    for (let column = 0; column < standardized.length; column++) {
      const value = standardized[column][row];
      if (value === null) {
        if (options.missingValues !== 'renormalize') return null;
        continue;
      }
      score += value * effective[column];
      definedWeight += effective[column];
      definedCount++;
    }
    if (options.missingValues === 'renormalize') {
      const totalWeight = effective.reduce((sum, weight) => sum + weight, 0);
      return definedCount > 0 && definedWeight > 0 ? (score * totalWeight) / definedWeight : null;
    }
    return score;
  });
}

/** Per-column multipliers applied to standardized values: weights, divided by their sum when normalized. */
export function getResidentSuitabilityEffectiveWeights(
  scoring: Pick<ResidentSuitabilityPlan, 'indicators'>,
  weightStandardization: 'normalize' | 'raw'
): number[] {
  const total = scoring.indicators.reduce((sum, indicator) => sum + indicator.weight, 0);
  return scoring.indicators.map(({weight}) =>
    weightStandardization === 'normalize' ? weight / total : weight
  );
}
