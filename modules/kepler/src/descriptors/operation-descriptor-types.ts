// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {
  GPUAnalysisDefaultRule,
  GPUAnalysisParameterCondition,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisInterpretationSpec,
  GPUAnalysisPresentationSpec,
  GPUAnalysisQuestion,
  GPUAnalysisRequirementSpec,
  GPUAnalysisRowSemantics,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// What a catalog binding means to a user, kept beside the binding and joined to it by id. The
// binding says how an operation runs (inputs, parameters, engines); the descriptor says which
// question it answers, which columns it reads and what they should measure, what must hold
// before it runs, which defaults come from the data, and how its result is shown and read.

/** Geometry kinds a catalog input can hold. */
export type KeplerOperationGeometry = 'point' | 'line' | 'polygon';

/** A column an input reads, and the binding parameter that receives the column name. */
export type KeplerOperationFieldDescriptor = GPUAnalysisFieldRoleSpec & {
  /**
   * Measurement levels the method is meant for when the input holds this geometry, replacing
   * `measurement`. A total (count or amount) per area mostly measures the size of the area, so
   * an area statistic that accepts amounts for point objects should list only rates for polygons.
   */
  measurementByGeometry?: Partial<
    Record<KeplerOperationGeometry, readonly GPUAnalysisMeasurementLevel[]>
  >;
  /**
   * `CatalogParameterSpec.name` of the string parameter that holds the column name. A role with
   * `multiple` writes a comma-separated list, as the binding's parameter documents.
   */
  parameter: string;
  /**
   * Optional computation inputs that must be chosen explicitly. Applicability offers candidates
   * without assigning one; parameter resolution uses an empty column name until a mapping or
   * override is supplied, even if the binding has a nonempty default column name.
   */
  // `explicit` also skips inference but preserves the binding default (for feature-ID fallback).
  // Point-coordinate policies use declared pointColumns, never generic numeric measurements.
  autoAssign?: false | 'explicit' | 'point-longitude' | 'point-latitude';
};

/** What one binding input stands for. */
export type KeplerOperationInputDescriptor = {
  /** `CatalogInputSpec.name`. */
  name: string;
  /** What the features are, in the ontology's data kinds (`points`, `polygons`, `network`, ...). */
  kind: GPUSpatialAnalysisDataKind;
  /** Row semantics the method is meant for; absent when any will do. */
  rowSemantics?: readonly GPUAnalysisRowSemantics[];
  fields: readonly KeplerOperationFieldDescriptor[];
};

/** Where one parameter's default comes from and how prominent it is. */
export type KeplerOperationParameterDescriptor = {
  /** `CatalogParameterSpec.name`. */
  name: string;
  /**
   * `constant`: the binding's `defaultValue` stands.
   * `data`: derived from the profile of the named input, multiplied by `factor` (default 1) and,
   * when `factorByParameter` is given, by the factor listed for that parameter's current value
   * (a kernel whose bandwidth means a support radius rather than a standard deviation). The
   * `column-*` rules read the column bound to `role`. The binding's `defaultValue` is the fallback.
   * `preset`: this layer's choice of a better starting value than the binding's default, with the
   * reason shown to the user (999 permutations; correction on). With `whenGeometry` it applies
   * only when the named input holds that geometry (queen contiguity for polygons).
   * `field-measurement`: chosen from what the column bound to `role` measures (an extensive
   * total or an intensive rate); the binding's default stands when the level is not listed.
   */
  defaultPolicy:
    | {from: 'constant'}
    | {
        from: 'data';
        input: string;
        rule: GPUAnalysisDefaultRule;
        factor?: number;
        role?: string;
        factorByParameter?: {parameter: string; factors: Readonly<Record<string, number>>};
        /**
         * Tried when `rule` cannot give a value for this data (no clear knee in the k-distance
         * curve), before the binding's constant, so the default still follows the data's scale.
         */
        fallback?: {rule: GPUAnalysisDefaultRule; factor?: number};
      }
    | {
        from: 'preset';
        value: CatalogParameterValue;
        reason: string;
        input?: string;
        whenGeometry?: KeplerOperationGeometry;
      }
    | {
        from: 'field-measurement';
        input: string;
        role: string;
        values: Partial<Record<GPUAnalysisMeasurementLevel, CatalogParameterValue>>;
      }
    | {
        /** Complete ISO text can resolve automatically; ambiguous epoch units need a choice. */
        from: 'time-encoding';
        input: string;
        role: string;
      };
  /** `basic` parameters are shown first; `advanced` ones are folded away. */
  level: 'basic' | 'advanced';
  /**
   * `rebuild` changes allocation, graph topology, prepared inputs or shader constants.
   * `live` changes only uniforms, runtime parameter buffers or a post-pass on an existing result.
   * This describes capability; hosts own scheduling. It does not change the execution ABI.
   */
  apply: 'live' | 'rebuild';
  /** A false-by-default boolean certification that must be true before the operation is ready. */
  confirmation?: {message: string; remedy: string};
  /**
   * Present when the parameter only has an effect while a shared condition holds
   * (one controller matching any listed value, or anyOf across several controllers). A distance
   * band, for example, only matters for distance-band neighbourhoods. A form hides or disables
   * an inactive parameter, and a report does not list it as chosen.
   */
  activeWhen?: GPUAnalysisParameterCondition;
};

/**
 * A geometry-only result that should retain a uniform host style. This explicitly records why
 * a column/scale plan would be misleading; it is never inferred from missing metadata.
 */
export type KeplerOperationUniformStyleException = {
  type: 'uniform-style';
  reason: string;
};

/** How one output table is shown, read and followed up. */
export type KeplerOperationOutputDescriptor = {
  /** `CatalogOutputSpec.tableId`. */
  tableId: string;
  /** What the rows are; `summary` for a table with no geometry. */
  kind: GPUSpatialAnalysisDataKind;
  /**
   * `CatalogInputSpec.name` whose geometry the output keeps, for operations that return the
   * features they were given (measures, transforms, clean-up). The output's kind is then the
   * kind of the dataset assigned to that input, and `kind` is only the usual case.
   */
  kindFrom?: string;
  /** Significance-derived values or categories require full-population inference before keeping. */
  inference?: boolean;
  presentation?: GPUAnalysisPresentationSpec;
  /** Mutually exclusive with presentation; shown in report notes. */
  presentationException?: KeplerOperationUniformStyleException;
  interpretation?: GPUAnalysisInterpretationSpec;
  /** Questions that are natural to ask of this output next. */
  next?: readonly GPUAnalysisQuestion[];
};

/** A requirement on one input of the binding. */
export type KeplerOperationRequirement = GPUAnalysisRequirementSpec & {
  /** `CatalogInputSpec.name` the requirement is about. */
  input: string;
  /** Aggregate failures may be unknown when execution discards rows without geometry. */
  statisticsScope?: 'retained-geometry';
  /** Resolve a count requirement's limit from the selected numeric parameter before evaluation. */
  limitFromParameter?: {parameter: string; absolute?: boolean; offset?: number};
  /**
   * Present when the requirement only applies while a shared parameter condition holds
   * (cost columns matter only in the property cost mode; non-negative values only for the G
   * statistics). It is not evaluated otherwise.
   */
  activeWhen?: GPUAnalysisParameterCondition;
};

export type KeplerOperationDescriptor = {
  /** `CatalogOperationBinding.id`. */
  bindingId: string;
  /** Questions the operation answers; the first is the primary one. At least one. */
  questions: readonly GPUAnalysisQuestion[];
  /** Short action phrase for a suggestion, at most 40 characters: `Find hot spots`. */
  action: string;
  /** 1: a leading suggestion for its data. 2: offered when it fits. 3: specialist or plumbing. */
  prominence: 1 | 2 | 3;
  inputs: readonly KeplerOperationInputDescriptor[];
  /** One entry per binding parameter that is not a field role. */
  parameters: readonly KeplerOperationParameterDescriptor[];
  requirements: readonly KeplerOperationRequirement[];
  /**
   * Pairs of input names whose datasets must cover the same area for the operation to mean
   * anything (points counted in polygons; two periods compared cell by cell). Operations that
   * relate places by distance or by a route (nearest facility, catchments, shortest paths) list
   * none: their inputs may lie apart.
   */
  overlappingInputs?: readonly (readonly [string, string])[];
  outputs: readonly KeplerOperationOutputDescriptor[];
  /**
   * Plain sentences a user must read before trusting the result, shown with the form: how this
   * binding differs from another form of the same method, what it does not do, what its defaults
   * imply. Not for restating the summary.
   */
  notes?: readonly string[];
};
