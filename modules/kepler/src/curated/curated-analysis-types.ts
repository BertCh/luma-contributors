// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisFieldRoleSpec,
  GPUAnalysisQuestion,
  GPUAnalysisRowSemantics,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';

// A curated analysis is a question-named, pre-configured way to run one catalog binding: typed
// dataset slots with field roles, preset parameters, a few exposed parameters and a sample. It
// is one step. Multi-step chains are workflow documents (`modules/kepler/src/workflows`); a
// curated analysis can be placed in one as a single step.

/** A dataset the analysis needs, by what it is and the columns it must offer. */
export type KeplerCuratedAnalysisSlot = {
  /** Stable slot id: `points`, `areas`, `network`. */
  name: string;
  label: string;
  kind: GPUSpatialAnalysisDataKind;
  rowSemantics?: readonly GPUAnalysisRowSemantics[];
  fields: readonly GPUAnalysisFieldRoleSpec[];
};

/** How a curated analysis runs today. */
export type KeplerCuratedAnalysisExecution =
  | {
      status: 'available';
      /** `CatalogOperationBinding.id` that runs it. */
      bindingId: string;
      /** Slot name to `CatalogInputSpec.name`. */
      inputs: Readonly<Record<string, string>>;
      /** `<slot>.<role>` to the binding parameter that receives the column name. */
      fieldParameters: Readonly<Record<string, string>>;
      /** Parameter values the curated analysis fixes; the rest keep their resolved defaults. */
      parameters: Readonly<Record<string, CatalogParameterValue>>;
    }
  | {
      status: 'needs-binding';
      /** The analysis-package recipe or contributor that implements it. */
      implementation: string;
      /** What is missing in the Kepler package, in one sentence. */
      reason: string;
    };

export type KeplerCuratedAnalysis = {
  /** Stable id: `hot-spot-map`. */
  id: string;
  question: GPUAnalysisQuestion;
  /** The question as a user would ask it: `Where are the significant hot and cold spots?` */
  title: string;
  /** Short name of the method: `Hot spot map`. */
  name: string;
  /** One paragraph: what is computed, in order, and what the result shows. */
  method: string;
  /** 1: first twelve. 2: the rest of the starting set. */
  wave: 1 | 2;
  slots: readonly KeplerCuratedAnalysisSlot[];
  execution: KeplerCuratedAnalysisExecution;
  /** Three to five binding parameter names shown first; empty when `needs-binding`. */
  exposedParameters: readonly string[];
  /** Id of a repository-authored synthetic sample (MIT); absent when there is none yet. */
  sampleId?: string;
};
