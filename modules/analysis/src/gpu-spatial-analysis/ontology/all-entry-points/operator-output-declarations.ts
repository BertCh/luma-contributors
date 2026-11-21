// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {freezeCatalog} from '../freeze-catalog';
import type {
  GPUAnalysisOperatorIdentity,
  GPUAnalysisOutputDeclaration,
  GPUAnalysisOutputDescription
} from '../spatial-analysis-ontology';

/** Modes with audited output layouts; absent keys are explicitly undisclosed. */
export const OPERATOR_OUTPUT_MODES = freezeCatalog({
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOrdinaryLeastSquares': ['default'],
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineChunk': ['chunk', 'substring']
} as Readonly<Record<string, readonly string[]>>);

/** Describes caller-owned allocations in format rows, including terminal offsets and loss status. */
export function describeOutputs(
  identity: GPUAnalysisOperatorIdentity,
  mode: string,
  sizes: Readonly<Record<string, number>>
): GPUAnalysisOutputDescription {
  const key = `${identity.package}:${identity.exportName}`;
  const modes = OPERATOR_OUTPUT_MODES[key];
  if (!modes) return freezeCatalog({declaration: 'absent', outputs: []});
  if (!modes.includes(mode))
    throw new Error(
      `describeOutputs ${identity.exportName} mode must be ${modes.join(', ')}; received ${mode}`
    );
  const size = (name: string, minimum = 1, maximum = 0xffffffff): number => {
    const value = sizes[name];
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum)
      throw new Error(
        `describeOutputs ${name} must be an integer in [${minimum}, ${maximum}]; received ${value}`
      );
    return value;
  };
  const outputs: GPUAnalysisOutputDeclaration[] = [];
  const constraints: {expression: string; checked: boolean}[] = [];
  const add = (
    path: string,
    format: GPUAnalysisOutputDeclaration['format'],
    capacity: number,
    activeLengthRule: string,
    status: readonly string[],
    optional = false,
    activeLength?: number,
    minimumCapacity = capacity,
    tail = 'unspecified beyond active length',
    meaning?: string
  ) => {
    outputs.push({
      path: `output.${path}`,
      format,
      capacity,
      minimumCapacity,
      activeLengthRule,
      status,
      optional,
      ...(activeLength === undefined ? {} : {activeLength}),
      tail,
      ...(meaning ? {meaning} : {})
    });
  };
  if (identity.exportName === 'GPUOrdinaryLeastSquares') {
    const coefficientCount = size('predictorCount', 1, 15) + 1;
    const rowCount = size('rowCount');
    // spatial-regression-solve.ts: sixteen including intercept; OLS accepts active-sized views.
    for (const path of ['coefficients', 'standardErrors', 'tStatistics'])
      add(
        path,
        'float32',
        16,
        'predictorCount + 1',
        ['output.status.code'],
        false,
        coefficientCount,
        coefficientCount
      );
    // ordinary-least-squares-parameters.ts: GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH = 22.
    add('summary', 'float32', 22, '22', ['output.status.code'], false, 22);
    add('status.code', 'uint32', 1, '1', [], false, 1);
    for (const path of ['residuals', 'fitted', 'predictions'])
      add(
        path,
        'float32',
        rowCount,
        path === 'predictions'
          ? 'rowCount (finite predictor rows including masked rows or missing responses; NaN for invalid predictors or failed fit)'
          : 'rowCount (NaN outside valid training rows or for failed fit)',
        ['output.status.code'],
        true,
        rowCount
      );
  } else {
    const lineCapacity =
      mode === 'substring' ? size('lineCount') : size('lineCapacity', 1, 0x7fffffff);
    const vertexCapacity = size('vertexCapacity');
    const checkConstraint = (expression: string, required: string[], evaluate: () => number) => {
      const checked = required.every(name => sizes[name] !== undefined);
      for (const name of required) if (sizes[name] !== undefined) size(name);
      if (checked && evaluate() > 0xffffffff)
        throw new Error(
          `describeOutputs ${expression} must be at most 4294967295; received ${evaluate()}`
        );
      constraints.push({expression, checked});
    };
    checkConstraint(
      'lineCount * (lineCapacity + 1)',
      ['lineCount'],
      () => sizes['lineCount'] * (lineCapacity + 1)
    );
    checkConstraint(
      '2 * lineCapacity + positionCount',
      ['positionCount'],
      () => 2 * lineCapacity + sizes['positionCount']
    );
    const status = ['output.status.count', 'output.status.overflow'];
    add(
      'positions',
      'float32x2',
      vertexCapacity,
      'output.lineOffsets[output.status.count] (or output.status.vertexCount when requested)',
      status
    );
    add('lineOffsets', 'uint32', lineCapacity + 1, 'output.status.count + 1', status);
    add(
      'sourceLineIndices',
      'uint32',
      lineCapacity,
      'output.status.count',
      status,
      true,
      undefined,
      lineCapacity,
      '0xffffffff past output.status.count'
    );
    add(
      'sourceRows',
      'uint32',
      vertexCapacity,
      'output.lineOffsets[output.status.count]',
      status,
      true
    );
    add(
      'measures',
      'float32',
      vertexCapacity,
      'output.lineOffsets[output.status.count]',
      status,
      true
    );
    for (const path of ['count', 'overflow', 'requiredCount', 'vertexCount', 'requiredVertexCount'])
      add(
        `status.${path}`,
        'uint32',
        1,
        '1',
        path === 'requiredVertexCount'
          ? ['output.status.requiredCount', 'output.status.count', 'output.status.overflow']
          : [],
        !['count', 'overflow'].includes(path),
        1,
        1,
        'unspecified beyond active length',
        path === 'requiredVertexCount'
          ? 'Counts vertices for retained pieces only. Use to resize vertex storage only when requiredCount equals count (no line loss); with line loss it is a lower bound, not the complete vertex requirement.'
          : undefined
      );
  }
  return freezeCatalog({
    declaration: 'declared',
    outputs,
    ...(constraints.length ? {constraints} : {})
  });
}
