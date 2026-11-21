// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '@luma-contributors/analysis/analysis-core';
import {parseCatalogTableInput} from '../analysis/catalog-data-input';
import {profileKeplerWorkbenchData} from '../descriptors/workbench-adapters';
import {makeWorkflowData} from './workflow-data';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {
  WORKFLOW_LIMITS,
  type WorkflowOperation,
  type WorkflowParameterSpec
} from './workflow-types';

export const MAXIMUM_WORKFLOW_TABLE_COLUMNS = 64;
export const MAXIMUM_WORKFLOW_TABLE_CELLS = 1000000;

/**
 * Original-schema scalar snapshots, including row_index and geometry names. The source-table
 * route admits 100000 total rows, MAXIMUM_WORKFLOW_TABLE_COLUMNS = 64 columns,
 * MAXIMUM_WORKFLOW_TABLE_CELLS = 1000000 cells and 262144 JSON characters per snapshot.
 */
export function makeWorkflowTableSourceOperation(): WorkflowOperation {
  const parameters: WorkflowParameterSpec[] = [
    {
      name: 'text',
      label: 'Table JSON',
      kind: 'text',
      maximumLength: WORKFLOW_LIMITS.maximumTextCharacters,
      defaultValue: ''
    }
  ];
  const read = (text: string) => {
    const input = parseCatalogTableInput(text, WORKFLOW_LIMITS.maximumSourceRows);
    validateCapacity(
      'source-table',
      'columnCount',
      Object.keys(input.table.columns).length,
      0,
      MAXIMUM_WORKFLOW_TABLE_COLUMNS
    );
    validateCapacity(
      'source-table',
      'cellCount',
      Object.keys(input.table.columns).length * input.table.rowCount,
      0,
      MAXIMUM_WORKFLOW_TABLE_CELLS
    );
    return input;
  };
  return {
    id: 'source-table',
    title: 'Table snapshot',
    summary: 'Read bounded scalar table JSON with its original column names.',
    origin: 'builtin',
    category: 'source',
    engine: 'host',
    version: '1',
    inputs: [],
    parameters,
    outputs: [
      {
        name: 'data',
        label: 'Table rows',
        kind: 'table',
        maximumRows: WORKFLOW_LIMITS.maximumSourceRows,
        identityColumns: []
      }
    ],
    capabilityIds: [],
    contributorNames: [],
    declaredBehavior: [
      'Preserves the original scalar columns, row order and identity without generated row_index columns.'
    ],
    limitation: `At most ${WORKFLOW_LIMITS.maximumSourceRows} rows, 64 columns, 1,000,000 cells and ${WORKFLOW_LIMITS.maximumTextCharacters} JSON characters.`,
    normalizeParameters: raw => {
      const values = normalizeWorkflowParameters(parameters, raw, {
        operationTitle: 'Table snapshot'
      });
      read(String(values['text']));
      return values;
    },
    run: async (context, _inputs, values) => {
      context.signal.throwIfAborted();
      const input = read(String(values['text']));
      const data = makeWorkflowData(input.table, {identityColumn: input.identityColumn});
      data.analysisProfile = profileKeplerWorkbenchData(data);
      return {
        outputs: {data},
        diagnostics: [],
        summary: {rows: input.table.rowCount},
        engine: 'host',
        dispatchedContributors: []
      };
    }
  };
}
