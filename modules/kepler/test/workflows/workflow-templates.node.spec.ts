// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {describe, expect, test} from 'vitest';
import {KEPLER_CATALOG_BINDINGS} from '../../src/analysis/catalog-bindings';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {countWorkflowVertices, getWorkflowFeatures} from '../../src/workflows/workflow-data';
import {
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {layoutWorkflow, updateWorkflowInput} from '../../src/workflows/workflow-edit';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {getWorkflowDownstreamSteps, planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {
  instantiateWorkflowTemplate,
  listWorkflowTemplates
} from '../../src/workflows/workflow-templates';
import {WorkflowError} from '../../src/workflows/workflow-types';
import {makeFixtureHost} from './workflow-fixtures';

const EXPECTED_OUTPUTS: Record<string, {rows: number; column: string; values: unknown[]}> = {
  'cluster-footprints': {rows: 3, column: 'point_count', values: [16, 16, 16]},
  'buffer-areas': {rows: 4, column: 'vertex_count', values: [17, 17, 17, 17]},
  'nearest-facilities': {rows: 4, column: 'demand_row_count', values: [7, 10, 13, 16]},
  'district-counts': {rows: 4, column: 'row_count', values: [7, 10, 13, 16]},
  'street-coverage': {rows: 4, column: 'line_count', values: [8, 8, 8, 8]},
  'voronoi-catchments': {rows: 4, column: 'count', values: [16, 13, 10, 7]}
};

function checkMeaningfulValues(templateId: string, table: CatalogOutputTable): void {
  const expected = EXPECTED_OUTPUTS[templateId];
  expect(table.rowCount).toBe(expected.rows);
  expect(table.columns[expected.column]).toEqual(expected.values);
  if (templateId === 'cluster-footprints') {
    expect(table.columns['group_value']).toEqual(['0', '1', '2']);
  }
  if (templateId === 'buffer-areas') {
    for (const value of table.columns['hectares']) expect(value).toBeCloseTo(9.91915456, 4);
  }
  if (templateId === 'nearest-facilities') {
    expect(table.columns['group_value']).toEqual([
      'facility-0',
      'facility-1',
      'facility-2',
      'facility-3'
    ]);
    expect(table.columns['point_count']).toEqual([7, 10, 13, 16]);
    for (const value of table.columns['demand_value_mean']) {
      expect(value).toBeGreaterThan(100);
      expect(value).toBeLessThan(400);
    }
  }
  if (templateId === 'district-counts') {
    expect(table.columns['source_id']).toEqual([
      'district-0',
      'district-1',
      'district-2',
      'district-3'
    ]);
    for (const value of table.columns['area']) expect(value).toBeGreaterThan(500000);
  }
  if (templateId === 'street-coverage') {
    expect(table.columns['polygon_id']).toEqual([
      'district-0',
      'district-1',
      'district-2',
      'district-3'
    ]);
    for (const value of table.columns['kilometers']) {
      expect(value).toBeGreaterThan(6);
      expect(value).toBeLessThan(6.5);
    }
  }
  if (templateId === 'voronoi-catchments') {
    const sums = table.columns['sum'] as readonly number[];
    expect(sums.reduce((total, value) => total + value, 0)).toBe(184);
    expect(sums).toEqual([64, 52, 40, 28]);
    expect(table.columns['polygon_id']).toEqual(['3:0', '2:0', '1:0', '0:0']);
    for (const value of table.columns['shape_length']) expect(value).toBeGreaterThan(2500);
  }
}

describe('workflow templates', () => {
  const templates = listWorkflowTemplates();
  const registry = getWorkflowOperationRegistry();

  test('offers six to eight distinct, laid-out, self-contained CPU workflows', () => {
    expect(templates.length).toBeGreaterThanOrEqual(6);
    expect(templates.length).toBeLessThanOrEqual(8);
    expect(templates.map(template => template.id)).toEqual(Object.keys(EXPECTED_OUTPUTS));
    expect(new Set(templates.map(template => template.id)).size).toBe(templates.length);
    expect(new Set(templates.map(template => template.title)).size).toBe(templates.length);
    expect(listWorkflowTemplates()).toBe(templates);
    expect(Object.isFrozen(templates)).toBe(true);
    expect(
      templates.filter(template => template.document.inputs.length > 0).length
    ).toBeGreaterThanOrEqual(2);
    expect(
      templates.filter(template =>
        template.document.steps.some(step => step.operationId.startsWith('table-'))
      ).length
    ).toBeGreaterThanOrEqual(3);
    for (const template of templates) {
      const {document} = template;
      expect(document.id).toBe(template.id);
      expect(document.title).toBe(template.title);
      expect(document.steps.length).toBeGreaterThanOrEqual(3);
      expect(new Set(document.steps.map(step => step.id)).size).toBe(document.steps.length);
      expect(layoutWorkflow(document)).toEqual(document);
      expect(
        document.steps.some(step =>
          Object.values(step.inputs).some(
            connection =>
              registry.get(
                document.steps.find(producer => producer.id === connection.stepId)!.operationId
              )!.category !== 'source'
          )
        )
      ).toBe(true);
      for (const step of document.steps) {
        expect(step.label?.length).toBeGreaterThan(0);
        const operation = registry.get(step.operationId)!;
        expect(operation).toBeDefined();
        if (operation.category === 'source') {
          expect(step.operationId).toBe('source-geojson');
          const binding = step.parameters['text'];
          expect(binding.from).toBe('static');
          if (binding.from !== 'static') throw new Error('Template source must have static text');
          const collection = JSON.parse(binding.value as string) as {
            features: Array<{id: string; geometry: {coordinates: unknown}}>;
          };
          expect(collection.features.length).toBeGreaterThan(0);
          expect(collection.features.length).toBeLessThan(400);
          expect(new Set(collection.features.map(feature => feature.id)).size).toBe(
            collection.features.length
          );
        } else {
          expect(['cpu-reference', 'gpu-or-cpu']).toContain(operation.engine);
        }
      }
      const sinks = document.steps.filter(
        step =>
          !document.steps.some(consumer =>
            Object.values(consumer.inputs).some(connection => connection.stepId === step.id)
          )
      );
      expect(sinks).toHaveLength(1);
      expect(sinks[0].display).toHaveLength(1);
    }
  });

  for (const template of templates) {
    test(`${template.title} round-trips and every step plans ready`, () => {
      const {document} = template;
      expect(parseWorkflowDocument(serializeWorkflowDocument(document))).toEqual(document);
      const plan = planWorkflow(document, {registry, host: makeFixtureHost()});
      expect(plan.issues.filter(issue => issue.severity === 'error')).toEqual([]);
      expect(plan.order).toHaveLength(document.steps.length);
      for (const step of Object.values(plan.steps)) {
        expect(step.issues.filter(issue => issue.severity === 'error')).toEqual([]);
        expect(step.status).toBe('ready');
        expect(step.key).toBeTypeOf('string');
      }
    });

    test(`${template.title} runs on CPU within every capacity and pins its final output`, async () => {
      const host = makeFixtureHost();
      const runner = new WorkflowRunner({registry, host, now: () => 1000});
      try {
        runner.setDocument(template.document);
        const record = await runner.run();
        expect(record.status, JSON.stringify(record)).toBe('done');
        const state = runner.getState();
        for (const step of template.document.steps) {
          const stepState = state.steps[step.id];
          expect(stepState.status, JSON.stringify(stepState.error)).toBe('done');
          const operation = registry.get(step.operationId)!;
          expect(stepState.result!.engine).toBe(
            operation.category === 'source' ? 'host' : 'cpu-reference'
          );
          for (const port of operation.outputs) {
            const data = stepState.result!.outputs[port.name];
            expect(data.table.rowCount).toBeLessThanOrEqual(port.maximumRows);
          }
          for (const [inputName, connection] of Object.entries(step.inputs)) {
            const input = operation.inputs.find(port => port.name === inputName)!;
            const data = state.steps[connection.stepId].result!.outputs[connection.output];
            expect(data.table.rowCount).toBeLessThanOrEqual(input.maximumRows);
            if (input.maximumVertices !== undefined) {
              expect(countWorkflowVertices(data)).toBeLessThanOrEqual(input.maximumVertices);
            }
          }
          if (operation.category === 'source') {
            for (const feature of getWorkflowFeatures(stepState.result!.outputs['data']).features) {
              const positions = JSON.stringify(feature.geometry);
              expect(positions).toContain('-87.');
              expect(positions).toContain('41.');
            }
          }
          for (const outputName of step.display ?? []) {
            const data = stepState.result!.outputs[outputName];
            expect(data.table.geometryColumn).toBeDefined();
            checkMeaningfulValues(template.id, data.table);
          }
        }
        expect(host.published).toHaveLength(1);
        expect(host.published[0].rowCount).toBe(EXPECTED_OUTPUTS[template.id].rows);
      } finally {
        await runner.destroy();
      }
    });
  }

  for (const template of templates.filter(candidate => candidate.document.inputs.length)) {
    test(`${template.title} re-keys and reruns only the promoted parameter's dependents`, async () => {
      const original = template.document;
      const before = planWorkflow(original, {registry});
      const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 1000});
      try {
        runner.setDocument(original);
        expect((await runner.run()).status).toBe('done');
        for (const input of original.inputs) {
          expect(input.kind).toBe('number');
          const edited = updateWorkflowInput(original, input.id, {
            value: (input.value as number) * 1.1
          });
          const after = planWorkflow(edited, {registry});
          const bound = original.steps.filter(step =>
            Object.values(step.parameters).some(
              binding => binding.from === 'input' && binding.inputId === input.id
            )
          );
          expect(bound.length).toBeGreaterThan(0);
          const affected = new Set(
            bound.flatMap(step => [step.id, ...getWorkflowDownstreamSteps(original, step.id)])
          );
          runner.setDocument(edited);
          for (const step of original.steps) {
            expect(after.steps[step.id].status).toBe('ready');
            if (affected.has(step.id)) {
              expect(after.steps[step.id].key).not.toBe(before.steps[step.id].key);
              expect(runner.getState().steps[step.id].status).toBe('stale');
            } else {
              expect(after.steps[step.id].key).toBe(before.steps[step.id].key);
              expect(runner.getState().steps[step.id].status).toBe('done');
            }
          }
          const record = await runner.run();
          expect(record.status, JSON.stringify(record)).toBe('done');
          expect(Object.values(runner.getState().steps).every(step => step.status === 'done')).toBe(
            true
          );
          if (template.id === 'buffer-areas') {
            const table = runner.getState().steps['hectares'].result!.outputs['calculated'].table;
            for (const value of table.columns['hectares'])
              expect(value).toBeCloseTo(9.91915456 * 1.1 ** 2, 4);
          } else {
            checkMeaningfulValues(
              template.id,
              runner.getState().steps['footprints'].result!.outputs['group-hulls'].table
            );
          }
        }
      } finally {
        await runner.destroy();
      }
    });
  }

  test.each([1e-9, 10, 59, 201, 1e9])('refuses unsupported cluster sample radius %s', value => {
    const document = instantiateWorkflowTemplate('cluster-footprints', []);
    expect(() => updateWorkflowInput(document, document.inputs[0].id, {value})).toThrowError(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message: expect.stringContaining(
          `Neighborhood radius: Value must be ${value < 60 ? 'at least 60' : 'at most 200'}`
        )
      })
    );
  });

  for (const template of templates) {
    test(`${template.title} includes a registered GPU contributor path`, () => {
      const operations = template.document.steps.map(step => registry.get(step.operationId)!);
      const accelerated = operations.filter(operation => operation.engine === 'gpu-or-cpu');
      expect(accelerated.length).toBeGreaterThan(0);
      expect(template.document.settings.engine).toBe('auto');
      for (const operation of accelerated) {
        const binding = KEPLER_CATALOG_BINDINGS.find(candidate => candidate.id === operation.id)!;
        expect(binding.runGpu).toBeTypeOf('function');
        expect(binding.gpuContributorNames.length).toBeGreaterThan(0);
        expect(
          template.document.steps.find(step => step.operationId === operation.id)!.engine
        ).not.toBe('cpu');
      }
    });

    test(`${template.title} attempts GPU execution when a device is supplied`, async () => {
      let accesses = 0;
      const device = new Proxy(
        {},
        {
          get() {
            accesses++;
            throw new Error('Template GPU execution probe');
          }
        }
      ) as Device;
      const host = makeFixtureHost();
      host.getDevice = () => device;
      const runner = new WorkflowRunner({registry, host, now: () => 1000});
      try {
        runner.setDocument(template.document);
        const record = await runner.run();
        expect(record.status, JSON.stringify(record)).toBe('done');
        expect(accesses).toBeGreaterThan(0);
        expect(
          record.steps.some(step => step.fallbackReason?.includes('Template GPU execution probe'))
        ).toBe(true);
      } finally {
        await runner.destroy();
      }
    });
  }

  test.each([
    60, 61, 80, 100, 150, 199, 200
  ])('cluster radius %s keeps the sample runnable', async value => {
    const document = instantiateWorkflowTemplate('cluster-footprints', []);
    expect(document.inputs[0]).toMatchObject({minimum: 60, maximum: 200});
    const edited = updateWorkflowInput(document, document.inputs[0].id, {value});
    const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 1000});
    try {
      runner.setDocument(edited);
      const record = await runner.run();
      expect(record.status, JSON.stringify(record)).toBe('done');
      const steps = runner.getState().steps;
      expect(Object.values(steps).map(step => step.status)).toEqual([
        'done',
        'done',
        'done',
        'done'
      ]);
      expect(steps['members'].result!.outputs['filtered'].table.rowCount).toBe(48);
      checkMeaningfulValues(
        'cluster-footprints',
        steps['footprints'].result!.outputs['group-hulls'].table
      );
    } finally {
      await runner.destroy();
    }
  });

  test.each([
    1e-9, 10
  ])('planning refuses persisted all-noise radius %s before execution', value => {
    const document = instantiateWorkflowTemplate('cluster-footprints', []);
    document.inputs[0].value = value;
    const plan = planWorkflow(document, {registry});
    expect(plan.steps['clusters'].status).toBe('not-configured');
    expect(plan.steps['clusters'].issues).toContainEqual(
      expect.objectContaining({code: 'parameter-out-of-range', severity: 'error'})
    );
    expect(plan.steps['members'].status).toBe('blocked');
    expect(plan.steps['footprints'].status).toBe('blocked');
  });

  test.each([50, 500])('buffer radius %s runs at the sample range boundary', async value => {
    const document = instantiateWorkflowTemplate('buffer-areas', []);
    expect(document.inputs[0]).toMatchObject({minimum: 50, maximum: 500});
    const edited = updateWorkflowInput(document, document.inputs[0].id, {value});
    const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 1000});
    try {
      runner.setDocument(edited);
      const record = await runner.run();
      expect(record.status, JSON.stringify(record)).toBe('done');
      expect(Object.values(runner.getState().steps).every(step => step.status === 'done')).toBe(
        true
      );
      const table = runner.getState().steps['hectares'].result!.outputs['calculated'].table;
      expect(table.rowCount).toBe(4);
      for (const area of table.columns['hectares'])
        expect(area).toBeCloseTo(9.91915456 * (value / 180) ** 2, 3);
    } finally {
      await runner.destroy();
    }
  });

  test('instantiation copies nested values without changing the cached template', () => {
    for (const template of templates) {
      const before = serializeWorkflowDocument(template.document);
      const copy = instantiateWorkflowTemplate(template.id, []);
      expect(copy).toEqual(template.document);
      expect(copy).not.toBe(template.document);
      copy.steps[0].position.x = -999;
      copy.steps[0].parameters['text'] = {from: 'static', value: 'changed'};
      const connected = copy.steps.find(step => Object.keys(step.inputs).length)!;
      Object.values(connected.inputs)[0].stepId = 'changed';
      if (copy.inputs.length) copy.inputs[0].value = 999;
      copy.steps.at(-1)!.display!.push('changed');
      expect(serializeWorkflowDocument(template.document)).toEqual(before);
      expect(instantiateWorkflowTemplate(template.id, [])).toEqual(template.document);
    }
  });

  test('makes ids and titles unique, including gaps and one-use iterables', () => {
    const template = templates[0];
    function* taken(): Generator<string> {
      yield template.id;
      yield `${template.id}-2`;
      yield `${template.id}-4`;
      yield template.title;
      yield `${template.title} 2`;
      yield `${template.title} 4`;
    }
    const copy = instantiateWorkflowTemplate(template.id, taken());
    expect(copy.id).toBe(`${template.id}-3`);
    expect(copy.title).toBe(`${template.title} 3`);
    const fromIds = instantiateWorkflowTemplate(template.id, [template.id, `${template.id}-2`]);
    expect(fromIds.id).toBe(`${template.id}-3`);
    expect(fromIds.title).toBe(`${template.title} 3`);
    const fromTitles = instantiateWorkflowTemplate(template.id, [
      template.title,
      `${template.title} 2`
    ]);
    expect(fromTitles.id).toBe(template.id);
    expect(fromTitles.title).toBe(`${template.title} 3`);
  });

  test('rejects an unknown template id with the documented error code and id', () => {
    let caught: unknown;
    try {
      instantiateWorkflowTemplate('missing-template', []);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(WorkflowError);
    expect((caught as WorkflowError).code).toBe('malformed');
    expect((caught as WorkflowError).message).toBe(
      'Workflow template "missing-template" does not exist.'
    );
  });
});
