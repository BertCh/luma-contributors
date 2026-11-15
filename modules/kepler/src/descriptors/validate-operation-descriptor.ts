// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_ANALYSIS_QUESTIONS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogOperationBinding} from '../analysis/catalog-binding-types';
import type {KeplerOperationDescriptor} from './operation-descriptor-types';

/** Longest action phrase a suggestion row can show. */
export const MAXIMUM_OPERATION_ACTION_LENGTH = 40;

/**
 * Checks a descriptor against the binding it describes and returns one sentence per mismatch;
 * an empty list means the two agree. Structural only: it does not run the binding.
 */
export function validateKeplerOperationDescriptor(
  descriptor: KeplerOperationDescriptor,
  binding: CatalogOperationBinding
): string[] {
  const issues: string[] = [];
  const report = (message: string) => issues.push(`${descriptor.bindingId}: ${message}`);
  if (descriptor.bindingId !== binding.id) {
    report(`describes ${descriptor.bindingId} but was checked against ${binding.id}`);
  }
  if (descriptor.questions.length === 0) report('questions is empty');
  for (const question of descriptor.questions) {
    if (!GPU_ANALYSIS_QUESTIONS.includes(question)) report(`unknown question ${question}`);
  }
  if (
    descriptor.action.length === 0 ||
    descriptor.action.length > MAXIMUM_OPERATION_ACTION_LENGTH
  ) {
    report(`action must have 1 to ${MAXIMUM_OPERATION_ACTION_LENGTH} characters`);
  }

  const bindingInputNames = binding.inputs.map(input => input.name);
  const describedInputNames = descriptor.inputs.map(input => input.name);
  for (const name of bindingInputNames) {
    if (!describedInputNames.includes(name)) report(`input ${name} is not described`);
  }
  for (const name of describedInputNames) {
    if (!bindingInputNames.includes(name)) report(`input ${name} is not a binding input`);
  }

  const parametersByName = new Map(
    binding.parameters.map(parameter => [parameter.name, parameter])
  );
  const validateActiveWhen = (
    owner: string,
    condition: KeplerOperationDescriptor['parameters'][number]['activeWhen']
  ) => {
    if (!condition) return;
    const controller = parametersByName.get(condition.parameter);
    if (!controller) {
      report(`${owner} activeWhen names unknown parameter ${condition.parameter}`);
      return;
    }
    if (condition.equals.length === 0) {
      report(`${owner} activeWhen equals must contain at least one possible value`);
    }
    for (const value of condition.equals) {
      let possible = false;
      switch (controller.kind) {
        case 'enum':
          possible = typeof value === 'string' && Boolean(controller.options?.includes(value));
          break;
        case 'boolean':
          possible = typeof value === 'boolean';
          break;
        case 'integer':
        case 'number':
          possible =
            typeof value === 'number' &&
            Number.isFinite(value) &&
            (controller.kind !== 'integer' || Number.isInteger(value)) &&
            (controller.minimum === undefined || value >= controller.minimum) &&
            (controller.maximum === undefined || value <= controller.maximum);
          break;
        case 'string':
          possible =
            typeof value === 'string' &&
            (controller.maximumLength === undefined || value.length <= controller.maximumLength);
          break;
        case 'number-list':
          // activeWhen scalar values cannot equal a list parameter.
          break;
      }
      if (!possible) {
        report(
          `${owner} activeWhen value ${String(value)} is outside the allowed values of parameter ${condition.parameter} (${controller.kind})`
        );
      }
    }
  };
  // A parameter may be named by the same role on several inputs (one value column read from
  // both the earlier and the later layer); it may not be named by two different roles.
  const fieldParameterNames = new Set<string>();
  const roleByFieldParameter = new Map<string, string>();
  for (const input of descriptor.inputs) {
    const roles = new Set<string>();
    for (const field of input.fields) {
      if (roles.has(field.role)) report(`input ${input.name} repeats role ${field.role}`);
      roles.add(field.role);
      const parameter = parametersByName.get(field.parameter);
      if (!parameter) {
        report(`role ${field.role} names parameter ${field.parameter}, which the binding lacks`);
      } else if (parameter.kind !== 'string') {
        report(`role ${field.role} names parameter ${field.parameter} of kind ${parameter.kind}`);
      }
      const earlierRole = roleByFieldParameter.get(field.parameter);
      if (earlierRole !== undefined && earlierRole !== field.role) {
        report(`parameter ${field.parameter} receives the roles ${earlierRole} and ${field.role}`);
      }
      roleByFieldParameter.set(field.parameter, field.role);
      fieldParameterNames.add(field.parameter);
      if (field.accepts.length === 0) report(`role ${field.role} accepts no column type`);
      if (
        typeof field.required === 'object' &&
        !parametersByName.has(field.required.whenParameter)
      ) {
        report(`role ${field.role} depends on unknown parameter ${field.required.whenParameter}`);
      }
      if (field.multiple && field.multiple.minimum > field.multiple.maximum) {
        report(`role ${field.role} has minimum above maximum`);
      }
    }
  }

  const describedParameterNames = new Set<string>();
  for (const parameter of descriptor.parameters) {
    validateActiveWhen(`parameter ${parameter.name}`, parameter.activeWhen);
    if (!parametersByName.has(parameter.name)) {
      report(`parameter ${parameter.name} is not a binding parameter`);
    }
    if (describedParameterNames.has(parameter.name)) report(`parameter ${parameter.name} repeats`);
    describedParameterNames.add(parameter.name);
    if (fieldParameterNames.has(parameter.name)) {
      report(`parameter ${parameter.name} is both a field role and a plain parameter`);
    }
    if (parameter.defaultPolicy.from === 'data') {
      const bindingParameter = parametersByName.get(parameter.name);
      if (
        bindingParameter &&
        bindingParameter.kind !== 'number' &&
        bindingParameter.kind !== 'integer'
      ) {
        report(
          `parameter ${parameter.name} derives a number but is of kind ${bindingParameter.kind}`
        );
      }
      if (!bindingInputNames.includes(parameter.defaultPolicy.input)) {
        report(
          `parameter ${parameter.name} derives from unknown input ${parameter.defaultPolicy.input}`
        );
      }
    }
  }
  for (const name of parametersByName.keys()) {
    if (!describedParameterNames.has(name) && !fieldParameterNames.has(name)) {
      report(`parameter ${name} is neither a field role nor described`);
    }
  }

  for (const requirement of descriptor.requirements) {
    validateActiveWhen(`requirement ${requirement.id}`, requirement.activeWhen);
    if (!bindingInputNames.includes(requirement.input)) {
      report(`requirement ${requirement.id} names unknown input ${requirement.input}`);
    }
    if (requirement.role !== undefined) {
      const input = descriptor.inputs.find(candidate => candidate.name === requirement.input);
      if (!input?.fields.some(field => field.role === requirement.role)) {
        report(`requirement ${requirement.id} names unknown role ${requirement.role}`);
      }
    }
  }

  for (const pair of descriptor.overlappingInputs ?? []) {
    for (const name of pair) {
      if (!bindingInputNames.includes(name)) {
        report(`overlappingInputs names unknown input ${name}`);
      }
    }
  }

  const bindingTableIds = binding.outputs.map(output => output.tableId);
  const describedTableIds = descriptor.outputs.map(output => output.tableId);
  for (const tableId of bindingTableIds) {
    if (!describedTableIds.includes(tableId)) report(`output ${tableId} is not described`);
  }
  for (const output of descriptor.outputs) {
    if (output.kindFrom !== undefined && !bindingInputNames.includes(output.kindFrom)) {
      report(`output ${output.tableId} kindFrom names unknown input ${output.kindFrom}`);
    }
    if (!bindingTableIds.includes(output.tableId)) {
      report(`output ${output.tableId} is not a binding output`);
    }
    if (output.presentation?.scale === 'class-scheme' && !output.presentation.classScheme) {
      report(`output ${output.tableId} uses a class scheme without naming it`);
    }
  }
  return issues;
}
