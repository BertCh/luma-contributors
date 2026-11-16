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
  binding: CatalogOperationBinding,
  options: {allowUnmappedColumnParameters?: boolean} = {}
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
    if ('anyOf' in condition) {
      if (condition.anyOf.length === 0)
        report(`${owner} activeWhen anyOf must contain at least one condition`);
      for (const branch of condition.anyOf) validateActiveWhen(owner, branch);
      return;
    }
    const parameterName = 'parameter' in condition ? condition.parameter : condition.whenParameter;
    const controller = parametersByName.get(parameterName);
    if (!controller) {
      report(`${owner} activeWhen names unknown parameter ${parameterName}`);
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
          `${owner} activeWhen value ${String(value)} is outside the allowed values of parameter ${parameterName} (${controller.kind})`
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
      if (typeof field.required === 'object') {
        validateActiveWhen(`role ${field.role}`, field.required);
      }
      if (field.multiple && field.multiple.minimum > field.multiple.maximum) {
        report(`role ${field.role} has minimum above maximum`);
      }
    }
  }

  const describedParameterNames = new Set<string>();
  for (const parameter of descriptor.parameters) {
    validateActiveWhen(`parameter ${parameter.name}`, parameter.activeWhen);
    if (parameter.confirmation) {
      const specification = parametersByName.get(parameter.name);
      if (
        specification?.kind !== 'boolean' ||
        specification.defaultValue !== false ||
        parameter.defaultPolicy.from !== 'constant'
      ) {
        report(
          `parameter ${parameter.name} confirmation requires a false-by-default boolean with a constant policy`
        );
      }
      if (!parameter.confirmation.message.trim() || !parameter.confirmation.remedy.trim()) {
        report(`parameter ${parameter.name} confirmation needs a message and remedy`);
      }
    }
    if (parameter.apply !== 'live' && parameter.apply !== 'rebuild') {
      report(`parameter ${parameter.name} must declare apply as live or rebuild`);
    }
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
  for (const parameter of descriptor.parameters) {
    if (parameter.defaultPolicy.from === 'time-encoding') {
      const policy = parameter.defaultPolicy;
      const input = descriptor.inputs.find(candidate => candidate.name === policy.input);
      if (!input?.fields.some(field => field.role === policy.role)) {
        report(
          `parameter ${parameter.name} derives time encoding from unknown input/role ${policy.input}/${policy.role}`
        );
      }
      const specification = parametersByName.get(parameter.name);
      if (
        specification?.kind !== 'enum' ||
        !['iso8601', 'unix-millisecond', 'unix-second'].every(value =>
          specification.options?.includes(value)
        )
      ) {
        report(
          `parameter ${parameter.name} time-encoding policy requires the ISO and Unix timestamp options`
        );
      }
    }
  }
  if (!options.allowUnmappedColumnParameters) {
    for (const parameter of binding.parameters) {
      // Catalog strings are column references when the name or UI label declares one. Free
      // strings (labels, expressions, separators) remain legal ordinary settings.
      const namesColumn =
        /(?:propert(?:y|ies)|columns?)$/i.test(parameter.name) ||
        /\bcolumn(?:s| name)?\b/i.test(parameter.label);
      if (parameter.kind === 'string' && namesColumn && !fieldParameterNames.has(parameter.name)) {
        report(`column-naming string parameter ${parameter.name} must be mapped to a field role`);
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
    for (const parameterName of [
      requirement.parameter,
      requirement.limitFromParameter?.parameter
    ]) {
      if (parameterName === undefined) continue;
      const parameter = parametersByName.get(parameterName);
      if (!parameter || (parameter.kind !== 'number' && parameter.kind !== 'integer'))
        report(
          `requirement ${requirement.id} names unknown or non-numeric parameter ${parameterName}`
        );
    }
    if (
      requirement.limitFromParameter &&
      (![
        'minimum-rows',
        'maximum-rows',
        'maximum-vertices',
        'minimum-distinct-values',
        'maximum-distinct-values'
      ].includes(requirement.id) ||
        !Number.isFinite(requirement.limitFromParameter.offset ?? 0))
    )
      report(
        `requirement ${requirement.id} limitFromParameter requires a count requirement and finite offset`
      );
    if (requirement.id === 'numeric-range') {
      if ((requirement.role === undefined) === (requirement.parameter === undefined))
        report('requirement numeric-range must name exactly one role or parameter');
      if (
        (requirement.minimum === undefined &&
          requirement.maximum === undefined &&
          requirement.integer !== true) ||
        (requirement.minimum !== undefined && !Number.isFinite(requirement.minimum)) ||
        (requirement.maximum !== undefined && !Number.isFinite(requirement.maximum)) ||
        (requirement.minimum !== undefined &&
          requirement.maximum !== undefined &&
          requirement.minimum > requirement.maximum)
      )
        report('requirement numeric-range requires finite ordered bounds or integer: true');
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
    if (output.presentationException) {
      if (output.presentation)
        report(
          `output ${output.tableId} declares both a presentation and a uniform-style exception`
        );
      if (!output.presentationException.reason?.trim())
        report(`output ${output.tableId} uniform-style exception needs a reason`);
    }
    if (output.presentation?.scale === 'class-scheme' && !output.presentation.classScheme) {
      report(`output ${output.tableId} uses a class scheme without naming it`);
    }
  }
  return issues;
}
