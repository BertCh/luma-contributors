// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** JSON identity inputs cannot contain values that stringify as null or nested containers. */
export function getGPUComputationIdentityIssues(props: {
  parameters?: unknown;
  seed?: unknown;
  choices?: unknown;
  adapter?: unknown;
  datasetVersion?: unknown;
  implementationRevision?: unknown;
  dependencyRevision?: unknown;
  graphs?: unknown;
}): {path: string; message: string}[] {
  const issues: {path: string; message: string}[] = [];
  const checkScalar = (value: unknown, path: string): void => {
    if (
      !(
        typeof value === 'string' ||
        typeof value === 'boolean' ||
        (typeof value === 'number' && Number.isFinite(value))
      )
    )
      issues.push({
        path,
        message: 'Expected a finite JSON scalar (string, boolean or finite number)'
      });
  };
  if (props.parameters !== undefined) {
    if (
      props.parameters === null ||
      typeof props.parameters !== 'object' ||
      Array.isArray(props.parameters) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(props.parameters))
    )
      issues.push({path: 'parameters', message: 'Expected a scalar parameter map'});
    else
      for (const [name, value] of Object.entries(props.parameters))
        checkScalar(value, `parameters.${name}`);
  }
  if (
    props.seed !== undefined &&
    !(
      typeof props.seed === 'string' ||
      (typeof props.seed === 'number' && Number.isFinite(props.seed))
    )
  )
    issues.push({path: 'seed', message: 'Expected a string or finite numeric seed'});
  const checkString = (value: unknown, path: string): void => {
    if (typeof value !== 'string') issues.push({path, message: 'Expected a string identity value'});
  };
  for (const name of ['datasetVersion', 'implementationRevision', 'dependencyRevision'] as const)
    if (props[name] !== undefined) checkString(props[name], name);
  if (props.dependencyRevision === '')
    issues.push({path: 'dependencyRevision', message: 'Expected a nonempty dependency revision'});
  if (props.adapter && typeof props.adapter === 'object') {
    const adapter = props.adapter as Record<string, unknown>;
    for (const name of ['backend', 'variant', 'vendor', 'architecture', 'description'])
      if (adapter[name] !== undefined) checkString(adapter[name], `adapter.${name}`);
  }
  if (Array.isArray(props.graphs))
    props.graphs.forEach((graph, index) => {
      if (!graph || typeof graph !== 'object') return;
      const entry = 'graph' in graph ? graph.graph : graph;
      if (!entry || typeof entry !== 'object') return;
      checkString(entry.id, `graphs[${index}].id`);
      if (Array.isArray(entry.nodes))
        entry.nodes.forEach((node: {id?: unknown}, row: number) =>
          checkString(node?.id, `graphs[${index}].nodes[${row}].id`)
        );
    });
  if (props.choices !== undefined && !Array.isArray(props.choices))
    issues.push({path: 'choices', message: 'Expected a choice array'});
  if (Array.isArray(props.choices))
    props.choices.forEach((choice: unknown, index) => {
      if (!choice || typeof choice !== 'object') {
        issues.push({path: `choices[${index}]`, message: 'Expected a choice object'});
        return;
      }
      const entry = choice as Record<string, unknown>;
      checkString(entry['stage'], `choices[${index}].stage`);
      checkString(entry['name'], `choices[${index}].name`);
      checkScalar(entry['value'], `choices[${index}].value`);
      if (entry['alternatives'] !== undefined && !Array.isArray(entry['alternatives']))
        issues.push({
          path: `choices[${index}].alternatives`,
          message: 'Expected a scalar alternatives array'
        });
      if (Array.isArray(entry['alternatives']))
        entry['alternatives'].forEach((value, alternativeIndex) =>
          checkScalar(value, `choices[${index}].alternatives[${alternativeIndex}]`)
        );
    });
  return issues;
}
