// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OutputDoc, ParamDoc} from '../../catalog/schema';

export function ParamTable({params, moreRequired}: {params: ParamDoc[]; moreRequired?: string[]}) {
  // The column only appears when at least one prop documents a default.
  const hasDefaults = params.some(param => param.default);
  return (
    <table className="param-table">
      <thead>
        <tr>
          <th>Prop</th>
          <th>Type</th>
          {hasDefaults ? <th>Default</th> : null}
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {params.map(param => (
          <tr key={param.name}>
            <td>
              <code className="param-table__name">{param.name}</code>
              {param.required ? (
                <span className="param-table__required" title="Required">
                  *
                </span>
              ) : null}
            </td>
            <td>
              <code className="param-table__type">{param.type}</code>
            </td>
            {hasDefaults ? (
              <td>
                {param.default ? <code>{param.default}</code> : <span aria-label="none">-</span>}
              </td>
            ) : null}
            <td>{param.description}</td>
          </tr>
        ))}
        {moreRequired && moreRequired.length > 0 ? (
          <tr>
            <td colSpan={hasDefaults ? 4 : 3}>
              and {moreRequired.length} more required:{' '}
              {moreRequired.map((name, index) => (
                <span key={name}>
                  {index > 0 ? ', ' : null}
                  <code>{name}</code>
                </span>
              ))}
            </td>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

/** Compact three-column table for what an operation returns. */
export function OutputTable({outputs}: {outputs: OutputDoc[]}) {
  return (
    <table className="param-table param-table--compact">
      <thead>
        <tr>
          <th>Field</th>
          <th>Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {outputs.map(output => (
          <tr key={output.name}>
            <td>
              <code className="param-table__name">{output.name}</code>
            </td>
            <td>
              <code className="param-table__type">{output.type}</code>
            </td>
            <td>{output.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
