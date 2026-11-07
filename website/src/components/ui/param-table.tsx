// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ParamDoc} from '../../catalog/schema';

export function ParamTable({params}: {params: ParamDoc[]}) {
  return (
    <table className="param-table">
      <thead>
        <tr>
          <th>Prop</th>
          <th>Type</th>
          <th>Default</th>
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
            <td>{param.default ? <code>{param.default}</code> : null}</td>
            <td>{param.description}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
