// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatCount} from '../../../../lens/format';
import {StatStrip} from '../../page-kit';
import contractData from './contract-data.json';
import './contract.css';

/** What the checker's baseline holds today: known departures, by rule. Generated, never typed. */
export function BaselineTable() {
  const {rules, totalFindings} = contractData;
  const largest = Math.max(...rules.map(rule => rule.findings));
  return (
    <>
      <StatStrip
        stats={[
          {value: formatCount(rules.length), label: 'rules with known departures'},
          {value: formatCount(totalFindings), label: 'findings in the baseline'}
        ]}
      />
      <div className="meta-panel meta-scroll" style={{marginTop: 'var(--space-4)'}}>
        <table className="contract-baseline" style={{minWidth: 520}}>
          <thead>
            <tr>
              <th scope="col">Rule</th>
              <th scope="col">Files</th>
              <th scope="col">Findings</th>
              <th scope="col">
                <span className="sr-only">Share</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rules.map(rule => (
              <tr key={rule.id}>
                <th scope="row">{rule.id}</th>
                <td>{formatCount(rule.files)}</td>
                <td>{formatCount(rule.findings)}</td>
                <td>
                  <span
                    className="contract-baseline__bar"
                    style={{width: `${(rule.findings / largest) * 100}%`}}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
