// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Term} from '../../page-kit';
import contractData from './contract-data.json';
import './contract.css';

/** A few words per port. The names and the full definitions are generated from status.ts. */
const GLOSS: Record<string, string> = {
  count: 'rows written',
  requiredCount: 'rows needed',
  overflow: 'output too small',
  candidateOverflow: 'scratch too small',
  uncertainCount: 'fallback taken',
  antimeridianUncertainCount: 'ring crosses 180',
  poleUncertainCount: 'ring encloses a pole',
  invalidCount: 'rows rejected',
  iterationCount: 'rounds run',
  converged: 'met its stop test',
  iterationLimitReached: 'budget used up',
  approximation: 'approximate result'
};

export function StatusTimeline() {
  return (
    <ol className="contract-timeline" aria-label="Construction, the GPU run, the caller">
      <li data-kind="setup">
        <span className="contract-timeline__when">1. Construction</span>
        <span className="contract-timeline__title">Can throw</span>
        <p>
          The constructor checks props and the CPU-side limits. A failure is an Error that names the
          contributor.
        </p>
      </li>
      <li data-kind="kernel">
        <span className="contract-timeline__when">2. The GPU run</span>
        <span className="contract-timeline__title">Writes status</span>
        <p>
          A kernel cannot throw. It writes named <Term id="status">status</Term> ports instead, one
          uint32 each:
        </p>
        <ul className="contract-chips">
          {contractData.ports.map(port => (
            <li key={port.name} className="contract-chip" title={port.doc}>
              <b>{port.name}</b>
              <span>{GLOSS[port.name] ?? port.doc}</span>
            </li>
          ))}
        </ul>
      </li>
      <li data-kind="caller">
        <span className="contract-timeline__when">3. The caller</span>
        <span className="contract-timeline__title">Reads if it wants</span>
        <p>
          A contributor never reads status back. The caller decides whether to read a port, and what
          to do about an overflow.
        </p>
      </li>
    </ol>
  );
}
