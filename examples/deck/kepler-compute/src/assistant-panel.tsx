// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';
import type {AssistantHost} from './assistant-host';
import type {AssistantRouteResult} from './assistant-operation-router';

const EXAMPLE_OPERATION = JSON.stringify(
  {type: 'setFilter', datasetId: 'dataset-id', field: 'value', range: [0, 10]},
  null,
  2
);

/** Approval surface for structured assistant operations; execution uses the same routes as manual UI. */
export function AssistantPanel({host}: {host: AssistantHost}): React.JSX.Element {
  const [text, setText] = useState(EXAMPLE_OPERATION);
  const [isApproved, setIsApproved] = useState(false);
  const [result, setResult] = useState<AssistantRouteResult | null>(null);
  const [parseError, setParseError] = useState('');
  const run = async () => {
    let operation: unknown;
    try {
      operation = JSON.parse(text);
    } catch (error) {
      setParseError(error instanceof Error ? error.message : String(error));
      return;
    }
    setParseError('');
    setResult(await host.runOperation(isApproved, operation));
    setIsApproved(false);
  };
  return (
    <section aria-label="Assistant operations" data-testid="assistant-panel">
      <h2>Assistant operations</h2>
      <label>
        Structured operation (JSON)
        <textarea
          rows={6}
          value={text}
          onChange={event => setText(event.target.value)}
          style={{width: '100%'}}
        />
      </label>
      <label>
        <input
          type="checkbox"
          checked={isApproved}
          onChange={event => setIsApproved(event.target.checked)}
        />{' '}
        I approve this exact operation
      </label>
      <button type="button" onClick={() => void run()}>
        Run operation
      </button>
      {parseError ? <p role="alert">{parseError}</p> : null}
      {result ? (
        <p role={result.accepted ? 'status' : 'alert'} data-testid="assistant-result">
          {result.accepted
            ? `Ran ${result.route.kind === 'native-action' ? result.route.actionName : result.route.capability}.`
            : `Rejected (${result.code}): ${result.reason}`}
        </p>
      ) : null}
    </section>
  );
}
