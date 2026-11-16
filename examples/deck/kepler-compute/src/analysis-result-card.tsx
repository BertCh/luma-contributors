// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState, type ReactNode} from 'react';
import {hasKeplerResultInference} from '@luma-contributors/kepler';
import type {
  KeplerResultInferenceScope,
  KeplerResultInferenceSettings,
  KeplerResultInferenceUpdate
} from '@luma-contributors/kepler';
import type {AnalysisResultCardModel, AnalysisResultNextAction} from './analysis-result-card-model';
import {AnalysisResultInferenceControls} from './analysis-result-inference-controls';

/** One explanatory result surface for the catalog, generated sheet and workbench. */
export function AnalysisResultCard({
  model,
  onNext,
  onAddToStack,
  onSaveWorkflow,
  inStack = false,
  actionsDisabledReason,
  showLegends = true,
  inferenceScope = 'full',
  onInferenceChange,
  onInferenceRerun,
  children
}: {
  model: AnalysisResultCardModel;
  onNext(action: AnalysisResultNextAction): void | Promise<void>;
  onAddToStack?(): void | Promise<void>;
  onSaveWorkflow?(): void | Promise<void>;
  inStack?: boolean;
  actionsDisabledReason?: string;
  showLegends?: boolean;
  inferenceScope?: KeplerResultInferenceScope;
  /** Resolve only after the host's map and tables accept this entire display. */
  onInferenceChange?(update: KeplerResultInferenceUpdate): void | Promise<void>;
  onInferenceRerun?(settings: KeplerResultInferenceSettings): void | Promise<void>;
  children?: ReactNode;
}) {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [inference, setInference] = useState<{
    runIdentity: string;
    update: KeplerResultInferenceUpdate;
  }>();
  // Match the controls' stable run key so harmless model reconstruction cannot split the display.
  const runIdentity = model.json;
  const update = inference?.runIdentity === runIdentity ? inference.update : undefined;
  const inferredReport = update?.report ?? model.report;
  const inferredPresentations = update?.presentations ?? model.presentations;
  const partialInference =
    inferenceScope !== 'full' && hasKeplerResultInference(inferredReport, inferredPresentations);
  const report = partialInference
    ? {
        ...inferredReport,
        sentence: `Significance is hidden for this ${inferenceScope} preview. Recompute on the full population to keep inference.`,
        outputs: []
      }
    : inferredReport;
  const presentations = partialInference ? [] : inferredPresentations;
  async function perform(action: () => void | Promise<void>, success: string) {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      await action();
      setMessage(success);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section
      data-testid="ax-result-card"
      aria-label={`${model.title} result`}
      style={{padding: 8, overflowWrap: 'anywhere'}}
    >
      <h3>{model.title}</h3>
      <p data-testid="ax-result-sentence">{report.sentence}</p>
      <AnalysisResultInferenceControls
        key={model.json}
        model={model}
        scope={inferenceScope}
        disabled={busy || Boolean(actionsDisabledReason)}
        onRerun={onInferenceRerun}
        onApply={
          onInferenceChange
            ? async nextUpdate => {
                await onInferenceChange(nextUpdate);
                setInference({runIdentity, update: nextUpdate});
              }
            : undefined
        }
      />
      {report.outputs
        .filter(output => output.sentence !== report.sentence)
        .map(output => (
          <p key={output.tableId}>{output.sentence}</p>
        ))}
      <p data-testid="ax-result-engine">{report.engine}</p>
      {report.fallbackReason && <p data-testid="ax-result-fallback">{report.fallbackReason}</p>}
      <details>
        <summary>What was chosen</summary>
        {report.chosen.length ? (
          <ul>
            {report.chosen.map(line => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        ) : (
          <p>The operation’s default settings were used.</p>
        )}
        <details>
          <summary>All executed settings</summary>
          <dl>
            {Object.entries(model.result.parameters).map(([name, value]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>{Array.isArray(value) ? value.join(', ') : String(value)}</dd>
              </div>
            ))}
          </dl>
        </details>
      </details>
      {report.notes.length > 0 && (
        <ul aria-label="Method notes">
          {report.notes.map(note => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      {report.cautions.length > 0 && (
        <ul aria-label="Cautions">
          {report.cautions.map(caution => (
            <li key={caution.id}>
              <strong>{caution.title}:</strong> {caution.text}
            </li>
          ))}
        </ul>
      )}
      {report.warnings.length > 0 && (
        <ul aria-label="Warnings">
          {report.warnings.map(warning => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      {report.diagnostics.length > 0 && (
        <details>
          <summary>Run diagnostics</summary>
          <ul>
            {report.diagnostics.map((diagnostic, index) => (
              <li key={`${diagnostic.code}:${index}`}>
                {diagnostic.severity}: {diagnostic.message} ({diagnostic.count})
              </li>
            ))}
          </ul>
        </details>
      )}
      {showLegends &&
        presentations.map(presentation => (
          <ul key={presentation.tableId} aria-label={`${presentation.tableId} legend`}>
            {presentation.legend.map((entry, index) => (
              <li key={index}>
                <span
                  aria-hidden="true"
                  style={{
                    display: 'inline-block',
                    width: 12,
                    height: 12,
                    marginRight: 6,
                    backgroundColor: `rgb(${entry.color.join(',')})`
                  }}
                />
                {entry.label}
                {entry.count === undefined ? '' : `: ${entry.count}`}
              </li>
            ))}
          </ul>
        ))}
      {model.next.length > 0 && (
        <div aria-label="Next">
          <h4>Next</h4>
          {model.next.map(action => (
            <div
              key={action.question}
              data-testid="ax-result-next"
              data-status={action.applicability.status}
            >
              <button
                type="button"
                disabled={
                  busy ||
                  Boolean(actionsDisabledReason) ||
                  action.applicability.status === 'blocked'
                }
                onClick={() => perform(() => onNext(action), 'Opened next analysis settings.')}
              >
                {action.question}: {action.label}
              </button>
              <p>
                {action.applicability.status === 'ready'
                  ? 'Ready to configure on this result.'
                  : action.applicability.missing
                      .map(item => `${item.what}. ${item.remedy}`)
                      .join(' ')}
              </p>
            </div>
          ))}
        </div>
      )}
      <div aria-label="Keep result">
        <button
          type="button"
          disabled={
            busy || partialInference || Boolean(actionsDisabledReason) || inStack || !onAddToStack
          }
          title={inStack ? 'This run is already in the analysis stack.' : undefined}
          onClick={() => onAddToStack && perform(onAddToStack, 'Added run settings to the stack.')}
        >
          {inStack ? 'In stack' : 'Add to stack'}
        </button>{' '}
        <button
          type="button"
          disabled={busy || partialInference || Boolean(actionsDisabledReason) || !onSaveWorkflow}
          onClick={() => onSaveWorkflow && perform(onSaveWorkflow, 'Saved as a workflow template.')}
        >
          Save as workflow
        </button>{' '}
        <button
          type="button"
          disabled={busy}
          onClick={() =>
            perform(
              () =>
                navigator.clipboard.writeText(
                  update
                    ? JSON.stringify(
                        {run: JSON.parse(model.json), inferenceDisplay: update.settings},
                        null,
                        2
                      )
                    : model.json
                ),
              'Copied result and provenance as JSON.'
            )
          }
        >
          Copy as JSON
        </button>
      </div>
      {actionsDisabledReason && <p>{actionsDisabledReason}</p>}
      {partialInference && <p>Run the full population before keeping an inferential result.</p>}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
      {children}
    </section>
  );
}
