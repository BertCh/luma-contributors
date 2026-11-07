// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState} from 'react';
import type {
  KeplerArrowDataset,
  KeplerComputeSession,
  KeplerDatasetSnapshot
} from '@luma-contributors/kepler';
import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma.gl/graph/arrow';
import {
  ArrowStreamController,
  type ArrowStreamPreview,
  type ArrowStreamProgress
} from './arrow-stream-controller';
import type {CoordinateRoles} from './source-publication';
import {parseRemoteStreamUrl} from './remote-stream-source';
import {DEFAULT_BACKOFF_OPTIONS, parseAutomaticPolicyConfig} from './automatic-recovery-policy';

const AUTOMATIC_POLICY = parseAutomaticPolicyConfig(globalThis.location?.search ?? '');
let streamSequence = 0;

/** Mounted inside the native Prepare Data workflow; unfinished prefixes stay local to this panel. */
export function ArrowStreamPanel({
  session,
  remoteSources = [],
  onPublishCompleted,
  getCompletionFence
}: {
  session: KeplerComputeSession;
  /** Published remote origins. Refetch always starts a new stream and a new dataset. */
  remoteSources?: readonly {label: string; url: string}[];
  onPublishCompleted(
    dataset: KeplerDatasetSnapshot<KeplerArrowDataset>,
    label: string,
    coordinates?: CoordinateRoles
  ): Promise<void>;
  getCompletionFence(): PromiseLike<unknown>;
}) {
  const controller = useRef<ArrowStreamController | undefined>(undefined);
  const generation = useRef(0);
  const retirement = useRef(Promise.resolve());
  const unsubscribe = useRef<(() => void) | undefined>(undefined);
  const [file, setFile] = useState<File>();
  const [remoteUrl, setRemoteUrl] = useState('');
  const [confirmPartial, setConfirmPartial] = useState(false);
  const [progress, setProgress] = useState<ArrowStreamProgress>();
  const [preview, setPreview] = useState<ArrowStreamPreview | null>(null);
  const [error, setError] = useState('');
  const [round, setRound] = useState(false);
  const [absoluteError, setAbsoluteError] = useState('0.00001');
  const [relativeError, setRelativeError] = useState('0.000001');
  const [longitude, setLongitude] = useState('');
  const [latitude, setLatitude] = useState('');
  const [altitude, setAltitude] = useState('');
  useEffect(() => {
    generation.current++;
    setProgress(undefined);
    setPreview(null);
    return () => {
      generation.current++;
      unsubscribe.current?.();
      const previous = controller.current;
      controller.current = undefined;
      // Callback captures its owning session/fence; a replacement session is never touched.
      retirement.current = previous?.dispose().catch(() => undefined) ?? Promise.resolve();
    };
  }, [session]);
  const run = (operation: Promise<unknown>) => {
    const current = generation.current;
    void operation.catch(failure => {
      if (generation.current === current)
        setError(failure instanceof Error ? failure.message : String(failure));
    });
  };
  const start = async (override?: string) => {
    const url = (override ?? remoteUrl).trim()
      ? parseRemoteStreamUrl((override ?? remoteUrl).trim()).href
      : undefined;
    if (!file && !url) return;
    if (!url && file!.size > 128 * 1024 * 1024)
      throw new Error('IPC stream exceeds the 128 MiB input limit');
    const current = ++generation.current;
    unsubscribe.current?.();
    await retirement.current;
    await controller.current?.dispose();
    if (generation.current !== current) return;
    const normalization: PrepareArrowTableForGPUAnalyticsOptions = {
      float64: round
        ? {
            mode: 'round',
            maximumAbsoluteError: Number(absoluteError),
            maximumRelativeError: Number(relativeError)
          }
        : {mode: 'exact'}
    };
    const next = new ArrowStreamController({
      session,
      id: `arrow-stream-${Date.now()}-${++streamSequence}`,
      normalization,
      maximumInputBytes: 128 << 20,
      maximumInputChunkBytes: 1 << 20,
      maximumQueuedBytes: 32 << 20,
      maximumBatchRows: 250000,
      getCompletionFence,
      remoteUrl: url,
      ...(AUTOMATIC_POLICY.remoteRetry
        ? {retry: {options: DEFAULT_BACKOFF_OPTIONS}, resume: {}}
        : {})
    });
    controller.current = next;
    const update = () => {
      if (generation.current !== current) return;
      setProgress(next.progress);
      setPreview(next.preview());
    };
    unsubscribe.current = next.subscribe(update);
    setError('');
    setConfirmPartial(false);
    setLongitude('');
    setLatitude('');
    setAltitude('');
    await (url ? next.startRemote() : next.start(file!.stream()));
  };
  const active = progress && !['failed', 'cancelled', 'published'].includes(progress.phase);
  const snapshot = controller.current?.publisher.snapshot;
  const coordinateFields = snapshot
    ? Object.keys(snapshot.value.columns).filter(
        name =>
          snapshot.value.columns[name].format === 'float32' &&
          !snapshot.value.prepared.dictionaries[name]
      )
    : [];
  return (
    <section aria-label="Stream Arrow IPC" data-testid="arrow-stream-panel">
      <h2>Stream Arrow IPC</h2>
      <p>
        Read sequential IPC record batches and upload each batch once. The first batch pauses for a
        source preview. Finish at end of input before adding the complete dataset to Kepler.
      </p>
      <p className="hint">
        IPC stream format only. Arrow file/Feather uses the existing file importer. Maximum input
        128 MiB; one admitted batch at a time, up to 32 MiB or 250,000 rows. Supported scalar,
        category and timestamp fields retain their original batches.
      </p>
      <fieldset disabled={Boolean(active)}>
        <label>
          Arrow IPC stream file
          <input
            type="file"
            accept=".arrows,.ipc,.arrow"
            onChange={event => setFile(event.target.files?.[0])}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={round}
            onChange={event => setRound(event.target.checked)}
          />
          Allow bounded Float64 rounding
        </label>
        {round && (
          <>
            <label>
              Maximum absolute error
              <input
                type="number"
                min="0"
                step="any"
                value={absoluteError}
                onChange={event => setAbsoluteError(event.target.value)}
              />
            </label>
            <label>
              Maximum relative error
              <input
                type="number"
                min="0"
                step="any"
                value={relativeError}
                onChange={event => setRelativeError(event.target.value)}
              />
            </label>
          </>
        )}
        <label>
          Remote Arrow IPC stream URL (optional, replaces the file)
          <input
            type="url"
            placeholder="https://example.com/data.arrows"
            value={remoteUrl}
            onChange={event => setRemoteUrl(event.target.value)}
          />
        </label>
        {remoteSources.length > 0 && (
          <label>
            Refetch published remote source (adds a new dataset)
            <select
              aria-label="Refetch remote source"
              value=""
              onChange={event => {
                setRemoteUrl(event.target.value);
                if (event.target.value) run(start(event.target.value));
              }}
            >
              <option value="">Choose a source</option>
              {remoteSources.map(source => (
                <option key={source.url} value={source.url}>
                  Refetch {source.label}
                </option>
              ))}
            </select>
          </label>
        )}
        <button disabled={!file && !remoteUrl.trim()} onClick={() => run(start())}>
          Open stream
        </button>
      </fieldset>
      {progress && (
        <>
          <p role="status">
            {progress.phase}: {progress.batches} batches, {progress.rows.toLocaleString()} admitted
            rows; {(progress.bytesRead / 1048576).toFixed(2)} MiB read.
          </p>
          {progress.resume && (
            <p role="status">
              Connection {progress.resume.phase} after {progress.resume.resumes} interruption(s) at{' '}
              {progress.resume.bytesDelivered.toLocaleString()} bytes.
            </p>
          )}
          {!['complete', 'publishing', 'published'].includes(progress.phase) && (
            <p>
              Incomplete prefix. Analysis, native layers and full export are unavailable until
              Finish.
            </p>
          )}
          <div className="actions">
            <button
              disabled={progress.phase !== 'paused'}
              onClick={() => run(controller.current!.next())}
            >
              Read next batch
            </button>
            <button
              disabled={progress.phase !== 'paused'}
              onClick={() => run(controller.current!.resume())}
            >
              Continue stream
            </button>
            <button
              disabled={progress.phase !== 'reading'}
              onClick={() => controller.current?.pause()}
            >
              Pause stream
            </button>
            <button
              disabled={
                !['opening', 'paused', 'reading', 'exhausted', 'finishing'].includes(progress.phase)
              }
              onClick={() => controller.current?.cancel()}
            >
              Cancel stream
            </button>
            <button
              disabled={progress.phase !== 'exhausted'}
              onClick={() => run(controller.current!.finish())}
            >
              Finish stream
            </button>
          </div>
          {progress.phase === 'paused' && progress.batches > 0 && (
            <fieldset>
              <legend>Save incomplete prefix</legend>
              <p className="hint">
                Closes the stream here and cancels the remaining input. The result is recorded as a
                partial population of {progress.rows.toLocaleString()} rows in {progress.batches}{' '}
                batches, never as the finite complete dataset.
              </p>
              <label>
                <input
                  type="checkbox"
                  checked={confirmPartial}
                  onChange={event => setConfirmPartial(event.target.checked)}
                />
                I understand this is an incomplete prefix
              </label>
              <button
                disabled={!confirmPartial}
                onClick={() => run(controller.current!.finishPartial('save-incomplete-prefix'))}
              >
                Save incomplete prefix
              </button>
            </fieldset>
          )}
          {progress.phase === 'exhausted' && (
            <p>
              End of input reached. Finish confirms that all admitted batches form the complete
              population.
            </p>
          )}
          {progress.phase === 'complete' && (
            <>
              {progress.partial && (
                <p role="note">
                  Partial population: {progress.partial.rowCount.toLocaleString()} rows,{' '}
                  {progress.partial.batchCount} batches. The saved manifest records it as partial.
                </p>
              )}
              <fieldset>
                <legend>Optional point coordinate roles</legend>
                <label>
                  Longitude
                  <select
                    value={longitude}
                    onChange={event => {
                      setLongitude(event.target.value);
                      if (!event.target.value) setAltitude('');
                    }}
                  >
                    <option value="">Table only</option>
                    {coordinateFields.map(name => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Latitude
                  <select
                    value={latitude}
                    onChange={event => {
                      setLatitude(event.target.value);
                      if (!event.target.value) setAltitude('');
                    }}
                  >
                    <option value="">Table only</option>
                    {coordinateFields.map(name => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Altitude (optional)
                  <select
                    value={altitude}
                    disabled={!longitude || !latitude}
                    onChange={event => setAltitude(event.target.value)}
                  >
                    <option value="">Ground level</option>
                    {coordinateFields.map(name => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
              </fieldset>
              <button
                disabled={Boolean(longitude) !== Boolean(latitude)}
                onClick={() =>
                  run(
                    controller.current!.publish(dataset =>
                      onPublishCompleted(
                        dataset,
                        controller.current?.progress.remoteUrl
                          ? new URL(controller.current.progress.remoteUrl).pathname
                              .split('/')
                              .pop() || 'Remote Arrow stream'
                          : (file?.name ?? 'Arrow stream'),
                        longitude && latitude
                          ? {longitude, latitude, ...(altitude ? {altitude} : {})}
                          : undefined
                      )
                    )
                  )
                }
              >
                Add completed stream
              </button>
            </>
          )}
          {progress.phase === 'published' && (
            <p>
              The complete dataset is available in Kepler. Closing this panel keeps it resident.
            </p>
          )}
        </>
      )}
      {preview && (
        <details open>
          <summary>
            {preview.incomplete ? 'Incomplete source preview' : 'Complete source preview'} — first{' '}
            {preview.rows.length} rows, up to 6 fields
          </summary>
          <div style={{overflowX: 'auto'}}>
            <table aria-label="Source preview rows">
              <thead>
                <tr>
                  {preview.fields.map(field => (
                    <th key={field}>{field}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, index) => (
                  <tr key={index}>
                    {row.map((value, column) => (
                      <td key={column}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Bounded original-source values, not a filtered result or full export. Values are limited
            to 120 characters.
          </p>
        </details>
      )}
      {(error || progress?.error) && (
        <p role="alert" className="error">
          {error || progress?.error}
        </p>
      )}
    </section>
  );
}
