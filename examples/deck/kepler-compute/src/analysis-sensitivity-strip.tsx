// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useRef, useState, useSyncExternalStore} from 'react';
import type {Device} from '@luma.gl/core';
import {
  AnalysisSensitivitySession,
  ANALYSIS_SENSITIVITY_CAUTION,
  type AnalysisSensitivityProvenance,
  type CatalogFeatureCollection,
  type CatalogOperationBinding,
  type KeplerDatasetRegistry
} from '@luma-contributors/kepler';
import {
  analysisSensitivityCleanupOwner,
  getAnalysisSensitivityRuntimeKey,
  type AnalysisSensitivityCleanupOwner,
  createAnalysisSensitivityMaps,
  getAnalysisSensitivityRadiusValues
} from './analysis-sensitivity-model';

export type AnalysisSensitivityControl = {
  binding: CatalogOperationBinding;
  device: Device;
  registry: KeplerDatasetRegistry;
  parameters: Readonly<Record<string, unknown>>;
  revisionKey: string;
  readInputs(): Promise<Record<string, CatalogFeatureCollection>>;
  isCurrent(): boolean;
  cleanupOwner?: AnalysisSensitivityCleanupOwner;
  onChoose(provenance: AnalysisSensitivityProvenance): void;
};
/** Observe recovery even when uniforms and dataset revision do not change. */
export function AnalysisSensitivityStrip({control}: {control: AnalysisSensitivityControl}) {
  useSyncExternalStore(
    listener => control.registry.subscribeMemory(listener),
    () => control.registry.deviceEpoch,
    () => control.registry.deviceEpoch
  );
  return (
    <AnalysisSensitivityRun key={getAnalysisSensitivityRuntimeKey(control)} control={control} />
  );
}
/** Reports failures from already retired strips, with persistent ownership for retry. */
export function AnalysisSensitivityCleanupStatus({
  owner = analysisSensitivityCleanupOwner
}: {
  owner?: AnalysisSensitivityCleanupOwner;
}) {
  useSyncExternalStore(owner.subscribe, owner.getRevision, owner.getRevision);
  if (!owner.failures.length) return null;
  return (
    <section role="alert" data-testid="ax-sensitivity-cleanup-failure">
      <p>Sensitivity cleanup failed. Its resources remain owned for retry.</p>
      {owner.failures.map((failure, index) => (
        <p key={index}>
          {failure instanceof AggregateError
            ? [failure.message, ...failure.errors.map(String)].join(': ')
            : String(failure)}
        </p>
      ))}
      <button type="button" onClick={() => void owner.retry()}>
        Retry sensitivity cleanup
      </button>
    </section>
  );
}
/** Geometry-only maps from six serialized runs of one compiled graph. */
function AnalysisSensitivityRun({control}: {control: AnalysisSensitivityControl}) {
  const [maps, setMaps] = useState<ReturnType<typeof createAnalysisSensitivityMaps>>();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const session = useRef<AnalysisSensitivitySession | undefined>(undefined);
  const controller = useRef<AbortController | undefined>(undefined);
  const mounted = useRef(true);
  const cleanupOwner = control.cleanupOwner ?? analysisSensitivityCleanupOwner;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controller.current?.abort();
      const retired = session.current;
      session.current = undefined;
      if (retired) void cleanupOwner.retire(retired);
    };
  }, [cleanupOwner]);
  const explore = async () => {
    if (controller.current || running || maps || !control.isCurrent()) return;
    const nextController = new AbortController();
    controller.current = nextController;
    setRunning(true);
    setError('');
    try {
      // Completed views retain their admission until unmount.
      await session.current?.destroy();
      session.current = undefined;
      const inputs = await control.readInputs();
      if (nextController.signal.aborted || !control.isCurrent()) return;
      const rowCount = Object.values(inputs).reduce(
        (total, input) => total + input.features.length,
        0
      );
      const sweep = new AnalysisSensitivitySession({
        ...control,
        inputs,
        parameter: 'radius',
        values: getAnalysisSensitivityRadiusValues(Number(control.parameters['radius'])),
        population: {
          scope: 'full-population',
          rowCount,
          sourceRowCount: rowCount,
          selectionKey: control.revisionKey
        }
      });
      session.current = sweep;
      const frames = await sweep.run(nextController.signal);
      if (frames && control.isCurrent() && mounted.current)
        setMaps(createAnalysisSensitivityMaps(frames));
      else if (mounted.current) await sweep.destroy();
      else await cleanupOwner.retire(sweep);
    } catch (failure) {
      if (mounted.current) setError(String(failure));
      const failed = session.current;
      if (failed) {
        if (!mounted.current) await cleanupOwner.retire(failed);
        else await failed.destroy().catch(cleanup => setError(String(cleanup)));
      }
    } finally {
      if (controller.current === nextController) controller.current = undefined;
      if (mounted.current) setRunning(false);
    }
  };
  return (
    <section aria-label="Radius sensitivity" data-testid="ax-sensitivity-strip">
      <button
        type="button"
        disabled={running || Boolean(maps) || !control.isCurrent()}
        onClick={() => void explore()}
        data-testid="ax-sensitivity-open"
      >
        Explore Radius
      </button>
      <p>{ANALYSIS_SENSITIVITY_CAUTION}</p>
      <p>
        Full selected population. Six maps share one extent and colour. No significance classes are
        displayed.
      </p>
      {running && (
        <button
          type="button"
          data-testid="ax-sensitivity-cancel"
          onClick={() => {
            controller.current?.abort();
            session.current?.cancel();
          }}
        >
          Cancel sensitivity
        </button>
      )}
      {error && <p role="alert">{error}</p>}
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 4}}>
        {maps?.maps.map(map => (
          <figure
            key={map.value}
            style={{margin: 0}}
            data-testid="ax-sensitivity-map"
            data-radius={map.value}
          >
            <svg
              role="img"
              aria-label={`Exploratory radius ${map.value}`}
              viewBox={maps.viewBox}
              width="100%"
              height="100"
              style={{background: '#18212b'}}
            >
              {map.paths.map((path, index) => (
                <path
                  key={index}
                  d={path}
                  fill="#45b6c4"
                  fillOpacity={0.35}
                  stroke="#45b6c4"
                  strokeWidth="1"
                  vectorEffect="non-scaling-stroke"
                  fillRule="evenodd"
                />
              ))}
            </svg>
            <figcaption>Radius {Number(map.value.toPrecision(4))}</figcaption>
            <button
              type="button"
              disabled={!control.isCurrent()}
              onClick={() => {
                try {
                  if (control.isCurrent()) control.onChoose(session.current!.choose(map.value));
                } catch (failure) {
                  setError(String(failure));
                }
              }}
            >
              Choose {Number(map.value.toPrecision(4))}
            </button>
          </figure>
        ))}
      </div>
    </section>
  );
}
