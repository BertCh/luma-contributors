// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCatalogBinding,
  getPolygonParts,
  type AnalysisSensitivityFrame
} from '@luma-contributors/kepler';

/** Six distinct values inside binding limits, around an accepted radius. */
export function getAnalysisSensitivityRadiusValues(radius: number): number[] {
  const specification = getCatalogBinding('geometry-point-buffer')!.parameters.find(
    parameter => parameter.name === 'radius'
  )!;
  const minimum = Math.max(specification.minimum!, radius / 2);
  const maximum = Math.min(specification.maximum!, radius * 2);
  return Array.from({length: 6}, (_, index) => minimum + ((maximum - minimum) * index) / 5);
}

/** Shared extent and fixed colour preserve visible scale differences across all six maps. */
export function createAnalysisSensitivityMaps(frames: readonly AnalysisSensitivityFrame[]) {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  const maps = frames.map(frame => {
    const paths = frame.result.tables.flatMap(table => {
      const geometry = table.geometryColumn;
      return geometry
        ? table.columns[geometry].flatMap(value => {
            if (typeof value !== 'string') return [];
            return getPolygonParts(JSON.parse(value)).map(polygon =>
              polygon
                .map(ring => {
                  for (const [longitude, latitude] of ring) {
                    minimumX = Math.min(minimumX, longitude);
                    maximumX = Math.max(maximumX, longitude);
                    minimumY = Math.min(minimumY, latitude);
                    maximumY = Math.max(maximumY, latitude);
                  }
                  return (
                    ring
                      .map(
                        ([longitude, latitude], index) =>
                          `${index === 0 ? 'M' : 'L'}${longitude},${-latitude}`
                      )
                      .join(' ') + ' Z'
                  );
                })
                .join(' ')
            );
          })
        : [];
    });
    return {value: frame.value, paths};
  });
  const width = Number.isFinite(maximumX - minimumX) ? Math.max(maximumX - minimumX, 1e-6) : 1;
  const height = Number.isFinite(maximumY - minimumY) ? Math.max(maximumY - minimumY, 1e-6) : 1;
  const padding = Math.max(width, height) * 0.05;
  const viewBox = [
    Number.isFinite(minimumX) ? minimumX - padding : 0,
    Number.isFinite(maximumY) ? -maximumY - padding : 0,
    width + 2 * padding,
    height + 2 * padding
  ].join(' ');
  return {maps, viewBox};
}
const sensitivityRuntimeIdentities = new WeakMap<object, number>();
let nextSensitivityRuntimeIdentity = 0;

/** Runtime identity is independent of dataset revisions and uniforms. */
export function getAnalysisSensitivityRuntimeKey(control: {
  device: object;
  registry: {deviceEpoch: number};
  cleanupOwner?: object;
  revisionKey: string;
  parameters: Readonly<Record<string, unknown>>;
}): string {
  const identity = (runtime: object) => {
    let value = sensitivityRuntimeIdentities.get(runtime);
    if (value === undefined) {
      value = ++nextSensitivityRuntimeIdentity;
      sensitivityRuntimeIdentities.set(runtime, value);
    }
    return value;
  };
  return JSON.stringify([
    identity(control.device),
    identity(control.registry),
    control.registry.deviceEpoch,
    control.cleanupOwner ? identity(control.cleanupOwner) : null,
    control.revisionKey,
    control.parameters
  ]);
}
type SensitivityCleanupSession = {destroy(): Promise<void>};
type SensitivityCleanupEntry = {
  session: SensitivityCleanupSession;
  error?: unknown;
  pending?: Promise<void>;
  retryTimer?: ReturnType<typeof setTimeout>;
};

/** Application ownership survives sheet unmount, runtime replacement and failed disposal. */
export class AnalysisSensitivityCleanupOwner {
  private readonly entries = new Map<SensitivityCleanupSession, SensitivityCleanupEntry>();
  private readonly listeners = new Set<() => void>();
  private revision = 0;
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
  readonly getRevision = (): number => this.revision;
  get failures(): readonly unknown[] {
    return Array.from(this.entries.values()).flatMap(entry =>
      entry.error === undefined ? [] : [entry.error]
    );
  }
  get pendingCount(): number {
    return this.entries.size;
  }
  /** Transfer ownership synchronously, before destroy or component disappearance. */
  retire(session: SensitivityCleanupSession): Promise<void> {
    let entry = this.entries.get(session);
    if (entry) return entry.pending ?? Promise.resolve();
    entry = {session};
    this.entries.set(session, entry);
    return this.dispose(entry, true);
  }
  /** Repeated failures remain visible for explicit retry after one automatic retry. */
  retry(): Promise<void> {
    return Promise.all(Array.from(this.entries.values(), entry => this.dispose(entry, false))).then(
      () => undefined
    );
  }
  private dispose(entry: SensitivityCleanupEntry, retryAutomatically: boolean): Promise<void> {
    if (entry.pending) return entry.pending;
    if (entry.retryTimer !== undefined) {
      clearTimeout(entry.retryTimer);
      entry.retryTimer = undefined;
    }
    entry.pending = Promise.resolve()
      .then(() => entry.session.destroy())
      .then(
        () => {
          this.entries.delete(entry.session);
        },
        error => {
          entry.error = error;
          if (retryAutomatically)
            entry.retryTimer = setTimeout(() => {
              entry.retryTimer = undefined;
              void this.dispose(entry, false);
            }, 250);
        }
      )
      .finally(() => {
        entry.pending = undefined;
        this.revision++;
        for (const listener of this.listeners) listener();
      });
    return entry.pending;
  }
}
// Kept by the application module rather than a component or a retiring runtime.
export const analysisSensitivityCleanupOwner = new AnalysisSensitivityCleanupOwner();
