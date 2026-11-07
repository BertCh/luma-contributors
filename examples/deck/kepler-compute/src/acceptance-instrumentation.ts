// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** DEV-only performance.mark/measure instrumentation consumed by the Playwright acceptance harness. */

export type AcceptanceInteractionKind = 'filter' | 'brush' | 'histogram' | 'style' | 'summary';

export type AcceptanceInteraction = {
  kind: AcceptanceInteractionKind;
  startMilliseconds: number;
  submittedFrameMilliseconds: number | null;
  summaryMilliseconds: number | null;
};

export type AcceptanceUploadMeasure = {
  name: string;
  durationMilliseconds: number;
  bytes: number;
};

export type AcceptanceMemoryUsage = {
  allocatedBytes: Readonly<Record<string, number>>;
  reservedBytes: Readonly<Record<string, number>>;
};

export type AcceptanceDeviceReportInput = {
  device: {
    info: object;
    limits: object;
    features: Iterable<string>;
  };
  memoryUsage: AcceptanceMemoryUsage;
  uploadBytes: number | null | undefined;
  readbackBytes: number | null | undefined;
};

const MAXIMUM_ENTRY_COUNT = 4096;

type PendingInteraction = AcceptanceInteraction & {id: number};

const interactions = new Map<number, PendingInteraction>();
let nextInteractionId = 1;
let uploadMeasures: AcceptanceUploadMeasure[] = [];
let frameTimestamps: number[] = [];
let isRecordingFrames = false;

function pushBounded<T>(entries: T[], entry: T): void {
  entries.push(entry);
  if (entries.length > MAXIMUM_ENTRY_COUNT) entries.shift();
}

function measureSafely(
  name: string,
  startMilliseconds: number,
  endMilliseconds: number,
  detail?: unknown
): void {
  try {
    performance.measure(name, {
      start: startMilliseconds,
      end: endMilliseconds,
      detail
    });
  } catch {
    // Instrumentation must never break the application.
  }
}

export function recordSubmittedFrame(): void {
  const timestamp = performance.now();
  try {
    performance.mark('kc:frame');
  } catch {
    // Ignore unsupported performance.mark.
  }
  if (isRecordingFrames) pushBounded(frameTimestamps, timestamp);
  for (const interaction of interactions.values()) {
    if (interaction.submittedFrameMilliseconds !== null) continue;
    if (timestamp < interaction.startMilliseconds) continue;
    interaction.submittedFrameMilliseconds = timestamp - interaction.startMilliseconds;
    measureSafely(`kc:${interaction.kind}:submitted`, interaction.startMilliseconds, timestamp);
  }
}

export function recordSummaryObserved(): void {
  const timestamp = performance.now();
  for (const interaction of interactions.values()) {
    if (interaction.summaryMilliseconds !== null) continue;
    if (timestamp < interaction.startMilliseconds) continue;
    interaction.summaryMilliseconds = timestamp - interaction.startMilliseconds;
    measureSafely(`kc:${interaction.kind}:summary`, interaction.startMilliseconds, timestamp);
  }
}

export function recordUploadMeasure(name: string, startMilliseconds: number, bytes: number): void {
  const endMilliseconds = performance.now();
  pushBounded(uploadMeasures, {
    name,
    durationMilliseconds: endMilliseconds - startMilliseconds,
    bytes
  });
  measureSafely(`kc:upload:${name}`, startMilliseconds, endMilliseconds, {
    bytes
  });
}

function sumValues(record: Readonly<Record<string, number>>): number {
  return Object.values(record).reduce((total, value) => total + value, 0);
}

function copyNumericLimits(limits: Record<string, unknown>): Record<string, number> {
  const result: Record<string, number> = {};
  // GPUSupportedLimits exposes its limits on the prototype, so use for...in.
  for (const key in limits) {
    const value = limits[key];
    if (typeof value === 'number') result[key] = value;
  }
  return result;
}

function copyAdapterInfo(info: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const key in info) {
    const value = info[key];
    if (typeof value !== 'function') result[key] = value;
  }
  return result;
}

export function installAcceptanceInstrumentation(
  getDeviceReportInput: () => AcceptanceDeviceReportInput
): void {
  const navigatorWithMemory = navigator as Navigator & {
    deviceMemory?: number;
  };
  (window as unknown as {__keplerComputeAcceptance: unknown}).__keplerComputeAcceptance = {
    version: 1,
    beginInteraction: (kind: AcceptanceInteractionKind): number => {
      const id = nextInteractionId++;
      const startMilliseconds = performance.now();
      try {
        performance.mark(`kc:${kind}:start:${id}`);
      } catch {
        // Ignore unsupported performance.mark.
      }
      interactions.set(id, {
        id,
        kind,
        startMilliseconds,
        submittedFrameMilliseconds: null,
        summaryMilliseconds: null
      });
      if (interactions.size > MAXIMUM_ENTRY_COUNT) {
        interactions.delete(interactions.keys().next().value as number);
      }
      return id;
    },
    getInteraction: (id: number): AcceptanceInteraction | null => {
      const interaction = interactions.get(id);
      if (!interaction) return null;
      return {
        kind: interaction.kind,
        startMilliseconds: interaction.startMilliseconds,
        submittedFrameMilliseconds: interaction.submittedFrameMilliseconds,
        summaryMilliseconds: interaction.summaryMilliseconds
      };
    },
    clearEntries: (): void => {
      interactions.clear();
      uploadMeasures = [];
      frameTimestamps = [];
      try {
        performance.clearMarks();
        performance.clearMeasures();
      } catch {
        // Ignore unsupported clear.
      }
    },
    startFrameRecording: (): void => {
      frameTimestamps = [];
      isRecordingFrames = true;
    },
    stopFrameRecording: (): {frameIntervalsMilliseconds: number[]} => {
      isRecordingFrames = false;
      const frameIntervalsMilliseconds: number[] = [];
      for (let index = 1; index < frameTimestamps.length; index++) {
        frameIntervalsMilliseconds.push(frameTimestamps[index] - frameTimestamps[index - 1]);
      }
      frameTimestamps = [];
      return {frameIntervalsMilliseconds};
    },
    getUploadMeasures: (): AcceptanceUploadMeasure[] =>
      uploadMeasures.map(measure => ({...measure})),
    getMemorySnapshot: () => {
      let input: AcceptanceDeviceReportInput | null = null;
      try {
        input = getDeviceReportInput();
      } catch {
        // Runtime may be unavailable during recovery.
      }
      const heap = (performance as unknown as {memory?: {usedJSHeapSize?: number}}).memory;
      return {
        registryAllocatedBytes: input ? sumValues(input.memoryUsage.allocatedBytes) : null,
        registryReservedBytes: input ? sumValues(input.memoryUsage.reservedBytes) : null,
        jsHeapUsedBytes: heap?.usedJSHeapSize ?? null,
        sourceUploadBytes: input?.uploadBytes ?? null,
        explorerReadbackBytes: input?.readbackBytes ?? null
      };
    },
    getDeviceReport: () => {
      const {device} = getDeviceReportInput();
      return {
        adapterInfo: copyAdapterInfo(device.info as Record<string, unknown>),
        limits: copyNumericLimits(device.limits as Record<string, unknown>),
        features: [...device.features],
        userAgent: navigator.userAgent,
        deviceMemoryGigabytes: navigatorWithMemory.deviceMemory ?? null,
        hardwareConcurrency: navigator.hardwareConcurrency ?? null
      };
    }
  };
}
