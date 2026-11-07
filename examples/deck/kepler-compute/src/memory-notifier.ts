// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MEMORY_CRITICAL_FRACTION,
  MEMORY_WARNING_FRACTION,
  formatMemoryBytes,
  type MemoryPressure
} from './memory-manager-model';

export type MemoryNotice = Readonly<{
  /** Stable key so the host can replace rather than stack repeated notices. */
  key: string;
  type: 'warning' | 'error' | 'info';
  message: string;
}>;

export type MemoryNotifierProps = {
  getUsage(): Readonly<{totalBytes: number; budgetBytes: number}>;
  /** Delivers a notice; the host forwards it to the Kepler notification store. */
  emit(notice: MemoryNotice): void;
};

/** Hysteresis band: pressure must fall this far below a threshold before it can warn again. */
const RELEASE_MARGIN = 0.05;

/**
 * Turns registry usage changes into at most one notice per upward threshold crossing, plus
 * explicit notices for refused admissions and evictions. Pure state machine; no timers.
 */
export class MemoryNotifier {
  private pressure: MemoryPressure = 'ok';

  constructor(private readonly props: MemoryNotifierProps) {}

  /** Call after every registry memory change. */
  check(): void {
    const {totalBytes, budgetBytes} = this.props.getUsage();
    // An unavailable (zero) budget, e.g. while a workspace is rebuilt, carries no pressure information.
    const fraction = budgetBytes > 0 ? totalBytes / budgetBytes : 0;
    const next = this.getNextPressure(fraction);
    if (next === this.pressure) return;
    const rising = rank(next) > rank(this.pressure);
    this.pressure = next;
    if (!rising || next === 'ok') return;
    const percent = Math.round(fraction * 100);
    this.props.emit({
      key: `memory-pressure-${next}`,
      type: next === 'critical' ? 'error' : 'warning',
      message: `GPU memory budget ${percent}% used (${formatMemoryBytes(totalBytes)} of ${formatMemoryBytes(budgetBytes)}). Open the Memory manager to unload datasets or raise the budget.`
    });
  }

  /** A reservation was refused before allocation. */
  reportRefusal(requestedBytes?: number): void {
    const {totalBytes, budgetBytes} = this.props.getUsage();
    this.props.emit({
      key: 'memory-refused',
      type: 'error',
      message: `Request${requestedBytes === undefined ? '' : ` for ${formatMemoryBytes(requestedBytes)}`} refused: ${formatMemoryBytes(totalBytes)} of ${formatMemoryBytes(budgetBytes)} is in use. Nothing was allocated.`
    });
  }

  reportEviction(datasetLabel: string, releasedBytes: number): void {
    this.props.emit({
      key: `memory-evicted-${datasetLabel}`,
      type: 'info',
      message: `Evicted ${datasetLabel} (${formatMemoryBytes(releasedBytes)}); it reloads from its retained source when used.`
    });
  }

  private getNextPressure(fraction: number): MemoryPressure {
    const warning =
      this.pressure === 'ok' ? MEMORY_WARNING_FRACTION : MEMORY_WARNING_FRACTION - RELEASE_MARGIN;
    const critical =
      this.pressure === 'critical'
        ? MEMORY_CRITICAL_FRACTION - RELEASE_MARGIN
        : MEMORY_CRITICAL_FRACTION;
    if (fraction >= critical) return 'critical';
    return fraction >= warning ? 'warning' : 'ok';
  }
}

function rank(pressure: MemoryPressure): number {
  return pressure === 'ok' ? 0 : pressure === 'warning' ? 1 : 2;
}
