// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerDatasetRegistry, type KeplerMemoryReservation} from '../runtime/dataset-registry';
import {normalizeCatalogParameters, validateCatalogInputs} from './catalog-bindings';
import {getCatalogFeatureInputs, type CatalogDataInput} from './catalog-data-input';
import {validateCatalogCoordinateSystem} from './catalog-geojson';
import type {
  CatalogOperationBinding,
  CatalogPreviewExecutor,
  CatalogRunResult
} from './catalog-binding-types';

export type AnalysisPreviewRequest = {
  parameters: Readonly<Record<string, unknown>>;
  inputs: Readonly<Record<string, CatalogDataInput>>;
  /** Must change for input, population, selection, roles or runtime/device replacement. */
  revisionKey: string;
  signal?: AbortSignal;
  /** Checked after compile, execution and immediately before committing. */
  isCurrent: () => boolean;
  /** Synchronous atomic display handoff. Throwing must leave the previous display intact. */
  commit: (result: CatalogRunResult) => void;
};

type RetainedPreview = {
  key: string;
  executor: CatalogPreviewExecutor;
  reservations: KeplerMemoryReservation[];
};

/**
 * Latest-request-wins execution for bindings that explicitly support retained previews.
 * Admission covers the old retained executor plus the new peak before create can allocate.
 * Builds and executions are serialized; cancellation prevents scheduling/publication, while GPU
 * completion is still awaited before destruction or release. Results own their CPU arrays.
 */
export class AnalysisPreviewSession {
  private retained?: RetainedPreview;
  private tail: Promise<unknown> = Promise.resolve();
  private controller?: AbortController;
  private destroyed = false;
  private destruction?: Promise<void>;
  private readonly deviceEpoch: number;
  private readonly retired = new Set<RetainedPreview>();
  private readonly cleanupErrors: unknown[] = [];

  constructor(
    private readonly options: {
      registry: KeplerDatasetRegistry;
      device: Device;
      binding: CatalogOperationBinding;
    }
  ) {
    this.deviceEpoch = options.registry.deviceEpoch;
    validateRequirement(
      'AnalysisPreviewSession',
      Boolean(options.binding.planPreview),
      `binding ${options.binding.id} must support retained previews`
    );
  }

  /** Returns false for cancelled, superseded or obsolete work. Other failures preserve the display. */
  request(request: AnalysisPreviewRequest): Promise<boolean> {
    if (this.destroyed) return Promise.resolve(false);
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    const abort = () => controller.abort();
    request.signal?.addEventListener('abort', abort, {once: true});
    if (request.signal?.aborted) controller.abort();
    const execution = this.tail.then(() => this.executeRequest(request, controller.signal));
    this.tail = execution.catch(() => undefined);
    return execution.finally(() => {
      request.signal?.removeEventListener('abort', abort);
      if (this.controller === controller) this.controller = undefined;
    });
  }

  cancel(): void {
    this.controller?.abort();
  }

  /** A committed display remains accepted if predecessor cleanup fails; its bytes stay reserved. */
  takeCleanupErrors(): unknown[] {
    return this.cleanupErrors.splice(0);
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.cancel();
    this.destruction = this.tail
      .then(async () => {
        const retained = this.retained;
        this.retained = undefined;
        if (retained) this.retired.add(retained);
        const outcomes = await Promise.allSettled(
          Array.from(this.retired, preview => this.releasePreview(preview))
        );
        const failures = outcomes.filter(outcome => outcome.status === 'rejected');
        if (failures.length)
          throw new AggregateError(
            failures.map(outcome => outcome.reason),
            'AnalysisPreviewSession cleanup failed'
          );
      })
      .catch(error => {
        this.destruction = undefined;
        throw error;
      });
    return this.destruction;
  }

  private async executeRequest(
    request: AnalysisPreviewRequest,
    signal: AbortSignal
  ): Promise<boolean> {
    const epoch = this.deviceEpoch;
    const current = () =>
      !this.destroyed &&
      !signal.aborted &&
      this.options.registry.deviceEpoch === epoch &&
      request.isCurrent();
    if (!current()) return false;
    const {binding, registry, device} = this.options;
    const parameters = normalizeCatalogParameters(binding, request.parameters);
    validateCatalogInputs(binding, request.inputs);
    validateCatalogCoordinateSystem(
      binding.id,
      parameters['coordinateSystem'],
      getCatalogFeatureInputs(request.inputs)
    );
    const plan = binding.planPreview!(request.inputs, parameters);
    validateCapacity('AnalysisPreviewSession', 'gpuByteLength', plan.gpuByteLength, 0);
    validateCapacity('AnalysisPreviewSession', 'cpuByteLength', plan.cpuByteLength, 0);
    const key = JSON.stringify([epoch, request.revisionKey, plan.key]);
    let next = this.retained;
    let replacement = false;
    try {
      if (!next || next.key !== key) {
        const reservations: KeplerMemoryReservation[] = [];
        try {
          reservations.push(registry.reserveMemory(plan.gpuByteLength, 'gpu', 'scratch'));
          reservations.push(registry.reserveMemory(plan.cpuByteLength, 'cpu', 'output'));
          // Both admissions have succeeded before even the first buffer or pipeline allocation.
          const executor = await plan.create({device, signal});
          // Capture cleanup ownership before initialization can allocate or reject.
          next = {key, executor, reservations};
          replacement = true;
        } catch (error) {
          for (const reservation of reservations) reservation.release();
          throw error;
        }
      }
      if (!current()) return false;
      if (replacement) {
        await next.executor.initialize?.(signal);
        if (!current()) return false;
      }
      const output = await next.executor.execute(parameters, signal);
      if (!current()) return false;
      for (const table of output.tables) {
        const specification = binding.outputs.find(candidate => candidate.tableId === table.id);
        validateRequirement(
          'AnalysisPreviewSession',
          Boolean(specification) && table.rowCount <= specification!.maximumRows,
          `${binding.id} output ${table.id} must fit maximumRows ${specification?.maximumRows ?? 0}`
        );
      }
      const result: CatalogRunResult = {
        ...output,
        operationId: binding.id,
        engine: 'gpu',
        parameters,
        dispatchedContributors: output.dispatchedContributors ?? binding.gpuContributorNames
      };
      if (!current()) return false;
      request.commit(result);
      const previous = this.retained;
      this.retained = next;
      replacement = false;
      if (previous && previous !== next) {
        try {
          await this.releasePreview(previous);
        } catch (error) {
          this.cleanupErrors.push(error);
        }
      }
      return true;
    } catch (error) {
      if (!current()) return false;
      throw error;
    } finally {
      if (replacement && next) {
        try {
          await this.releasePreview(next);
        } catch (error) {
          // releasePreview retains this owner and its admission for destruction retries.
          // Preserve the initialization/execution failure (or cancellation) reported above.
          this.cleanupErrors.push(error);
        }
      }
    }
  }

  private async releasePreview(preview: RetainedPreview): Promise<void> {
    this.retired.add(preview);
    // A failing destroy must keep its reservation: freeing bytes still in use is unsafe.
    await preview.executor.destroy();
    for (const reservation of preview.reservations) reservation.release();
    this.retired.delete(preview);
  }
}
