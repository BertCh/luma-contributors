// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisColumnProfile} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {KeplerDatasetProfileCache} from './dataset-profile-cache';
import {withKeplerDatasetProfileStatistics} from './dataset-profile';
import type {KeplerDatasetProfile, KeplerDatasetProfileProgress} from './dataset-profile-types';

export type KeplerDatasetProfileStageContext = {
  profile: KeplerDatasetProfile;
  signal: AbortSignal;
};

/** Providers finish their submitted work and readbacks before settling, including on cancellation. */
export type KeplerDatasetProfileStageRunner = {
  columns?(context: KeplerDatasetProfileStageContext): Promise<readonly GPUAnalysisColumnProfile[]>;
  spacing?(context: KeplerDatasetProfileStageContext): Promise<KeplerDatasetProfile['spacing']>;
  contiguity?(
    context: KeplerDatasetProfileStageContext
  ): Promise<KeplerDatasetProfile['contiguity']>;
};

export type KeplerDatasetProfileJob = {
  finished: Promise<KeplerDatasetProfile | undefined>;
  cancel(): void;
};

/** Metadata is immediately usable; each later stage refines the same revision and source generation. */
export function createKeplerDatasetProfileStages(options: {
  cache: KeplerDatasetProfileCache;
  onProgress?: (progress: KeplerDatasetProfileProgress) => void;
}) {
  const jobs = new Map<string, AbortController>();
  return {
    cancel(datasetId: string): void {
      jobs.get(datasetId)?.abort();
    },
    start(request: {
      metadata: KeplerDatasetProfile;
      runner: KeplerDatasetProfileStageRunner;
      getRevision: () => number | undefined;
      signal?: AbortSignal;
    }): KeplerDatasetProfileJob {
      const {metadata, runner} = request;
      jobs.get(metadata.datasetId)?.abort();
      const controller = new AbortController();
      jobs.set(metadata.datasetId, controller);
      const cancel = () => controller.abort();
      request.signal?.addEventListener('abort', cancel, {once: true});
      if (request.signal?.aborted) cancel();
      const started = performance.now();
      const timings: KeplerDatasetProfileProgress['timings'] = {};
      let published: KeplerDatasetProfile | undefined;
      let generation: number | undefined;
      const current = () =>
        !controller.signal.aborted &&
        jobs.get(metadata.datasetId) === controller &&
        request.getRevision() === metadata.revision &&
        (!published || options.cache.getGeneration(metadata.datasetId) === generation);
      const announce = (stage: KeplerDatasetProfileProgress['stage'], error?: unknown) => {
        if (jobs.get(metadata.datasetId) !== controller) return;
        options.onProgress?.({
          datasetId: metadata.datasetId,
          revision: metadata.revision,
          stage,
          elapsedMilliseconds: performance.now() - started,
          timings: {...timings},
          profile: published,
          ...(error === undefined ? {} : {error})
        });
      };
      if (current()) {
        published = options.cache.set(metadata);
        generation = options.cache.getGeneration(metadata.datasetId);
        announce('metadata');
      }
      const finished = Promise.resolve().then(async () => {
        try {
          for (const stage of ['columns', 'spacing', 'contiguity'] as const) {
            if (!current() || !published) break;
            const provider = runner[stage];
            if (!provider) continue;
            announce(stage);
            if (!current()) break;
            const profile = options.cache.get(metadata.datasetId, metadata.revision)!;
            const stageStarted = performance.now();
            const value = await provider({profile, signal: controller.signal});
            timings[stage] = performance.now() - stageStarted;
            if (!current()) break;
            const latest = options.cache.get(metadata.datasetId, metadata.revision)!;
            const refined = options.cache.refine(
              withKeplerDatasetProfileStatistics(latest, {[stage]: value}),
              generation
            );
            if (!refined) {
              announce('stale');
              return undefined;
            }
            published = refined;
            announce(stage);
          }
          if (!current()) {
            announce(controller.signal.aborted ? 'cancelled' : 'stale');
            return undefined;
          }
          announce('complete');
          return options.cache.get(metadata.datasetId, metadata.revision);
        } catch (error) {
          if (!current()) {
            announce(controller.signal.aborted ? 'cancelled' : 'stale');
            return undefined;
          }
          announce('failed', error);
          throw error;
        } finally {
          request.signal?.removeEventListener('abort', cancel);
          if (jobs.get(metadata.datasetId) === controller) jobs.delete(metadata.datasetId);
        }
      });
      return {finished, cancel};
    }
  };
}
