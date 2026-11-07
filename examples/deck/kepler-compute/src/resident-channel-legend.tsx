// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {memo, useEffect, useReducer} from 'react';
import {LayerLegendContentFactory} from '@kepler.gl/components';
import {ResidentConsumerNotice} from './resident-consumer-notice';
import {getResidentChannelLegendModel} from './resident-channel-legend-model';
import {getAnalysisLegendModel, type AnalysisLegendSummary} from './analysis-legend';

export type ResidentChannelLegendOptions = {
  isResidentDataset(datasetId: string): boolean;
  /** GPU-derived finite domain of a size/radius/height field, or null while unavailable. */
  getChannelDomain?(datasetId: string, fieldName: string): readonly number[] | null;
  /** Stated reason the domain is unavailable (refused or empty); null while pending. */
  getChannelDomainFailure?(datasetId: string, fieldName: string): string | null;
  /** Active analysis result for the dataset, when it colors the map. */
  getAnalysis?(datasetId: string): {mode: string; summary: AnalysisLegendSummary | null} | null;
  subscribe?(listener: () => void): () => void;
};
type NativeContentProps = Parameters<ReturnType<typeof LayerLegendContentFactory>>[0];
type ChannelKind = 'size' | 'radius' | 'height';

const CHANNEL_SCALE_KEYS: Record<string, ChannelKind> = {
  size: 'size',
  radius: 'radius',
  height: 'height',
  strokeWidth: 'size'
};

function rgbCSS(color: readonly [number, number, number]): string {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

/** Wraps the native legend content, adding resident size/radius/height and analysis legends. */
export function createResidentChannelLegendFactory(options: ResidentChannelLegendOptions) {
  function ResidentChannelLegendFactory(
    ...dependencies: Parameters<typeof LayerLegendContentFactory>
  ) {
    const NativeContent = LayerLegendContentFactory(...dependencies);
    return memo(function ResidentLegendContent(props: NativeContentProps) {
      const [, refresh] = useReducer((revision: number) => revision + 1, 0);
      useEffect(() => options.subscribe?.(refresh), []);
      const layer = props.layer;
      const datasetId = layer.config.dataId;
      const config: Record<string, any> = layer.config;
      if (!datasetId || !options.isResidentDataset(datasetId)) return <NativeContent {...props} />;
      const channels = Object.values(layer.getLegendVisualChannels() as Record<string, any>)
        .filter(channel => CHANNEL_SCALE_KEYS[channel.key] && config[channel.field])
        .filter(channel => !channel.condition || channel.condition(layer.config));
      const analysis = options.getAnalysis?.(datasetId);
      const analysisModel = analysis
        ? getAnalysisLegendModel(analysis.mode, analysis.summary)
        : null;
      return (
        <>
          <NativeContent {...props} />
          {channels.map(channel => {
            const kind = CHANNEL_SCALE_KEYS[channel.key];
            const fieldName = config[channel.field]?.name as string | undefined;
            const domain = fieldName
              ? (options.getChannelDomain?.(datasetId, fieldName) ?? null)
              : null;
            const failure = fieldName
              ? (options.getChannelDomainFailure?.(datasetId, fieldName) ?? null)
              : null;
            if (failure)
              return (
                <ResidentConsumerNotice
                  key={channel.key}
                  consumer="legend"
                  status="unsupported"
                  reason={failure}
                />
              );
            const range = layer.config.visConfig[channel.range] as readonly number[] | undefined;
            if (!domain || !range || range.length < 2)
              return (
                <ResidentConsumerNotice
                  key={channel.key}
                  consumer="legend"
                  status="pending"
                  reason={`Computing GPU ${kind} legend...`}
                />
              );
            const model = getResidentChannelLegendModel({
              scale: String(config[channel.scale]),
              domain,
              range: [range[0], range[1]]
            });
            if (model.status !== 'ready')
              return (
                <ResidentConsumerNotice
                  key={channel.key}
                  consumer="legend"
                  status={model.status}
                  reason={model.reason}
                />
              );
            return (
              <div key={channel.key} data-resident-channel-legend={kind}>
                {model.ticks.map((tick, index) => (
                  <div key={index} style={{display: 'flex', alignItems: 'center', gap: 6}}>
                    <span
                      aria-hidden="true"
                      style={{
                        display: 'inline-block',
                        width: kind === 'radius' ? tick.size * 2 : Math.max(2, tick.size),
                        height: kind === 'radius' ? tick.size * 2 : kind === 'height' ? 6 : 6,
                        borderRadius: kind === 'radius' ? '50%' : 0,
                        border: '1px solid currentColor'
                      }}
                    />
                    <span>{Number(tick.value.toPrecision(5)).toLocaleString('en-US')}</span>
                  </div>
                ))}
              </div>
            );
          })}
          {analysisModel && (
            <div data-resident-analysis-legend>
              <strong>{analysisModel.title}</strong>
              {analysisModel.entries.map((entry, index) => (
                <div key={index} style={{display: 'flex', alignItems: 'center', gap: 6}}>
                  <span
                    aria-hidden="true"
                    style={{
                      display: 'inline-block',
                      width: 16,
                      height: 12,
                      background: rgbCSS(entry.color)
                    }}
                  />
                  <span>{entry.label}</span>
                </div>
              ))}
              {analysisModel.missingCount !== 0 && (
                <div data-resident-legend-missing>
                  {typeof analysisModel.missingCount === 'number'
                    ? `No result: ${analysisModel.missingCount.toLocaleString('en-US')} rows`
                    : 'No result (null, invalid or excluded rows are not colored)'}
                </div>
              )}
            </div>
          )}
        </>
      );
    });
  }
  ResidentChannelLegendFactory.deps = LayerLegendContentFactory.deps;
  return ResidentChannelLegendFactory;
}
