// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  prepareArrowTableForGPUAnalytics,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma.gl/graph/arrow';
import {isExcludedRoute, readFieldRoute} from './source-type-routes';

export type SourceFieldPreview = {
  name: string;
  sourceType: string;
  targetType?: string;
  nullable: boolean;
  selected: boolean;
  kind: 'scalar' | 'category' | 'time' | 'unsupported';
  categoryCount?: number;
  timeUnit?: string;
  timeZone?: string | null;
  maximumAbsoluteError?: number;
  maximumRelativeError?: number;
  reason?: string;
  /** Type route chosen by `source-type-routes`, e.g. "converted: Every digit is kept exactly". */
  route?: string;
};

export type ArrowSourcePreview = {
  rowCount: number;
  batchRowCounts: readonly number[];
  fields: readonly SourceFieldPreview[];
  canUpload: boolean;
  reason?: string;
};

/** CPU ingestion preview using the same adapter as upload. Never allocates or reads GPU resources. */
export function previewArrowSource(
  table: arrow.Table,
  normalization: PrepareArrowTableForGPUAnalyticsOptions = {},
  sourceOnlyColumns: readonly string[] = []
): ArrowSourcePreview {
  const names = table.schema.fields.map(field => field.name);
  const routeOf = (name: string) =>
    readFieldRoute(table.schema.fields.find(field => field.name === name)!);
  const selected =
    normalization.columns ??
    names.filter(name => {
      const route = routeOf(name);
      return !sourceOnlyColumns.includes(name) && !(route && isExcludedRoute(route.kind));
    });
  if (new Set(names).size !== names.length || new Set(selected).size !== selected.length)
    throw new Error('Source and selected field names must be unique');
  if (selected.some(name => !names.includes(name)))
    throw new Error('Unknown selected source field');
  if (selected.some(name => sourceOnlyColumns.includes(name)))
    throw new Error('Source role columns cannot be scalar-normalized');
  const fields: SourceFieldPreview[] = table.schema.fields
    .filter(field => !sourceOnlyColumns.includes(field.name))
    .map(field => {
      const route = readFieldRoute(field);
      const common = {
        name: field.name,
        sourceType: route?.sourceType ?? String(field.type),
        nullable: field.nullable,
        selected: selected.includes(field.name),
        ...(route ? {route: route.route} : {})
      };
      if (route && isExcludedRoute(route.kind))
        return {...common, kind: 'unsupported', reason: route.route};
      try {
        const prepared = prepareArrowTableForGPUAnalytics(table, {
          ...normalization,
          columns: [field.name]
        });
        const time = prepared.timeColumns[field.name];
        if (time)
          return {
            ...common,
            kind: 'time',
            targetType: 'int64 time words',
            timeUnit: time.unit,
            timeZone: time.timeZone
          };
        const conversion = prepared.conversions[0];
        const dictionary = prepared.dictionaries[field.name];
        return {
          ...common,
          kind: dictionary ? 'category' : 'scalar',
          targetType: conversion.targetType,
          maximumAbsoluteError: conversion.maximumAbsoluteError,
          maximumRelativeError: conversion.maximumRelativeError,
          ...(dictionary ? {categoryCount: dictionary.values.length} : {})
        };
      } catch (error) {
        return {
          ...common,
          kind: 'unsupported',
          reason: error instanceof Error ? error.message : String(error)
        };
      }
    });
  const unsupported = fields.filter(field => field.selected && field.kind === 'unsupported');
  const reason = unsupported.length
    ? `Choose a supported conversion or exclude these fields: ${unsupported.map(field => field.name).join(', ')}.`
    : undefined;
  return {
    rowCount: table.numRows,
    batchRowCounts: table.batches.map(batch => batch.numRows),
    fields,
    canUpload: !reason,
    reason
  };
}
