// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** Checks lexical identifier tokens against the full W3C table, independently of an interpreter. */
export function getUtilizationReservedIdentifiers(source: string): string[] {
  const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
  return [...new Set(withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(identifier =>
    WGSL_RESERVED_WORDS.has(identifier)
  );
}
