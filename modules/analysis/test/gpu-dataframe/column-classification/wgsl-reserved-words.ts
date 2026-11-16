// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** Returns forbidden tokens after comments are removed; checks uses as well as declarations. */
export function getReservedWGSLIdentifiers(source: string): string[] {
  const code = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ');
  return [...new Set(code.match(/\b[A-Za-z_][A-Za-z_0-9]*\b/g) ?? [])].filter(identifier =>
    WGSL_RESERVED_WORDS.has(identifier)
  );
}
