// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {defineConfig, mergeConfig, type Alias, type UserConfig} from 'vite';

const EXAMPLES_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const MODULES_DIRECTORY = resolve(EXAMPLES_DIRECTORY, '../modules');

type PackageExportTarget = string | {import?: string; default?: string; types?: string};

type ExamplePackageJson = {
  name?: string;
  exports?: Record<string, PackageExportTarget>;
};

/** Options for {@link createExampleViteConfig}. */
export type ExampleViteConfigOptions = {
  /**
   * Directory name used for the website deployment, for example `city-scene`. When set, builds in
   * `--mode website` use `<WEBSITE_BASE_URL>/standalone-examples/<name>/` as base URL and other
   * modes use a relative base.
   */
  websiteExampleName?: string;
  /** Extra Vite configuration merged over the shared defaults (arrays are concatenated). */
  config?: UserConfig;
};

/** Maps a package export target such as `./dist/graph/models/index.js` to its source path. */
function getSourcePath(packageDirectory: string, exportTarget: PackageExportTarget): string | null {
  const target =
    typeof exportTarget === 'string'
      ? exportTarget
      : (exportTarget.import ?? exportTarget.default ?? exportTarget.types);
  if (!target?.startsWith('./dist/') || target.includes('*')) {
    return null;
  }
  const sourceBase = join(packageDirectory, 'src', target.slice('./dist/'.length)).replace(
    /(\.d)?\.(js|cjs|ts)$/,
    ''
  );
  const candidates = [`${sourceBase}.ts`, `${sourceBase}.tsx`, `${sourceBase}/index.ts`];
  return candidates.find(candidate => existsSync(candidate)) ?? null;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
}

let cachedAliases: Alias[] | null = null;
let cachedModuleImportPaths: string[] | null = null;

/**
 * Creates an alias for every package and subpath export of every package in `modules/`, so
 * standalone examples always run against local source. Derived from each `package.json`
 * `exports` map, so new subpaths need no per-example configuration.
 */
export function getModuleAliases(): Alias[] {
  if (cachedAliases) {
    return cachedAliases;
  }
  const aliases: Alias[] = [];
  const moduleImportPaths: string[] = [];
  const fallbackAliases: Alias[] = [];
  for (const moduleName of readdirSync(MODULES_DIRECTORY)) {
    const packageDirectory = join(MODULES_DIRECTORY, moduleName);
    const packageJsonPath = join(packageDirectory, 'package.json');
    if (!existsSync(packageJsonPath)) {
      continue;
    }
    const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8')) as ExamplePackageJson;
    if (!packageJson.name || !packageJson.exports) {
      continue;
    }
    // Unlisted deep imports keep resolving to source, as they did with the former prefix aliases.
    fallbackAliases.push({
      find: new RegExp(`^${escapeRegExp(packageJson.name)}/(.+)$`),
      replacement: `${join(packageDirectory, 'src')}/$1`
    });
    for (const [exportKey, exportTarget] of Object.entries(packageJson.exports)) {
      const sourcePath = getSourcePath(packageDirectory, exportTarget);
      if (!sourcePath) {
        continue;
      }
      const importPath =
        exportKey === '.' ? packageJson.name : `${packageJson.name}${exportKey.slice(1)}`;
      // Exact match so that `@luma.gl/engine` never captures `@luma.gl/engine/models`.
      aliases.push({
        find: new RegExp(`^${escapeRegExp(importPath)}$`),
        replacement: sourcePath
      });
      moduleImportPaths.push(importPath);
    }
  }
  cachedAliases = [...aliases, ...fallbackAliases];
  cachedModuleImportPaths = moduleImportPaths;
  return cachedAliases;
}

/**
 * Import paths of every local module. They are excluded from dependency pre-bundling and
 * deduplicated, so that node_modules packages such as deck.gl pre-bundled by Vite share the one
 * source copy of luma.gl instead of bundling a second instance.
 */
export function getModuleImportPaths(): string[] {
  getModuleAliases();
  return cachedModuleImportPaths ?? [];
}

/** Creates the Vite configuration shared by all standalone examples. */
export function createExampleViteConfig(options: ExampleViteConfigOptions = {}) {
  return defineConfig(({mode}) => {
    const baseConfig: UserConfig = {
      resolve: {alias: getModuleAliases(), dedupe: getModuleImportPaths()},
      optimizeDeps: {exclude: getModuleImportPaths()},
      server: {open: true}
    };
    if (options.websiteExampleName) {
      const websiteBaseUrl = (process.env['WEBSITE_BASE_URL'] || '/').replace(/\/?$/, '/');
      baseConfig.base =
        mode === 'website'
          ? `${websiteBaseUrl}standalone-examples/${options.websiteExampleName}/`
          : './';
    }
    return options.config ? mergeConfig(baseConfig, options.config) : baseConfig;
  });
}
