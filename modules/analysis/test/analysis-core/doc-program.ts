// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

// Shared API-page discovery for navigation, snippet checks and generated Parameters blocks,
// plus a TypeScript program using the repository's compiler options and optional in-memory files.

export const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
export const DOCS_ROOT = join(REPOSITORY_ROOT, 'docs/api');

/** Discovers every API Markdown page, relative to the root, in deterministic path order. */
export function discoverApiDocPages(root = DOCS_ROOT): string[] {
  const visit = (directory: string, prefix: string): string[] =>
    readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
      const page = `${prefix}${entry.name}`;
      return entry.isDirectory()
        ? visit(join(directory, entry.name), `${page}/`)
        : entry.isFile() && entry.name.endsWith('.md')
          ? [page]
          : [];
    });
  return visit(root, '').sort();
}

export const ANALYSIS_DOC_PAGES = discoverApiDocPages();

type ContentsItem = string | {items: ContentsItem[]};

/** Every discovered API page must have exactly one sidebar entry, and every target must exist. */
export function getSidebarPageIssues(
  pages: readonly string[] = ANALYSIS_DOC_PAGES,
  contents: ContentsItem[] = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'docs/table-of-contents.json'), 'utf8')
  )
): string[] {
  const getPages = (items: ContentsItem[]): string[] =>
    items.flatMap(item =>
      typeof item === 'string'
        ? item.startsWith('api/')
          ? [`${item.slice(4)}.md`]
          : []
        : getPages(item.items)
    );
  const sidebarPages = getPages(contents);
  return [
    ...pages.flatMap(page => {
      const count = sidebarPages.filter(target => target === page).length;
      return count === 1 ? [] : [`${page}: expected one sidebar entry, found ${count}`];
    }),
    ...sidebarPages.filter(page => !pages.includes(page)).map(page => `${page}: missing API page`)
  ];
}

/** Public entries used by documentation fixtures. */
export const ANALYSIS_ENTRIES = [
  '@luma-contributors/analysis/analysis-core',
  '@luma-contributors/analysis/gpu-spatial-analysis',
  '@luma-contributors/analysis/gpu-network',
  '@luma-contributors/analysis/gpu-terrain',
  '@luma-contributors/analysis/raster-analysis',
  '@luma-contributors/analysis/dataframe-analysis',
  '@luma-contributors/analysis/geospatial-analysis',
  '@luma-contributors/analysis/pointcloud-analysis',
  '@luma-contributors/analysis/computation-record',
  '@luma.gl/graph/gpu-raster',
  '@luma.gl/graph/gpu-dataframe',
  '@luma.gl/graph/geospatial',
  '@luma.gl/graph/gpu-tables',
  '@luma.gl/graph/gpu-core',
  '@luma.gl/core'
] as const;

/** Reads the root compiler options (strict, bundler resolution, `@luma.gl/*` paths). */
export function getRepositoryCompilerOptions(): typescript.CompilerOptions {
  const configPath = join(REPOSITORY_ROOT, 'tsconfig.json');
  const config = typescript.readConfigFile(configPath, typescript.sys.readFile);
  const parsed = typescript.parseJsonConfigFileContent(
    config.config,
    typescript.sys,
    REPOSITORY_ROOT
  );
  return {
    ...parsed.options,
    noEmit: true,
    composite: false,
    incremental: false,
    declaration: false,
    // Snippets show isolated steps; an unused local is not an API error.
    noUnusedLocals: false
  };
}

/**
 * Creates a program whose root files are the given in-memory files (absolute path → text). Every
 * other file is read from disk, so package-name imports resolve to the current sources.
 */
export function createVirtualProgram(
  virtualFiles: Map<string, string>,
  options: typescript.CompilerOptions = getRepositoryCompilerOptions(),
  oldProgram?: typescript.Program
): typescript.Program {
  const host = typescript.createCompilerHost(options, true);
  const readFile = host.readFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  const getSourceFile = host.getSourceFile.bind(host);
  host.readFile = fileName => virtualFiles.get(fileName) ?? readFile(fileName);
  host.fileExists = fileName => virtualFiles.has(fileName) || fileExists(fileName);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const text = virtualFiles.get(fileName);
    return text === undefined
      ? getSourceFile(fileName, languageVersion, onError, shouldCreate)
      : typescript.createSourceFile(fileName, text, languageVersion, true);
  };
  // Validate literal imports before repository `paths` can make private leaves look public.
  // Only roots are fenced documentation: source dependencies retain repository resolution.
  host.resolveModuleNames = (names, containingFile) =>
    names.map(name => {
      if (virtualFiles.has(containingFile) && getPublicPackageSpecifierIssue(name))
        return undefined;
      return typescript.resolveModuleName(name, containingFile, options, host).resolvedModule;
    });
  return typescript.createProgram([...virtualFiles.keys()], options, host, oldProgram);
}

/** Reads one analysis doc page. */
export function readDocPage(page: string): string {
  const path = join(DOCS_ROOT, page);
  if (!existsSync(path)) {
    throw new Error(`missing doc page ${page}`);
  }
  return readFileSync(path, 'utf8');
}

/** Formats a diagnostic as `message` with its line and column in the file. */
export function formatDiagnostic(diagnostic: typescript.Diagnostic): string {
  const message = typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
  if (!diagnostic.file || diagnostic.start === undefined) {
    return message;
  }
  const {line, character} = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  return `${line + 1}:${character + 1} TS${diagnostic.code} ${message}`;
}

export const FIRST_RUN_HEADINGS = [
  'Shared first-run lifecycle',
  'First-run terrain derivatives',
  'First-run exact distance field',
  'First-run time-window filter',
  'First-run point-cloud statistical filter'
] as const;

/** Literal bodies and original Markdown lines, including executable invocation lines. */
export function getFirstRunFences(text = readDocPage('README.md')) {
  return FIRST_RUN_HEADINGS.map(heading => getReadmeFence(heading, text));
}

function getReadmeFence(heading: string, text: string) {
  const headingOffset = text.indexOf(`### ${heading}\n`);
  if (headingOffset < 0) throw new Error(`README.md: missing ${heading}`);
  const section = text.slice(headingOffset).split(/\n#{1,3} /)[0];
  const matches = [...section.matchAll(/^```ts\n([\s\S]*?)^```/gm)];
  if (matches.length !== 1) throw new Error(`README.md: expected one fence in ${heading}`);
  const match = matches[0];
  const fenceLine = text.slice(0, headingOffset + match.index!).split('\n').length;
  const code = match[1].trimEnd();
  return {heading, code, fenceLine};
}

/** The Deck integration is checked separately from the four executable family examples. */
export function getDeckOwnedDeviceFence(text = readDocPage('README.md')) {
  return getReadmeFence('Using a Deck-owned device', text);
}

function pairFirstRunFence(
  shared: ReturnType<typeof getReadmeFence>,
  example: ReturnType<typeof getReadmeFence>
) {
  return {
    heading: example.heading,
    page: 'README.md',
    fenceLine: example.fenceLine,
    code: `${shared.code}\n\n${example.code}`,
    markdownLines: [
      ...shared.code.split('\n').map((_, index) => shared.fenceLine + index + 1),
      shared.fenceLine,
      ...example.code.split('\n').map((_, index) => example.fenceLine + index + 1)
    ]
  };
}

export function getPairedDeckOwnedDeviceFence(text = readDocPage('README.md')) {
  return pairFirstRunFence(getFirstRunFences(text)[0], getDeckOwnedDeviceFence(text));
}

export function getPairedFirstRunFences(text = readDocPage('README.md')) {
  const [shared, ...examples] = getFirstRunFences(text);
  return examples.map(example => pairFirstRunFence(shared, example));
}

/** Match the manifest contract before mapping an analysis package import to checkout source. */
export function getPublicPackageSpecifierIssue(specifier: string): string | undefined {
  const packageName = '@luma-contributors/analysis';
  if (specifier !== packageName && !specifier.startsWith(`${packageName}/`)) return undefined;
  const manifest = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/package.json'), 'utf8')
  ) as {exports: Record<string, unknown>};
  const subpath = `.${specifier.slice(packageName.length)}`;
  return Object.hasOwn(manifest.exports, subpath)
    ? undefined
    : `unpublished package specifier ${specifier}`;
}
