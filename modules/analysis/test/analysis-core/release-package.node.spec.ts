// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';

const PACKAGE_DIRECTORY = fileURLToPath(new URL('../../', import.meta.url));
const WORKSPACE_DIRECTORY = fileURLToPath(new URL('../../../../', import.meta.url));
const packageManifest = JSON.parse(readFileSync(join(PACKAGE_DIRECTORY, 'package.json'), 'utf8'));

function collectPackedFiles(): string[] {
  const cacheDirectory = mkdtempSync(join(tmpdir(), 'analysis-package-inventory-'));
  try {
    const result = spawnSync(
      'npm',
      [
        'pack',
        '--dry-run',
        '--offline',
        '--ignore-scripts',
        '--json',
        '--workspaces=false',
        '--cache',
        cacheDirectory
      ],
      {cwd: PACKAGE_DIRECTORY, encoding: 'utf8', timeout: 30_000, maxBuffer: 16 * 1024 * 1024}
    );
    expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
    const packs = JSON.parse(result.stdout) as {files: {path: string}[]}[];
    expect(packs.length).toBe(1);
    return packs[0].files.map(file => file.path);
  } finally {
    rmSync(cacheDirectory, {recursive: true, force: true});
  }
}

describe('analysis release package', () => {
  it.each([
    ['AGENTS.md', '## Relationship to luma.gl', '## Plans and docs'],
    ['dev-docs/roadmap.md', '# Roadmap', 'Companions:'],
    ['dev-docs/upstream-first-recording.md', '# Computation records', '## Public graph data']
  ])('keeps the current dependency statement in %s consistent with the integration target', (path, start, end) => {
    const rootManifest = JSON.parse(
      readFileSync(join(WORKSPACE_DIRECTORY, 'package.json'), 'utf8')
    );
    const text = readFileSync(join(WORKSPACE_DIRECTORY, path), 'utf8');
    const section = text.slice(text.indexOf(start), text.indexOf(end));
    const versions = Array.from(section.matchAll(/10\.0\.0-alpha\.\d+/g), match => match[0]);
    if (versions.length) {
      for (const version of versions)
        expect(version, path).toBe(rootManifest.devDependencies['@luma.gl/graph']);
    } else {
      expect(section, path).toContain('release-package.md');
    }
  });

  it('describes the unreleased package without conflating runtime and typecheck exclusions', () => {
    const changelog = readFileSync(join(WORKSPACE_DIRECTORY, 'CHANGELOG.md'), 'utf8');
    expect(changelog).toContain('Unreleased (0.1.0-alpha.0)');
    expect(changelog).not.toMatch(/`@luma-contributors\/analysis`\s*\(published\)/);
    expect(changelog).toContain('excludedFromRuns');
    expect(changelog).toContain('excludedFromTypecheck');
    expect(changelog).not.toMatch(/without the \d+ kepler specs/);
  });

  it.each([
    'README.md',
    'modules/analysis/README.md'
  ])('documents required peers and the shipped licence inventory in %s', path => {
    const readme = readFileSync(join(WORKSPACE_DIRECTORY, path), 'utf8');
    expect(readme, path).toContain('apache-arrow');
    expect(readme, path).toMatch(/required peer/i);
    expect(readme, path).toContain(packageManifest.license);
    expect(readme, path).toContain('THIRD-PARTY-NOTICES.md');
    expect(readme, path).toContain('release-package.md');
  });

  it('requires Arrow for native imports of the existing public barrels', () => {
    expect(packageManifest.peerDependencies['apache-arrow']).toBe('>=17.0.0');
    expect(packageManifest.peerDependenciesMeta?.['apache-arrow']?.optional ?? false).toBe(false);
  });

  it('ships a notice inventory covering every source SPDX expression and non-MIT file', () => {
    expect(packageManifest.license).toBe('MIT AND Apache-2.0 AND BSD-3-Clause');
    const notices = readFileSync(join(PACKAGE_DIRECTORY, 'THIRD-PARTY-NOTICES.md'), 'utf8');
    for (const file of ['LICENSE', 'LICENSES', 'THIRD-PARTY-NOTICES.md']) {
      expect(packageManifest.files).toContain(file);
    }
    const packedFiles = collectPackedFiles();
    for (const file of [
      'LICENSE',
      'LICENSES/Apache-2.0.txt',
      'LICENSES/BSD-3-Clause.txt',
      'LICENSES/H3-NOTICE.txt',
      'THIRD-PARTY-NOTICES.md'
    ]) {
      expect(packedFiles).toContain(file);
    }
    for (const file of ['LICENSE', 'LICENSES/Apache-2.0.txt', 'LICENSES/BSD-3-Clause.txt']) {
      expect(readFileSync(join(PACKAGE_DIRECTORY, file), 'utf8').length).toBeGreaterThan(1000);
    }
    const sources = packedFiles.filter(path => path.startsWith('src/'));
    expect(sources.length).toBeGreaterThan(0);
    for (const source of sources) {
      const text = readFileSync(join(PACKAGE_DIRECTORY, source), 'utf8');
      const expression = text
        .match(/SPDX-License-Identifier: ([^\r\n]+)/)?.[1]
        ?.replace(/\s*(?:\*\/|-->)\s*$/, '')
        .trim();
      if (/\.(?:[cm]?ts|tsx)$/.test(source)) expect(expression, source).toBeDefined();
      if (!expression) continue;
      expect(notices, source).toContain(`\`${expression}\``);
      if (expression !== 'MIT') {
        const relativePath = source;
        const inventoryLine = notices
          .split('\n')
          .find(line => line.includes(`\`${relativePath}\``));
        expect(inventoryLine, relativePath).toBeDefined();
        expect(inventoryLine, relativePath).toContain(`\`${expression}\``);
        for (const holder of text.matchAll(/SPDX-FileCopyrightText: ([^\r\n]+)/g)) {
          expect(notices, relativePath).toContain(holder[1]);
        }
      }
    }
  }, 30_000);
});
