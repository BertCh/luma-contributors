// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Publishes the static site to GitHub Pages: builds it for the project sub-path and force-pushes
// the build as one commit to the `gh-pages` branch, which Pages serves (Settings > Pages >
// "Deploy from a branch", gh-pages, root). The branch holds only the latest build, so the
// repository does not grow with each deploy. The main working tree and index are not touched.
//
// The site is built here, not in CI, because it reads `modules/*/dist` (run `yarn build` first
// when package sources changed) and the checks that open every page need a real GPU
// (`node scripts/check-all-pages.mjs --base /<repository>`).
//
// Usage (from the website folder):
//   node scripts/deploy-pages.mjs                 build, then publish
//   node scripts/deploy-pages.mjs --skip-build    publish the existing dist/client (it must have
//                                                 been built with the same SITE_BASE_PATH)
// Options: --remote origin, --branch gh-pages, --base /<repository>/ (default: from the remote URL),
//   --dry-run (build and stage the commit, do not push)

import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildDirectory = path.join(websiteDirectory, 'dist/client');

function parseArguments(argv) {
  const options = {remote: 'origin', branch: 'gh-pages', base: null, skipBuild: false, dryRun: false};
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--remote') options.remote = argv[++index];
    else if (argument === '--branch') options.branch = argv[++index];
    else if (argument === '--base') options.base = argv[++index];
    else if (argument === '--skip-build') options.skipBuild = true;
    else if (argument === '--dry-run') options.dryRun = true;
    else throw new Error(`deploy-pages: unknown argument ${argument}`);
  }
  return options;
}

function runGit(directory, gitArguments) {
  return execFileSync('git', gitArguments, {cwd: directory, encoding: 'utf8'}).trim();
}

/** `/<repository>/` for a GitHub remote URL, the path GitHub Pages serves a project site from. */
function getBasePathFromRemote(remoteUrl) {
  const match = /[/:]([^/]+?)(?:\.git)?$/.exec(remoteUrl);
  if (!match) {
    throw new Error(`deploy-pages: cannot read a repository name from ${remoteUrl}; pass --base`);
  }
  return `/${match[1]}/`;
}

function normalizeBasePath(basePath) {
  return `/${basePath.replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');
}

/** Fails when the build in `dist/client` was not made for `basePath`. */
function validateBuildBasePath(basePath) {
  const indexFile = path.join(buildDirectory, 'index.html');
  if (!fs.existsSync(indexFile)) {
    throw new Error(`deploy-pages: ${indexFile} does not exist; build the site first`);
  }
  const html = fs.readFileSync(indexFile, 'utf8');
  if (!html.includes(`="${basePath}assets/`)) {
    throw new Error(
      `deploy-pages: dist/client was not built for ${basePath}; rebuild with SITE_BASE_PATH=${basePath}`
    );
  }
}

const options = parseArguments(process.argv.slice(2));
const remoteUrl = runGit(websiteDirectory, ['remote', 'get-url', options.remote]);
const basePath = normalizeBasePath(options.base ?? getBasePathFromRemote(remoteUrl));
const sourceCommit = runGit(websiteDirectory, ['rev-parse', '--short', 'HEAD']);
const isSourceDirty = runGit(websiteDirectory, ['status', '--porcelain', '--', '.', '../modules', '../docs']) !== '';

if (!options.skipBuild) {
  execFileSync('yarn', ['build'], {
    cwd: websiteDirectory,
    stdio: 'inherit',
    env: {...process.env, SITE_BASE_PATH: basePath}
  });
}
validateBuildBasePath(basePath);

const stagingDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'gridlight-pages-'));
try {
  fs.cpSync(buildDirectory, stagingDirectory, {recursive: true});
  // Pages must serve the files as they are: no Jekyll pass over the build.
  fs.writeFileSync(path.join(stagingDirectory, '.nojekyll'), '');
  const fileCount = fs.readdirSync(stagingDirectory, {recursive: true}).length;

  runGit(stagingDirectory, ['init', '--quiet', '--initial-branch', options.branch]);
  runGit(stagingDirectory, ['add', '--all']);
  const message = `Deploy website from ${sourceCommit}${isSourceDirty ? ' (with uncommitted changes)' : ''}`;
  const authorName = runGit(websiteDirectory, ['config', 'user.name']);
  const authorEmail = runGit(websiteDirectory, ['config', 'user.email']);
  runGit(stagingDirectory, [
    '-c',
    `user.name=${authorName}`,
    '-c',
    `user.email=${authorEmail}`,
    'commit',
    '--quiet',
    '--message',
    message
  ]);
  if (options.dryRun) {
    process.stdout.write(`deploy-pages: dry run, ${fileCount} entries staged for ${basePath}, nothing pushed\n`);
  } else {
    runGit(stagingDirectory, ['push', '--force', '--quiet', remoteUrl, `HEAD:refs/heads/${options.branch}`]);
    process.stdout.write(
      `deploy-pages: pushed ${fileCount} entries for ${basePath} to ${options.remote}/${options.branch} (${message})\n`
    );
  }
} finally {
  fs.rmSync(stagingDirectory, {recursive: true, force: true});
}
