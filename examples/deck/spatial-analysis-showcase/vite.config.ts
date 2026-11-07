import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import type {Plugin} from 'vite';
import {createExampleViteConfig} from '../../vite.config.shared';

/** Identifies the emitted code, including the compiled graph package actually bundled. */
function recordBuildEvidence(): Plugin {
  return {
    name: 'showcase-build-evidence',
    writeBundle(options, bundle) {
      const files = Object.keys(bundle)
        .sort()
        .map(fileName => {
          const content = readFileSync(join(options.dir!, fileName));
          return {
            fileName,
            bytes: Buffer.byteLength(content),
            sha256: createHash('sha256').update(content).digest('hex')
          };
        });
      writeFileSync(
        join(options.dir!, 'build-evidence.json'),
        `${JSON.stringify(
          {
            schemaVersion: 1,
            builtAt: new Date().toISOString(),
            bundleSha256: createHash('sha256').update(JSON.stringify(files)).digest('hex'),
            scope: 'Emitted chunks and assets; public dataset files use their dataset manifests.',
            files
          },
          null,
          2
        )}\n`
      );
    }
  };
}

const websiteBaseUrl = process.env['WEBSITE_BASE_URL'] || '/';

export default createExampleViteConfig({
  config: {
    base: websiteBaseUrl,
    plugins: [recordBuildEvidence()],
    server: {port: 5312, strictPort: true, open: false}
  }
});
