// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useState} from 'react';

export type CodeBlockProps = {
  code: string;
  language?: string;
  /** Pre-highlighted HTML (for example a Shiki <pre> from the build). Rendered as is. */
  html?: string;
};

export function CodeBlock({code, language = 'ts', html}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function copyCode(): Promise<void> {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard unavailable (insecure context or denied): ignore.
    }
  }

  return (
    <div className="code-block">
      {html ? (
        <div dangerouslySetInnerHTML={{__html: html}} />
      ) : (
        <pre>
          <code className={`language-${language}`}>{code}</code>
        </pre>
      )}
      <button type="button" className="code-block__copy" onClick={copyCode} aria-label="Copy code">
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  );
}
