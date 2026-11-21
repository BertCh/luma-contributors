// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Minimal typings for markdown-it 14 (the package ships none and `@types/markdown-it` is not installed).
declare module 'markdown-it' {
  type Token = {
    type: string;
    tag: string;
    content: string;
    info: string;
    attrs: [string, string][] | null;
    children: Token[] | null;
    block: boolean;
    map: [number, number] | null;
    meta: unknown;
    attrGet: (name: string) => string | null;
    attrSet: (name: string, value: string) => void;
  };
  type StateBlock = {
    src: string;
    bMarks: number[];
    eMarks: number[];
    tShift: number[];
    sCount: number[];
    blkIndent: number;
    line: number;
    push(type: string, tag: string, nesting: number): Token;
  };
  type BlockRule = (
    state: StateBlock,
    startLine: number,
    endLine: number,
    silent: boolean
  ) => boolean;
  type RulerOptions = {alt?: string[]};
  type BlockRuler = {
    before(beforeName: string, ruleName: string, rule: BlockRule, options?: RulerOptions): void;
    after(afterName: string, ruleName: string, rule: BlockRule, options?: RulerOptions): void;
    at(ruleName: string, rule: BlockRule, options?: RulerOptions): void;
  };
  type Renderer = {
    rules: Record<
      string,
      | ((tokens: Token[], index: number, options: unknown, env: unknown, self: Renderer) => string)
      | undefined
    >;
    renderToken: (tokens: Token[], index: number, options: unknown) => string;
  };
  type Options = {html?: boolean; linkify?: boolean; typographer?: boolean; breaks?: boolean};
  export default class MarkdownIt {
    constructor(options?: Options);
    block: {ruler: BlockRuler};
    renderer: Renderer;
    use<PluginArguments extends unknown[]>(
      plugin: (markdown: MarkdownIt, ...pluginArguments: PluginArguments) => void,
      ...pluginArguments: PluginArguments
    ): this;
    render(source: string, env?: unknown): string;
    renderInline(source: string, env?: unknown): string;
    utils: {escapeHtml: (text: string) => string};
  }
}
