/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  Content,
  Part,
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Candidate,
  FunctionCall,
  FunctionDeclaration,
  Tool,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

function mapMessages(
  contents: Content[],
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  return contents.map(
    (c) =>
      ({
        role: (c.role as 'user' | 'assistant' | 'system') || 'user',
        content:
          c.parts
            ?.map((p) => (typeof p === 'string' ? p : p.text || ''))
            .join('') || '',
      }) as OpenAI.Chat.Completions.ChatCompletionMessageParam,
  );
}

function mapTools(
  decls: FunctionDeclaration[],
): OpenAI.Chat.Completions.ChatCompletionTool[] {
  return decls.map(
    (d) =>
      ({
        type: 'function',
        function: {
          name: d.name,
          description: d.description,
          parameters: d.parameters,
        },
      }) as OpenAI.Chat.Completions.ChatCompletionTool,
  );
}

function mapResponse(
  message: OpenAI.Chat.Completions.ChatCompletionMessage,
): GenerateContentResponse {
  const parts: Part[] = [];
  if (message.content) {
    parts.push({ text: message.content });
  }
  if (message.tool_calls) {
    for (const tc of message.tool_calls) {
      const args = tc.function.arguments
        ? JSON.parse(tc.function.arguments)
        : {};
      const fnCall: FunctionCall = { id: tc.id, name: tc.function.name, args };
      parts.push({ functionCall: fnCall });
    }
  }
  const cand: Candidate = { content: { role: 'model', parts } };
  return { candidates: [cand] } as GenerateContentResponse;
}

export class OpenAIContentGenerator implements ContentGenerator {
  private client: OpenAI;
  private model: string;
  constructor(apiKey: string, model: string) {
    this.client = new OpenAI({ apiKey });
    this.model = model;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = mapMessages(request.contents as Content[]);
    const tools = (request.config?.tools?.[0] as Tool | undefined)
      ?.functionDeclarations;
    const res = await this.client.chat.completions.create({
      model: this.model,
      messages,
      tools: tools ? mapTools(tools) : undefined,
      tool_choice: tools ? 'auto' : undefined,
    });
    const message = res.choices[0].message;
    return mapResponse(message);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = mapMessages(request.contents as Content[]);
    const tools = (request.config?.tools?.[0] as Tool | undefined)
      ?.functionDeclarations;
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages,
      tools: tools ? mapTools(tools) : undefined,
      tool_choice: tools ? 'auto' : undefined,
      stream: true,
    });
    const asyncIter = (async function* () {
      for await (const chunk of stream) {
        const delta = chunk.choices[0].delta;
        if (!delta) continue;
        const message = {
          role: 'assistant',
          content: delta.content || '',
          tool_calls:
            delta.tool_calls as unknown as OpenAI.Chat.Completions.ChatCompletionMessageToolCall[],
        } as OpenAI.Chat.Completions.ChatCompletionMessage;
        yield mapResponse(message);
      }
    })();
    return asyncIter;
  }

  async countTokens(_req: CountTokensParameters): Promise<CountTokensResponse> {
    return { totalTokens: 0 } as CountTokensResponse;
  }

  async embedContent(
    _req: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    return { embeddings: [] } as EmbedContentResponse;
  }
}
