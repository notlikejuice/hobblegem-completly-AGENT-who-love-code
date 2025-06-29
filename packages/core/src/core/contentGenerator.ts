/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  GoogleGenAI,
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { getEffectiveModel } from './modelCheck.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE_PERSONAL = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  USE_OPENAI = 'openai-api-key',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
};

export async function createContentGeneratorConfig(
  model: string | undefined,
  authType: AuthType | undefined,
  config?: { getModel?: () => string },
): Promise<ContentGeneratorConfig> {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const openAiKey = process.env.OPENAI_API_KEY;
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION;

  // Use runtime model from config if available, otherwise fallback to parameter or default
  const effectiveModel = config?.getModel?.() || model || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
  };

  // if we are using google auth nothing else to validate for now
  if (authType === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    return contentGeneratorConfig;
  }

  //
  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    !!googleApiKey &&
    googleCloudProject &&
    googleCloudLocation
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );

    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_OPENAI && openAiKey) {
    contentGeneratorConfig.apiKey = openAiKey;
    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };
  if (config.authType === AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
    return createCodeAssistContentGenerator(httpOptions, config.authType);
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  if (config.authType === AuthType.USE_OPENAI) {
    const { OpenAI } = await import('openai');
    const client = new OpenAI({ apiKey: config.apiKey });
    return {
      generateContent: async ({ model, contents, config: genCfg }) => {
        const messages = (contents as Content[]).map((c) => ({
          role: c.role === 'model' ? 'assistant' : c.role,
          content: c.parts?.map((p) => p.text).join('') ?? '',
        })) as any;
        const res = await client.chat.completions.create({
          model,
          messages,
          temperature: genCfg?.temperature,
          top_p: genCfg?.topP,
        });
        return {
          candidates: [
            { content: { role: 'model', parts: [{ text: res.choices[0]?.message?.content || '' }] } },
          ],
          usageMetadata: {
            inputTokenCount: res.usage?.prompt_tokens,
            outputTokenCount: res.usage?.completion_tokens,
            totalTokenCount: res.usage?.total_tokens,
          },
        } as unknown as GenerateContentResponse;
      },
      generateContentStream: async ({ model, contents, config: genCfg }) => {
        const messages = (contents as Content[]).map((c) => ({
          role: c.role === 'model' ? 'assistant' : c.role,
          content: c.parts?.map((p) => p.text).join('') ?? '',
        })) as any;
        const stream = await client.chat.completions.create({
          model,
          messages,
          temperature: genCfg?.temperature,
          top_p: genCfg?.topP,
          stream: true,
        });
        async function* gen() {
          let text = '';
          for await (const part of stream) {
            const delta = part.choices[0]?.delta?.content;
            if (delta) {
              text += delta;
              yield {
                candidates: [
                  { content: { role: 'model', parts: [{ text }] } },
                ],
              } as unknown as GenerateContentResponse;
            }
          }
        }
        return gen();
      },
      countTokens: async ({ model, contents }) => {
        const messages = (contents as Content[]).map((c) => ({
          role: c.role === 'model' ? 'assistant' : c.role,
          content: c.parts?.map((p) => p.text).join('') ?? '',
        })) as any;
        const res = await client.chat.completions.create({
          model,
          messages,
          max_tokens: 1,
        });
        return { totalTokens: res.usage?.total_tokens } as CountTokensResponse;
      },
      embedContent: async ({ model, contents }) => {
        const input = contents as any;
        const res = await client.embeddings.create({ model, input });
        return {
          embeddings: res.data.map((d) => ({ values: d.embedding })) as any,
        } as EmbedContentResponse;
      },
    } as ContentGenerator;
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}
