// This module takes care of all communication with the OpenAI-compatible API
// using Obsidian's requestUrl to bypass CORS restrictions in the browser context.
// Supports both the OpenAI Responses API and the Chat Completions API used by
// x.ai, OpenRouter, and other OpenAI-compatible providers.

import { Notice, Platform, requestUrl } from "obsidian";
import type { ChatCompletionMessageParam } from "openai/resources";
import type AitPlugin from "../main";

const OPENAI_BASE = "https://api.openai.com/v1";

export default class OpenAiApi {
	private plugin: AitPlugin;

	constructor(plugin: AitPlugin) {
		this.plugin = plugin;
	}

	// returns the base URL to use, stripping any trailing slash
	private baseUrl(overrideBaseURL?: string | null): string {
		const raw =
			overrideBaseURL ??
			(this.plugin.settings.defaultEndpoint !== ""
				? this.plugin.settings.defaultEndpoint
				: OPENAI_BASE);
		return raw.replace(/\/+$/, "");
	}

	// The Responses API is only available on the OpenAI platform.
	// All other providers (x.ai, OpenRouter, etc.) use Chat Completions.
	private useResponsesApi(base: string): boolean {
		return base.includes("api.openai.com");
	}

	// validates the current settings (API Key and Model)
	validateSettings = (): boolean => {
		if (!this.plugin.settings.defaultApiKey) {
			this.plugin.log("validateSettings", "defaultApiKey is not set");
			return false;
		}
		if (!this.plugin.settings.defaultModel) {
			this.plugin.log("validateSettings", "defaultModel is not set");
			return false;
		}
		return true;
	};

	// executes a single prompt and returns the completion
	// Automatically selects Responses API (OpenAI) or Chat Completions API
	// (x.ai, OpenRouter, and other OpenAI-compatible providers).
	chat = async (
		promptOrMessages: string | ChatCompletionMessageParam[],
		model?: string | null,
		systemMessage?: string | null,
		maxTokens?: number,
		maxOutgoingCharacters?: number,
		baseURL?: string | null,
		apiKey?: string | null,
		organization?: string | null,
	): Promise<string> => {
		if (!this.validateSettings()) {
			new Notice(
				"Check your setting for valid API key, model and endpoint.",
				10000,
			);
			return "";
		}

		// Normalise to a messages array
		const messages: ChatCompletionMessageParam[] =
			typeof promptOrMessages === "string"
				? [{ role: "user", content: promptOrMessages }]
				: [...promptOrMessages];

		// Extract system message
		// Priority: systemMessage argument > system entry in array > settings default
		let instructions: string = this.plugin.settings.defaultSystemMessage;
		const systemIndex = messages.findIndex((m) => m.role === "system");
		if (systemIndex > -1) {
			const sysMsg = messages[systemIndex];
			if (typeof sysMsg.content === "string") instructions = sysMsg.content;
			messages.splice(systemIndex, 1);
		}
		if (systemMessage) instructions = systemMessage;

		// Only user/assistant messages (no system) remain
		const input = messages;

		// Test the character count to make sure it does not exceed maxOutgoingCharacters
		const jsonString = JSON.stringify(input);
		const characterCount = jsonString.length;

		const maxCharacters = maxOutgoingCharacters
			? maxOutgoingCharacters
			: this.plugin.settings.defaultMaxOutgoingCharacters;
		if (characterCount > maxCharacters) {
			new Notice(
				`${this.plugin.APP_ABBREVIARTION}: Character count exceeds the limit of ${maxCharacters.toString()} for outgoing AI requests and the current character count is ${characterCount.toString()}. This setting can be changed in Settings or as part of the command.`,
				15000,
			);
			this.plugin.log(
				`chat', 'Character count exceeds limt. Defined limit is ${maxCharacters.toString()} and the current character count is ${characterCount.toString()}.`,
			);
			return "";
		}

		try {
			const resolvedApiKey = apiKey ?? this.plugin.settings.defaultApiKey;
			const resolvedModel = model ?? this.plugin.settings.defaultModel;
			const base = this.baseUrl(baseURL);
			const resolvedMaxTokens =
				maxTokens ?? this.plugin.settings.defaultMaxNumTokens;

			const headers: Record<string, string> = {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${resolvedApiKey}`,
			};
			if (organization) headers["OpenAI-Organization"] = organization;

			let url: string;
			let body: Record<string, unknown>;

			if (this.useResponsesApi(base)) {
				// ── OpenAI Responses API ──────────────────────────────
				url = `${base}/responses`;
				body = {
					model: resolvedModel,
					instructions: instructions,
					input: input,
					max_output_tokens: resolvedMaxTokens,
				};
				if (this.plugin.settings.enableWebSearch) {
					body["tools"] = [{ type: "web_search" }];
				}
			} else {
				// ── Chat Completions API (x.ai, OpenRouter, etc.) ────
				url = `${base}/chat/completions`;
				const allMessages: ChatCompletionMessageParam[] = [
					{ role: "system", content: instructions },
					...input,
				];
				body = {
					model: resolvedModel,
					messages: allMessages,
					max_tokens: resolvedMaxTokens,
				};
				if (this.plugin.settings.enableWebSearch) {
					if (base.includes("x.ai")) {
						// x.ai / Grok: enable live search via search_parameters
						body["search_parameters"] = { mode: "auto" };
					}
					// Other providers: web search not universally available,
					// skip silently so the request still succeeds.
				}
			}

			const res = await requestUrl({
				url,
				method: "POST",
				headers,
				body: JSON.stringify(body),
				throw: false,
			});

			if (res.status >= 400) {
				const errText = res.text ?? String(res.status);
				throw new Error(`HTTP ${res.status.toString()}: ${errText}`);
			}

			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			const data = res.json;

			if (this.plugin.settings.debugToConsole) {
				this.plugin.log("chat", {
					url,
					instructions,
					input,
					response: data,
					outgoingCharacterCountMax: maxCharacters,
					outgoingCharacerCountActual: characterCount,
				});
			}

			// ── Token usage display ──────────────────────────────────
			if (
				((Platform.isDesktop &&
					this.plugin.settings.displayTokenUsageDesktop) ??
					false) ||
				((Platform.isMobile && this.plugin.settings.displayTokenUsageMobile) ??
					false)
			) {
				// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
				const usage = data?.usage as Record<string, number> | undefined;
				if (usage) {
					// Responses API uses input_tokens/output_tokens,
					// Chat Completions uses prompt_tokens/completion_tokens.
					const inTok = usage["input_tokens"] ?? usage["prompt_tokens"] ?? 0;
					const outTok = usage["output_tokens"] ?? usage["completion_tokens"] ?? 0;
					const totalTok = usage["total_tokens"] ?? inTok + outTok;
					const displayMessage =
						`${this.plugin.APP_ABBREVIARTION}:\n` +
						`Tokens: ${inTok.toString()}/${outTok.toString()}/${totalTok.toString()}\n` +
						`Character count: ${characterCount.toString()}`;
					new Notice(displayMessage, 8000);
				}
			}

			// ── Extract output text ──────────────────────────────────
			let outputText = "";

			if (this.useResponsesApi(base)) {
				// Responses API: output_text is an SDK-only convenience
				// property absent from raw JSON. Walk the output array.
				// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
				if (typeof data?.output_text === "string") {
					outputText = data.output_text as string;
				} else if (Array.isArray(data?.output)) {
					const parts: string[] = [];
					// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
					for (const item of data.output as Array<Record<string, unknown>>) {
						if (item["type"] === "message" && Array.isArray(item["content"])) {
							for (const block of item["content"] as Array<Record<string, unknown>>) {
								if (block["type"] === "output_text" && typeof block["text"] === "string") {
									parts.push(block["text"] as string);
								}
							}
						}
					}
					outputText = parts.join("");
				}
			} else {
				// Chat Completions API: choices[0].message.content
				// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
				const choices = data?.choices as Array<Record<string, unknown>> | undefined;
				if (Array.isArray(choices) && choices.length > 0) {
					const message = choices[0]["message"] as Record<string, unknown> | undefined;
					if (message && typeof message["content"] === "string") {
						outputText = message["content"] as string;
					}
				}
			}

			if (!outputText) {
				console.warn(
					`${this.plugin.APP_ABBREVIARTION}: chat() returned empty output. ` +
					`API: ${this.useResponsesApi(base) ? "Responses" : "ChatCompletions"}. ` +
					`Status: ${String(res.status)}. ` +
					// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
					`Response keys: ${data ? Object.keys(data).join(", ") : "null"}. ` +
					`Raw response:`,
					data,
				);
			}
			return outputText;
		} catch (error) {
			console.error(
				`${this.plugin.APP_ABBREVIARTION}: chat() caught error:`,
				error,
			);
			new Notice(
				`${this.plugin.APP_ABBREVIARTION} Error: ${String(error)}`,
				20000,
			);
			return "";
		}
	};

	// for the current endpoint, returns a list of available models
	availableModels = async (): Promise<string[]> => {
		if (!this.validateSettings()) {
			new Notice(
				`${this.plugin.APP_ABBREVIARTION} Check your setting for valid API key, model and endpoint.`,
				10000,
			);
			return [];
		}

		try {
			const url = `${this.baseUrl()}/models`;
			const res = await requestUrl({
				url,
				method: "GET",
				headers: {
					"Authorization": `Bearer ${this.plugin.settings.defaultApiKey}`,
				},
				throw: false,
			});

			if (res.status >= 400) {
				throw new Error(`HTTP ${res.status.toString()}: ${res.text}`);
			}

			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			const data = res.json;

			if (this.plugin.settings.debugToConsole) {
				this.plugin.log("availableModels", data);
			}

			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return
			return (data?.data ?? []).map((m: { id: string }) => m.id);
		} catch (error) {
			new Notice(
				`${this.plugin.APP_ABBREVIARTION} Error: ${String(error)}`,
				20000,
			);
			return [];
		}
	};
}
