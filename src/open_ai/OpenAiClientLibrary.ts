// This module takes care of all communication with OpenAI-compatible APIs
// using Obsidian's requestUrl to bypass CORS restrictions in the browser context.
// Uses the Chat Completions API (/chat/completions) which is supported by all
// providers: OpenAI, x.ai, OpenRouter, and other compatible services.

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

	// executes a single prompt and returns the completion via Chat Completions API
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

		// Ensure system message is present.
		// Priority: systemMessage argument > system entry in array > settings default
		let instructions: string = this.plugin.settings.defaultSystemMessage;
		const systemIndex = messages.findIndex((m) => m.role === "system");
		if (systemIndex > -1) {
			const sysMsg = messages[systemIndex];
			if (typeof sysMsg.content === "string") instructions = sysMsg.content;
			messages.splice(systemIndex, 1);
		}
		if (systemMessage) instructions = systemMessage;

		// Build the full messages array with the system message at the front
		const allMessages: ChatCompletionMessageParam[] = [
			{ role: "system", content: instructions },
			...messages,
		];

		// Test the character count to make sure it does not exceed maxOutgoingCharacters
		const jsonString = JSON.stringify(allMessages);
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
			const url = `${base}/chat/completions`;

			const body: Record<string, unknown> = {
				model: resolvedModel,
				messages: allMessages,
				max_tokens: maxTokens ?? this.plugin.settings.defaultMaxNumTokens,
			};

			// Web search: each provider has its own mechanism
			if (this.plugin.settings.enableWebSearch) {
				if (base.includes("api.openai.com")) {
					// OpenAI: web_search_options (requires search-capable models
					// like gpt-4o-search-preview or gpt-4o-mini-search-preview)
					body["web_search_options"] = {};
				} else if (base.includes("x.ai")) {
					// x.ai / Grok: live search via search_parameters
					body["search_parameters"] = { mode: "auto" };
				}
				// Other providers: web search not universally available,
				// skip silently so the request still succeeds.
			}

			const headers: Record<string, string> = {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${resolvedApiKey}`,
			};
			if (organization) headers["OpenAI-Organization"] = organization;

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
					messages: allMessages,
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
					const inTok = usage["prompt_tokens"] ?? 0;
					const outTok = usage["completion_tokens"] ?? 0;
					const totalTok = usage["total_tokens"] ?? inTok + outTok;
					const displayMessage =
						`${this.plugin.APP_ABBREVIARTION}:\n` +
						`Tokens: ${inTok.toString()}/${outTok.toString()}/${totalTok.toString()}\n` +
						`Character count: ${characterCount.toString()}`;
					new Notice(displayMessage, 8000);
				}
			}

			// ── Extract output text from choices[0].message.content ─
			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
			const choices = data?.choices as Array<Record<string, unknown>> | undefined;
			let outputText = "";
			if (Array.isArray(choices) && choices.length > 0) {
				const message = choices[0]["message"] as Record<string, unknown> | undefined;
				if (message && typeof message["content"] === "string") {
					outputText = message["content"] as string;
				}
			}

			if (!outputText) {
				console.warn(
					`${this.plugin.APP_ABBREVIARTION}: chat() returned empty output. ` +
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
