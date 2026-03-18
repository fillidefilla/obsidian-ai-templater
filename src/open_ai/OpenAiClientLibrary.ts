// This module takes care of all communication with the OpenAI Client Library
// https://platform.openai.com/docs/api-reference

import { Notice, Platform } from "obsidian";
import { OpenAI } from "openai";
import type { ClientOptions } from "openai";
import type { ChatCompletionMessageParam } from "openai/resources";
import type AitPlugin from "../main";

export default class OpenAiApi {
	private plugin: AitPlugin;

	constructor(plugin: AitPlugin) {
		this.plugin = plugin;
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
		const openai = new OpenAI({
			apiKey: apiKey ?? this.plugin.settings.defaultApiKey,
			dangerouslyAllowBrowser: true,
		} as ClientOptions);

		if (baseURL) openai.baseURL = baseURL;
		if (organization) openai.organization = organization;

		if (!this.validateSettings()) {
			new Notice(
				"Check your setting for valid API key, model and endpoint.",
				10000,
			);
			return "";
		}

		if (this.plugin.settings.defaultEndpoint !== "")
			openai.baseURL = this.plugin.settings.defaultEndpoint;

		// Normalise to a messages array
		const messages: ChatCompletionMessageParam[] =
			typeof promptOrMessages === "string"
				? [{ role: "user", content: promptOrMessages }]
				: promptOrMessages;

		// Extract system message from the array (Responses API requires it in
		// the separate `instructions` field, not inside `input`).
		// Priority: systemMessage argument > system entry in array > settings default
		let instructions: string = this.plugin.settings.defaultSystemMessage;
		const systemIndex = messages.findIndex((m) => m.role === "system");
		if (systemIndex > -1) {
			const sysMsg = messages[systemIndex];
			if (typeof sysMsg.content === "string") instructions = sysMsg.content;
			messages.splice(systemIndex, 1);
		}
		if (systemMessage) instructions = systemMessage;

		// Only user/assistant messages go into `input`
		const input = messages as unknown as Record<string, unknown>[];

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
			const requestParams: Record<string, unknown> = {
				model: model ?? this.plugin.settings.defaultModel,
				instructions: instructions,
				input: input,
				max_output_tokens: maxTokens ?? this.plugin.settings.defaultMaxNumTokens,
			};

			if (this.plugin.settings.enableWebSearch) {
				// web_search tool works with any capable model on both OpenAI and x.ai
				requestParams["tools"] = [{ type: "web_search" }];
			}

			const response = await (openai.responses.create as (params: Record<string, unknown>) => Promise<Record<string, unknown>>)(requestParams);

			if (this.plugin.settings.debugToConsole) {
				const logMessage = {
					instructions: instructions,
					input: input,
					response: response,
					clientOptions: openai,
					outgoingCharacterCountMax: maxCharacters,
					outgoingCharacerCountActual: characterCount,
				};
				this.plugin.log("chat", logMessage);
			}

			if (
				((Platform.isDesktop &&
					this.plugin.settings.displayTokenUsageDesktop) ??
					false) ||
				((Platform.isMobile && this.plugin.settings.displayTokenUsageMobile) ??
					false)
			) {
				const usage = response["usage"] as Record<string, number> | undefined;
				if (usage) {
					const displayMessage =
						`${this.plugin.APP_ABBREVIARTION}:\n` +
						`Tokens: ${usage["input_tokens"].toString()}/${usage["output_tokens"].toString()}/${usage["total_tokens"].toString()}\n` +
						`Character count: ${characterCount.toString()}`;
					new Notice(displayMessage, 8000);
				}
			}

			const outputText = response["output_text"];
			return typeof outputText === "string" ? outputText : "";
		} catch (error) {
			new Notice(
				`${this.plugin.APP_ABBREVIARTION} Error: ${String(error)}`,
				20000,
			);
			return "";
		}
	};

	// for the current endpoint, returns a list of available models
	availableModels = async (): Promise<string[]> => {
		const openai = new OpenAI({
			apiKey: this.plugin.settings.defaultApiKey,
			dangerouslyAllowBrowser: true,
		} as ClientOptions);

		if (!this.validateSettings()) {
			new Notice(
				`${this.plugin.APP_ABBREVIARTION} Check your setting for valid API key, model and endpoint.`,
				10000,
			);
			return [];
		}

		const models = await openai.models.list();

		if (this.plugin.settings.debugToConsole) {
			this.plugin.log("availableModels", models);
		}

		return models.data.map((model) => model.id);
	};
}
