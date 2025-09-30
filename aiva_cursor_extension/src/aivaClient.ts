import axios, { AxiosInstance, AxiosResponse } from 'axios';
import * as vscode from 'vscode';

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

export interface CompletionOptions {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
    stop?: string[];
}

export class AIVAClient {
    private client: AxiosInstance;
    private serverUrl: string;
    private modelName: string;

    constructor(serverUrl: string, modelName: string) {
        this.serverUrl = serverUrl;
        this.modelName = modelName;

        this.client = axios.create({
            baseURL: serverUrl,
            timeout: 30000, // 30 second timeout
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.code === 'ECONNREFUSED') {
                    throw new Error('Cannot connect to AIVA server. Make sure it\'s running.');
                }
                if (error.response?.status === 503) {
                    throw new Error('AIVA server is initializing. Please wait.');
                }
                throw error;
            }
        );
    }

    updateConfig(serverUrl: string, modelName: string): void {
        this.serverUrl = serverUrl;
        this.modelName = modelName;
        this.client.defaults.baseURL = serverUrl;
    }

    async healthCheck(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.data.status === 'healthy';
        } catch (error) {
            throw new Error('Health check failed');
        }
    }

    async chatCompletion(
        messages: ChatMessage[],
        options: CompletionOptions = {}
    ): Promise<string> {
        const config = vscode.workspace.getConfiguration('aiva');

        const requestData = {
            model: this.modelName,
            messages: messages,
            max_tokens: options.maxTokens || config.get('chat.maxTokens', 1024),
            temperature: options.temperature || config.get('temperature', 0.7),
            top_p: options.topP || 0.9,
            frequency_penalty: options.frequencyPenalty || 0.0,
            presence_penalty: options.presencePenalty || 0.0,
            stop: options.stop,
            stream: false,
        };

        try {
            const response: AxiosResponse = await this.client.post('/v1/chat/completions', requestData);
            return response.data.choices[0].message.content;
        } catch (error: any) {
            throw new Error(`Chat completion failed: ${error.message}`);
        }
    }

    async completeCode(
        context: string,
        language: string,
        options: CompletionOptions = {}
    ): Promise<string> {
        const config = vscode.workspace.getConfiguration('aiva');

        const prompt = `You are an expert ${language} programmer. Complete the following ${language} code. Only provide the completion without explanation:

${context}`;

        const requestData = {
            model: this.modelName,
            prompt: prompt,
            max_tokens: options.maxTokens || config.get('completion.maxTokens', 512),
            temperature: options.temperature || 0.3, // Lower temperature for code completion
            top_p: options.topP || 0.9,
            frequency_penalty: options.frequencyPenalty || 0.0,
            presence_penalty: options.presencePenalty || 0.0,
            stop: options.stop || ['\n\n', '```'],
            echo: false,
        };

        try {
            const response: AxiosResponse = await this.client.post('/v1/completions', requestData);
            return response.data.choices[0].text.trim();
        } catch (error: any) {
            throw new Error(`Code completion failed: ${error.message}`);
        }
    }

    async explainCode(code: string, language: string): Promise<string> {
        const messages: ChatMessage[] = [
            {
                role: 'system',
                content: 'You are an expert programmer. Explain the provided code clearly and concisely.'
            },
            {
                role: 'user',
                content: `Explain this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\``
            }
        ];

        return await this.chatCompletion(messages);
    }

    async refactorCode(code: string, language: string): Promise<string> {
        const messages: ChatMessage[] = [
            {
                role: 'system',
                content: 'You are an expert programmer. Refactor the provided code to be more readable, efficient, and maintainable. Keep the same functionality.'
            },
            {
                role: 'user',
                content: `Refactor this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nProvide only the refactored code without explanation.`
            }
        ];

        return await this.chatCompletion(messages);
    }

    async optimizeCode(code: string, language: string): Promise<string> {
        const messages: ChatMessage[] = [
            {
                role: 'system',
                content: 'You are an expert programmer. Optimize the provided code for performance while maintaining correctness and readability.'
            },
            {
                role: 'user',
                content: `Optimize this ${language} code for performance:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nProvide only the optimized code without explanation.`
            }
        ];

        return await this.chatCompletion(messages);
    }

    async debugCode(code: string, language: string): Promise<string> {
        const messages: ChatMessage[] = [
            {
                role: 'system',
                content: 'You are an expert debugger. Analyze the provided code for potential bugs, issues, and improvements.'
            },
            {
                role: 'user',
                content: `Debug and analyze this ${language} code for issues:\n\n\`\`\`${language}\n${code}\n\`\`\`\n\nIdentify any bugs, potential issues, and suggest fixes.`
            }
        ];

        return await this.chatCompletion(messages);
    }

    async analyzeProject(projectPath: string): Promise<string> {
        // This would typically use the tool system, but for now we'll use chat
        const messages: ChatMessage[] = [
            {
                role: 'system',
                content: 'You are an expert software architect. Analyze the project structure and provide insights.'
            },
            {
                role: 'user',
                content: `Analyze this project structure. What type of project is this? What technologies are used? Any recommendations for improvement?\n\nProject path: ${projectPath}`
            }
        ];

        return await this.chatCompletion(messages);
    }

    async getStats(): Promise<any> {
        try {
            const response: AxiosResponse = await this.client.get('/stats');
            return response.data;
        } catch (error: any) {
            throw new Error(`Failed to get stats: ${error.message}`);
        }
    }
}
