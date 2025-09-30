import React, { useRef, useState, useEffect } from 'react';
import Editor, { Monaco } from '@monaco-editor/react';
import { Button } from '@/components/ui/button';
import { gptApi } from '@/lib/api';
import { Play, Save, Lightbulb, Loader2 } from 'lucide-react';

interface EditorProps {
  file?: {
    name: string;
    path: string;
    content: string;
    language: string;
  };
}

const CodeEditor: React.FC<EditorProps> = ({ file }) => {
  const [code, setCode] = useState(file?.content || '');
  const [language, setLanguage] = useState(file?.language || 'typescript');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<Monaco | null>(null);

  useEffect(() => {
    if (file) {
      setCode(file.content);
      setLanguage(file.language);
    }
  }, [file]);

  const handleEditorDidMount = (editor: any, monaco: Monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Configure Monaco Editor
    monaco.editor.defineTheme('aiva-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '6A9955' },
        { token: 'keyword', foreground: '569CD6' },
        { token: 'string', foreground: 'CE9178' },
        { token: 'number', foreground: 'B5CEA8' },
      ],
      colors: {
        'editor.background': '#1e1e1e',
        'editor.foreground': '#d4d4d4',
        'editor.lineHighlightBackground': '#2d2d30',
        'editor.selectionBackground': '#264f78',
      }
    });

    monaco.editor.setTheme('aiva-dark');

    // Add IntelliSense-like completion
    monaco.languages.registerCompletionItemProvider(language, {
      provideCompletionItems: async (model: any, position: any) => {
        const word = model.getWordUntilPosition(position);
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        };

        try {
          // Get code completion from GPT
          const response = await gptApi.completeCode(
            model.getValueInRange({
              startLineNumber: Math.max(1, position.lineNumber - 5),
              startColumn: 1,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            }),
            language,
            model.getValue()
          );

          if (response.completion) {
            return {
              suggestions: [
                {
                  label: response.completion,
                  kind: monaco.languages.CompletionItemKind.Function,
                  insertText: response.completion,
                  range: range,
                  documentation: 'AI-generated completion',
                },
              ],
            };
          }
        } catch (error) {
          console.error('Completion error:', error);
        }

        return { suggestions: [] };
      },
    });

    // Add keyboard shortcut for AI suggestions (Ctrl+Space)
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Space, async () => {
      const position = editor.getPosition();
      const model = editor.getModel();
      if (!position || !model) return;

      setIsLoading(true);
      try {
        const context = model.getValueInRange({
          startLineNumber: Math.max(1, position.lineNumber - 10),
          startColumn: 1,
          endLineNumber: position.lineNumber,
          endColumn: position.column,
        });

        const response = await gptApi.completeCode(context, language);
        if (response.completion) {
          const range = {
            startLineNumber: position.lineNumber,
            startColumn: position.column,
            endLineNumber: position.lineNumber,
            endColumn: position.column,
          };

          editor.executeEdits('ai-completion', [{
            range,
            text: response.completion,
          }]);
        }
      } catch (error) {
        console.error('AI completion error:', error);
      } finally {
        setIsLoading(false);
      }
    });
  };

  const handleCodeChange = (value: string | undefined) => {
    setCode(value || '');
  };

  const handleRunCode = () => {
    // This would integrate with a code runner
    console.log('Running code:', code);
  };

  const handleSaveFile = () => {
    // This would save the file
    console.log('Saving file:', file?.path, code);
  };

  const handleGetAISuggestions = async () => {
    if (!code.trim()) return;

    setIsLoading(true);
    try {
      const response = await gptApi.chat([{
        role: 'user',
        content: `Please provide code improvement suggestions for this ${language} code:\n\n${code}`
      }]);

      setSuggestions(response.response.split('\n').filter((s: string) => s.trim()));
    } catch (error) {
      console.error('AI suggestions error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-2 border-b bg-muted/50">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">
            {file?.name || 'Untitled'}
          </span>
          <span className="text-xs text-muted-foreground">
            {language}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="outline"
            onClick={handleGetAISuggestions}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            ) : (
              <Lightbulb className="h-3 w-3 mr-1" />
            )}
            AI Suggestions
          </Button>
          <Button size="sm" variant="outline" onClick={handleSaveFile}>
            <Save className="h-3 w-3 mr-1" />
            Save
          </Button>
          <Button size="sm" onClick={handleRunCode}>
            <Play className="h-3 w-3 mr-1" />
            Run
          </Button>
        </div>
      </div>

      {/* Editor */}
      <div className="flex-1 relative">
        <Editor
          height="100%"
          language={language}
          value={code}
          onChange={handleCodeChange}
          onMount={handleEditorDidMount}
          options={{
            minimap: { enabled: true },
            fontSize: 14,
            lineNumbers: 'on',
            roundedSelection: false,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            wordWrap: 'on',
            suggestOnTriggerCharacters: true,
            acceptSuggestionOnEnter: 'on',
            quickSuggestions: {
              other: true,
              comments: true,
              strings: true,
            },
            parameterHints: {
              enabled: true,
            },
            hover: {
              enabled: true,
            },
          }}
        />
      </div>

      {/* AI Suggestions Panel */}
      {suggestions.length > 0 && (
        <div className="border-t bg-muted/50 p-3">
          <h4 className="text-sm font-medium mb-2">AI Suggestions:</h4>
          <ul className="text-xs space-y-1">
            {suggestions.map((suggestion, index) => (
              <li key={index} className="text-muted-foreground">
                • {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default CodeEditor;
