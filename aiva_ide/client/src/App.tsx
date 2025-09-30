import { Switch, Route } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import Sidebar from "@/components/Sidebar";
import Editor from "@/components/Editor";
import Terminal from "@/components/Terminal";
import AIChat from "@/components/AIChat";
import FileExplorer from "@/components/FileExplorer";
import "./App.css";

const queryClient = new QueryClient();

function MainLayout() {
  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="h-12 bg-background border-b flex items-center px-4">
        <h1 className="text-lg font-semibold">AIVA IDE</h1>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-sm text-muted-foreground">AI-Powered Development</span>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          {/* Sidebar */}
          <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
            <Sidebar />
          </ResizablePanel>

          <ResizableHandle />

          {/* Main Editor Area */}
          <ResizablePanel defaultSize={60}>
            <ResizablePanelGroup direction="vertical">
              {/* File Explorer and Editor */}
              <ResizablePanel defaultSize={70}>
                <ResizablePanelGroup direction="horizontal">
                  <ResizablePanel defaultSize={25} minSize={20}>
                    <FileExplorer />
                  </ResizablePanel>
                  <ResizableHandle />
                  <ResizablePanel defaultSize={75}>
                    <Editor />
                  </ResizablePanel>
                </ResizablePanelGroup>
              </ResizablePanel>

              <ResizableHandle />

              {/* Terminal */}
              <ResizablePanel defaultSize={30} minSize={20}>
                <Terminal />
              </ResizablePanel>
            </ResizablePanelGroup>
          </ResizablePanel>

          <ResizableHandle />

          {/* AI Chat Panel */}
          <ResizablePanel defaultSize={20} minSize={15} maxSize={35}>
            <AIChat />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="aiva-theme">
        <TooltipProvider>
          <Switch>
            <Route path="/" component={MainLayout} />
            <Route component={() => <div>404 - Page not found</div>} />
          </Switch>
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
