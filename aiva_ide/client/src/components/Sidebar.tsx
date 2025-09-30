import React from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  FolderOpen,
  FileText,
  Terminal,
  MessageSquare,
  Settings,
  Search,
  GitBranch,
  Package
} from 'lucide-react';

const Sidebar = () => {
  const menuItems = [
    { icon: FolderOpen, label: 'Explorer', active: true },
    { icon: Search, label: 'Search' },
    { icon: GitBranch, label: 'Source Control' },
    { icon: Package, label: 'Extensions' },
    { icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="h-full bg-background border-r flex flex-col">
      {/* Top menu */}
      <div className="p-2 border-b">
        <div className="flex flex-col gap-1">
          {menuItems.map((item) => (
            <Button
              key={item.label}
              variant={item.active ? "secondary" : "ghost"}
              size="sm"
              className="w-full justify-start p-2"
            >
              <item.icon className="h-4 w-4 mr-2" />
              <span className="sr-only">{item.label}</span>
            </Button>
          ))}
        </div>
      </div>

      {/* Bottom status */}
      <div className="mt-auto p-2 border-t">
        <div className="text-xs text-muted-foreground text-center">
          AIVA IDE v1.0
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
