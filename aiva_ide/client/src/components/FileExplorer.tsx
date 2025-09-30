import React from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import {
  Folder,
  FolderOpen,
  FileText,
  FileCode,
  FileJson,
  Settings,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

interface FileNode {
  name: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  expanded?: boolean;
}

const mockFileTree: FileNode[] = [
  {
    name: 'src',
    type: 'folder',
    expanded: true,
    children: [
      {
        name: 'components',
        type: 'folder',
        expanded: true,
        children: [
          { name: 'App.tsx', type: 'file' },
          { name: 'Editor.tsx', type: 'file' },
          { name: 'Terminal.tsx', type: 'file' }
        ]
      },
      {
        name: 'lib',
        type: 'folder',
        children: [
          { name: 'utils.ts', type: 'file' }
        ]
      },
      { name: 'main.tsx', type: 'file' },
      { name: 'App.css', type: 'file' }
    ]
  },
  { name: 'package.json', type: 'file' },
  { name: 'tsconfig.json', type: 'file' },
  { name: 'README.md', type: 'file' }
];

const getFileIcon = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'tsx':
    case 'ts':
    case 'js':
    case 'jsx':
      return FileCode;
    case 'json':
      return FileJson;
    case 'md':
      return FileText;
    default:
      return FileText;
  }
};

interface FileTreeNodeProps {
  node: FileNode;
  level: number;
  onToggle: (node: FileNode) => void;
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node, level, onToggle }) => {
  const Icon = node.type === 'folder'
    ? (node.expanded ? FolderOpen : Folder)
    : getFileIcon(node.name);

  return (
    <div>
      <Button
        variant="ghost"
        className={`w-full justify-start h-6 px-${level * 4 + 2} text-xs`}
        onClick={() => node.type === 'folder' && onToggle(node)}
      >
        {node.type === 'folder' && (
          <span className="mr-1">
            {node.expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </span>
        )}
        <Icon className="h-3 w-3 mr-2" />
        {node.name}
      </Button>
      {node.type === 'folder' && node.expanded && node.children && (
        <div>
          {node.children.map((child, index) => (
            <FileTreeNode
              key={index}
              node={child}
              level={level + 1}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const FileExplorer = () => {
  const [fileTree, setFileTree] = React.useState<FileNode[]>(mockFileTree);

  const handleToggle = (node: FileNode) => {
    const updateTree = (nodes: FileNode[]): FileNode[] => {
      return nodes.map(n => {
        if (n === node) {
          return { ...n, expanded: !n.expanded };
        }
        if (n.children) {
          return { ...n, children: updateTree(n.children) };
        }
        return n;
      });
    };

    setFileTree(updateTree(fileTree));
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-2 border-b">
        <h3 className="text-sm font-semibold">Explorer</h3>
      </div>

      {/* File Tree */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {fileTree.map((node, index) => (
            <FileTreeNode
              key={index}
              node={node}
              level={0}
              onToggle={handleToggle}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default FileExplorer;
