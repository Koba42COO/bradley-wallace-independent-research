import React from 'react';

const Terminal = () => {
  return (
    <div className="h-full bg-black text-green-400 p-4 font-mono text-sm">
      <div className="mb-2">
        <span className="text-blue-400">$ </span>
        <span>Welcome to AIVA IDE Terminal</span>
      </div>
      <div className="mb-2">
        <span className="text-blue-400">$ </span>
        <span>npm install</span>
      </div>
      <div className="text-yellow-400">
        Installing dependencies...
      </div>
      <div className="mt-4">
        <span className="text-blue-400">$ </span>
        <span className="animate-pulse">_</span>
      </div>
    </div>
  );
};

export default Terminal;
