#!/usr/bin/env python3
"""
Convert markdown podcast synopsis to HTML
"""
import re
import html as html_escape

def convert_md_to_html(md_text):
    """Convert markdown to HTML with proper formatting"""
    lines = md_text.split('\n')
    html_lines = []
    in_code_block = False
    code_lang = ''
    in_list = False
    list_type = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line
        
        # Code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                lang_match = re.match(r'```(\w+)', line)
                code_lang = lang_match.group(1) if lang_match else ''
                html_lines.append(f'<pre><code class="language-{code_lang}">')
                in_code_block = True
            else:
                html_lines.append('</code></pre>')
                in_code_block = False
            i += 1
            continue
        
        if in_code_block:
            html_lines.append(html_escape.escape(line) + '\n')
            i += 1
            continue
        
        # Headers
        if line.startswith('# '):
            html_lines.append(f'<h1>{convert_inline_md(line[2:])}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{convert_inline_md(line[3:])}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{convert_inline_md(line[4:])}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{convert_inline_md(line[5:])}</h4>')
        # Horizontal rule
        elif line.strip() == '---':
            html_lines.append('<hr>')
        # Lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list or list_type != 'ul':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            item = line.strip()[2:].strip()
            html_lines.append(f'<li>{convert_inline_md(item)}</li>')
        elif re.match(r'^\d+\. ', line.strip()):
            if not in_list or list_type != 'ol':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            item = re.sub(r'^\d+\. ', '', line.strip())
            html_lines.append(f'<li>{convert_inline_md(item)}</li>')
        # Empty line
        elif not line.strip():
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
                list_type = None
            html_lines.append('')
        # Regular paragraphs
        else:
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
                list_type = None
            
            # Check for special boxes (handle multi-line)
            if '**Analogy Answer**:' in line:
                # Start analogy answer box
                content = line.replace('**Analogy Answer**:', '').strip()
                if content:
                    html_lines.append(f'<div class="analogy-answer"><strong>Analogy Answer:</strong> {convert_inline_md(content)}')
                else:
                    html_lines.append('<div class="analogy-answer"><strong>Analogy Answer:</strong>')
                # Continue reading lines until next question or section
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(('**Analogy Answer**:', '**Concrete Example**:', '###', '##', '#')) and lines[i].strip():
                    html_lines.append(f'<p>{convert_inline_md(lines[i])}</p>')
                    i += 1
                html_lines.append('</div>')
                continue
            elif '**Concrete Example**:' in line:
                # Start example box
                content = line.replace('**Concrete Example**:', '').strip()
                if content:
                    html_lines.append(f'<div class="example-box"><strong>Concrete Example:</strong> {convert_inline_md(content)}')
                else:
                    html_lines.append('<div class="example-box"><strong>Concrete Example:</strong>')
                # Continue reading lines
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(('**Analogy Answer**:', '**Concrete Example**:', '###', '##', '#')) and lines[i].strip():
                    html_lines.append(f'<p>{convert_inline_md(lines[i])}</p>')
                    i += 1
                html_lines.append('</div>')
                continue
            else:
                html_lines.append(f'<p>{convert_inline_md(line)}</p>')
        
        i += 1
    
    if in_list:
        html_lines.append(f'</{list_type}>')
    
    return '\n'.join(html_lines)

def convert_inline_md(text):
    """Convert inline markdown to HTML"""
    # Escape HTML first
    text = html_escape.escape(text)
    
    # Bold (must come before italic)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic (single asterisk, not double)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', text)
    # Code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # URLs to links
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank">\1</a>', text)
    
    return text

import sys
import os

# ... (rest of imports)

# ... (convert_md_to_html and convert_inline_md functions remain the same)

if __name__ == '__main__':
    # Default values
    md_file = 'DROPPING_BOMBS_PODCAST_SYNOPSIS.md'
    html_file = 'DROPPING_BOMBS_PODCAST_SYNOPSIS.html'
    
    # Check for command line arguments
    if len(sys.argv) >= 2:
        md_file = sys.argv[1]
        if len(sys.argv) >= 3:
            html_file = sys.argv[2]
        else:
            html_file = md_file.replace('.md', '.html')
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found")
        sys.exit(1)

    print(f"Converting {md_file} to {html_file}...")

    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to HTML
    html_content = convert_md_to_html(md_content)

    # Create full HTML document
    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_file.replace('.md', '').replace('_', ' ').title()}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 50px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            border-radius: 10px;
        }}
        h1 {{
            color: #667eea;
            border-bottom: 5px solid #764ba2;
            padding-bottom: 15px;
            margin-bottom: 40px;
            font-size: 2.8em;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            margin-top: 50px;
            margin-bottom: 25px;
            font-size: 2.2em;
            border-left: 6px solid #667eea;
            padding: 15px 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        h3 {{
            color: #667eea;
            margin-top: 35px;
            margin-bottom: 20px;
            font-size: 1.6em;
        }}
        h4 {{
            color: #666;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        p {{
            margin-bottom: 18px;
            text-align: justify;
            font-size: 1.05em;
        }}
        ul, ol {{
            margin-left: 40px;
            margin-bottom: 25px;
        }}
        li {{
            margin-bottom: 10px;
            line-height: 1.8;
        }}
        strong {{
            color: #2c3e50;
            font-weight: 700;
        }}
        em {{
            color: #667eea;
            font-style: italic;
        }}
        code {{
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            color: #e74c3c;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 25px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 25px 0;
            border-left: 5px solid #667eea;
        }}
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        hr {{
            border: none;
            border-top: 3px solid #ecf0f1;
            margin: 50px 0;
        }}
        a {{
            color: #667eea;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .analogy-answer {{
            background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f2 100%);
            border-left: 5px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .example-box {{
            background: linear-gradient(135deg, #fff9e6 0%, #ffeaa7 100%);
            border: 3px solid #f39c12;
            border-left: 5px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>'''

    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    print(f'✅ HTML file created: {html_file}')

