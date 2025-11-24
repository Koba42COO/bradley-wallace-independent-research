#!/usr/bin/env python3
"""
Convert Podcast Synopsis HTML to PDF
Uses multiple methods to ensure compatibility
"""

import subprocess
import sys
from pathlib import Path

def try_weasyprint(html_path, pdf_path):
    """Try using weasyprint"""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  ⚠️  WeasyPrint error: {e}")
        return False

def try_pdfkit(html_path, pdf_path):
    """Try using pdfkit (requires wkhtmltopdf)"""
    try:
        import pdfkit
        pdfkit.from_file(str(html_path), str(pdf_path))
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  ⚠️  PDFKit error: {e}")
        return False

def try_chrome_headless(html_path, pdf_path):
    """Try using Chrome headless (if available)"""
    try:
        # Try Chrome
        chrome_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            'google-chrome',
            'chromium'
        ]
        
        chrome = None
        for path in chrome_paths:
            if Path(path).exists() or subprocess.run(['which', path.split('/')[-1]], 
                                                     capture_output=True).returncode == 0:
                chrome = path
                break
        
        if chrome:
            cmd = [
                chrome,
                '--headless',
                '--disable-gpu',
                '--print-to-pdf=' + str(pdf_path),
                'file://' + str(html_path.absolute())
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and pdf_path.exists():
                return True
    except Exception as e:
        pass
    return False

def try_safari_print(html_path, pdf_path):
    """Try using Safari via AppleScript (macOS)"""
    try:
        import platform
        if platform.system() != 'Darwin':
            return False
        
        # Create AppleScript to print HTML to PDF
        script = f'''
        tell application "Safari"
            activate
            open POSIX file "{html_path.absolute()}"
            delay 2
            tell application "System Events"
                keystroke "p" using command down
                delay 1
                keystroke "s" using command down
                delay 1
                keystroke "{pdf_path.name}"
                delay 1
                keystroke return
            end tell
        end tell
        '''
        # This is complex, skip for now
        return False
    except:
        return False

def create_simple_pdf(html_path, pdf_path):
    """Create a simple PDF using reportlab as fallback"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        import re
        
        # Read HTML and extract text
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Simple HTML to text extraction
        from html.parser import HTMLParser
        
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.in_script = False
                self.in_style = False
            
            def handle_starttag(self, tag, attrs):
                if tag in ['script', 'style']:
                    self.in_script = True
                if tag == 'br':
                    self.text.append('\n')
                if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    self.text.append('\n\n')
                if tag == 'p':
                    self.text.append('\n')
            
            def handle_endtag(self, tag):
                if tag in ['script', 'style']:
                    self.in_script = False
                if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']:
                    self.text.append('\n')
            
            def handle_data(self, data):
                if not self.in_script:
                    self.text.append(data)
        
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = ''.join(parser.text)
        
        # Clean up text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Create PDF
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        story = []
        styles = getSampleStyleSheet()
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace('\n', '<br/>'), styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  ⚠️  ReportLab error: {e}")
        return False

def main():
    """Main conversion function"""
    
    html_path = Path("PODCAST_SYNOPSIS_FINAL_PRIME_ZETA_CONSCIOUSNESS.html")
    pdf_path = Path("PODCAST_SYNOPSIS_FINAL_PRIME_ZETA_CONSCIOUSNESS.pdf")
    
    if not html_path.exists():
        print(f"❌ HTML file not found: {html_path}")
        return 1
    
    print("📄 Converting Podcast Synopsis to PDF...")
    print(f"   Input: {html_path}")
    print(f"   Output: {pdf_path}")
    print()
    
    # Try different methods
    methods = [
        ("WeasyPrint", try_weasyprint),
        ("PDFKit (wkhtmltopdf)", try_pdfkit),
        ("Chrome Headless", try_chrome_headless),
        ("ReportLab (simple)", create_simple_pdf),
    ]
    
    for method_name, method_func in methods:
        print(f"🔄 Trying {method_name}...")
        try:
            if method_func(html_path, pdf_path):
                print(f"✅ Success! PDF created using {method_name}")
                print(f"   File: {pdf_path.absolute()}")
                print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
                return 0
        except Exception as e:
            print(f"  ⚠️  {method_name} failed: {e}")
            continue
    
    print()
    print("❌ All PDF conversion methods failed.")
    print()
    print("💡 INSTALLATION OPTIONS:")
    print()
    print("Option 1: Install WeasyPrint (recommended)")
    print("  pip3 install weasyprint")
    print()
    print("Option 2: Install PDFKit + wkhtmltopdf")
    print("  pip3 install pdfkit")
    print("  brew install wkhtmltopdf  # macOS")
    print()
    print("Option 3: Use Chrome headless")
    print("  Install Google Chrome browser")
    print()
    print("Option 4: Manual conversion")
    print("  1. Open PODCAST_SYNOPSIS_FINAL_PRIME_ZETA_CONSCIOUSNESS.html in browser")
    print("  2. Print to PDF (Cmd+P, Save as PDF)")
    print()
    
    return 1

if __name__ == "__main__":
    sys.exit(main())

