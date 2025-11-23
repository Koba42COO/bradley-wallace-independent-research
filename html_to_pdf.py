#!/usr/bin/env python3
"""
Convert HTML podcast synopsis to PDF
"""
import sys
import os

# Try multiple PDF conversion methods
def convert_to_pdf(html_file, pdf_file):
    """Convert HTML to PDF using available method"""
    
    # Method 1: Try weasyprint
    try:
        from weasyprint import HTML
        print("Using weasyprint...")
        HTML(filename=html_file).write_pdf(pdf_file)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Weasyprint error: {e}")
    
    # Method 2: Try pdfkit (requires wkhtmltopdf)
    try:
        import pdfkit
        print("Using pdfkit...")
        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_file, pdf_file, options=options)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Pdfkit error: {e}")
    
    # Method 3: Try playwright
    try:
        from playwright.sync_api import sync_playwright
        print("Using playwright...")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f'file://{os.path.abspath(html_file)}')
            page.pdf(path=pdf_file, format='Letter', margin={'top': '0.75in', 'right': '0.75in', 'bottom': '0.75in', 'left': '0.75in'})
            browser.close()
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Playwright error: {e}")
    
    return False

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        html_file = sys.argv[1]
        pdf_file = sys.argv[2] if len(sys.argv) >= 3 else html_file.replace('.html', '.pdf')
    else:
        html_file = 'DROPPING_BOMBS_PODCAST_SYNOPSIS.html'
        pdf_file = 'DROPPING_BOMBS_PODCAST_SYNOPSIS.pdf'
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found")
        sys.exit(1)
    
    print(f"Converting {html_file} to {pdf_file}...")
    
    if convert_to_pdf(html_file, pdf_file):
        print(f"✅ PDF created: {pdf_file}")
        print(f"File size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
    else:
        print("❌ No PDF conversion tool available")
        print("\nTo install a PDF converter, try one of:")
        print("  pip install weasyprint")
        print("  pip install pdfkit  # (also need: brew install wkhtmltopdf)")
        print("  pip install playwright  # (then: playwright install chromium)")
        sys.exit(1)

