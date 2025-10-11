#!/usr/bin/env python3
"""
Generate icons for the GPT Teams Archive Extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Create a simple brain/AI themed icon"""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Colors
    brain_color = (102, 126, 234)  # #667eea
    accent_color = (118, 75, 162)  # #764ba2

    # Draw brain-like shape (simplified)
    center = size // 2
    radius = size // 3

    # Main brain shape
    draw.ellipse([center - radius, center - radius, center + radius, center + radius],
                fill=brain_color)

    # Neural network connections
    connection_color = accent_color + (180,)  # Semi-transparent

    # Horizontal connections
    draw.line([center - radius//2, center, center + radius//2, center],
             fill=connection_color, width=max(1, size//32))

    # Vertical connections
    draw.line([center, center - radius//2, center, center + radius//2],
             fill=connection_color, width=max(1, size//32))

    # Diagonal connections
    draw.line([center - radius//3, center - radius//3, center + radius//3, center + radius//3],
             fill=connection_color, width=max(1, size//32))
    draw.line([center + radius//3, center - radius//3, center - radius//3, center + radius//3],
             fill=connection_color, width=max(1, size//32))

    # Add small nodes at intersections
    node_radius = max(1, size//24)
    nodes = [
        (center, center),
        (center - radius//2, center),
        (center + radius//2, center),
        (center, center - radius//2),
        (center, center + radius//2),
    ]

    for x, y in nodes:
        draw.ellipse([x - node_radius, y - node_radius, x + node_radius, y + node_radius],
                    fill=accent_color)

    # Save as PNG
    img.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

def main():
    """Generate all required icon sizes"""
    icons_dir = os.path.dirname(__file__) + '/icons'

    # Create icons in different sizes
    sizes = [16, 48, 128]

    for size in sizes:
        filename = f"{icons_dir}/icon{size}.png"
        create_icon(size, filename)

    print("\nIcon generation complete!")
    print(f"Icons saved to: {icons_dir}")

if __name__ == '__main__':
    main()
