import requests
import json

# Search for parrot communication research
query = "parrot communication vocal learning"
url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=10&fields=title,abstract,authors,year,venue,citationCount,externalIds"

response = requests.get(url)
data = response.json()

print('🐦 PARROT COMMUNICATION RESEARCH DATABASE')
print('=' * 50)

if 'data' in data:
    for i, paper in enumerate(data['data'], 1):
        print(f'{i}. {paper.get("title", "No title")}')
        authors = [a.get('name', '') for a in paper.get('authors', [])]
        print(f'   Authors: {", ".join(authors[:3])} et al.')
        print(f'   Year: {paper.get("year", "N/A")}')
        print(f'   Citations: {paper.get("citationCount", 0)}')
        print(f'   Venue: {paper.get("venue", "N/A")}')
        if paper.get('abstract'):
            abstract = paper['abstract'][:150] + '...' if len(paper['abstract']) > 150 else paper['abstract']
            print(f'   Abstract: {abstract}')
        print()
