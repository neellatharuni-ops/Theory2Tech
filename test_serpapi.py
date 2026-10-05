from serpapi_service import search_google


results = search_google("B-Tree database")

print(f"Found {len(results)} results.")

for result in results[:5]:
    print()
    print("Title:", result.get("title"))
    print("Link:", result.get("link"))