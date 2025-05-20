from serpapi import GoogleSearch

API_KEY = "bf6519e83a7841c6c927e454a592499d59ae0e34db2473eca9b4055453fea92f"
QUERY_FILE = "query.txt"
OUTPUT_FILE = "sites_v2.txt"

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    QUERY = f.read().strip()

urls = []

for page in range(0, 10):
    start_val = page * 100
    print(f"🔍 Fetching results {start_val+1}–{start_val+100}...")

    params = {
        "q": QUERY,
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "api_key": API_KEY,
        "engine": "google",
        "num": 100,
        "start": start_val
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic = results.get("organic_results", [])
    if not organic:
        print(f"⚠️ No more results at start={start_val}. Ending early.")
        break

    for result in organic:
        url = result.get("link")
        urls.append(url)

unique_urls = list(set(urls))

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for u in unique_urls:
        f.write(u + "\n")

print(f"\n✅ Saved {len(unique_urls)} unique URLs to {OUTPUT_FILE}")
