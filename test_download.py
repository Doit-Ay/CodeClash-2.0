import requests

url = "https://doitay-my.sharepoint.com/personal/adityayadav_doitay_onmicrosoft_com/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fadityayadav_doitay_onmicrosoft_com%2FDocuments%2Fimages%2F3000000004.png"
r = requests.get(url, timeout=15, allow_redirects=True)
print(f"Status: {r.status_code}")
ct = r.headers.get("Content-Type", "unknown")
print(f"Content-Type: {ct}")
print(f"Content-Length: {len(r.content)}")
print(f"Is image: {'image' in ct.lower()}")
