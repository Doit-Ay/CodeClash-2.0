"""
Script to download images from a SharePoint shared folder using the sharing URL.
Uses the Microsoft Graph sharing API to resolve the shared folder and list/download files.
"""
import base64
import requests
import os
import sys

SHARE_URL = "https://doitay-my.sharepoint.com/:f:/g/personal/adityayadav_doitay_onmicrosoft_com/IgALtdyrt-xyS4T5-xyX9Ud8AU8TwHHHcEvNZC0XbBjUs1Q?e=os5MNn"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_images")

def encode_sharing_url(url):
    """Encode a sharing URL for the Microsoft Graph API."""
    encoded = base64.urlsafe_b64encode(url.encode()).decode()
    # Remove trailing '=' and prepend 'u!'
    return "u!" + encoded.rstrip("=")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Encode the sharing URL for Graph API
    encoded = encode_sharing_url(SHARE_URL)
    
    # Try to use Graph API to resolve the share
    graph_url = f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem/children"
    
    print(f"Encoded sharing token: {encoded}")
    print(f"Graph API URL: {graph_url}")
    
    # Try unauthenticated (for publicly shared links)
    resp = requests.get(graph_url, timeout=15)
    print(f"Graph API status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        items = data.get("value", [])
        print(f"Found {len(items)} items")
        
        for item in items:
            name = item.get("name", "")
            download_url = item.get("@microsoft.graph.downloadUrl", "")
            if download_url and name.endswith(".png"):
                print(f"Downloading: {name}")
                img_resp = requests.get(download_url, timeout=30)
                if img_resp.status_code == 200:
                    filepath = os.path.join(OUTPUT_DIR, name)
                    with open(filepath, "wb") as f:
                        f.write(img_resp.content)
                    print(f"  Saved: {filepath} ({len(img_resp.content)} bytes)")
                else:
                    print(f"  Failed: {img_resp.status_code}")
    else:
        print(f"Graph API returned: {resp.status_code}")
        print(resp.text[:500])
        print()
        print("Trying alternative approach with root driveItem...")
        
        # Try to get the root driveItem first
        root_url = f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem"
        resp2 = requests.get(root_url, timeout=15)
        print(f"Root driveItem status: {resp2.status_code}")
        if resp2.status_code == 200:
            print(resp2.json())
        else:
            print(resp2.text[:500])

if __name__ == "__main__":
    main()
