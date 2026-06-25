---
title: Domains Only
category: Data Feeds
endpoint: GET https://api-v1.whoisdatacenter.com/api/v2/domains-only
method: GET
source_url: https://whoisdatacenter.com/api-docs/#domains-only-feed
---

# Domains Only

## Description

Returns a bare list of domain names without full WHOIS data.

## Metadata

- **HTTP Method:** `GET`
- **Endpoint Path:** `https://api-v1.whoisdatacenter.com/api/v2/domains-only`
- **Credits Deducted:** 1 Credit
- **Rate Limit:** 1 Credit
- **Authentication:** Required
- **Input Type:** Date
- **Output Type:** Domain name list

## Request

`GET https://api-v1.whoisdatacenter.com/api/v2/domains-only`

## Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `date` | `string` | No | Example value: `2026-05-14`. Pass as a URL-encoded query parameter. |
| `format` | `string` | No | Response format. One of `json`, `xml`, `csv`. Defaults to `json`. |

## Examples & Notes

*Get a plain domain list for a given date.*

## Code Examples

### cURL Request

```bash
# Date/TLD lookup — JSON
curl -X GET "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Accept: application/json"
```

### Python Request

```python
import requests

response = requests.get(
  "https://api-v1.whoisdatacenter.com/api/v2/domains-only",
  params={
    "date": "2026-05-14"
  },
  headers={"Authorization": "Bearer YOUR_API_KEY"}
)
data = response.json()
print(data)
```

### Node.js Request

```javascript
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### JavaScript Request

```javascript
// Browser fetch — requires CORS
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### jQuery Request

```javascript
$.ajax({
  url: "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  dataType: "json",
  success: function(data) {
    console.log(data);
  }
});
```

### PHP Request

```php
$ch = curl_init("https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14");
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_HTTPHEADER => ["Authorization: Bearer YOUR_API_KEY"]
]);
$data = json_decode(curl_exec($ch), true);
```

### Ruby Request

```ruby
require "net/http"
require "json"

uri = URI("https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14")
req = Net::HTTP::Get.new(uri)
req["Authorization"] = "Bearer YOUR_API_KEY"
res = Net::HTTP.start(uri.host, uri.port, use_ssl: true) { |h| h.request(req) }
data = JSON.parse(res.body)
```

### Java Request

```java
import java.net.URI;
import java.net.http.*;

HttpClient client = HttpClient.newHttpClient();
HttpRequest req = HttpRequest.newBuilder()
  .uri(URI.create("https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14"))
  .header("Authorization", "Bearer YOUR_API_KEY")
  .GET()
  .build();
HttpResponse<String> res = client.send(req,
  HttpResponse.BodyHandlers.ofString());
```

### C# Request

```csharp
using System.Net.Http;

var client = new HttpClient();
client.DefaultRequestHeaders.Add("Authorization", "Bearer YOUR_API_KEY");
var json = await client.GetStringAsync("https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14");
```

### Go Request

```go
package main

import (
  "encoding/json"
  "net/http"
)

req, _ := http.NewRequest("GET", "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14", nil)
req.Header.Set("Authorization", "Bearer YOUR_API_KEY")
res, _ := http.DefaultClient.Do(req)
defer res.Body.Close()
var data map[string]any
json.NewDecoder(res.Body).Decode(&data)
```

### Swift Request

```swift
import Foundation

var req = URLRequest(url: URL(string: "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14")!)
req.httpMethod = "GET"
req.setValue("Bearer YOUR_API_KEY",
  forHTTPHeaderField: "Authorization")
URLSession.shared.dataTask(with: req) { data, _, _ in
  if let d = data {
    let json = try? JSONSerialization.jsonObject(with: d)
  }
}.resume()
```

### C Request

```bash
#include <curl/curl.h>

int main(void) {
  CURL *c = curl_easy_init();
  struct curl_slist *h = NULL;
  h = curl_slist_append(h, "Authorization: Bearer YOUR_API_KEY");
  curl_easy_setopt(c, CURLOPT_URL, "https://api-v1.whoisdatacenter.com/api/v2/domains-only?date=2026-05-14");
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
  curl_easy_perform(c);
  curl_easy_cleanup(c);
}
```

