---
title: Bulk Export
category: Bulk Operations
endpoint: POST https://api-v1.whoisdatacenter.com/api/v2/bulk/export
method: POST
source_url: https://whoisdatacenter.com/api-docs/#bulk-export
---

# Bulk Export

## Description

Exports bulk WHOIS results to CSV, JSON, or JSONL.

## Metadata

- **HTTP Method:** `POST`
- **Endpoint Path:** `https://api-v1.whoisdatacenter.com/api/v2/bulk/export`
- **Credits Deducted:** 1 Credit per domain
- **Rate Limit:** 1 Credit per domain
- **Authentication:** Required
- **Input Type:** Filters or job input
- **Output Type:** Export file / URL

## Request

`POST https://api-v1.whoisdatacenter.com/api/v2/bulk/export`

## Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `body` | `JSON` | **Yes** | POST a JSON body with your input: `{"tld":"com","from":"2025-01-01","to":"2025-12-31"}` |
| `format` | `string` | No | Response format. One of `json`, `xml`, `csv`. Defaults to `json`. |

## Examples & Notes

*Export 10,000 WHOIS records to one CSV file.*

## Code Examples

### cURL Request

```bash
# POST with JSON body
curl -X POST "https://api-v1.whoisdatacenter.com/api/v2/bulk/export" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tld":"com","from":"2025-01-01","to":"2025-12-31"}'
```

### Python Request

```python
import requests

response = requests.post(
  "https://api-v1.whoisdatacenter.com/api/v2/bulk/export",
  json={"tld":"com","from":"2025-01-01","to":"2025-12-31"},
  headers={"Authorization": "Bearer YOUR_API_KEY"}
)
data = response.json()
```

### Node.js Request

```javascript
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/bulk/export", {
    method: "POST",
    headers: {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({"tld":"com","from":"2025-01-01","to":"2025-12-31"})
  }
);
const data = await res.json();
```

### JavaScript Request

```javascript
// POST from browser
const res = await fetch("https://api-v1.whoisdatacenter.com/api/v2/bulk/export", {
  method: "POST",
  headers: {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({"tld":"com","from":"2025-01-01","to":"2025-12-31"})
});
const data = await res.json();
```

### jQuery Request

```javascript
$.ajax({
  url: "https://api-v1.whoisdatacenter.com/api/v2/bulk/export",
  method: "POST",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  contentType: "application/json",
  data: JSON.stringify({"tld":"com","from":"2025-01-01","to":"2025-12-31"}),
  success: function(data) {
    console.log(data);
  }
});
```

### PHP Request

```php
$ch = curl_init("https://api-v1.whoisdatacenter.com/api/v2/bulk/export");
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_POST => true,
  CURLOPT_POSTFIELDS => json_encode({"tld":"com","from":"2025-01-01","to":"2025-12-31"}),
  CURLOPT_HTTPHEADER => ["Authorization: Bearer YOUR_API_KEY", "Content-Type: application/json"]
]);
$data = json_decode(curl_exec($ch), true);
```

### Ruby Request

```ruby
require "net/http"
require "json"

uri = URI("https://api-v1.whoisdatacenter.com/api/v2/bulk/export")
req = Net::HTTP::Post.new(uri)
req["Authorization"] = "Bearer YOUR_API_KEY"
req.body = ({"tld":"com","from":"2025-01-01","to":"2025-12-31"}).to_json
req["Content-Type"] = "application/json"
res = Net::HTTP.start(uri.host, uri.port, use_ssl: true) { |h| h.request(req) }
data = JSON.parse(res.body)
```

### Java Request

```java
import java.net.URI;
import java.net.http.*;

HttpClient client = HttpClient.newHttpClient();
HttpRequest req = HttpRequest.newBuilder()
  .uri(URI.create("https://api-v1.whoisdatacenter.com/api/v2/bulk/export"))
  .header("Authorization", "Bearer YOUR_API_KEY")
  .POST(HttpRequest.BodyPublishers.ofString("{\"tld\":\"com\",\"from\":\"2025-01-01\",\"to\":\"2025-12-31\"}"))
  .build();
HttpResponse<String> res = client.send(req,
  HttpResponse.BodyHandlers.ofString());
```

### C# Request

```csharp
using System.Net.Http;

var client = new HttpClient();
client.DefaultRequestHeaders.Add("Authorization", "Bearer YOUR_API_KEY");
var content = new StringContent("{\"tld\":\"com\",\"from\":\"2025-01-01\",\"to\":\"2025-12-31\"}",
  Encoding.UTF8, "application/json");
var res = await client.PostAsync("https://api-v1.whoisdatacenter.com/api/v2/bulk/export", content);
```

### Go Request

```go
package main

import (
  "strings"
  "net/http"
)

body := strings.NewReader(`{"tld":"com","from":"2025-01-01","to":"2025-12-31"}`)
req, _ := http.NewRequest("POST", "https://api-v1.whoisdatacenter.com/api/v2/bulk/export", body)
req.Header.Set("Authorization", "Bearer YOUR_API_KEY")
req.Header.Set("Content-Type", "application/json")
res, _ := http.DefaultClient.Do(req)
```

### Swift Request

```swift
import Foundation

var req = URLRequest(url: URL(string: "https://api-v1.whoisdatacenter.com/api/v2/bulk/export")!)
req.httpMethod = "POST"
req.setValue("Bearer YOUR_API_KEY",
  forHTTPHeaderField: "Authorization")
req.httpBody = "{\"tld\":\"com\",\"from\":\"2025-01-01\",\"to\":\"2025-12-31\"}".data(using: .utf8)
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
  curl_easy_setopt(c, CURLOPT_URL, "https://api-v1.whoisdatacenter.com/api/v2/bulk/export");
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
  curl_easy_perform(c);
  curl_easy_cleanup(c);
}
```

