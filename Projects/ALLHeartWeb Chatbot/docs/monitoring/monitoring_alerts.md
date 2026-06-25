---
title: Check WHOIS Alerts
category: Monitoring & Alerts
endpoint: POST https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts
method: POST
source_url: https://whoisdatacenter.com/api-docs/#monitoring-alerts
---

# Check WHOIS Alerts

## Description

Compares the latest WHOIS against the stored baseline and returns any changed fields.

## Metadata

- **HTTP Method:** `POST`
- **Endpoint Path:** `https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts`
- **Credits Deducted:** 1 Credit
- **Rate Limit:** 1 Credit
- **Authentication:** Required
- **Input Type:** Domain
- **Output Type:** Change report

## Request

`POST https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts`

## Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `body` | `JSON` | **Yes** | POST a JSON body with your input: `{"domain":"yahoo.com"}` |
| `format` | `string` | No | Response format. One of `json`, `xml`, `csv`. Defaults to `json`. |

## Examples & Notes

*Check if yahoo.com ownership changed since last check.*

## Code Examples

### cURL Request

```bash
# POST with JSON body
curl -X POST "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"domain":"yahoo.com"}'
```

### Python Request

```python
import requests

response = requests.post(
  "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts",
  json={"domain":"yahoo.com"},
  headers={"Authorization": "Bearer YOUR_API_KEY"}
)
data = response.json()
```

### Node.js Request

```javascript
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts", {
    method: "POST",
    headers: {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({"domain":"yahoo.com"})
  }
);
const data = await res.json();
```

### JavaScript Request

```javascript
// POST from browser
const res = await fetch("https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts", {
  method: "POST",
  headers: {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({"domain":"yahoo.com"})
});
const data = await res.json();
```

### jQuery Request

```javascript
$.ajax({
  url: "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts",
  method: "POST",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  contentType: "application/json",
  data: JSON.stringify({"domain":"yahoo.com"}),
  success: function(data) {
    console.log(data);
  }
});
```

### PHP Request

```php
$ch = curl_init("https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts");
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_POST => true,
  CURLOPT_POSTFIELDS => json_encode({"domain":"yahoo.com"}),
  CURLOPT_HTTPHEADER => ["Authorization: Bearer YOUR_API_KEY", "Content-Type: application/json"]
]);
$data = json_decode(curl_exec($ch), true);
```

### Ruby Request

```ruby
require "net/http"
require "json"

uri = URI("https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts")
req = Net::HTTP::Post.new(uri)
req["Authorization"] = "Bearer YOUR_API_KEY"
req.body = ({"domain":"yahoo.com"}).to_json
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
  .uri(URI.create("https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts"))
  .header("Authorization", "Bearer YOUR_API_KEY")
  .POST(HttpRequest.BodyPublishers.ofString("{\"domain\":\"yahoo.com\"}"))
  .build();
HttpResponse<String> res = client.send(req,
  HttpResponse.BodyHandlers.ofString());
```

### C# Request

```csharp
using System.Net.Http;

var client = new HttpClient();
client.DefaultRequestHeaders.Add("Authorization", "Bearer YOUR_API_KEY");
var content = new StringContent("{\"domain\":\"yahoo.com\"}",
  Encoding.UTF8, "application/json");
var res = await client.PostAsync("https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts", content);
```

### Go Request

```go
package main

import (
  "strings"
  "net/http"
)

body := strings.NewReader(`{"domain":"yahoo.com"}`)
req, _ := http.NewRequest("POST", "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts", body)
req.Header.Set("Authorization", "Bearer YOUR_API_KEY")
req.Header.Set("Content-Type", "application/json")
res, _ := http.DefaultClient.Do(req)
```

### Swift Request

```swift
import Foundation

var req = URLRequest(url: URL(string: "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts")!)
req.httpMethod = "POST"
req.setValue("Bearer YOUR_API_KEY",
  forHTTPHeaderField: "Authorization")
req.httpBody = "{\"domain\":\"yahoo.com\"}".data(using: .utf8)
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
  curl_easy_setopt(c, CURLOPT_URL, "https://api-v1.whoisdatacenter.com/api/v2/monitoring/alerts");
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
  curl_easy_perform(c);
  curl_easy_cleanup(c);
}
```

