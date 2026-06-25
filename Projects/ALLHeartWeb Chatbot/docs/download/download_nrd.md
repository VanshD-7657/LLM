---
title: Download NRD
category: Download
endpoint: GET | POST https://api-v1.whoisdatacenter.com/api/v2/download/nrd
method: GET | POST
source_url: https://whoisdatacenter.com/api-docs/#download-nrd
---

# Download NRD

## Description

Downloads a ZIP archive of newly registered domains for a date.

## Metadata

- **HTTP Method:** `GET | POST`
- **Endpoint Path:** `https://api-v1.whoisdatacenter.com/api/v2/download/nrd`
- **Credits Deducted:** 200,000 Credits
- **Rate Limit:** 200,000 Credits
- **Authentication:** Required
- **Input Type:** Date
- **Output Type:** ZIP archive

## Request

`GET | POST https://api-v1.whoisdatacenter.com/api/v2/download/nrd`

## Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `date` | `string` | No | Example value: `2026-05-14`. Pass as a URL-encoded query parameter. |
| `apiKey` | `string` | **Yes** | Example value: `YOUR_API_KEY`. Pass as a URL-encoded query parameter. |

## Examples & Notes

*Download all new domains for today.*

## Code Examples

### cURL Request

```bash
# Date/TLD lookup — JSON
curl -X GET "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Accept: application/json"
```

### Python Request

```python
import requests

response = requests.get(
  "https://api-v1.whoisdatacenter.com/api/v2/download/nrd",
  params={
    "date": "2026-05-14",
    "apiKey": "YOUR_API_KEY"
  },
  headers={"Authorization": "Bearer YOUR_API_KEY"}
)
data = response.json()
print(data)
```

### Node.js Request

```javascript
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### JavaScript Request

```javascript
// Browser fetch — requires CORS
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### jQuery Request

```javascript
$.ajax({
  url: "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  dataType: "json",
  success: function(data) {
    console.log(data);
  }
});
```

### PHP Request

```php
$ch = curl_init("https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY");
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

uri = URI("https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY")
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
  .uri(URI.create("https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY"))
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
var json = await client.GetStringAsync("https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY");
```

### Go Request

```go
package main

import (
  "encoding/json"
  "net/http"
)

req, _ := http.NewRequest("GET", "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY", nil)
req.Header.Set("Authorization", "Bearer YOUR_API_KEY")
res, _ := http.DefaultClient.Do(req)
defer res.Body.Close()
var data map[string]any
json.NewDecoder(res.Body).Decode(&data)
```

### Swift Request

```swift
import Foundation

var req = URLRequest(url: URL(string: "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY")!)
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
  curl_easy_setopt(c, CURLOPT_URL, "https://api-v1.whoisdatacenter.com/api/v2/download/nrd?date=2026-05-14&apiKey=YOUR_API_KEY");
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
  curl_easy_perform(c);
  curl_easy_cleanup(c);
}
```

