---
title: Reverse WHOIS by Address
category: Reverse WHOIS
endpoint: GET https://api-v1.whoisdatacenter.com/api/v2/address
method: GET
source_url: https://whoisdatacenter.com/api-docs/#reverse-whois-by-address
---

# Reverse WHOIS by Address

## Description

Returns all domains linked to a registrant street address.

## Metadata

- **HTTP Method:** `GET`
- **Endpoint Path:** `https://api-v1.whoisdatacenter.com/api/v2/address`
- **Credits Deducted:** 5 Credits
- **Rate Limit:** 5 Credits
- **Authentication:** Required
- **Input Type:** Address
- **Output Type:** Domain list

## Request

`GET https://api-v1.whoisdatacenter.com/api/v2/address`

## Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `address` | `string` | **Yes** | Example value: `1600 Amphitheatre Parkway`. Pass as a URL-encoded query parameter. |
| `format` | `string` | No | Response format. One of `json`, `xml`, `csv`. Defaults to `json`. |

## Examples & Notes

*Find domains registered at 1600 Amphitheatre Parkway.*

## Code Examples

### cURL Request

```bash
# Address lookup — JSON
curl -X GET "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Accept: application/json"
```

### Python Request

```python
import requests

response = requests.get(
  "https://api-v1.whoisdatacenter.com/api/v2/address",
  params={
    "address": "1600 Amphitheatre Parkway"
  },
  headers={"Authorization": "Bearer YOUR_API_KEY"}
)
data = response.json()
print(data)
```

### Node.js Request

```javascript
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### JavaScript Request

```javascript
// Browser fetch — requires CORS
const res = await fetch(
  "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway",
  { headers: { "Authorization": "Bearer YOUR_API_KEY" } }
);
const data = await res.json();
```

### jQuery Request

```javascript
$.ajax({
  url: "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway",
  headers: { "Authorization": "Bearer YOUR_API_KEY" },
  dataType: "json",
  success: function(data) {
    console.log(data);
  }
});
```

### PHP Request

```php
$ch = curl_init("https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway");
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

uri = URI("https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway")
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
  .uri(URI.create("https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway"))
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
var json = await client.GetStringAsync("https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway");
```

### Go Request

```go
package main

import (
  "encoding/json"
  "net/http"
)

req, _ := http.NewRequest("GET", "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway", nil)
req.Header.Set("Authorization", "Bearer YOUR_API_KEY")
res, _ := http.DefaultClient.Do(req)
defer res.Body.Close()
var data map[string]any
json.NewDecoder(res.Body).Decode(&data)
```

### Swift Request

```swift
import Foundation

var req = URLRequest(url: URL(string: "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway")!)
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
  curl_easy_setopt(c, CURLOPT_URL, "https://api-v1.whoisdatacenter.com/api/v2/address?address=1600 Amphitheatre Parkway");
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
  curl_easy_perform(c);
  curl_easy_cleanup(c);
}
```

