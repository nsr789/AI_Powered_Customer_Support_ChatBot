import logging
from typing import Any, Iterable, List

import requests
from requests import HTTPError
# ...

def fetch_products() -> List[dict[str, Any]]:
    """Hit FakeStore API and return raw JSON list. Fall back to static data."""
    try:
        resp = requests.get(_FAKESTORE_PRODUCTS_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]
    except (HTTPError, requests.RequestException) as err:
        log.warning("FakeStore API unavailable (%s). Using offline sample.", err)
        return [
            {
                "id": 1,
                "title": "Sample Tee",
                "description": "Offline sample product",
                "category": "clothing",
                "price": 9.99,
                "image": "",
            }
        ]
