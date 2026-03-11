from __future__ import annotations

import base64
import re
import time
from pathlib import Path

import pandas as pd
import requests


GRAPHQL_ENDPOINT = "https://www.ratemyprofessors.com/graphql"
HEADERS_TEMPLATE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://www.ratemyprofessors.com",
    "Authorization": "Basic dGVzdDp0ZXN0",
}
RATINGS_QUERY = """
query RatingsList($id: ID!, $cursor: String) {
  node(id: $id) {
    ... on Teacher {
      ratings(first: 20, after: $cursor) {
        edges {
          node {
            clarityRating
            class
            comment
            date
            difficultyRating
            legacyId
            ratingTags
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""


def post_with_retry(
    url: str,
    payload: dict,
    headers: dict[str, str],
    max_retries: int = 5,
) -> requests.Response:
    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            time.sleep(delay)
            delay *= 2
            if attempt == max_retries - 1:
                raise

    raise RuntimeError("Exceeded retry limit while fetching reviews.")


def legacy_to_relay(legacy_id: int | str) -> str:
    return base64.b64encode(f"Teacher-{legacy_id}".encode()).decode()


def clean_comment(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def load_data(teacher_id: int | str) -> list[dict]:
    headers = {
        **HEADERS_TEMPLATE,
        "Referer": f"https://www.ratemyprofessors.com/professor/{teacher_id}",
    }
    teacher_global_id = legacy_to_relay(teacher_id)
    all_reviews: list[dict] = []
    cursor = None

    while True:
        payload = {
            "query": RATINGS_QUERY,
            "variables": {"id": teacher_global_id, "cursor": cursor},
        }
        response = post_with_retry(GRAPHQL_ENDPOINT, payload, headers)
        data = response.json()

        reviews = data["data"]["node"]["ratings"]
        for edge in reviews["edges"]:
            node = edge["node"]
            node["comment"] = clean_comment(node.get("comment"))
            all_reviews.append(node)

        if not reviews["pageInfo"]["hasNextPage"]:
            break
        cursor = reviews["pageInfo"]["endCursor"]

    return all_reviews


def get_reviews_for_professors(teacher_ids: list[int] | list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for teacher_id in teacher_ids:
        reviews = load_data(teacher_id)
        for review in reviews:
            row = dict(review)
            row["profId"] = teacher_id
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    professors_path = project_dir / "all_professors_rmp.csv"
    output_path = project_dir / "rmp_all_schools_reviews_small.csv"

    df = pd.read_csv(professors_path)
    teacher_ids = df["legacyId"].tolist()
    reviews_df = get_reviews_for_professors(teacher_ids)
    reviews_df.to_csv(output_path, index=False)
    print(f"wrote {len(reviews_df)} reviews to {output_path}")


if __name__ == "__main__":
    main()
