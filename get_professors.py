from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


SCHOOL_IDS = {
    "UCLA": "U2Nob29sLTEwNzU=",
    "UCB": "U2Nob29sLTEwNzI=",
    "UCSD": "U2Nob29sLTEwNzk=",
    "UCI": "U2Nob29sLTEwNzQ=",
    "UCSB": "U2Nob29sLTEwNzc=",
    "UCSC": "U2Nob29sLTEwNzg=",
    "UCD": "U2Nob29sLTEwNzM=",
    "UCR": "U2Nob29sLTEwNzY=",
}
SCHOOL_LINK_IDS = {
    "UCLA": "1075",
    "UCB": "1072",
    "UCI": "1074",
    "UCSD": "1079",
    "UCSB": "1077",
    "UCD": "1073",
    "UCSC": "1078",
    "UCR": "1076",
}
SCHOOLS = ["UCLA", "UCB", "UCI", "UCSD", "UCR", "UCSC", "UCSB", "UCD"]

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
TEACHER_SEARCH_QUERY = """
query TeacherSearch($schoolID: ID!, $text: String!, $cursor: String) {
  newSearch {
    teachers(
      after: $cursor
      first: 20
      query: { text: $text, schoolID: $schoolID, fallback: true }
    ) {
      edges {
        node {
          legacyId
          firstName
          lastName
          department
          avgRating
          numRatings
          wouldTakeAgainPercent
          avgDifficulty
          school {
            name
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      resultCount
    }
  }
}
"""


def get_professors(
    schools: list[str] | tuple[str, ...] = SCHOOLS,
    school_ids: dict[str, str] = SCHOOL_IDS,
    school_link_ids: dict[str, str] = SCHOOL_LINK_IDS,
) -> list[dict]:
    all_teachers: list[dict] = []
    for school in schools:
        headers = {
            **HEADERS_TEMPLATE,
            "Referer": (
                "https://www.ratemyprofessors.com/search/professors/"
                f"{school_link_ids[school]}?q=*"
            ),
        }
        cursor = None

        while True:
            payload = {
                "query": TEACHER_SEARCH_QUERY,
                "variables": {
                    "schoolID": school_ids[school],
                    "text": "",
                    "cursor": cursor,
                },
            }
            response = requests.post(
                GRAPHQL_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            teachers = data["data"]["newSearch"]["teachers"]
            for edge in teachers["edges"]:
                node = edge["node"]
                school_obj = node.get("school")
                node["school"] = school_obj.get("name", "") if isinstance(school_obj, dict) else (school_obj or "")
                all_teachers.append(node)

            if not teachers["pageInfo"]["hasNextPage"]:
                break
            cursor = teachers["pageInfo"]["endCursor"]

        print(f"done with {school}")
    return all_teachers


def build_professors_dataframe(all_teachers: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_teachers)
    return df[
        [
            "legacyId",
            "firstName",
            "lastName",
            "school",
            "department",
            "avgRating",
            "numRatings",
            "wouldTakeAgainPercent",
            "avgDifficulty",
        ]
    ]


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "all_professors_rmp.csv"
    all_teachers = get_professors()
    df = build_professors_dataframe(all_teachers)
    df.to_csv(output_path, index=False)
    print(f"wrote {len(df)} professors to {output_path}")


if __name__ == "__main__":
    main()
