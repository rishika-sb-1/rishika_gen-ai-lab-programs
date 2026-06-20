# Program 9: Institution Details from Wikipedia Using Pydantic

# !pip install pydantic wikipedia-api requests beautifulsoup4

from pydantic import BaseModel
from typing import List, Optional
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import re

class InstitutionProfile(BaseModel):
    founder: Optional[str] = "Unknown"
    Established: Optional[int] = 0
    branches: List[str] = ["Unknown"]
    employee_count: Optional[int] = 0
    summary: str

def fetch_wikipedia_data(institution_name: str) -> dict:
    user_agent = "MyInstitutionInfoBot/1.0 (contact: your-email@example.com)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")

    page = wiki_wiki.page(institution_name)

    if not page.exists():
        raise ValueError(f"Wikipedia page for '{institution_name}' not found.")

    summary = page.summary[:300]

    wiki_url = f"https://en.wikipedia.org/wiki/{institution_name.replace(' ', '_')}"
    headers = {"User-Agent": user_agent}
    response = requests.get(wiki_url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    infobox = soup.find("table", {"class": "infobox"})

    data = {
        "founder": "Unknown",
        "Established": 0,
        "branches": ["Unknown"],
        "employee_count": 0,
        "summary": summary
    }

    if infobox:
        for row in infobox.find_all("tr"):
            header = row.find("th")
            value = row.find("td")

            if header and value:
                key = header.text.strip()
                val = value.text.strip()

                if key in ["Founder", "Founders", "Founder(s)"]:
                    data["founder"] = val

                elif key in ["Established", "Founded", "Formation"]:
                    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", val)
                    if match:
                        data["Established"] = int(match.group(0))

                elif key in ["Total staff", "Employees", "Staff"]:
                    match = re.search(r"(\d{3,5})", val)
                    if match:
                        data["employee_count"] = int(match.group(0))

                elif key in ["Address", "Location"]:
                    data["branches"] = [val.split(",")[0]]

    if data["founder"] == "Unknown":
        dbpedia_data = fetch_dbpedia_data(institution_name)
        data.update(dbpedia_data)

    return data

def fetch_dbpedia_data(institution_name: str) -> dict:
    dbpedia_url = f"https://dbpedia.org/data/{institution_name.replace(' ', '_')}.json"
    headers = {"User-Agent": "MyInstitutionInfoBot/1.0"}

    response = requests.get(dbpedia_url, headers=headers)

    if response.status_code != 200:
        return {}

    dbpedia_json = response.json()
    entity_url = f"http://dbpedia.org/resource/{institution_name.replace(' ', '_')}"

    data = {"founder": "Unknown"}

    if entity_url in dbpedia_json:
        entity = dbpedia_json[entity_url]

        if "http://dbpedia.org/ontology/foundedBy" in entity:
            data["founder"] = entity["http://dbpedia.org/ontology/foundedBy"][0]["value"]

    return data

def create_institution_profile(institution_name: str) -> InstitutionProfile:
    data = fetch_wikipedia_data(institution_name)

    profile = InstitutionProfile(
        founder=data["founder"],
        Established=data["Established"],
        branches=data["branches"],
        employee_count=data["employee_count"],
        summary=data["summary"]
    )

    return profile

if __name__ == "__main__":
    institution_name = input("Enter institution name: ")

    try:
        profile = create_institution_profile(institution_name)
        print(profile.model_dump_json(indent=2))
    except ValueError as e:
        print(e)
