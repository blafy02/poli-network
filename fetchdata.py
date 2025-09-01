import os
import requests
from google.cloud import bigquery
from google.oauth2 import service_account

# CONFIGURATION
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")  # Set this in your environment
BQ_PROJECT = os.environ.get("BQ_PROJECT")  # Set this in your environment
BQ_DATASET = os.environ.get("BQ_DATASET")  # Set this in your environment
BQ_TABLE = os.environ.get("BQ_TABLE")      # Set this in your environment
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # Path to service account JSON

# Congress.gov API endpoint
CONGRESS_API_URL = "https://api.congress.gov/v3/bill"

def fetch_bills(offset=0, limit=250):
    params = {
        "api_key": 0lOneIIOTPLOcvTC8e4n8VJnT9OhQVEPMx2lQAod,
        "offset": offset,
        "limit": limit,
        "format": "json"
    }
    resp = requests.get(CONGRESS_API_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("bills", [])

def get_existing_bill_ids(bq_client):
    query = f"SELECT billId FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
    query_job = bq_client.query(query)
    return set(row.billId for row in query_job)

def insert_new_bills(bq_client, bills, existing_ids):
    rows_to_insert = []
    for bill in bills:
        bill_id = bill.get("billId")
        if bill_id not in existing_ids:
            rows_to_insert.append({
                "billId": bill_id,
                "title": bill.get("title"),
                "congress": bill.get("congress"),
                "introducedDate": bill.get("introducedDate"),
                "billType": bill.get("billType"),
                "originChamber": bill.get("originChamber"),
                "url": bill.get("url")
            })
    if rows_to_insert:
        errors = bq_client.insert_rows_json(f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}", rows_to_insert)
        if errors:
            print("Encountered errors while inserting rows: ", errors)
        else:
            print(f"Inserted {len(rows_to_insert)} new bills.")
    else:
        print("No new bills to insert.")

def main():
    # Authenticate BigQuery client
    credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
    bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT)

    # Fetch existing bill IDs
    existing_ids = get_existing_bill_ids(bq_client)

    # Fetch new bills from Congress.gov
    offset = 0
    limit = 20
    while True:
        bills = fetch_bills(offset=offset, limit=limit)
        if not bills:
            break
        insert_new_bills(bq_client, bills, existing_ids)
        # Update existing_ids with newly inserted bills
        existing_ids.update(bill.get("billId") for bill in bills)
        offset += limit

if __name__ == "__main__":
    main()