import os
import datetime
import sqlalchemy

from google.cloud.sql.connector import Connector
from google.cloud import secretmanager

import requests

def get_secret(secret_id, project_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

def main():
    """
    Main function to fetch bill data and store it in Cloud SQL.
    """
    print("Starting bill data ingestion process...")

    # --- 1. Configuration ---
    project_id = os.environ.get("GCP_PROJECT_ID")
    db_user = os.environ.get("DB_USER")          # e.g., 'postgres'
    db_name = os.environ.get("DB_NAME")          # e.g., 'bills'
    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME") # e.g., '[PROJECT_ID]:[REGION]:[INSTANCE_NAME]'

    # Fetch secrets
    try:
        api_key = get_secret("CONGRESS_API_KEY", project_id)
        db_password = get_secret("DB_PASSWORD", project_id)
        print("Successfully fetched secrets.")
    except Exception as e:
        print(f"Error fetching secrets: {e}")
        return # Exit if secrets cannot be fetched

    # --- 2. Database Connection ---
    connector = Connector()

    def get_conn() -> sqlalchemy.engine.base.Connection:
        conn = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_password,
            db=db_name,
        )
        return conn

    try:
        pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=get_conn,
        )
        db_conn = pool.connect()
        print("Successfully connected to the database.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return # Exit if DB connection fails


    # --- 3. Create Table if it Doesn't Exist ---
    metadata = sqlalchemy.MetaData()
    bills_table = sqlalchemy.Table(
        "congressional_bills",
        metadata,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
        sqlalchemy.Column("bill_number", sqlalchemy.String(255), unique=True, nullable=False),
        sqlalchemy.Column("title", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("sponsor_name", sqlalchemy.String(255)),
        sqlalchemy.Column("introduced_date", sqlalchemy.Date),
        sqlalchemy.Column("latest_action_text", sqlalchemy.Text),
        sqlalchemy.Column("latest_action_date", sqlalchemy.Date),
        sqlalchemy.Column("bill_url", sqlalchemy.String(1024)),
        sqlalchemy.Column("fetched_at", sqlalchemy.DateTime, default=datetime.datetime.utcnow),
    )
    metadata.create_all(pool)
    print("Table 'congressional_bills' is ready.")


    # --- 4. Fetch Data from Congress.gov API ---
    # We'll fetch bills updated in the last day.
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    api_url = f"[https://api.congress.gov/v3/bill?fromDateTime=](https://api.congress.gov/v3/bill?fromDateTime=){yesterday}&sort=updateDate+desc"
    
    headers = {"X-Api-Key": api_key}
    
    print(f"Fetching data from URL: {api_url}")
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        bills = data.get("bills", [])
        print(f"Found {len(bills)} new or updated bills.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return # Exit if API call fails
    

    # --- 5. Insert or Update Data in Database ---
    insert_count = 0
    for bill in bills:
        # Check if bill already exists
        stmt_check = sqlalchemy.select(bills_table).where(bills_table.c.bill_number == bill['number'])
        existing_bill = db_conn.execute(stmt_check).fetchone()

        bill_data = {
            "bill_number": bill['number'],
            "title": bill['title'],
            "sponsor_name": bill.get('sponsors', [{}])[0].get('fullName', 'N/A'),
            "introduced_date": bill.get('introducedDate'),
            "latest_action_text": bill.get('latestAction', {}).get('text'),
            "latest_action_date": bill.get('latestAction', {}).get('actionDate'),
            "bill_url": bill.get('url')
        }

        if existing_bill:
            # If it exists, update it
            stmt_update = sqlalchemy.update(bills_table).where(bills_table.c.bill_number == bill['number']).values(bill_data)
            db_conn.execute(stmt_update)
        else:
            # If it doesn't exist, insert it
            stmt_insert = sqlalchemy.insert(bills_table).values(bill_data)
            db_conn.execute(stmt_insert)
            insert_count += 1
    
    # The database driver has its own transaction management.
    # We commit the changes to the database.
    db_conn.commit()

    print(f"Process complete. Inserted {insert_count} new bills. Updated {len(bills) - insert_count} bills.")

    # --- 6. Cleanup ---
    db_conn.close()
    connector.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()
