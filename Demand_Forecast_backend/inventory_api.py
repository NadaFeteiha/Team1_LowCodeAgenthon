# inventory_api.py

from flask import Blueprint, jsonify
import psycopg2
import pandas as pd

inventory_api = Blueprint('inventory_api', __name__)

def get_db_connection():
    return psycopg2.connect(
   
    )

@inventory_api.route('/api/inventory', methods=['GET'])
def get_inventory():
    try:
        conn = get_db_connection()

        query = """
            SELECT 
                inventory_id,
                item_type,
                item_name,
                vendor_id,
                lead_time_days,
                avg_daily_consumption,
                minimum_required,
                maximum_capacity,
                initial_stock,
                unit_cost,
                expiry_date
            FROM inventory_master
            ORDER BY item_name ASC;
        """

        df = pd.read_sql(query, conn)
        conn.close()

        items = df.to_dict(orient='records')

        return jsonify({"success": True, "items": items})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
