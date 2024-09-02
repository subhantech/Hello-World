import tkinter as tk
import psycopg2
from tkinter import ttk
from datetime import date, timedelta

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="stockdata",
    user="postgres",
    password="Subhan$007",
    port="5432"
)
cur = conn.cursor()

# Fetch symbols from the daily_prices table
cur.execute("SELECT DISTINCT symbol FROM daily_prices")
symbols = [row[0] for row in cur.fetchall()]

# Create the alerts table if it doesn't exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20),
        condition TEXT,
        created_on DATE,
        triggered BOOLEAN DEFAULT FALSE
    )
""")
conn.commit()

# --- Functions ---

def create_alert():
    """Creates a new alert and inserts it into the database."""
    symbol = symbol_var.get()
    condition = condition_var.get()
    operator = operator_var.get()
    value = value_entry.get()

    if not all([symbol, condition, operator, value]):
        tk.messagebox.showwarning("Missing Input", "Please fill all fields.")
        return

    try:
        value = float(value)  # Ensure value is a number
    except ValueError:
        tk.messagebox.showerror("Invalid Input", "Value must be a number.")
        return

    created_on = date.today()
    condition_str = f"{condition} {operator} {value}"

    try:
        cur.execute(
            "INSERT INTO alerts (symbol, condition, created_on, triggered) VALUES (%s, %s, %s, %s)",
            (symbol, condition_str, created_on, False)
        )
        conn.commit()
        print("Alert created successfully!")
        display_created_alerts()  # Update the displayed alerts
    except psycopg2.Error as e:
        tk.messagebox.showerror("Database Error", f"Error creating alert: {e}")

def check_triggered_alerts():
    """Checks all untriggered alerts against the latest stock data."""
    cur.execute("SELECT id, symbol, condition, created_on FROM alerts WHERE triggered = FALSE")
    untriggered_alerts = cur.fetchall()


    for alert in untriggered_alerts:
        alert_id, symbol, condition, created_on = alert 
        condition_parts = condition.split()
        price_type, operator, value = condition_parts[0], condition_parts[1], float(condition_parts[2])

        cur.execute(f"SELECT {price_type} FROM daily_prices WHERE symbol = %s ORDER BY date DESC LIMIT 1", (symbol,))
        latest_price = cur.fetchone()

        if latest_price and eval(f"{latest_price[0]} {operator} {value}"):
            cur.execute("UPDATE alerts SET triggered = TRUE WHERE id = %s", (alert_id,))
            conn.commit()
    display_triggered_alerts()  # Update the displayed alerts

def delete_alert(alert_id):
    """Deletes an alert from the database."""
    try:
        cur.execute("DELETE FROM alerts WHERE id = %s", (alert_id,))
        conn.commit()
        print("Alert deleted successfully!")
        display_created_alerts()  # Update the displayed alerts
    except psycopg2.Error as e:
        tk.messagebox.showerror("Database Error", f"Error deleting alert: {e}")

def display_alerts(treeview, triggered=False):
    """Fetches and displays alerts in the given treeview."""
    treeview.delete(*treeview.get_children())
    cur.execute("SELECT id, symbol, created_on, condition FROM alerts WHERE triggered = %s ORDER BY created_on DESC", (triggered,))
    alerts = cur.fetchall()
    for alert in alerts:
        treeview.insert("", "end", values=alert)

def display_created_alerts():
    """Displays untriggered alerts."""
    display_alerts(created_alerts_treeview, triggered=False)

def display_triggered_alerts():
    """Displays triggered alerts."""
    display_alerts(triggered_alerts_treeview, triggered=True)

# --- GUI Setup ---

root = tk.Tk()
root.title("Stock Alerts")

# --- Alert Creation Frame ---
create_frame = tk.LabelFrame(root, text="Create New Alert")
create_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# Symbol
symbol_label = tk.Label(create_frame, text="Symbol:")
symbol_label.grid(row=0, column=0)
symbol_var = tk.StringVar()
symbol_entry = ttk.Combobox(create_frame, textvariable=symbol_var, values=symbols)
symbol_entry.grid(row=0, column=1)

# Condition
condition_label = tk.Label(create_frame, text="Condition:")
condition_label.grid(row=1, column=0)
condition_var = tk.StringVar(value="close")  # Default to closing price
condition_dropdown = ttk.Combobox(create_frame, textvariable=condition_var, values=["open", "close", "high", "low", "volume"])
condition_dropdown.grid(row=1, column=1)

# Operator
operator_label = tk.Label(create_frame, text="Operator:")
operator_label.grid(row=2, column=0)
operator_var = tk.StringVar(value=">")
operator_dropdown = ttk.Combobox(create_frame, textvariable=operator_var, values=[">", "<", ">=", "<=", "="])
operator_dropdown.grid(row=2, column=1)

# Value
value_label = tk.Label(create_frame, text="Value:")
value_label.grid(row=3, column=0)
value_entry = tk.Entry(create_frame)
value_entry.grid(row=3, column=1)

# Create Button
create_button = tk.Button(create_frame, text="Create Alert", command=create_alert)
create_button.grid(row=4, column=0, columnspan=2, pady=(5, 0))

# --- Alerts Display Frame ---
alerts_frame = tk.LabelFrame(root, text="Active Alerts")
alerts_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Untriggered Alerts
created_alerts_treeview = ttk.Treeview(alerts_frame, columns=("id", "symbol", "created_on", "condition"), show="headings")
created_alerts_treeview.heading("id", text="ID")
created_alerts_treeview.heading("symbol", text="Symbol")
created_alerts_treeview.heading("created_on", text="Created On")
created_alerts_treeview.heading("condition", text="Condition")
created_alerts_treeview.pack(side="left", fill="both", expand=True)

# Delete Button for Untriggered Alerts
delete_button = tk.Button(alerts_frame, text="Delete Alert", command=lambda: delete_alert(created_alerts_treeview.item(created_alerts_treeview.selection())['values'][0]) if created_alerts_treeview.selection() else None)
delete_button.pack(side="left", padx=(5, 0))

# Triggered Alerts
triggered_alerts_treeview = ttk.Treeview(alerts_frame, columns=("id", "symbol", "created_on", "condition"), show="headings")
triggered_alerts_treeview.heading("id", text="ID")
triggered_alerts_treeview.heading("symbol", text="Symbol")
triggered_alerts_treeview.heading("created_on", text="Created On")
triggered_alerts_treeview.heading("condition", text="Condition")
triggered_alerts_treeview.pack(side="right", fill="both", expand=True)

# --- Check and Display Alerts ---
check_triggered_alerts()
display_created_alerts()
display_triggered_alerts()

root.mainloop()

# Close the database connection
cur.close()
conn.close()
