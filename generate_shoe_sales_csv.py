import pandas as pd
import random
from datetime import datetime, timedelta

# ==================================================
# CONFIGURATION
# ==================================================
NUM_RECORDS = 1000
START_DATE = datetime(2024, 7, 14)

brands = ["Nike", "Adidas", "Puma", "Bata", "Reebok", "Skechers", "Woodland"]
shoe_types = ["Sports", "Casual", "Formal"]
store_locations = [
    "Ahmedabad", "Surat", "Vadodara", "Rajkot",
    "Mumbai", "Pune", "Delhi", "Bengaluru"
]
customer_genders = ["Male", "Female"]
payment_modes = ["Cash", "Card", "UPI", "Wallet"]

# ==================================================
# DATA GENERATION
# ==================================================
data = []

for i in range(1, NUM_RECORDS + 1):
    brand = random.choice(brands)
    shoe_type = random.choice(shoe_types)
    store = random.choice(store_locations)
    gender = random.choice(customer_genders)
    payment = random.choice(payment_modes)

    price = random.randint(2000, 7000)
    quantity = random.randint(1, 5)
    discount = random.choice([0, 5, 10, 15, 20, 25])

    total_sales = int(price * quantity * (1 - discount / 100))

    date = START_DATE + timedelta(days=random.randint(0, 180))

    data.append([
        i,
        date.strftime("%d-%m-%Y"),  # DD-MM-YYYY format
        brand,
        shoe_type,
        price,
        quantity,
        discount,
        total_sales,
        gender,
        store,
        payment
    ])

# ==================================================
# CREATE DATAFRAME
# ==================================================
columns = [
    "Order_ID",
    "Date",
    "Shoe_Brand",
    "Shoe_Type",
    "Price",
    "Quantity",
    "Discount",
    "Total_Sales",
    "Customer_Gender",
    "Store_Location",
    "Payment_Mode"
]

df = pd.DataFrame(data, columns=columns)

# ==================================================
# SAVE CSV
# ==================================================
df.to_csv("shoe_store_sales.csv", index=False)

print("✅ shoe_store_sales.csv generated successfully with 1000 records!")
print("📅 Date range starts from 14-07-2024")
