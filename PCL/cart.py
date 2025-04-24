import streamlit as st
from pyzbar.pyzbar import decode
from PIL import Image
import requests
import sqlite3
import io

# Initialize session state variables
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'last_scanned_barcode' not in st.session_state:
    st.session_state.last_scanned_barcode = None
if 'last_product_info' not in st.session_state:
    st.session_state.last_product_info = None
if 'show_manual_entry' not in st.session_state:
    st.session_state.show_manual_entry = False

# DB setup
def create_connection():
    conn = sqlite3.connect("products.db")
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            barcode TEXT PRIMARY KEY,
            name TEXT,
            price REAL,
            brand TEXT,
            category TEXT,
            ingredients TEXT,
            nutrition TEXT,
            source TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_product_from_db(barcode):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE barcode=?", (barcode,))
    row = cursor.fetchone()
    conn.close()
    return row

def save_product_to_db(barcode, name, price, brand="Manual", category="Manual", ingredients="", nutrition="{}", source="Manual"):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO products (barcode, name, price, brand, category, ingredients, nutrition, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (barcode, name, price, brand, category, ingredients, nutrition, source))
    conn.commit()
    conn.close()

def fetch_product_from_api(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == 1:
            product = data["product"]
            return {
                "name": product.get("product_name", "Unknown"),
                "price": 0.0,
                "brand": product.get("brands", "Unknown"),
                "category": product.get("categories", "Unknown"),
                "ingredients": product.get("ingredients_text", ""),
                "nutrition": product.get("nutriments", {}),
                "source": "API"
            }
    return None

def decode_barcode(image):
    decoded_objects = decode(image)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')
    return None

def add_to_cart(product_info):
    st.session_state.cart.append(product_info)

def clear_cart():
    st.session_state.cart = []

def display_cart():
    st.subheader("Cart")
    total = 0.0
    for product in st.session_state.cart:
        st.write(f"{product['name']} - ₹{product['price']:.2f}")
        total += product['price']
    st.write(f"**Total: ₹{total:.2f}**")

def render_product_info(product_info):
    st.subheader("Product Info")
    st.write(f"**Name:** {product_info['name']}")
    st.write(f"**Brand:** {product_info['brand']}")
    st.write(f"**Category:** {product_info['category']}")
    st.write(f"**Ingredients:** {product_info['ingredients']}")
    st.write(f"**Price:** ₹{product_info['price']:.2f}")
    st.write(f"**Source:** {product_info['source']}")
    if st.button("Add to Cart"):
        add_to_cart(product_info)

def main():
    st.title("SmartCart - Barcode Scanner")
    create_table()

    uploaded_file = st.file_uploader("Upload Barcode Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        barcode = decode_barcode(image)
        if barcode:
            st.success(f"Detected Barcode: {barcode}")
            st.session_state.last_scanned_barcode = barcode

            product_info = fetch_product_from_api(barcode)

            if product_info:
                    save_product_to_db(barcode, product_info['name'], product_info['price'], product_info['brand'], product_info['category'], product_info['ingredients'], str(product_info['nutrition']), product_info['source'])
            else:
                product_row = get_product_from_db(barcode)
                if product_row:
                    product_info = {
                        "barcode": product_row[0],
                        "name": product_row[1],
                        "price": product_row[2],
                        "brand": product_row[3],
                        "category": product_row[4],
                        "ingredients": product_row[5],
                        "nutrition": product_row[6],
                        "source": product_row[7]
                    }
                else:
                    st.warning("Product not found. Please enter details manually.")
                    st.session_state.show_manual_entry = True
                    product_info = None

            if product_info:
                st.session_state.last_product_info = product_info
                render_product_info(product_info)
        else:
            st.error("No barcode detected. Try another image.")

    if st.session_state.show_manual_entry and st.session_state.last_scanned_barcode:
        with st.form("manual_entry_form"):
            name = st.text_input("Product Name")
            price = st.number_input("Price (in ₹)", min_value=0.0, step=0.5)
            if st.form_submit_button("Save"):
                save_product_to_db(st.session_state.last_scanned_barcode, name, price)
                st.session_state.last_product_info = {
                    "name": name,
                    "price": price,
                    "brand": "Manual Entry",
                    "category": "Manual Entry",
                    "ingredients": "N/A",
                    "nutrition": {},
                    "source": "Manual"
                }
                st.session_state.show_manual_entry = False
                st.rerun()

    if st.session_state.last_product_info:
        st.write("\n---\n")
        if st.button("Add to Cart Again"):
            add_to_cart(st.session_state.last_product_info)

    display_cart()
    if st.button("Clear Cart"):
        clear_cart()

if __name__ == "__main__":
    main()