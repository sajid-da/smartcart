import streamlit as st
import pandas as pd
import requests
from pyzbar import pyzbar
import cv2
import numpy as np
import os
import random

DB_FILE = 'products.csv'
OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product/{}.json"


def load_db():
    """Loads the product database from CSV."""
    if os.path.exists(DB_FILE):
        try:
            df = pd.read_csv(DB_FILE)
            if not df.empty:
                return df.set_index('barcode')
            return pd.DataFrame(columns=['name', 'price']).set_index(pd.Index([], name='barcode'))
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['name', 'price']).set_index(pd.Index([], name='barcode'))
        except Exception as e:
            st.error(f"Error loading database: {e}")
            return pd.DataFrame(columns=['name', 'price']).set_index(pd.Index([], name='barcode'))
    else:
        df = pd.DataFrame(columns=['barcode', 'name', 'price'])
        df.to_csv(DB_FILE, index=False)
        return df.set_index('barcode')

def save_product_to_db(barcode, name, price=None):
    """Saves or updates a product in the CSV database."""
    try:
        # Generate random price if not provided
        if price is None:
            price = round(random.uniform(70, 1000), 2)
        
        db = load_db().reset_index()
        
        # Check if product exists
        if barcode in db['barcode'].values:
            # Update existing product
            db.loc[db['barcode'] == barcode, ['name', 'price']] = [name, price]
            st.success(f"Updated price for '{name}' to ₹{price:.2f}")
        else:
            # Add new product
            new_product = pd.DataFrame({
                'barcode': [barcode],
                'name': [name],
                'price': [price]
            })
            db = pd.concat([db, new_product], ignore_index=True)
            st.success(f"Added '{name}' with price ₹{price:.2f}")
        
        # Save to CSV
        db.to_csv(DB_FILE, index=False)
        st.session_state.product_db = load_db()
    except Exception as e:
        st.error(f"Error saving to database: {e}")


# --- API Function ---
def fetch_product_from_off(barcode):
    """Fetches product details from OpenFoodFacts API."""
    try:
        # Check local DB first
        local_db = load_db()
        if barcode in local_db.index:
            product_info = local_db.loc[barcode].to_dict()
            return product_info

        response = requests.get(OFF_API_URL.format(barcode))
        response.raise_for_status()
        data = response.json()
        if data.get("status") == 1 and data.get("product"):
            product = data["product"]
            return {
                "name": product.get("product_name", "N/A"),
                "brand": product.get("brands", "N/A"),
                "category": product.get("categories", "N/A"),
                "ingredients": product.get("ingredients_text", "N/A"),
                "nutrition": {
                    "energy": product.get("nutriments", {}).get("energy-kcal_100g", "N/A"),
                    "fat": product.get("nutriments", {}).get("fat_100g", "N/A"),
                    "carbs": product.get("nutriments", {}).get("carbohydrates_100g", "N/A"),
                    "protein": product.get("nutriments", {}).get("proteins_100g", "N/A"),
                    "sugar": product.get("nutriments", {}).get("sugars_100g", "N/A"),
                    "salt": product.get("nutriments", {}).get("salt_100g", "N/A")
                },
                "price": None
            }
        return None
    except Exception as e:
        st.error(f"Error fetching product: {e}")
        return None

def decode_barcode(image):
    """Decodes barcodes from an uploaded image or camera frame."""
    try:
        if isinstance(image, np.ndarray):
            # If image is already a numpy array (from webcam)
            img = image
        else:
            
            filestr = image.read()
            npimg = np.frombuffer(filestr, np.uint8)
            
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not decode image. Please ensure it's a valid image file.")
                return None

        barcodes = pyzbar.decode(img)
        if barcodes:
           
            return barcodes[0].data.decode('utf-8')
        else:
            return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def analyze_diet_compatibility(product_info):
    """Analyzes if a product is suitable for different diet types."""
    if not product_info or not product_info.get('nutrition'):
        return None
    
    nutrition = product_info.get('nutrition', {})
    recommendations = []
    
    # Low Sugar Diet
    sugar = float(nutrition.get('sugar', 0))
    if sugar <= 5:
        recommendations.append("✅ Suitable for low-sugar diet")
    else:
        recommendations.append("❌ High in sugar")
    
    # Low Fat Diet
    fat = float(nutrition.get('fat', 0))
    if fat <= 3:
        recommendations.append("✅ Suitable for low-fat diet")
    else:
        recommendations.append("❌ High in fat")
    
    # High Protein Diet
    protein = float(nutrition.get('protein', 0))
    if protein >= 20:
        recommendations.append("✅ Good for high-protein diet")
    else:
        recommendations.append("❌ Low in protein")
    
    # Low Sodium Diet
    salt = float(nutrition.get('salt', 0))
    if salt <= 0.3:
        recommendations.append("✅ Suitable for low-sodium diet")
    else:
        recommendations.append("❌ High in sodium")
    
    return recommendations

def find_similar_products(current_product, product_db):
    """Finds similar products based on category."""
    if current_product is None or product_db.empty:
        return None
    
    similar_products = []
    current_category = current_product.get('category', '')
    
    # Find products in the same category
    for _, product in product_db.iterrows():
        if product.get('category') == current_category and product.get('name') != current_product.get('name'):
            similar_products.append({
                'name': product.get('name', 'N/A'),
                'price': product.get('price', 'N/A'),
                'category': product.get('category', 'N/A')
            })
            if len(similar_products) >= 3:  # Limit to 3 products
                break
    
    return similar_products

def process_barcode(barcode):
    """Process a scanned barcode and update product information."""
    if barcode in st.session_state.product_db.index:
        # Product exists in local DB
        product_info = st.session_state.product_db.loc[barcode].to_dict()
        product_info.update({
            'source': 'Local DB',
            'brand': 'Local Database',
            'category': 'Local Database',
            'ingredients': 'Local Database',
            'nutrition': {
                'energy': 'N/A',
                'fat': 'N/A',
                'carbs': 'N/A',
                'protein': 'N/A',
                'sugar': 'N/A',
                'salt': 'N/A'
            }
        })
        return product_info
    else:
        # Try OpenFoodFacts
        with st.spinner("Searching OpenFoodFacts..."):
            product_info = fetch_product_from_off(barcode)
            if product_info:
                product_info['source'] = 'OpenFoodFacts'
                # Save with auto-generated price
                save_product_to_db(barcode, product_info['name'])
                # Reload DB and get price
                st.session_state.product_db = load_db()
                if barcode in st.session_state.product_db.index:
                    product_info['price'] = st.session_state.product_db.loc[barcode, 'price']
                return product_info
            else:
                # Create new product with auto-generated price
                save_product_to_db(barcode, "Unknown Product")
                st.session_state.product_db = load_db()
                if barcode in st.session_state.product_db.index:
                    product_info = st.session_state.product_db.loc[barcode].to_dict()
                    product_info.update({
                        'source': 'Local DB',
                        'brand': 'Local Database',
                        'category': 'Local Database',
                        'ingredients': 'Local Database',
                        'nutrition': {
                            'energy': 'N/A',
                            'fat': 'N/A',
                            'carbs': 'N/A',
                            'protein': 'N/A',
                            'sugar': 'N/A',
                            'salt': 'N/A'
                        }
                    })
                    return product_info
    return None

st.set_page_config(page_title="SmartCart", layout="wide")
st.title("🛒 SmartCart")

if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'product_db' not in st.session_state:
    st.session_state.product_db = load_db()
if 'last_scanned_barcode' not in st.session_state:
    st.session_state.last_scanned_barcode = None
if 'last_product_info' not in st.session_state:
    st.session_state.last_product_info = None
if 'show_manual_entry' not in st.session_state:
    st.session_state.show_manual_entry = False
if 'use_camera' not in st.session_state:
    st.session_state.use_camera = False

col1, col2 = st.columns([2, 1]) 

with col1:
    st.header("Scan Product")
    
    st.session_state.use_camera = st.checkbox("Use Webcam", value=st.session_state.use_camera)
    
    product_info_placeholder = st.empty()
    
    if st.session_state.use_camera:
        
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        if st.button("Scan Barcode"):
            ret, frame = cap.read()
            if ret:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                
                barcode = decode_barcode(frame)
                if barcode:
                    st.info(f"Detected Barcode: {barcode}")
                    st.session_state.last_scanned_barcode = barcode
                    product_info = process_barcode(barcode)
                    if product_info:
                        st.session_state.last_product_info = product_info
                    else:
                        st.error("Failed to process product information.")
                else:
                    st.error("No barcode detected in the uploaded image.")
                    st.session_state.last_scanned_barcode = None
                    st.session_state.last_product_info = None
            else:
                st.error("Failed to access webcam")
        
        cap.release()
    else:
        
        uploaded_file = st.file_uploader("Upload Barcode Image", type=["png", "jpg", "jpeg"], key="file_uploader")
        
        if uploaded_file is not None:
            barcode = decode_barcode(uploaded_file)
            st.session_state.last_scanned_barcode = barcode
            st.session_state.last_product_info = None
            st.session_state.show_manual_entry = False

            if barcode:
                st.info(f"Detected Barcode: {barcode}")
                st.session_state.last_scanned_barcode = barcode
                product_info = process_barcode(barcode)
                if product_info:
                    st.session_state.last_product_info = product_info
                else:
                    st.error("Failed to process product information.")
            else:
                st.error("No barcode detected in the uploaded image.")
                st.session_state.last_scanned_barcode = None
                st.session_state.last_product_info = None
                st.session_state.show_manual_entry = False


    # --- Display Product Info / Manual Entry ---
    with product_info_placeholder.container():
        if st.session_state.last_product_info:
            st.subheader("Product Details")
            info = st.session_state.last_product_info
            st.markdown(f"**Name:** {info.get('name', 'N/A')}")
            st.markdown(f"**Brand:** {info.get('brand', 'N/A')}")
            st.markdown(f"**Category:** {info.get('category', 'N/A')}")
            st.markdown(f"**Price:** {f'₹{info.get("price"):.2f}' if info.get('price') is not None else 'N/A'}")
            st.markdown(f"**Source:** {info.get('source', 'N/A')}")
            
            # Display diet compatibility analysis
            diet_recommendations = analyze_diet_compatibility(info)
            if diet_recommendations:
                with st.expander("Diet Compatibility Analysis"):
                    for rec in diet_recommendations:
                        st.write(rec)
            
            # Display ingredients if available
            if info.get('ingredients'):
                with st.expander("Ingredients"):
                    st.write(info.get('ingredients'))
            
            # Display nutritional information if available
            if info.get('nutrition'):
                with st.expander("Nutritional Information (per 100g)"):
                    nutrition = info.get('nutrition', {})
                    st.markdown(f"**Energy:** {nutrition.get('energy', 'N/A')} kcal")
                    st.markdown(f"**Fat:** {nutrition.get('fat', 'N/A')}g")
                    st.markdown(f"**Carbohydrates:** {nutrition.get('carbs', 'N/A')}g")
                    st.markdown(f"**Protein:** {nutrition.get('protein', 'N/A')}g")
                    st.markdown(f"**Sugar:** {nutrition.get('sugar', 'N/A')}g")
                    st.markdown(f"**Salt:** {nutrition.get('salt', 'N/A')}g")
            
            # Display similar products
            similar_products = find_similar_products(info, st.session_state.product_db)
            if similar_products:
                with st.expander("Similar Products You Might Like"):
                    for product in similar_products:
                        st.markdown(f"**{product['name']}**")
                        st.markdown(f"Category: {product['category']}")
                        st.markdown(f"Price: ₹{product['price']:.2f}")
                        st.divider()

            if st.button("Add to Cart", key="add_cart_button"):
                # Get price from product info
                price = info.get('price')
                if price is None:
                    # Generate random price if not set
                    price = round(random.uniform(70, 1000), 2)
                    # Update product info with new price
                    info['price'] = price
                    # Save to database
                    save_product_to_db(st.session_state.last_scanned_barcode, info.get('name', 'Unknown Product'), price)
                
                cart_item = {
                    "barcode": st.session_state.last_scanned_barcode,
                    "name": info.get('name', 'N/A'),
                    "price": float(price),
                    "quantity": 1
                }
                
                # Check if item already in cart
                found = False
                for item in st.session_state.cart:
                    if item['barcode'] == cart_item['barcode']:
                        item['quantity'] += 1
                        found = True
                        break
                if not found:
                    st.session_state.cart.append(cart_item)
                st.success(f"Added {info.get('name', 'N/A')} to cart.")
                st.rerun()


        # --- Manual Entry Form ---
        if st.session_state.show_manual_entry and st.session_state.last_scanned_barcode:
            st.subheader(f"Add/Update Product: {st.session_state.last_scanned_barcode}")
            with st.form(key="manual_entry_form"):
                manual_name = st.text_input("Product Name", value=st.session_state.last_product_info.get('name', '') if st.session_state.last_product_info else '')
                manual_price = st.number_input("Product Price (₹)", min_value=0.01, format="%.2f", value=st.session_state.last_product_info.get('price') if st.session_state.last_product_info and st.session_state.last_product_info.get('price') is not None else 0.01)
                submit_manual = st.form_submit_button("Save to Local DB")

                if submit_manual:
                    if manual_name and manual_price > 0:
                        save_product_to_db(st.session_state.last_scanned_barcode, manual_name, manual_price)
                        # Update last_product_info with manually entered data
                        st.session_state.last_product_info = {
                            'name': manual_name,
                            'price': manual_price,
                            'source': 'Manual Entry'
                        }
                        st.session_state.show_manual_entry = False
                        st.rerun()
                    else:
                        st.error("Please enter both name and a valid price.")


with col2:
    st.header("Shopping Cart")
    cart_placeholder = st.empty()

    if not st.session_state.cart:
        cart_placeholder.info("Your cart is empty.")
    else:
        cart_df = pd.DataFrame(st.session_state.cart)
        cart_df['Total'] = cart_df['price'] * cart_df['quantity']

        # Display Cart Items with +/- buttons and remove button
        new_cart = []
        items_to_remove = [] # Keep track of indices to remove

        for i, item in enumerate(st.session_state.cart):
            item_cols = st.columns([3, 1, 1, 1, 1, 1]) # Name, Price, Qty-, Qty, Qty+, Remove
            item_cols[0].write(f"{item['name']}")
            item_cols[1].write(f"₹{item['price']:.2f}")
            if item_cols[2].button("-", key=f"dec_{i}"):
                if item['quantity'] > 1:
                    item['quantity'] -= 1
                    st.rerun()
                else:
                    # Mark for removal if quantity becomes 0
                    items_to_remove.append(i)

            item_cols[3].write(f"{item['quantity']}")

            if item_cols[4].button("+", key=f"inc_{i}"):
                item['quantity'] += 1
                st.rerun()

            if item_cols[5].button("🗑️", key=f"rem_{i}"):
                items_to_remove.append(i)

        # Process removals after iterating
        if items_to_remove:
             # Remove items in reverse order to avoid index issues
            for index in sorted(items_to_remove, reverse=True):
                 del st.session_state.cart[index]
            st.rerun()


        # Recalculate cart total after potential updates
        if st.session_state.cart: # Check if cart is not empty after removals
            cart_df = pd.DataFrame(st.session_state.cart)
            cart_df['Total'] = cart_df['price'] * cart_df['quantity']
            total_price = cart_df['Total'].sum()
            st.subheader(f"Total: ₹{total_price:.2f}")
        else:
            cart_placeholder.info("Your cart is empty.") # Update placeholder if cart becomes empty


    if st.session_state.cart:
        if st.button("Clear Cart", key="clear_cart"):
            st.session_state.cart = []
            st.rerun()

st.divider()
st.subheader("Local Product Database")
st.dataframe(st.session_state.product_db, use_container_width=True)

st.caption("Powered by Streamlit, OpenFoodFacts, pyzbar, OpenCV")
