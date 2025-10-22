from firebase_admin import credentials, firestore, initialize_app
from sentence_transformers import SentenceTransformer
import datetime, firebase_admin

# ✅ Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    initialize_app(cred)

db = firestore.client()
model = SentenceTransformer("all-MiniLM-L6-v2")  # 🧩 lightweight model

def generate_listing_embeddings():
    listings = db.collection("listings").stream()
    print("🌾 Generating embeddings for listings...\n")

    for doc in listings:
        data = doc.to_dict()
        listing_id = doc.id
        farmer_id = data.get("farmerId")

        # ⚠️ Skip listings without a valid farmer
        if not farmer_id:
            print(f"⚠️ Skipping {listing_id}: No farmerId found.")
            continue

        # 🌍 Fetch farmer’s location
        farmer_ref = db.collection("users").document(farmer_id).get()
        if farmer_ref.exists:
            farmer_data = farmer_ref.to_dict()
            location_data = farmer_data.get("location", {})

            # 🧠 Handle both dict and string formats
            if isinstance(location_data, dict):
                address = location_data.get("address", "")
                city = location_data.get("city", "")
                province = location_data.get("province", "")
                phone = location_data.get("phone", "")
            else:
                # If stored as string
                address = location_data
                city = province = phone = ""
        else:
            print(f"⚠️ Farmer not found for {listing_id}. Using listing location as fallback.")
            location_data = data.get("location", {})
            if isinstance(location_data, dict):
                address = location_data.get("address", "")
                city = location_data.get("city", "")
                province = location_data.get("province", "")
                phone = location_data.get("phone", "")
            else:
                address = location_data
                city = province = phone = ""

        # 🧩 Combine data for embedding
        text = (
            f"{data.get('cropName', '')}, "
            f"{data.get('category', '')}, "
            f"{address}, {city}, {province}, "
            f"price {data.get('pricePerUnit', '')}"
        )

        # 🔢 Generate embedding
        vector = model.encode(text).tolist()

        # 🗃️ Store in Firestore
        db.collection("embeddings").document(listing_id).set({
            "listingId": listing_id,
            "embedding": vector,
            "cropName": data.get("cropName"),
            "farmerId": farmer_id,
            "location": {
                "address": address,
                "city": city,
                "province": province,
            },
            "pricePerUnit": data.get("pricePerUnit"),
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        print(f"✅ Embedding stored for {listing_id} ({data.get('cropName', '')})")

    print("\n🎉 All embeddings updated successfully!")

if __name__ == "__main__":
    generate_listing_embeddings()
