from firebase_admin import credentials, firestore, initialize_app
from sentence_transformers import SentenceTransformer
import datetime, firebase_admin

# Init Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    initialize_app(cred)

db = firestore.client()
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, local model

def generate_listing_embeddings():
    listings = db.collection("listings").stream()
    for doc in listings:
        data = doc.to_dict()
        listing_id = doc.id

        # Combine features for embedding
        text = f"{data.get('cropName', '')}, {data.get('category', '')}, {data.get('location', '')}, price {data.get('pricePerUnit', '')}"

        # Generate embedding
        vector = model.encode(text).tolist()

        # Store embedding
        db.collection("embeddings").document(listing_id).set({
            "listingId": listing_id,
            "embedding": vector,
            "cropName": data.get("cropName"),
            "location": data.get("location"),
            "pricePerUnit": data.get("pricePerUnit"),
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        print(f"✅ Embedding stored for {listing_id}")

if __name__ == "__main__":
    generate_listing_embeddings()
