from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import numpy as np
import time

# --- 1️⃣ Conexión a Milvus ---
connections.connect("default", host="milvus", port="19530")

# --- 2️⃣ Parámetros generales ---
collection_name = "violaciones_cev_vectors"
embedding_dim = 384  # Dimensión del modelo multilingual-MiniLM

# --- 3️⃣ Elimina colección previa si existe ---
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# --- 4️⃣ Define esquema ---
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="text_raw", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
]
schema = CollectionSchema(fields, description="Corpus embeddings multilingües")

# --- 5️⃣ Crear colección ---
collection = Collection(name=collection_name, schema=schema)
print(f"✅ Colección creada: {collection_name}")

# --- 6️⃣ Modelo de embeddings multilingüe ---
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# --- 7️⃣ Cargar corpus JSONL ---
data_path = "/data/corpus/books_preprocessed.jsonl"
texts, ids = [], []

with open(data_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        text = obj.get("text_raw", "")
        if text:
            ids.append(i)
            texts.append(text)

print(f"📚 Documentos leídos: {len(texts)}")

# --- 8️⃣ Generar embeddings ---
print("🧠 Generando embeddings multilingües...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# --- 9️⃣ Insertar en Milvus ---
print("📤 Insertando en Milvus...")
collection.insert([ids, texts, embeddings])
collection.flush()
print(f"✅ Insertados {len(texts)} documentos en Milvus.")

# --- 🔟 Crear índice (para búsquedas rápidas) ---
index_params = {
    "metric_type": "IP",  # Inner Product (cosine similarity)
    "index_type": "HNSW",  # Alta velocidad para búsquedas semánticas
    "params": {"M": 32, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)
print("✅ Índice creado correctamente.")

# --- 1️⃣1️⃣ Verificar búsqueda ---
collection.load()

query_text = "violación final"
query_vec = model.encode([query_text])

results = collection.search(
    data=query_vec,
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"ef": 128}},
    limit=5,
    output_fields=["text_raw"]
)

print("\n🔍 Resultados de búsqueda para:", query_text)
for hit in results[0]:
    print(f" - (score={hit.score:.3f}) {hit.entity.get('text_raw')[:120]}")

print("\n✅ Validación de índice y búsqueda completada.")
