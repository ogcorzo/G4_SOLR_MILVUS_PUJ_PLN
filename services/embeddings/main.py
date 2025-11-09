import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
import os
import time

# --- Configuración de conexión a Milvus ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

COLLECTION_NAME = "document_embeddings"

# --- Definición del esquema si no existe ---
def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"✅ La colección '{COLLECTION_NAME}' ya existe.")
        return Collection(COLLECTION_NAME)
    
    print(f"🧱 Creando colección '{COLLECTION_NAME}' ...")

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048)
    ]
    schema = CollectionSchema(fields, description="Embeddings multilingües de documentos")
    collection = Collection(COLLECTION_NAME, schema)
    return collection

# --- Carga de modelo de embeddings ---
print("🧠 Cargando modelo de embeddings multilingües...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- Crear colección ---
collection = create_collection()

# --- Cargar corpus ---
corpus_path = "/app/data/corpus.jsonl"
print(f"📄 Leyendo corpus desde {corpus_path}")
documents = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line.strip())
        if "id" in doc and "text" in doc:
            documents.append(doc)

print(f"✅ Se cargaron {len(documents)} documentos")

# --- Generar embeddings e insertar ---
ids = [d["id"] for d in documents]
texts = [d["text"] for d in documents]
print("⚙️ Generando embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

print("📥 Insertando datos en Milvus...")
collection.insert([ids, embeddings.tolist(), texts])
collection.flush()

# --- Crear índice para búsquedas ---
print("⚙️ Creando índice vectorial (IVF_FLAT + L2)...")
collection.create_index("embedding", {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
})
collection.load()

print("✅ Índice creado y colección lista para consultas.")

# --- Prueba de búsqueda ---
query_text = "violaciones de derechos humanos"
query_vec = model.encode([query_text])
results = collection.search(
    data=query_vec,
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"]
)

print(f"\n🔍 Resultados de búsqueda para: '{query_text}'\n")
for hits in results:
    for hit in hits:
        print(f"→ (score={hit.distance:.4f}) {hit.entity.get('text')[:100]}...")
