from __future__ import annotations
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np
import uuid


app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class Document:
  def __init__(self, text: str, metadata: Dict[str, str]):
    self.text = text
    self.metadata = metadata

class SearchResult:
  def __init__(self, score: float, document: Document):
    self.score = score
    self.document = document

class FilteredVectorStore:
  def __init__(self, embedding_model: SentenceTransformer):
    self.embedding_model = embedding_model
    self.documents = []
    self.embeddings = []

  def add_documents(self, documents: List[Document]):
    self.documents.extend(documents)
    texts = [doc.text for doc in documents]
    new_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
    self.embeddings.extend(new_embeddings)

  def search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None) -> List[SearchResult]:
    query_embedding = self.embedding_model.encode([query],convert_to_numpy=True)

    filtered_docs = []
    filtered_embeddings = []

    for doc, emb in zip(self.documents, self.embeddings):

      include_doc = True

      # Si existe filtro por metadata
      if metadata_filter is not None:

        # Por cada filtro, si valor no coincide, no se incluye
        for key, value, in metadata_filter.items():
          if doc.metadata.get(key) != value:
            include_doc = False
            break

      # Filtrar los documentos según sus metadatos
      if include_doc:
        filtered_docs.append(doc)
        filtered_embeddings.append(emb)

    if len(filtered_docs) == 0:
      return []

    # Cos calculo
    similarities = cosine_similarity(query_embedding,np.array(filtered_embeddings))[0]

    # Ordenados indice y resultados
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
      results.append(
          SearchResult(
              score=float(similarities[idx]),
              document=filtered_docs[idx]
          )
      )

    return results
  

# Instancia global del store 
vector_store = FilteredVectorStore(model)

# Nuestra base de datos
documents_db = {}

def chunking(text, chunk_size=400):
    # Pedazos del texto original 
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks

# Para que se requiera el metadata
class Metadata(BaseModel):
    author: str
    category: str
    source: str

# Lo que esperara para el POST
class DocumentRequest(BaseModel):
    text: str
    metadata: Metadata

# Uso de FilteredVectorStore
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    metadata_filter: Optional[Dict[str, str]] = None

# POST documents: Crear documentos 
@app.post("/documents")
def create_document(doc: DocumentRequest):

    # Generar ID único
    document_id = str(uuid.uuid4())

    # Guardar documento original
    documents_db[document_id] = {
        "text": doc.text,
        "metadata": doc.metadata.dict()
    }

    # Chunking si texto es grande
    if len(doc.text) > 500:
        chunks = chunking(doc.text)
    else:
        chunks = [doc.text]

    vector_documents = []

    # Crear documentos para vector store
    for chunk in chunks:

        metadata = doc.metadata.dict()

        # Agregar ID original
        metadata["document_id"] = document_id

        vector_documents.append(
            Document(
                text=chunk,
                metadata=metadata
            )
        )

    # Agregar chunks al vector store
    vector_store.add_documents(vector_documents)

    return {
        "message": "Documento agregado",
        "document_id": document_id,
        "chunks": len(chunks)
    }

# GET documents/{id}: Obtener un documento específico (completo, no solo fragmentos) 
@app.get("/documents/{document_id}")
def get_document(document_id: str):

    # Verificar si existe
    if document_id not in documents_db:
        return {
            "error": "Documento no encontrado"
        }

    # Regresar documento completo
    return {
        "document_id": document_id,
        "text": documents_db[document_id]["text"],
        "metadata": documents_db[document_id]["metadata"]
    }

# POST documents/search: Realiza la búsqueda semántica (incluye el porcentaje de similitud, el texto y los metadatos)
@app.post("/documents/search")
def search_documents(search_request: SearchRequest):

    # Búsqueda semántica
    results = vector_store.search(
        query=search_request.query,
        top_k=search_request.top_k,
        metadata_filter=search_request.metadata_filter
    )

    response = []

    # Resultados a JSON
    for r in results:

        response.append({
            "score": round(r.score * 100, 2),
            "text": r.document.text,
            "metadata": r.document.metadata
        })

    return {
        "query": search_request.query,
        "results": response
    }