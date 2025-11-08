# G4_SOLR_MILVUS_PUJ_PLN
Taller de RAG SOLR y MILVUS para la clase de PLN de la PUJ

# Carga de la información a SOLR

Para cargar la información a SOLR siga estos pasos:
1. Ingrese al contenedor utilizando ```docker exec -it <id_del_contenedor> bash```
2. Verifique la existencia del libro jsonl dentro del contenedor utilizando ```ls /data/corpus```
3. Ejecute el siguiente comando:
```
curl -X POST -H "Content-Type: application/json" \
     --data-binary @/data/corpus/books_for_solr.jsonl \
     "http://localhost:8983/solr/violaciones_cev_core/update?commit=true"
     
```