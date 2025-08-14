import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import ComplexField, SearchIndex, SearchFieldDataType, SimpleField, SearchableField
from dotenv import load_dotenv

load_dotenv()

search_service_name = os.getenv("SEARCH_SERVICE_NAME")
search_api_key = os.getenv("SEARCH_API_KEY")
search_index_name = os.getenv("SEARCH_INDEX_NAME")

# Configurar os clientes de índice e busca
endpoint = f"https://{search_service_name}.search.windows.net"
credential = AzureKeyCredential(search_api_key)
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
search_client = SearchClient(endpoint=endpoint, index_name=search_index_name, credential=credential)

# Definir o índice
def create_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="name", type=SearchFieldDataType.String, searchable=True, filterable=True),
        SearchableField(name="description", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="price", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True)
    ]
    
    index = SearchIndex(name=search_index_name, fields=fields)
    result = index_client.create_index(index)
    print(f"Índice criado: {result.name}")
    
# Carregar dados no índice
def upload_documents():
    documents = [
        {"id": "1", "name": "Laptop", "description": "A high performance laptop", "price": 1500, "category": "Electronics"},
        {"id": "2", "name": "Coffee Maker", "description": "A coffee maker with timer", "price": 50, "category": "Home Appliances"},
        {"id": "3", "name": "Headphones", "description": "Noise-cancelling headphones", "price": 200, "category": "Electronics"}
    ]
    
    result = search_client.upload_documents(documents)
    print(f"Documentos carregados: {result}")
    
# Realizar uma consulta de busca
def search_documents():
    search_term = "timer" # Deixe vazio para apenas filtrar e preenchido para buscar
    filter_expression = "category eq 'Electronics'" # Deixe vazio para apenas buscar e preenchido para filtrar
    results = search_client.search(
        search_text=search_term
        # , filter=filter_expression
        )
    
    for result in results:
        # print(result)
        print(f"ID: {result['id']}, Name: {result['name']}, Description: {result['description']}, Price: {result['price']}, Category: {result['category']}")

# Executar a função index primeiro depois as outras
# Lembre-se de comentar a linha onde é chamado o método create_index() para que ele seja criado apenas uma vez. 
# A mesma coisa deve ser feita com o método de upload de dados, o “upload_documents”, caso a intenção não seja repetir os dados.
# create_index()
# upload_documents()
search_documents()