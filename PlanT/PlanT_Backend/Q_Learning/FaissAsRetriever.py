import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class FaissAsRetriever:
    def __init__(self, db_path, csv_path):
        self.db_path = db_path
        self.csv_path = csv_path
        # Add some infomation in metadata
        self.metadata_columns = ['id', 'category', 'duration']
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli', # 임시 임베딩 모델
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
            )
        self.allow_dangerous_deserialization = True
        self.vectorstore = None
        self.encoding = 'utf-8'
    
    def load_or_create_vectorstore(self):
        """vectorstore가 로컬에 존재하면 load하고, 존재하지 않으면 새로 생성합니다."""
        if os.path.exists(self.db_path):
            self._load_vectorstore()
        else:
            self._create_and_save_vectorstore()
    
    def _load_vectorstore(self):
        self.vectorstore = FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=self.allow_dangerous_deserialization)
        print("Loaded existing FAISS database from local storage.")
    
    def _create_and_save_vectorstore(self):
        loader = CSVLoader(self.csv_path, metadata_columns=self.metadata_columns, encoding=self.encoding)
        docs = loader.load()
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.vectorstore.save_local(self.db_path)
        print("Created and saved new FAISS database to local storage.")
    
    def search(self, query):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not loaded. Call load_or_create_vectorstore() first.")
        # Create custom retriever instance
        retriever = self.vectorstore.as_retriever(search_type='mmr', search_kwargs= {'k': 100, 'fetch_k': 200, 'lambda_mult': 0.5})
        trendy_pois = retriever.invoke(query)

        return trendy_pois

    def faissRetriever(query):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_directory, 'vectorstore/faiss')
        csv_path = os.path.join(current_directory, 'total_poi_vectordb.csv')
        
        vectorstore_manager = FaissAsRetriever(db_path, csv_path)
        vectorstore_manager.load_or_create_vectorstore()
        
        results = vectorstore_manager.search(query)

        id_lst = []
        metadata_lst = []
        for i in results:
            id_lst.append(i.metadata['id'])
            metadata_lst.append(i.metadata)

        # poi_info = {
        #     "id": id_lst,
        #     "metadata": metadata_lst,
        # }

        return id_lst

## Output
query = "바다 여행" #query example
poi_result = FaissAsRetriever.faissRetriever(query) # query 넣으면 해당되는 poi id 리스트가 반환됩니다.
print(poi_result)