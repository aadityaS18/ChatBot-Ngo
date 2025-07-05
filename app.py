from flask import Flask, request, jsonify
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Cohere

app = Flask(__name__)

# Load Cohere embeddings and FAISS vector store
embedding = CohereEmbeddings(
    cohere_api_key="zJNR0dOqvUypmGWWcCZJdyaH5WCa1uyViiF3qHv8",
    user_agent="your-app"
)

vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# Load Cohere LLM
llm = Cohere(
    cohere_api_key="zJNR0dOqvUypmGWWcCZJdyaH5WCa1uyViiF3qHv8",
    model="command",
    max_tokens=500
)

# Setup RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# Flask endpoint
@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    response = qa.run(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
