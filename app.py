import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# --- Initialization ---
# Create a Flask web server
app = Flask(__name__)
# Enable CORS to allow requests from your frontend
CORS(app)

# Load environment variables (like your GOOGLE_API_KEY)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- RAG Configuration ---
VECTOR_STORE_PATH = "vectordatabase"

# Define the Prompt Template
prompt_template = """
You are an assistant who works for my gym business, i want you to answer every query that the user asks and reply in brief, if asked anything out of context then
ask the user to contant the gym owner at the given phone number. The membership cost for 1st month is 2500rs then 1000rs from the second month.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# --- Pre-load RAG Components ---
# To avoid reloading on every request, we load these once when the app starts.
try:
    print("Loading vector store and embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_k=3)
    print("Components loaded successfully.")
except Exception as e:
    print(f"Error loading RAG components: {e}")
    db = None
    retriever = None

# --- API Endpoint ---
@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    API endpoint to receive a question and return a RAG-generated answer.
    """
    if not retriever:
        return jsonify({"error": "RAG components are not available."}), 500

    # Get the question from the JSON request body
    data = request.get_json()
    query = data.get("question")

    if not query:
        return jsonify({"error": "No question provided."}), 400

    print(f"Received query: {query}")

    # 1. Retrieve context
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Set up the LLM and Chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)
    chain = LLMChain(llm=llm, prompt=prompt)

    # 3. Generate the response
    try:
        response = chain.invoke({"context": context, "question": query})
        answer = response.get("text", "Sorry, I could not generate an answer.")
        print(f"Generated answer: {answer}")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return jsonify({"error": "Failed to generate an answer."}), 500


# To run this app locally for testing:
if __name__ == "__main__":
    app.run(debug=True, port=5000)