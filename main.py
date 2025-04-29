import os
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.get_urls_updated import scrape_urls
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import PyPDF2
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Global session states
vector_store = None
chat_history = [AIMessage(content="Hello and welcome to Gleuhr. What can I help you with today?")]
max_depth = 5  # fixed
website_url = "https://gleuhr.com/"  # fixed

# --- Utility Functions ---

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_vectorstore_from_url(url, max_depth, pdf_text=None):
    try:
        os.makedirs('src/chroma', exist_ok=True)
        os.makedirs('src/scrape', exist_ok=True)

        documents = []
        if url:
            urls = scrape_urls(url, max_depth)
            valid_urls = [u for u in urls if u.startswith("http://") or u.startswith("https://")]
            loader = WebBaseLoader(valid_urls)
            documents = loader.load()

        if pdf_text:
            documents.append(Document(page_content=pdf_text, metadata={"source": "uploaded_pdf"}))

        if not documents:
            raise ValueError("No documents were created")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        document_chunks = text_splitter.split_documents(documents)

        if not document_chunks:
            raise ValueError("No document chunks were created")

        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

        vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding,
            persist_directory="src/chroma"
        )

        vector_store.persist()
        return vector_store, len(document_chunks)

    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        return None, 0

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=openai_api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation and the content retrieved, generate a relevant and accurate response.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=openai_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """## Gleuhr Chatbot Conversation Guidelines

### Bot Objective
The chatbot serves as Gleuhr's first point of contact, handling user inquiries, empathizing with their skin concerns, suggesting clinical treatments, qualifying leads primarily by location, and confirming expert follow-up.

---

### Communication Style
•⁠  ⁠*Warm & Friendly:* Maintain a conversational, empathetic, and reassuring tone.
•⁠  ⁠*Concise & Clear:* Provide direct and easily understandable information.
•⁠  ⁠*Affirmative & Encouraging:* Always highlight positive outcomes and reassure the user.
•⁠  ⁠*Credible & Trustworthy:* Frequently mention successful results (e.g., "thousands of satisfied clients") to build credibility.

---

### Chatbot Conversation Flow

#### Step 1: Greeting & Empathy
Start by warmly acknowledging the user's concern to immediately build trust.

*Example:*
"Hi [Name]! I completely understand how frustrating pigmentation can be. You're in the right place—Gleuhr has successfully helped thousands achieve clear, even-toned skin."

---

#### Step 2: Suggest Clinical Treatment & Qualify (Location)
Briefly suggest relevant clinical treatments emphasizing their effectiveness, and immediately qualify the user by asking for their location.

*Example:*
"We have specialized clinical treatments like our advanced Collagen Treatment, which deeply targets pigmentation and delivers excellent results even in stubborn cases. Could you please share your location? This will help us suggest the best way forward for you."

*Note:*
•⁠  ⁠Prioritize clinical treatments.
•⁠  ⁠Only mention products explicitly if requested by the user or if their location is clearly outside North India.

---

#### Step 3: Expert Follow-Up & Call Assurance (Conclusive)
Confirm clear understanding and assure the user that the most suitable skin expert will contact them shortly.

*Example:*
"Thanks for sharing these details! I've clearly understood your concern and have allocated your information to our most suitable senior skin expert. They'll reach out to you shortly to discuss a personalized solution. You're in great hands!"

---

### Important Qualification Questions
•⁠  ⁠*Primary (Always Ask):* Location (for practical clinic visits).

*Ask Only if Relevant:*
•⁠  ⁠Previous treatments tried (only if user explicitly mentions unsuccessful past treatments).
•⁠  ⁠Specific expectations or urgency (only if indicated by user context, e.g., "need quick results for a special event").

---

### General Guidelines
•⁠  ⁠Keep conversations focused, concise, and friendly.
•⁠  ⁠Avoid overly technical details unless specifically asked by the user.
•⁠  ⁠Always position Gleuhr treatments as credible, effective, and proven through client success stories.
•⁠  ⁠Avoid sounding robotic or sales-heavy; maintain genuine empathy and warmth.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Context: {context}"),
        ("human", "Answer:")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    ai_response = response.get('answer', '').split("Answer:", 1)[-1].strip()
    return ai_response

# --- API Models ---

class InitializeResponse(BaseModel):
    message: str
    total_documents: int

class UserInput(BaseModel):
    input_text: str

class ChatResponse(BaseModel):
    response_text: str

# --- API Endpoints ---

@app.post("/initialize", response_model=InitializeResponse)
async def initialize_chatbot():
    global vector_store

    try:
        # Load fixed PDF
        pdf_file_path = "pdf/Product Guide.pdf"
        pdf_text = None
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as f:
                pdf_text = extract_text_from_pdf(f)

        # Create vector store
        vector_store, len_docs = get_vectorstore_from_url(website_url, max_depth, pdf_text)

        if vector_store:
            return InitializeResponse(message="Initialization successful!", total_documents=len_docs)
        else:
            return JSONResponse(status_code=500, content={"message": "Failed to initialize."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Initialization failed: {str(e)}"})

from fastapi import HTTPException
from fastapi.responses import JSONResponse

@app.post("/chat", response_model=ChatResponse)
async def chat(user_input: UserInput):
    global chat_history, vector_store

    # Check if vector_store is initialized
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please initialize first by calling /initialize.")

    # Check if user_input has input_text
    if not user_input.input_text:
        raise HTTPException(status_code=422, detail="Field 'input_text' is required.")

    user_message = user_input.input_text.strip()

    # Extra check in case input is just spaces
    if not user_message:
        raise HTTPException(status_code=422, detail="Field 'input_text' cannot be empty.")

    # Get the response from your model
    response = get_response(user_message)

    # Update the chat history
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response))

    return ChatResponse(response_text=response)

@app.get("/")
async def root():
    return {"message": "FastAPI Chatbot is running 🚀"}
