import base64
import os
import json
import gradio as gr
from dotenv import load_dotenv
import chromadb
import docx
import tempfile
import fitz  # PyMuPDF for PDF handling
import re
import uuid
import requests
import plotly.graph_objects as go


# Load environment variables from .env file
load_dotenv()

# Global variables
conversation_history = []
health_metrics = {
    'Physical Health': 3,
    'Mental Wellness': 3,
    'Sleep Quality': 3,
    'Nutrition': 3,
    'Exercise': 3
}

# Flag to track if documents are available in the knowledge base
documents_available = False

# Custom OpenAI-compatible client using requests
class CustomOpenAIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Conte+nt-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
    def embeddings_create(self, input, model):
        """Create embeddings using the OpenAI compatible API"""
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": input,
            "model": model
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to generate embeddings: {response.text}")
        
        result = response.json()
        return result
    
    def chat_completions_create(self, model, messages, max_tokens=2000, temperature=0.7, n=1):
        """Create chat completions using the OpenAI compatible API"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to generate chat completion: {response.text}")
        
        result = response.json()
        
        # Create a compatible response object
        class Choice:
            def __init__(self, message):
                self.message = type('Message', (), {'content': message['content']})
                
        class Response:
            def __init__(self, choices):
                self.choices = [Choice(choice['message']) for choice in choices]
        
        return Response(result['choices'])

# Initialize custom client
client = CustomOpenAIClient(
    base_url="https://api.studio.nebius.com/v1",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# Custom embedding function for ChromaDB
class CustomEmbeddingFunction:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        
    def __call__(self, texts):
        embeddings = []
        # Process in batches to avoid API limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            for text in batch_texts:
                result = self.client.embeddings_create(input=text, model=self.model_name)
                embeddings.append(result['data'][0]['embedding'])
        return embeddings

# Initialize ChromaDB
def init_chroma_db():
    # Create a persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Ensure collection exists with correct embedding function
    collection_name = "healthcare_knowledge"

    try:
        # First try to get the collection
        collection = chroma_client.get_collection(collection_name)
        # Check if documents exist
        global documents_available
        documents_available = collection.count() > 0
        print(f"Collection exists with {collection.count()} documents")
    except chromadb.errors.CollectionNotFoundError:
        # Create new collection if it doesn't exist
        collection = chroma_client.create_collection(collection_name)
        documents_available = False
        print("Created new collection")

    return chroma_client, collection

# Initialize Chroma client and collection
chroma_client, collection = init_chroma_db()

# Text extraction functions for various document types
def extract_text_from_pdf(file_path):
    """Extract text from PDF files using PyMuPDF (fitz)"""
    text_chunks = []
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Break text into paragraphs
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                # Skip empty paragraphs
                clean_paragraph = paragraph.strip()
                if clean_paragraph and len(clean_paragraph) > 20:  # Avoid tiny chunks
                    text_chunks.append(clean_paragraph)
        
        doc.close()
    except Exception as e:
        print(f"Error extracting PDF: {str(e)}")
        text_chunks.append(f"Error processing PDF: {str(e)}")
    
    return text_chunks

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_images(file_path):
    text_chunks = []
    try:
        image_base64 = encode_image_to_base64(file_path)

        response = client.chat_completions_create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this medical report image and extract all the information you see"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            max_tokens=2000  # Increased for better extraction
        )
        
        extracted_text = response.choices[0].message.content
        # Split response into meaningful chunks
        text_chunks = [chunk.strip() for chunk in extracted_text.split('\n\n') if chunk.strip()]
        
    except Exception as e:
        print(f"Error extracting Image: {str(e)}")
        text_chunks.append(f"Error processing Image: {str(e)}")
    
    return text_chunks

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    text_chunks = []
    
    try:
        doc = docx.Document(file_path)
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text and len(text) > 20:  # Avoid empty/tiny paragraphs
                text_chunks.append(text)
                
    except Exception as e:
        print(f"Error extracting DOCX: {str(e)}")
        text_chunks.append(f"Error processing DOCX: {str(e)}")
    
    return text_chunks

def extract_text_from_txt(file_path):
    """Extract text from TXT files"""
    text_chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by paragraphs or lines
        chunks = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        for chunk in chunks:
            clean_chunk = chunk.strip()
            if clean_chunk and len(clean_chunk) > 20:  # Avoid tiny chunks
                text_chunks.append(clean_chunk)
                
    except Exception as e:
        print(f"Error extracting TXT: {str(e)}")
        text_chunks.append(f"Error processing TXT: {str(e)}")
    
    return text_chunks

def clear_collection():
    """Safely clear the collection by getting all IDs and deleting them"""
    global documents_available
    
    try:
        # Get all document IDs
        results = collection.query(
            query_texts=[""],  # Empty query to match everything
            n_results=10000,   # Large number to get all documents
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        if results and 'ids' in results and results['ids']:
            # Delete documents by IDs
            doc_ids = results['ids'][0]
            if doc_ids:
                collection.delete(ids=doc_ids)
                print(f"Deleted {len(doc_ids)} documents from collection")
            documents_available = False
            return True
        else:
            print("No documents found to delete")
            documents_available = False
            return True
    except Exception as e:
        print(f"Error clearing collection: {str(e)}")
        return False

def process_uploaded_file(file, mode):
    """Process various types of documents and store chunks in Chroma DB"""
    global documents_available
    
    if file is None:
        return "No file selected."
    
    # Create a temporary file to work with
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    
    # Copy uploaded file to temp file
    with open(temp_file.name, 'wb') as f:
        with open(file.name, 'rb') as source_file:
            f.write(source_file.read())
    
    # Extract text based on file type
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext == '.pdf':
        text_chunks = extract_text_from_pdf(temp_file.name)
    elif file_ext == '.docx':
        text_chunks = extract_text_from_docx(temp_file.name)
    elif file_ext == '.txt':
        text_chunks = extract_text_from_txt(temp_file.name)
    elif file_ext in ('.jpg', '.png', '.jpeg', '.webp'):
        text_chunks = extract_text_from_images(temp_file.name)
    else:
        os.unlink(temp_file.name)
        return f"Unsupported file type: {file_ext}"
    
    # Clean up temp file
    os.unlink(temp_file.name)
    
    # Clear collection if replacing existing content
    if mode == "Replace existing content":
        if not clear_collection():
            return "Error clearing existing documents. Please try again."
    
    # Add documents to ChromaDB
    if text_chunks:
        ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
        metadata = [{"source": file.name, "chunk_index": i} for i in range(len(text_chunks))]
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(text_chunks), batch_size):
            batch_end = min(i + batch_size, len(text_chunks))
            collection.add(
                documents=text_chunks[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadata[i:batch_end]
            )
        
        documents_available = True
        return f"✓ File processed successfully\n- Document: {file.name}\n- Chunks added: {len(text_chunks)}\n- Total documents in DB: {collection.count()}"
    else:
        return f"No valid text content extracted from {file.name}"

def rewrite_query(user_input, conversation_history, model_name):
    """Rewrite user query based on conversation history"""
    # Compile the recent conversation history into a single string
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    
    # Create a more healthcare-focused prompt
    prompt = f"""Rewrite the following medical/healthcare query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core medical inquiry and intent of the original query
    - Expand with relevant medical terminology that might appear in healthcare documents
    - Include symptoms, conditions, medications or treatments mentioned in the conversation
    - Make the query more specific for medical knowledge retrieval
    - DONT EVER ANSWER the Original query, but focus on reformulating it for better context retrieval
    
    Return ONLY the rewritten query text, with no additional explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    
    # Generate the rewritten query
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat_completions_create(
        model=model_name,
        messages=messages,
        max_tokens=200,
        temperature=0.1,
    )
    
    rewritten_query = response.choices[0].message.content
    return rewritten_query

def get_relevant_context(query, top_k=10):
    """Retrieve relevant context from ChromaDB"""
    global documents_available
    
    # Skip context retrieval if no documents are available
    if not documents_available:
        print("No documents available in knowledge base, skipping context retrieval")
        return []
        
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and 'documents' in results and results['documents']:
            return results['documents'][0]  # Return list of documents
        return []
    except Exception as e:
        print(f"Error querying ChromaDB: {str(e)}")
        return []
    
def init_health_metrics():
    return {
        'Physical Health': {'score': 3, 'evidence': []},
        'Mental Wellness': {'score': 3, 'evidence': []},
        'Sleep Quality': {'score': 3, 'evidence': []},
        'Nutrition': {'score': 3, 'evidence': []},
        'Exercise': {'score': 3, 'evidence': []}
    }

health_metrics = init_health_metrics()

def update_health_metrics(conversation, current_metrics):
    """Analyze conversation using LLM to update health metrics"""
    text = " ".join([msg['content'] for msg in conversation[-4:]])
    
    # Enhanced prompt with explicit formatting requirements
    prompt = f"""Analyze this health conversation and update metrics accordingly. 
    Use EXACTLY these metric names: Physical Health, Mental Wellness, Sleep Quality, Nutrition, Exercise.
    
    Conversation excerpt:
    {text}
    
    Return JSON format with:
    - metric: One of the exact metric names listed above
    - adjustment: Numeric value between -1.0 and +1.0
    - evidence: Short reason
    
    Example valid response:
    {{"assessments": [{{"metric": "Physical Health", "adjustment": -0.5, "evidence": "Mentioned chronic pain"}}]}}
    
    Return ONLY valid JSON. No extra text or formatting."""
    
    try:
        response = client.chat_completions_create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400  
        )

        raw_response = response.choices[0].message.content
        
        # Clean response to extract valid JSON
        json_str = raw_response.replace("```json", "").replace("```", "").strip()
        result = json.loads(json_str)
        
        updated_metrics = {k: {'score': v['score'], 'evidence': v['evidence'].copy()} 
                         for k, v in current_metrics.items()}
        
        for assessment in result.get('assessments', []):
            metric = assessment.get('metric', '')
            adjustment = assessment.get('adjustment', 0)
            
            if metric in updated_metrics and isinstance(adjustment, (int, float)):
                # Ensure score stays within 1-5 range
                new_score = updated_metrics[metric]['score'] + adjustment
                updated_metrics[metric]['score'] = max(1.0, min(5.0, new_score))
                updated_metrics[metric]['evidence'].append(
                    f"{assessment.get('evidence', '')} (Δ{adjustment:.1f})"
                )
                print(f"Updated {metric}: {new_score:.1f}")  # Debug logging
            else:
                print(f"Ignored invalid metric update: {assessment}")

        return updated_metrics
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed. Raw response: {raw_response}")
        return current_metrics
    except Exception as e:
        print(f"Metric analysis error: {str(e)}")
        return current_metrics

def create_wellness_chart(metrics):
    """Create an interactive wellness dashboard with precise scores"""
    categories = ['Physical Health', 'Mental Wellness', 'Sleep Quality', 'Nutrition', 'Exercise']
    scores = [metrics[c] for c in categories]
    
    # Create figure with exact values
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker_color=['#3498DB', '#2ECC71', '#F1C40F', '#E74C3C', '#9B59B6'],
        text=[f"{s:.1f}/5" for s in scores],  # Show decimal values
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Health Metrics Overview",
        xaxis=dict(
            range=[0,5], 
            title="Score",
            dtick=0.5  # Show half-point increments
        ),
        yaxis=dict(title="Category"),
        template="plotly_white",
        height=400
    )
    return fig

def get_health_tips(conversation, metrics):
    """Generate personalized tips using LLM for display in Gradio components"""
    # Convert metrics to readable format
    metric_summary = "\n".join([f"{k}: {v['score']}/5" for k, v in metrics.items()])
    
    prompt = f"""Generate personalized health advice based on these metrics:
    
    Health Metrics:
    {metric_summary}

    Recent Conversation Context:
    {conversation[-3:]}

    Create 3-5 practical recommendations with emojis. Use this format:
    - [Recommendation] [category-emoji] [reason from metrics]
    
    Example:
    - Take a 10-minute walk after meals 🚶‍♂️ (Physical Health: 2.5/5)
    
    Return ONLY the bullet points, no introduction or conclusion text."""
    
    try:
        response = client.chat_completions_create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        
        # Get the raw response
        return response.choices[0].message.content
    except Exception as e:
        print(f"Tips generation error: {str(e)}")
        return "❌ Could not generate tips. Please try another query."
    

def healthcare_chat(user_input, system_message, model_name, conversation_history):
    """Handle healthcare-focused conversation using the Nebius model"""
    global health_metrics
    conversation_history.append({"role": "user", "content": user_input})

    # Initialize response data
    response_data = {
        "original_query": user_input,
        "rewritten_query": "",
        "context": "",
        "response": "",
        "conversation_history": conversation_history,
        "doctor_recommendation": ""
    }

    # Rewrite query if we have conversation history
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_input, conversation_history, model_name)
        response_data["rewritten_query"] = rewritten_query
    else:
        rewritten_query = user_input

    # Retrieve relevant context
    relevant_context = get_relevant_context(rewritten_query)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        response_data["context"] = context_str

    # Append context to user input if available
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Medical Context:\n" + context_str

    # Update conversation history with context
    conversation_history[-1]["content"] = user_input_with_context

    # Construct messages for the model
    messages = [{"role": "system", "content": system_message}]
    for msg in conversation_history:
        messages.append({"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]})

    # Get response from the model
    response = client.chat_completions_create(
        model=model_name,
        messages=messages,
        max_tokens=2000,
    )

    # Extract assistant's response
    assistant_response = response.choices[0].message.content

    # Append response to conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})
    response_data["response"] = assistant_response

    # If we have context, determine doctor recommendation
    if relevant_context:
        doctor_type = analyze_doctor_type(relevant_context)
        response_data["doctor_recommendation"] = doctor_type
    
    health_metrics = update_health_metrics(conversation_history, health_metrics)
    response_data["health_metrics"] = health_metrics

    return response_data

# Add this after health_metrics definition

def reset_context():
    """Reset the conversation context"""
    global conversation_history, health_metrics
    conversation_history = []
    health_metrics = init_health_metrics()
    
    # Verify reset
    print("Reset Health Metrics:")
    for metric, data in health_metrics.items():
        print(f"- {metric}: {data['score']}")
    
    return f"Conversation reset successfully. Knowledge base contains {collection.count()} documents."

def reset_all():
    """Reset everything including the document collection"""
    global conversation_history, health_metrics, documents_available
    
    # Clear conversation history
    conversation_history = []
    
    # Reset health metrics
    health_metrics = init_health_metrics()
    
    # Clear the document collection
    clear_collection()
    documents_available = False
    
    return "Conversation and document collection reset successfully."

def analyze_doctor_type(medical_context):
    """Analyze medical context to recommend appropriate specialist"""
    if not medical_context:
        return ""
    
    context_str = "\n".join(medical_context)
    
    prompt = """Based on this medical information, identify the most appropriate medical specialist for the patient.
    Consider:
    1. Primary conditions and symptoms mentioned
    2. Current medications and treatments
    3. Medical history and risk factors
    
    Return ONLY the most relevant specialist type as a single word (e.g., 'Cardiologist', 'Neurologist', 'Gastroenterologist').
    If there's insufficient information for a specific recommendation, return 'Primary Care Physician'."""
    
    messages = [
        {"role": "system", "content": "You are a medical triage expert. Your task is to identify the most appropriate specialist based on the medical context provided."},
        {"role": "user", "content": f"Determine the most appropriate specialist for this medical context. Return ONLY the specialist name.\n\nMedical Context:\n{context_str}\n\nSpecialist:"}
    ]
    
    response = client.chat_completions_create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=50,
        temperature=0.1,
    )
    
    specialist = response.choices[0].message.content
    return specialist



def chat_interface(message, history, context_box, rewritten_query_box, doctor_box):
    """Handle user interaction and update UI components"""
    # Define healthcare-specific system message
    system_message = """
    You are a knowledgeable healthcare assistant specializing in providing reliable medical information.
    
    Guidelines:
    1. Focus on evidence-based medical information and treatment approaches
    2. Clearly indicate when information is general advice versus specific medical guidance
    3. Recommend appropriate consultation with healthcare professionals when needed
    4. Explain medical terminology in accessible language
    5. Do not diagnose conditions or prescribe treatments
    6. Be sensitive to patient concerns and medical context
    7. When referring to medical research, note limitations and relevance
    8. Make sure you act as a medical professional's assistant and should be only used by them
    
    Always maintain a professional, empathetic tone and prioritize patient safety.
    """

    # Get response from healthcare chat function
    response_data = healthcare_chat(
        message,
        system_message,
        "mistralai/Mixtral-8x22B-Instruct-v0.1-fast",
        conversation_history
    )

    # Update UI components
    context_display = "No relevant medical context found in knowledge base."
    if response_data["context"]:
        context_display = response_data["context"]

    rewritten_query_display = "Original query used."
    if response_data["rewritten_query"]:
        rewritten_query_display = response_data["rewritten_query"]
    
    doctor_recommendation = "Not enough information to recommend a specialist."
    if response_data["doctor_recommendation"]:
        doctor_recommendation = f"Recommended specialist: {response_data['doctor_recommendation']}"

    health_metrics = response_data["health_metrics"]
    health_tips = get_health_tips(conversation_history, health_metrics)

    display_metrics = {k: v['score'] for k, v in health_metrics.items()}
    health_chart = create_wellness_chart(display_metrics)
    
    print("Current Health Metrics:")
    for metric, data in health_metrics.items():
        print(f"- {metric}: {data['score']:.1f}")   

    return (
    response_data["response"], 
        context_display,
        rewritten_query_display,
        doctor_recommendation,
        health_chart,
        get_health_tips(conversation_history, health_metrics)
)

# Create the Gradio interface
def create_gradio_interface():
    # Initialize collection count for status
    collection_count = collection.count()
    global documents_available
    documents_available = collection_count > 0
    
    init_status = f"✓ System initialized successfully\n✓ ChromaDB connected\n✓ Documents in knowledge base: {collection_count}\n\nReady for queries!"
    
    # Gradio Blocks interface setup
    with gr.Blocks(theme='JohnSmith9982/small_and_pretty') as demo:
        gr.Markdown("""
        # BOCTOR
        This system helps answer medical queries using a knowledge base of healthcare information.
        Upload your medical documents to enhance the knowledge base.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # Chatbot interface
                chatbot = gr.Chatbot(height=500, show_label=False)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your medical query here...",
                    container=False,
                    lines=3
                )
                with gr.Row():
                    clear = gr.Button("Clear Conversation", size="sm")
                    submit = gr.Button("Submit Query", size="sm")

                with gr.Accordion("Personal Wellness Tracker", open=True):
                    health_chart = gr.Plot(
                        label="Health Metrics Dashboard",
                        show_label=False
                    )
                    health_tips = gr.Markdown(
                    label="Personalized Health Tips",
                    value="Your tips will appear here..."
                    )

            with gr.Column(scale=2):
                # Knowledge base management
                with gr.Accordion("Healthcare Knowledge Base", open=True):
                    file_upload = gr.File(
                        label="Upload Medical Document",
                        file_types=[".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".webp"],
                        file_count="single"
                    )
                    upload_mode = gr.Radio(
                        choices=["Replace existing content", "Append to existing content"],
                        value="Replace existing content",  # Changed default to replace
                        label="Upload Mode"
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        value="No file uploaded yet.",
                        interactive=False
                    )
                    reset_db_btn = gr.Button("Clear Knowledge Base", size="sm")

                # Query and context details
                with gr.Accordion("Query Processing Details", open=True):
                    rewritten_query_box = gr.Textbox(
                        label="Enhanced Medical Query",
                        value="No query processed yet.",
                        interactive=False
                    )
                    context_box = gr.Textbox(
                        label="Retrieved Medical Context",
                        value="No medical context retrieved yet.",
                        interactive=False,
                        lines=8
                    )
                    doctor_box = gr.Textbox(
                        label="Specialist Recommendation",
                        value="No specialist recommendation yet.",
                        interactive=False
                    )
                
                # System status
                system_status = gr.Textbox(
                    label="System Status",
                    value=init_status,
                    interactive=False
                )

        # File upload handler
        file_upload.upload(
            process_uploaded_file,
            inputs=[file_upload, upload_mode],
            outputs=[upload_status]
        )
        
        # User message handling
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        # Bot response generation
        def bot(history, context_box, rewritten_query_box, doctor_box):
            user_message = history[-1][0]
            # Now expecting 6 return values
            bot_message, context, rewritten, doctor, health_chart, health_tips = chat_interface(
                user_message, history, context_box, rewritten_query_box, doctor_box
            )
            history[-1][1] = bot_message
            return history, context, rewritten, doctor, health_chart, health_tips
                
        # Clear conversation handler
        def clear_conversation():
            reset_status = reset_context()
            return (
                None, 
                "No medical context retrieved yet.", 
                "No query processed yet.",
                "No specialist recommendation available.",
                go.Figure(),  # Empty plot
                "No health data available yet.",
                f"Conversation cleared.\n{reset_status}"
            )
        
        # Event handlers
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, context_box, rewritten_query_box, doctor_box], 
            [chatbot, context_box, rewritten_query_box, doctor_box, health_chart, health_tips]
        )
        
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, context_box, rewritten_query_box, doctor_box], 
            [chatbot, context_box, rewritten_query_box, doctor_box, health_chart, health_tips]
        )
        
        clear.click(clear_conversation, None, 
           [chatbot, context_box, rewritten_query_box, doctor_box, health_chart, health_tips, system_status], 
           queue=False)
        
        # Add handler for reset_db_btn
        reset_db_btn.click(
            reset_all,
            None,
            [system_status]
        )
    
    return demo

# Launch the Gradio app
if __name__ == "__main__":
    demo = create_gradio_interface()
    port = int(os.environ.get("PORT", 10000))  
    demo.launch(share=True, server_port=port)
