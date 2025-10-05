## company overview section of the report
import faiss
import torch
import base64
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPProcessor, CLIPModel
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize CLIP model for embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_data_store={}

with open("image_data_store.json","r",encoding='utf-8') as f:
    image_data_store=json.load(f)


def embed_text(text):
    """Embed text using CLIP model"""
    inputs = processor(
        text=text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

class CustomCLIPEmbeddings:
    """Wrapper class to make CLIP embeddings compatible with LangChain"""
    def __init__(self, embed_text_func):
        self.embed_text_func = embed_text_func 
    
    def embed_query(self, query):
        return self.embed_text_func(query)
    
    def embed_documents(self, texts):
        return [self.embed_text_func(text) for text in texts]

# Initialize embeddings
embeddings = CustomCLIPEmbeddings(embed_text_func=embed_text)

# Load the vector store
try:
    vector_store = FAISS.load_local(
        folder_path='ReportVectorDB',
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector database loaded successfully!")
except Exception as e:
    print(f"❌ Error loading vector database: {e}")
    exit(1)

def docs_retrieval_query(query: str, k: int = 10):
    """Retrieve documents using CLIP embedding similarity search"""
    try:
        query_embedding = embeddings.embed_query(query)
        retrieved_docs = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
        return retrieved_docs
    except Exception as e:
        print(f"❌ Error during document retrieval: {e}")
        return []

# Enhanced Prompt Templates
COMPANY_OVERVIEW_MULTIMODAL_PROMPT = """
You are a professional business analyst creating a comprehensive Company Overview section for an annual report.

CONTEXT PROVIDED:
{context}

TASK:
Generate a detailed Company Overview section using EXACTLY this structure and format:

# COMPANY OVERVIEW

## 1.1 Basic Information
**Company Name:** [Extract the full legal company name]
**Establishment Date:** [Find founding/incorporation date with year]
**Headquarters Location:** [Provide complete headquarters address]

## 1.2 Core Competencies
**Innovation Advantages:**
- [List 3-5 key technological innovations and R&D strengths]
- [Highlight unique technical capabilities]

**Product Advantages:**
- [Describe flagship products and their market advantages]
- [Mention key technological differentiators]

**Brand Recognition:**
- [Describe market position and brand reputation]
- [Mention industry leadership and market share if available]

**Reputation Ratings:**
- [Note any industry awards, recognitions, or certifications]
- [Mention customer satisfaction or industry rankings]

## 1.3 Mission & Vision
**Mission Statement:** [Extract the company's core mission and purpose]
**Vision Statement:** [Extract the company's future vision and aspirations]
**Core Values:** [List 3-5 fundamental corporate values and principles]

ANALYSIS GUIDELINES:
1. Extract information PRECISELY from the provided context - do not invent or assume
2. For images: Analyze corporate logos, headquarters photos, technology demonstrations
3. For tables: Extract key milestones, technology specifications, or market positions
4. Use professional business language appropriate for an annual report
5. If specific information is unavailable, state "Information not specified in available data"
6. Focus on the most relevant and impactful information for each section
7. Maintain consistency in formatting and bullet point structure

IMPORTANT: Base all information SOLELY on the provided context. Do not use external knowledge.
"""

COMPANY_OVERVIEW_TEXT_ONLY_PROMPT = """
You are a professional business analyst creating a comprehensive Company Overview section for an annual report.

CONTEXT PROVIDED:
{context}

TASK:
Generate a detailed Company Overview section using EXACTLY this structure and format:

# COMPANY OVERVIEW

## 1.1 Basic Information
**Company Name:** [Extract the full legal company name]
**Establishment Date:** [Find founding/incorporation date with year]
**Headquarters Location:** [Provide complete headquarters address]

## 1.2 Core Competencies
**Innovation Advantages:**
- [List 3-5 key technological innovations and R&D strengths]
- [Highlight unique technical capabilities and patents]

**Product Advantages:**
- [Describe flagship products and their competitive advantages]
- [Mention technological differentiators and market leadership]

**Brand Recognition:**
- [Describe market position, brand strength, and industry reputation]
- [Note any dominant market positions or industry recognition]

**Reputation Ratings:**
- [Mention industry awards, certifications, or quality recognitions]
- [Note any customer satisfaction metrics or industry rankings]

## 1.3 Mission & Vision
**Mission Statement:** [Extract the company's core mission and business purpose]
**Vision Statement:** [Extract the company's future vision and strategic direction]
**Core Values:** [List 3-5 fundamental corporate values and ethical principles]

CONTENT GUIDELINES:
- Extract information PRECISELY from the provided context
- Use professional, concise business language
- Focus on the most relevant and verified information
- If information is unavailable, state "Information not specified in available data"
- Maintain consistent formatting throughout
- Ensure all bullet points are parallel and professional

CRITICAL: Base ALL information ONLY on the provided context. Do not invent or assume any details.
"""

def create_multimodal_context(query, retrieved_docs):
    """Create a comprehensive multimodal context with enhanced structure"""
    content = []
    
    # Add the main instruction
    content.append({
        "type": "text",
        "text": COMPANY_OVERVIEW_MULTIMODAL_PROMPT.format(context="")
    })
    
    # Separate document types
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    table_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "table"]
    
    # Add comprehensive text context with categorization
    if text_docs:
        content.append({
            "type": "text",
            "text": "\n" + "="*60 + "\nTEXT CONTENT ANALYSIS\n" + "="*60
        })
        
        # Categorize text content
        basic_info_texts = []
        competency_texts = []
        mission_texts = []
        other_texts = []
        
        for doc in text_docs:
            content_lower = doc.page_content.lower()
            page_info = f"[Page {doc.metadata['page']}]: {doc.page_content}"
            
            if any(keyword in content_lower for keyword in ['incorporat', 'founded', 'headquarters', 'established', 'santa clara']):
                basic_info_texts.append(page_info)
            elif any(keyword in content_lower for keyword in ['gpu', 'technology', 'innovation', 'ai', 'accelerated computing', 'patent']):
                competency_texts.append(page_info)
            elif any(keyword in content_lower for keyword in ['mission', 'vision', 'value', 'purpose', 'strategy']):
                mission_texts.append(page_info)
            else:
                other_texts.append(page_info)
        
        # Add categorized text
        if basic_info_texts:
            content.append({
                "type": "text",
                "text": "\nBASIC COMPANY INFORMATION:\n" + "\n".join(basic_info_texts)
            })
        
        if competency_texts:
            content.append({
                "type": "text", 
                "text": "\nTECHNOLOGY & COMPETENCIES:\n" + "\n".join(competency_texts)
            })
        
        if mission_texts:
            content.append({
                "type": "text",
                "text": "\nMISSION & STRATEGY:\n" + "\n".join(mission_texts)
            })
        
        if other_texts:
            content.append({
                "type": "text",
                "text": "\nADDITIONAL CONTEXT:\n" + "\n".join(other_texts)
            })
    
    # Add table context
    if table_docs:
        content.append({
            "type": "text",
            "text": "\n" + "="*60 + "\nTABLE DATA\n" + "="*60
        })
        
        for i, doc in enumerate(table_docs[:2]):  # Limit to 2 most relevant tables
            content.append({
                "type": "text",
                "text": f"Table {i+1} (Page {doc.metadata['page']}):\n{doc.page_content}"
            })
    
    # Add images with analysis guidance
    if image_docs and image_data_store:
        content.append({
            "type": "text", 
            "text": "\n" + "="*60 + "\nVISUAL CONTENT ANALYSIS\n" + "="*60
        })
        
        for doc in image_docs[:3]:  # Limit to 3 most relevant images
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                content.append({
                    "type": "text",
                    "text": f"\nImage from Page {doc.metadata['page']}: Analyze this image for:"
                })
                content.append({
                    "type": "text", 
                    "text": "- Corporate branding and identity elements"
                })
                content.append({
                    "type": "text",
                    "text": "- Technology demonstrations or product visuals"
                })
                content.append({
                    "type": "text", 
                    "text": "- Any textual content visible in the image"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data_store[image_id]}"
                    }
                })
    
    return HumanMessage(content=content)

def generate_company_overview_multimodal(retrieved_docs):
    """Generate Company Overview using multimodal model with enhanced prompts"""
    
    multimodal_message = create_multimodal_context("Generate Company Overview", retrieved_docs)
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            google_api_key=api_key
        )
        
        print("🤖 Generating comprehensive Company Overview with multimodal analysis...")
        response = llm.invoke([multimodal_message])
        
        return response.content
        
    except Exception as e:
        return f"❌ Error generating Company Overview: {str(e)}"

def generate_company_overview_text_only(retrieved_docs):
    """Generate Company Overview using text-only approach with enhanced prompt"""
    
    # Combine and structure text content
    context_parts = []
    
    for doc in retrieved_docs:
        doc_type = doc.metadata.get('type', 'text')
        page_num = doc.metadata.get('page', 'N/A')
        
        if doc_type == 'text':
            context_parts.append(f"[Page {page_num} - Text]: {doc.page_content}")
        elif doc_type == 'table':
            context_parts.append(f"[Page {page_num} - Table]: {doc.page_content}")
        elif doc_type == 'image':
            context_parts.append(f"[Page {page_num} - Image Reference]: {doc.page_content}")
    
    full_context = "\n\n".join(context_parts)
    
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=COMPANY_OVERVIEW_TEXT_ONLY_PROMPT
    )
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=api_key
        )
        
        formatted_prompt = prompt_template.format(context=full_context)
        print("🤖 Generating Company Overview from text context...")
        response = llm.invoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Optimized query for Company Overview
query = """
Retrieve comprehensive information about:
- Company legal name, incorporation details, founding history
- Headquarters location and corporate address
- Core technologies, innovations, GPU architecture, AI platforms
- Mission statements, vision statements, corporate values
- Brand identity, market positioning, industry recognition
- Key products, technological advantages, competitive differentiators
- Corporate logos, branding elements, headquarters images
- Technology demonstrations, product visuals, innovation highlights
"""

print("🔍 Searching for Company Overview information...")
retrieved_docs = docs_retrieval_query(query, 12)  # Get more documents for comprehensive coverage

if retrieved_docs:
    print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
    
    # Detailed document analysis
    doc_types = {}
    page_coverage = set()
    
    for doc in retrieved_docs:
        doc_type = doc.metadata.get('type', 'text')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        page_coverage.add(doc.metadata.get('page', 'unknown'))
    
    print("📊 Document Analysis:")
    for doc_type, count in doc_types.items():
        print(f"   - {doc_type.capitalize()}: {count} documents")
    print(f"   - Pages covered: {len(page_coverage)}")
    
    # Choose generation method
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get('type') == 'image']
    
    if image_docs and image_data_store:
        print("🖼️  Using multimodal analysis with enhanced prompts...")
        company_overview = generate_company_overview_multimodal(retrieved_docs)
    else:
        print("📄 Using enhanced text-only analysis...")
        company_overview = generate_company_overview_text_only(retrieved_docs)
    
    # Display results
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE COMPANY OVERVIEW")
    print("="*80)
    print(company_overview)
    print("="*80)
    
    # Save to file with timestamp
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"company_overview_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Documents analyzed: {len(retrieved_docs)}\n")
            f.write("="*50 + "\n")
            f.write(company_overview)
        
        print(f"💾 Company Overview saved to '{filename}'")
        
    except Exception as e:
        print(f"⚠️  Could not save to file: {e}")
        
else:
    print("❌ No relevant documents retrieved for Company Overview")

print("\n" + "="*80)
print("✅ Company Overview generation completed!")
print("="*80)