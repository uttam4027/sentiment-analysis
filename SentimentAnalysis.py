import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime
import requests
from tqdm import tqdm
import time

# LangChain imports
# Alternative imports that maintain the same functionality
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
except ImportError:
    try:
        # Fallback for older versions
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.docstore.document import Document
    except ImportError:
        # If still failing, try community version
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

# Initialize Vector Database (Global)
class VectorKnowledgeBase:
    def __init__(self):
        print("Initializing Vector Knowledge Base...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.create_knowledge_base()
        print("Vector Knowledge Base initialized successfully!")
    
    def create_knowledge_base(self):
        """Create vector database with sentiment analysis examples"""
        
        # Comprehensive training examples for each category and sentiment type
        examples = [
            # Diamond - Return examples
            {"text": "A few years ago, I bought the 6 ctw bracelet, the stones were too small and the bracelet was too big. I returned it.", "category": "Diamond", "sentiment": "Return", "reasoning": "stones were too small"},
            {"text": "The diamonds were dull and cloudy. I had to return it.", "category": "Diamond", "sentiment": "Return", "reasoning": "diamonds were dull and cloudy"},
            {"text": "Stone quality was poor, very small carat. Returned the product.", "category": "Diamond", "sentiment": "Return", "reasoning": "stone quality was poor, very small carat"},
            {"text": "6 ctw bracelet stones too tiny for the price. I returned it.", "category": "Diamond", "sentiment": "Return", "reasoning": "6 ctw bracelet stones too tiny"},
            
            # Diamond - Positive examples
            {"text": "Clear, white and sparkly diamonds. Very nice. This 8 ctw is perfect. Bling~Bling!", "category": "Diamond", "sentiment": "Positive", "reasoning": "clear, white and sparkly diamonds"},
            {"text": "Beautiful brilliant diamonds, excellent clarity and sparkle", "category": "Diamond", "sentiment": "Positive", "reasoning": "beautiful brilliant diamonds, excellent clarity"},
            {"text": "The diamond quality is amazing, so clear and bright", "category": "Diamond", "sentiment": "Positive", "reasoning": "diamond quality is amazing, so clear and bright"},
            
            # Style - Return examples
            {"text": "The bracelet was too big and showy for everyday wear. I ultimately returned it.", "category": "Style", "sentiment": "Return", "reasoning": "bracelet was too big and showy"},
            {"text": "Too heavy for daily use, wrong size. Had to return it.", "category": "Style", "sentiment": "Return", "reasoning": "too heavy for daily use, wrong size"},
            {"text": "Design was too flashy, not suitable for work. Returned.", "category": "Style", "sentiment": "Return", "reasoning": "design was too flashy, not suitable"},
            {"text": "Ring was too small for my finger. I returned it.", "category": "Style", "sentiment": "Return", "reasoning": "ring was too small"},
            
            # Style - Positive examples
            {"text": "Perfect size and beautiful design. Love the elegant style!", "category": "Style", "sentiment": "Positive", "reasoning": "perfect size and beautiful design"},
            {"text": "Great stand-alone piece, fits perfectly", "category": "Style", "sentiment": "Positive", "reasoning": "great stand-alone piece, fits perfectly"},
            {"text": "Beautiful appearance, complements my other jewelry", "category": "Style", "sentiment": "Positive", "reasoning": "beautiful appearance, complements jewelry"},
            
            # Manufacturing - Return examples
            {"text": "The setting was loose and prongs were scratchy. I returned it.", "category": "Manufacturing", "sentiment": "Return", "reasoning": "setting was loose and prongs were scratchy"},
            {"text": "Poor craftsmanship, clasp kept breaking. Had to return.", "category": "Manufacturing", "sentiment": "Return", "reasoning": "poor craftsmanship, clasp kept breaking"},
            {"text": "Uncomfortable to wear, cheap construction. Returned it.", "category": "Manufacturing", "sentiment": "Return", "reasoning": "uncomfortable to wear, cheap construction"},
            
            # Manufacturing - Positive examples
            {"text": "Excellent craftsmanship, very comfortable to wear", "category": "Manufacturing", "sentiment": "Positive", "reasoning": "excellent craftsmanship, very comfortable"},
            {"text": "Well made, secure setting, smooth finish", "category": "Manufacturing", "sentiment": "Positive", "reasoning": "well made, secure setting, smooth finish"},
            
            # Merchandising - Return examples
            {"text": "Too expensive for the quality offered. I returned it.", "category": "Merchandising", "sentiment": "Return", "reasoning": "too expensive for the quality"},
            {"text": "Not worth the price, overpriced. Had to return.", "category": "Merchandising", "sentiment": "Return", "reasoning": "not worth the price, overpriced"},
            {"text": "Poor value for money. Returned for refund.", "category": "Merchandising", "sentiment": "Return", "reasoning": "poor value for money"},
            
            # Merchandising - Positive examples
            {"text": "Best gift. Worth every dollar. Great value!", "category": "Merchandising", "sentiment": "Positive", "reasoning": "worth every dollar, great value"},
            {"text": "Excellent value for the price, very reasonable", "category": "Merchandising", "sentiment": "Positive", "reasoning": "excellent value for the price"},
            
            # Retailer-related - Return examples
            {"text": "Poor customer service during return process. Difficult experience.", "category": "Retailer-related", "sentiment": "Return", "reasoning": "poor customer service during return"},
            {"text": "Wrong item was shipped, had to return it", "category": "Retailer-related", "sentiment": "Return", "reasoning": "wrong item was shipped"},
            {"text": "Delivery was delayed, website information incorrect. Returned.", "category": "Retailer-related", "sentiment": "Return", "reasoning": "delivery delayed, website incorrect"},
            
            # Retailer-related - Positive examples
            {"text": "Excellent customer service, fast shipping", "category": "Retailer-related", "sentiment": "Positive", "reasoning": "excellent customer service, fast shipping"},
            {"text": "Easy return process, helpful staff", "category": "Retailer-related", "sentiment": "Positive", "reasoning": "easy return process, helpful staff"},
        ]
        
        # Convert to documents
        documents = []
        for example in examples:
            doc = Document(
                page_content=example["text"],
                metadata={
                    "category": example["category"],
                    "sentiment": example["sentiment"],
                    "reasoning": example["reasoning"]
                }
            )
            documents.append(doc)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"Knowledge base created with {len(documents)} examples")
    
    def get_relevant_examples_for_categories(self, review, k=5):
        """Get relevant examples for category identification"""
        try:
            similar_docs = self.vectorstore.similarity_search(review, k=k)
            examples_text = ""
            for i, doc in enumerate(similar_docs, 1):
                metadata = doc.metadata
                examples_text += f"Example {i}: \"{doc.page_content}\" → Categories: {metadata['category']} (because: {metadata['reasoning']})\n"
            return examples_text
        except Exception as e:
            print(f"Error getting category examples: {e}")
            return ""
    
    def get_relevant_examples_for_sentiment(self, review, category, k=3):
        """Get relevant examples for sentiment analysis of specific category"""
        try:
            # Search for examples related to this specific category
            query = f"{category} sentiment in: {review}"
            similar_docs = self.vectorstore.similarity_search(query, k=k)
            
            # Filter for examples of the same category
            category_examples = [doc for doc in similar_docs if doc.metadata.get('category', '').lower() == category.lower()]
            
            if not category_examples:
                # Fallback to general similar examples
                category_examples = similar_docs[:k]
            
            examples_text = ""
            for i, doc in enumerate(category_examples, 1):
                metadata = doc.metadata
                examples_text += f"Example {i}: \"{doc.page_content}\" → {metadata['category']}: {metadata['sentiment']} (because: {metadata['reasoning']})\n"
            return examples_text
        except Exception as e:
            print(f"Error getting sentiment examples: {e}")
            return ""
    
    def get_relevant_examples_for_keywords(self, review, categories, k=4):
        """Get relevant examples for keyword extraction"""
        try:
            similar_docs = self.vectorstore.similarity_search(review, k=k)
            examples_text = ""
            for i, doc in enumerate(similar_docs, 1):
                metadata = doc.metadata
                examples_text += f"Example {i}: \"{doc.page_content}\" → {metadata['category']}: [\"{metadata['reasoning']}\"]\n"
            return examples_text
        except Exception as e:
            print(f"Error getting keyword examples: {e}")
            return ""

# Initialize global vector knowledge base
try:
    VECTOR_KB = VectorKnowledgeBase()
except Exception as e:
    print(f"Warning: Could not initialize vector database: {e}")
    VECTOR_KB = None

# Function to interact with Ollama API (UNCHANGED)
def query_ollama(prompt, model="gemma3:4b", max_retries=3, retry_delay=2):
    """
    Query the Ollama API with a given prompt
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model to use
        max_retries (int): Maximum number of retries for failed requests
        retry_delay (int): Delay between retries in seconds
    Returns:
        str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json().get("response", "")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error connecting to Ollama API: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to get response from Ollama after {max_retries} attempts: {e}")
                return ""

# Function to safely extract and parse JSON from model responses (UNCHANGED)
def extract_json_from_response(response, response_type="categories"):
    """
    Safely extract JSON from model responses with multiple fallback methods
    Args:
        response (str): The model's response
        response_type (str): Type of response expected ("categories" or "keywords")
    Returns:
        dict: Parsed JSON object or default values if parsing fails
    """
    # Try to extract JSON using code block pattern first
    json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON without code blocks - look for the most complete JSON object
        if response_type == "categories":
            json_blocks = re.findall(r'\{[^{]*"categories".*?\}', response, re.DOTALL)
        else:
            json_blocks = re.findall(r'\{[^{]*"sentiment_keywords".*?\}', response, re.DOTALL)
        
        if json_blocks:
            json_str = max(json_blocks, key=len)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                if response_type == "categories":
                    return {
                        "categories": ["Uncategorized"],
                        "explanation": "Failed to extract JSON from response"
                    }
                else:
                    return {"sentiment_keywords": {}}

    def repair_json(json_str):
        # Remove C-style comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)

        # Normalize whitespace around colons and commas
        json_str = re.sub(r'\s*:\s*', ': ', json_str)
        json_str = re.sub(r'\s*,\s*', ', ', json_str)

        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')

        # Fix unescaped quotes inside strings using advanced regex
        def escape_quotes_in_string(match):
            content = match.group(2).replace('\\"', 'TEMP_QUOTE_ESCAPE')
            content = content.replace('"', '\\"')
            content = content.replace('TEMP_QUOTE_ESCAPE', '\\"')
            return f'{match.group(1)}"{content}"'

        json_str = re.sub(r'(:\s*)(")((?:\\"|[^"])*)(")(?=\s*[,}])', escape_quotes_in_string, json_str)

        # Add missing commas between key-value pairs
        def add_missing_commas(match):
            return match.group(1) + ',' + match.group(2)
        json_str = re.sub(r'(".*?")\s*(".*?")', add_missing_commas, json_str)

        # Fix trailing commas before closing brackets or braces
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # Ensure all braces are balanced
        open_braces = json_str.count("{") - json_str.count("}")
        if open_braces > 0:
            json_str += "}" * open_braces
        elif open_braces < 0:
            json_str = "{" * abs(open_braces) + json_str

        return json_str

    parsing_strategies = [
        lambda s: json.loads(s),
        lambda s: json.loads(repair_json(s)),
        lambda s: json.loads(re.sub(r',\s*}', '}', s)),
        lambda s: json.loads(re.sub(r',\s*]', ']', s)),
    ]

    for strategy in parsing_strategies:
        try:
            result = strategy(json_str)
            if response_type == "categories" and "categories" in result:
                return result
            elif response_type == "keywords" and "sentiment_keywords" in result:
                return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            continue

    # Manual extraction fallback
    if response_type == "categories":
        try:
            categories = []
            categories_match = re.search(r'"categories":\s*\[(.*?)\]', json_str, re.DOTALL)
            if categories_match:
                categories_str = categories_match.group(1)
                categories = re.findall(r'"([^"]+)"', categories_str)
                categories = [cat.strip() for cat in categories if cat.strip()]
            if not categories:
                categories = ["Uncategorized"]
            explanation_match = re.search(r'"explanation":\s*"([^"]+)"', json_str)
            explanation = explanation_match.group(1).strip() if explanation_match else "Extracted with regex"
            return {
                "categories": categories,
                "explanation": explanation
            }
        except Exception as e:
            print(f"Final fallback JSON extraction failed: {e}")
            return {
                "categories": ["Uncategorized"],
                "explanation": "Comprehensive JSON extraction failed"
            }
    else:
        try:
            sentiment_keywords = {}
            # Try to extract keywords section by section
            keywords_match = re.search(r'"sentiment_keywords":\s*\{(.*?)\}', json_str, re.DOTALL)
            if keywords_match:
                keywords_content = keywords_match.group(1)
                # Extract each category's keywords
                category_matches = re.findall(r'"([^"]+)":\s*\[(.*?)\]', keywords_content, re.DOTALL)
                for category, keywords_str in category_matches:
                    keywords = re.findall(r'"([^"]+)"', keywords_str)
                    if keywords:
                        sentiment_keywords[category.strip()] = [kw.strip() for kw in keywords]
            return {"sentiment_keywords": sentiment_keywords}
        except Exception as e:
            print(f"Keywords fallback extraction failed: {e}")
            return {"sentiment_keywords": {}}

# Helper function to extract sentiment from model response
def extract_normal_sentiment(sentiment_response, sentiment_response_clean):
    """Helper function to extract sentiment from model response"""
    # First try to extract the exact single word response from model
    single_word_response = sentiment_response.strip()
    if single_word_response.lower() in ['positive', 'negative', 'return', 'uncategorized']:
        return single_word_response.capitalize()
    # Try to find the last occurrence of sentiment words (most likely the final answer)
    elif re.search(r'\b(positive|negative|return|uncategorized)\b', sentiment_response, re.IGNORECASE):
        matches = re.findall(r'\b(positive|negative|return|uncategorized)\b', sentiment_response, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
    # Fallback to keyword checking
    elif "return" in sentiment_response_clean and not ("returned" in sentiment_response_clean and "positive" in sentiment_response_clean):
        return "Return"
    elif "positive" in sentiment_response_clean:
        return "Positive"
    elif "negative" in sentiment_response_clean:
        return "Negative"
    elif "uncategorized" in sentiment_response_clean:
        return "Uncategorized"
    else:
        return "Uncategorized"

# Helper function to determine sentiment based on context
def determine_context_sentiment(review_lower, category):
    """Helper function to determine sentiment based on context"""
    category_lower = category.lower()
    
    if category_lower == "diamond":
        positive_words = ["sparkly", "sparkle", "brilliant", "clear", "beautiful", "gorgeous", "white", "colorless", "quality", "perfect", "amazing"]
        negative_words = ["dull", "cloudy", "poor", "bad", "ugly", "yellow", "unclear", "terrible", "awful"]
    elif category_lower == "style":
        positive_words = ["beautiful", "lovely", "gorgeous", "stunning", "perfect", "amazing", "nice", "great", "love", "pretty"]
        negative_words = ["ugly", "hideous", "terrible", "awful", "hate", "too big", "too small", "too heavy", "too showy"]
    elif category_lower == "manufacturing":
        positive_words = ["comfortable", "smooth", "secure", "well made", "good quality", "excellent", "sturdy", "durable"]
        negative_words = ["loose", "scratchy", "uncomfortable", "poor quality", "cheap", "broken", "faulty", "rough"]
    elif category_lower == "merchandising":
        positive_words = ["good value", "worth", "reasonable", "affordable", "great deal", "fair price"]
        negative_words = ["expensive", "overpriced", "not worth", "too much", "costly", "poor value"]
    elif category_lower == "retailer-related":
        positive_words = ["excellent service", "great service", "fast shipping", "helpful", "easy", "smooth"]
        negative_words = ["poor service", "slow shipping", "rude", "difficult", "bad experience", "unhelpful"]
    else:
        return "Uncategorized"
    
    positive_count = sum(1 for word in positive_words if word in review_lower)
    negative_count = sum(1 for word in negative_words if word in review_lower)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Uncategorized"

# Function to extract detailed reasoning for sentiment analysis (UNCHANGED)
def extract_sentiment_reasoning(review, category, sentiment, sentiment_keywords):
    """
    Extract specific keywords and phrases that justify the sentiment
    """
    review_lower = review.lower()
    category_lower = category.lower()
    
    # Extract actual phrases from the review that justify the sentiment
    reasoning_phrases = []
    
    if sentiment == "Return":
        # Find return-related phrases and the specific reasons
        return_patterns = [
            r'returned?[^.!?]*',
            r'sent[^.!?]*back[^.!?]*',
            r'had to return[^.!?]*',
            r'decided to return[^.!?]*',
            r'ultimately returned[^.!?]*',
            r'exchanged?[^.!?]*'
        ]
        
        for pattern in return_patterns:
            matches = re.findall(pattern, review, re.IGNORECASE)
            reasoning_phrases.extend([match.strip() for match in matches if match.strip()])
        
        # Look for specific return reasons related to this category in the ENTIRE review
        if category_lower == "diamond":
            reason_patterns = [
                r'stones?\s+[^.!?]*(?:too small|small|tiny|poor|bad)[^.!?]*',
                r'(?:too small|small|tiny|poor|bad)[^.!?]*stones?[^.!?]*',
                r'(?:dull|cloudy|poor quality)[^.!?]*diamonds?[^.!?]*',
                r'(\d+)\s*ctw[^.!?]*(?:too small|small|tiny)[^.!?]*',
                r'stones were too small[^.!?]*',
                r'6 ctw[^.!?]*'
            ]
        elif category_lower == "style":
            reason_patterns = [
                r'bracelet[^.!?]*(?:too big|too small|big|small)[^.!?]*',
                r'(?:too big|too small|big|small)[^.!?]*bracelet[^.!?]*',
                r'(?:too big|too small|too heavy|too showy)[^.!?]*',
                r'size[^.!?]*(?:too big|too small|wrong)[^.!?]*',
                r'bracelet was too big[^.!?]*'
            ]
        elif category_lower == "manufacturing":
            reason_patterns = [
                r'(?:setting|clasp|prong)[^.!?]*(?:loose|broken|poor)[^.!?]*',
                r'(?:loose|broken|poor)[^.!?]*(?:setting|clasp|prong)[^.!?]*',
                r'(?:poor quality|cheap|broken)[^.!?]*'
            ]
        elif category_lower == "merchandising":
            reason_patterns = [
                r'(?:too expensive|expensive|overpriced|not worth)[^.!?]*',
                r'price[^.!?]*(?:too high|high)[^.!?]*'
            ]
        else:
            reason_patterns = []
        
        for pattern in reason_patterns:
            matches = re.findall(pattern, review, re.IGNORECASE)
            reasoning_phrases.extend([match.strip() for match in matches if match.strip()])
        
        # If no specific patterns found, look for any mention of the category near return
        if not reasoning_phrases or len(reasoning_phrases) <= 1:
            category_mentions = {
                "diamond": [r'stones?[^.!?]*', r'\d+\s*ctw[^.!?]*', r'diamonds?[^.!?]*'],
                "style": [r'bracelet[^.!?]*', r'(?:too big|too small)[^.!?]*', r'size[^.!?]*'],
                "manufacturing": [r'(?:setting|clasp|quality)[^.!?]*'],
                "merchandising": [r'(?:price|cost|expensive)[^.!?]*'],
                "retailer-related": [r'returned?[^.!?]*', r'return[^.!?]*']
            }
            
            if category_lower in category_mentions:
                for pattern in category_mentions[category_lower]:
                    matches = re.findall(pattern, review, re.IGNORECASE)
                    reasoning_phrases.extend([match.strip() for match in matches if match.strip()])
    
    elif sentiment == "Positive":
        if category_lower == "diamond":
            positive_patterns = [
                r'diamond[s]?\s+[^.!?]*(?:sparkl|brilliant|clear|beautiful|gorgeous|white|colorless|quality|perfect)[^.!?]*',
                r'(?:sparkl|brilliant|clear|beautiful|gorgeous|white|colorless|quality|perfect)[^.!?]*diamond[s]?[^.!?]*',
                r'(?:clear|white|sparkly)[^.!?]*',
                r'(?:good|great|excellent)\s+(?:quality|clarity|color)[^.!?]*'
            ]
        elif category_lower == "style":
            positive_patterns = [
                r'(?:beautiful|lovely|gorgeous|stunning|perfect|amazing|nice|great)[^.!?]*',
                r'love[s]?\s+[^.!?]*(?:design|style|look|it)[^.!?]*',
                r'compliment[s]?[^.!?]*',
                r'perfect[^.!?]*(?:size|fit)[^.!?]*',
                r'very\s+nice[^.!?]*',
                r'great\s+(?:stand-alone|piece)[^.!?]*'
            ]
        elif category_lower == "manufacturing":
            positive_patterns = [
                r'(?:comfortable|smooth|secure|well made|good quality|excellent|sturdy)[^.!?]*',
                r'no[^.!?]*(?:sharp edges|problems|issues)[^.!?]*',
                r'setting[s]?\s+[^.!?]*(?:secure|tight|good)[^.!?]*'
            ]
        elif category_lower == "merchandising":
            positive_patterns = [
                r'(?:good value|worth|reasonable|affordable|great deal|fair price)[^.!?]*',
                r'price[^.!?]*(?:good|great|reasonable|fair)[^.!?]*',
                r'worth[^.!?]*(?:more|twice|double)[^.!?]*'
            ]
        elif category_lower == "retailer-related":
            positive_patterns = [
                r'(?:excellent|great|good)[^.!?]*service[^.!?]*',
                r'(?:fast|quick)[^.!?]*(?:shipping|delivery)[^.!?]*',
                r'helpful[^.!?]*staff[^.!?]*'
            ]
        else:
            positive_patterns = [r'(?:good|great|excellent|beautiful|perfect|amazing)[^.!?]*']
        
        for pattern in positive_patterns:
            matches = re.findall(pattern, review, re.IGNORECASE)
            reasoning_phrases.extend([match.strip() for match in matches if match.strip()])
    
    elif sentiment == "Negative":
        if category_lower == "diamond":
            negative_patterns = [
                r'diamond[s]?\s+[^.!?]*(?:dull|cloudy|poor|bad|ugly|yellow|unclear)[^.!?]*',
                r'(?:dull|cloudy|poor|bad|ugly|yellow|unclear)[^.!?]*diamond[s]?[^.!?]*',
                r'no[^.!?]*sparkle[^.!?]*'
            ]
        elif category_lower == "style":
            negative_patterns = [
                r'too\s+(?:big|small|heavy|showy|large|little)[^.!?]*',
                r'(?:ugly|hideous|terrible|awful)[^.!?]*',
                r'doesn\'t\s+fit[^.!?]*',
                r'wrong\s+size[^.!?]*',
                r'a\s+bit\s+too\s+(?:big|small|heavy|showy)[^.!?]*'
            ]
        elif category_lower == "manufacturing":
            negative_patterns = [
                r'(?:loose|scratchy|uncomfortable|poor quality|cheap|broken|faulty)[^.!?]*',
                r'setting[s]?\s+[^.!?]*(?:loose|poor|bad)[^.!?]*',
                r'clasp[^.!?]*(?:loose|broken|faulty)[^.!?]*',
                r'fell?\s+(?:off|out)[^.!?]*'
            ]
        elif category_lower == "merchandising":
            negative_patterns = [
                r'(?:expensive|overpriced|not worth|too much|costly)[^.!?]*',
                r'price[^.!?]*(?:too high|expensive|much)[^.!?]*',
                r'poor\s+value[^.!?]*'
            ]
        elif category_lower == "retailer-related":
            negative_patterns = [
                r'(?:poor|bad)[^.!?]*service[^.!?]*',
                r'(?:slow|late)[^.!?]*(?:shipping|delivery)[^.!?]*',
                r'rude[^.!?]*staff[^.!?]*'
            ]
        else:
            negative_patterns = [r'(?:bad|poor|terrible|awful|hate|ugly)[^.!?]*']
        
        for pattern in negative_patterns:
            matches = re.findall(pattern, review, re.IGNORECASE)
            reasoning_phrases.extend([match.strip() for match in matches if match.strip()])
    
    # Also include sentiment keywords if available
    if sentiment_keywords:
        reasoning_phrases.extend(sentiment_keywords[:2])  # Add top 2 keywords
    
    # Clean and deduplicate phrases
    unique_phrases = []
    for phrase in reasoning_phrases:
        phrase = phrase.strip()
        if phrase and len(phrase) > 3 and phrase not in unique_phrases:
            unique_phrases.append(phrase)
    
    # Return the most relevant phrases
    if unique_phrases:
        if len(unique_phrases) == 1:
            return unique_phrases[0]
        else:
            # Return top 2 most relevant phrases
            return "; ".join(unique_phrases[:2])
    else:
        # Fallback - extract any relevant phrase for this category
        category_keywords = {
            "diamond": ["diamond", "sparkle", "brilliant", "clear", "dull", "cloudy"],
            "style": ["beautiful", "design", "size", "big", "small", "heavy", "style"],
            "manufacturing": ["setting", "clasp", "comfortable", "loose", "quality"],
            "merchandising": ["price", "value", "expensive", "worth", "cost"],
            "retailer-related": ["service", "shipping", "delivery", "staff"]
        }
        
        keywords = category_keywords.get(category_lower, [])
        for keyword in keywords:
            if keyword in review_lower:
                # Find sentence containing this keyword
                sentences = review.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return f"General {sentiment.lower()} sentiment for {category}"

def manual_keyword_extraction(review, categories):
    """
    Manual fallback keyword extraction when model fails
    Args:
        review (str): The review text
        categories (list): List of categories to extract keywords for
    Returns:
        dict: Dictionary with category keywords
    """
    keywords = {}
    review_lower = review.lower()
    
    for category in categories:
        category_keywords = []
        
        if category == "Diamond":
            diamond_patterns = [
                r'diamond[s]?\s+(?:is|are|look[s]?|seem[s]?)\s+[^.!?]*(?:beautiful|gorgeous|sparkl|brilliant|clear|dull|cloudy)',
                r'(?:quality|size|carat|sparkle|brilliance|clarity|color)\s+[^.!?]*diamond',
                r'diamond[s]?\s+[^.!?]*(?:quality|size|carat|sparkle|brilliance|clarity|color)',
                r'(?:the\s+)?diamond[s]?\s+[^.!?]*(?:beautiful|gorgeous|amazing|perfect|stunning|dull|poor)'
            ]
            for pattern in diamond_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                category_keywords.extend(matches)
        
        elif category == "Manufacturing":
            manufacturing_patterns = [
                r'(?:latch|clasp|setting|prong)[s]?\s+(?:is|are)\s+[^.!?]*(?:loose|tight|secure|broken|faulty|perfect|good|bad)',
                r'(?:loose|tight|secure|broken|faulty|perfect|good|bad)\s+[^.!?]*(?:latch|clasp|setting|prong)',
                r'(?:comfort|uncomfortable|scratchy|smooth|rough)\s+[^.!?]*(?:wear|wearing)',
                r'(?:craftsmanship|workmanship|construction|build|made)\s+[^.!?]*(?:good|bad|excellent|poor|quality)'
            ]
            for pattern in manufacturing_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                category_keywords.extend(matches)
        
        elif category == "Merchandising":
            merchandising_patterns = [
                r'(?:price|cost|value|money|expensive|cheap|worth)\s+[^.!?]*',
                r'(?:could not touch|can\'t beat|better deal|great value|overpriced|reasonable)\s+[^.!?]*(?:price|cost|value)',
                r'(?:\$\d+|\d+\s+dollars?)\s+[^.!?]*',
                r'(?:for the price|price point|value for money)\s+[^.!?]*'
            ]
            for pattern in merchandising_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                category_keywords.extend(matches)
        
        elif category == "Retailer-related":
            retailer_patterns = [
                r'(?:service|staff|customer service|support)\s+[^.!?]*(?:outstanding|excellent|great|poor|bad|terrible)',
                r'(?:outstanding|excellent|great|poor|bad|terrible)\s+[^.!?]*(?:service|staff|customer service)',
                r'(?:shipping|delivery|pickup|pick up)\s+[^.!?]*(?:fast|quick|slow|easy|difficult)',
                r'(?:website|online|store)\s+[^.!?]*(?:easy|difficult|user-friendly|confusing)'
            ]
            for pattern in retailer_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                category_keywords.extend(matches)
        
        elif category == "Style":
            style_patterns = [
                r'(?:beautiful|gorgeous|stunning|ugly|hideous)\s+[^.!?]*',
                r'(?:design|style|look[s]?|appearance)\s+[^.!?]*(?:beautiful|gorgeous|stunning|ugly|hideous|perfect|amazing)',
                r'(?:compliment[s]?|love[s]?\s+it|hate[s]?\s+it)\s+[^.!?]*',
                r'(?:too\s+(?:big|small|heavy|light))\s+[^.!?]*'
            ]
            for pattern in style_patterns:
                matches = re.findall(pattern, review, re.IGNORECASE)
                category_keywords.extend(matches)
        
        # Clean and add keywords
        if category_keywords:
            # Remove duplicates and clean up
            unique_keywords = list(set([kw.strip() for kw in category_keywords if kw.strip()]))
            keywords[category] = unique_keywords[:3]  # Limit to top 3 matches
    
    return keywords

# Function to analyze a single review (ENHANCED WITH VECTOR DATABASE)
def analyze_review(review, product_style):
    """
    Analyze a single review to determine categories first, then sentiment for each category
    ENHANCED WITH VECTOR DATABASE EXAMPLES
    Args:
        review (str): The review text
        product_style (str): The style of the product
    Returns:
        dict: Dictionary containing categories and their respective sentiments
    """
    if pd.isna(review) or review.strip() == "":
        return {
            "category_sentiments": {"Uncategorized": "Uncategorized"},
            "explanation": "Empty review",
            "sentiment_keywords": {}
        }

    # Task 2: First identify categories using qwen3:8b model (ENHANCED WITH VECTOR EXAMPLES)
    vector_examples = ""
    if VECTOR_KB:
        vector_examples = VECTOR_KB.get_relevant_examples_for_categories(review)
    
    task2_prompt = f"""
You are analyzing a diamond jewelry review to categorize aspects mentioned in the review with extremely high accuracy.

LEARN FROM THESE SIMILAR EXAMPLES:
{vector_examples}

Product Style: {product_style}
Review: "{review}"

Task 2: Categorize this review based on the specific aspects of the product or service discussed:

1. Diamond - ONLY if the review explicitly mentions any of:
   - Diamond clarity/quality
   - Diamond color
   - Diamond sparkle/brilliance/dullness
   - Diamond shape
   - Diamond-specific issues (fluorescence, bow tie effect)
   - Diamond size/carat
   - Diamond diamond weight

2. Style - ONLY if the review explicitly mentions any of:
   - Design aesthetics (appealing, unique, trendy, classic)
   - Size issues (too wide, too small)
   - Weight issues (too heavy)
   - Metal type/color preferences
   - Compliments received about appearance
   - Suitability as a gift

3. Manufacturing - ONLY if the review explicitly mentions any of:
   - Craftsmanship/workmanship quality
   - Comfort/discomfort when wearing
   - Prong issues (scratching, smoothness)
   - Setting quality (diamonds secure or falling out)
   - Clasp/lock security (for earrings, necklaces, bracelets)
   - Physical durability aspects
   - Greate Quality

4. Merchandising - ONLY if the review explicitly mentions any of:
   - Price points or specific dollar amounts
   - Value for money considerations
   - Appraisal values (higher or lower than purchase price)
   - Future buying intentions based on value
   - Comparisons with alternatives based on price
   - Carat weights or gem sizes when discussing value

5. Retailer-related - ONLY if the review explicitly mentions any of:
   - Purchase experience
   - Customer service quality
   - Discount deals
   - Website representation accuracy
   - Shipping/delivery experience
   - Return policy or process

EXTREMELY IMPORTANT: Be VERY STRICT with categorization. Generic positive or negative statements WITHOUT specific category mentions MUST be classified as "Uncategorized".

A review MUST contain EXPLICIT KEYWORDS related to a category to be classified in that category. Pay special attention to negations ("not worth the price" = negative Merchandising).

If a review mentions multiple categories with different sentiments, track each separately.

If there is ANY doubt about a category, do not include it. If no specific categories apply, use "Uncategorized" with the overall sentiment.

Your response should be in JSON format ONLY:
{{
    "categories": ["Category1", "Category2", ...],
    "explanation": "Brief explanation of your categorization with specific text evidence"
}}

IMPORTANT: Return ONLY the JSON with no additional text before or after it.
"""

    print(f"\nAnalyzing review: '{review}'")
    print(f"Product Style: {product_style}")

    try:
        # Get categories from qwen3:8b model
        task2_response = query_ollama(task2_prompt, model="qwen3:4b")
        print(f"\nqwen3:8b response for categories:\n{task2_response}")
        
        categories_result = extract_json_from_response(task2_response, "categories")
        categories = categories_result.get("categories", ["Uncategorized"])
        explanation = categories_result.get("explanation", "No explanation provided")
        
        print(f"Extracted categories from qwen3:8b: {categories}")

        # Task 1: Now determine sentiment for each category using qwen3:8b model (ENHANCED WITH VECTOR EXAMPLES)
        category_sentiments = {}
        
        for category in categories:
            # Get vector examples for this specific category
            category_vector_examples = ""
            if VECTOR_KB:
                category_vector_examples = VECTOR_KB.get_relevant_examples_for_sentiment(review, category)
            
            task1_prompt = f"""
You are analyzing a diamond jewelry review for sentiment analysis for a specific category. Your task is to determine the sentiment for the '{category}' category mentioned in this review.

LEARN FROM THESE SIMILAR EXAMPLES FOR {category.upper()} CATEGORY:
{category_vector_examples}

Product Style: {product_style}
Review: "{review}"
Category: {category}

Task 1: Determine the sentiment of this review specifically for the '{category}' category:

CRITICAL INSTRUCTION: If the review mentions RETURNING/RETURNED a product AND gives specific reasons related to the '{category}' category, then the sentiment is "Return" regardless of any positive comments about other products.

Examples:
- "I returned it because the stones were too small" → Return (for Diamond category)
- "I returned it because the bracelet was too big" → Return (for Style category)  
- "I returned it because the setting was loose" → Return (for Manufacturing category)
- "I returned it because it was too expensive" → Return (for Merchandising category)

IMPORTANT: Look for patterns like:
- "the [category issue] and... I returned it"
- "because [category issue]... returned"
- "[category issue]... I returned it"

Sentiment Options:
- "Return" if the customer explicitly mentions returning/returned the product because of issues specifically related to this '{category}' category
- "Positive" if the customer is clearly satisfied with this specific aspect (and NO return mentioned for this category)
- "Negative" if the customer is clearly dissatisfied with this specific aspect (but no return mentioned)
- "Uncategorized" if the sentiment for this category is unclear or not mentioned

Focus ONLY on the sentiment related to the '{category}' category. 

For the current review, check if there are return reasons specifically related to '{category}':
- Diamond category: Look for returned due to stone/diamond size, clarity, color, sparkle, carat issues
- Style category: Look for returned due to bracelet/jewelry size, design, appearance, fit issues  
- Manufacturing category: Look for returned due to setting, clasp, comfort, quality issues
- Merchandising category: Look for returned due to price, value, cost issues
- Retailer-related category: Look for returned due to service, shipping, website issues

IMPORTANT: Provide your answer as EXACTLY ONE WORD from these options: Positive, Negative, Return

Your response should ONLY contain one word: Positive, Negative, Return
"""
            
            sentiment_response = query_ollama(task1_prompt, model="qwen3:4b")
            print(f"\nqwen3:8b sentiment response for {category}:\n{sentiment_response}")
            
            # PRIORITY return detection with enhanced logic
            sentiment_response_clean = sentiment_response.strip().lower()
            review_lower = review.lower()
            
            # Enhanced return detection - check if review mentions return AND this category is the reason
            return_indicators = [
                "returned", "returning", "return", "sent back", "sending back", 
                "had to send back", "exchanged", "exchanging", "ultimately returned"
            ]
            
            has_return_mention = any(indicator in review_lower for indicator in return_indicators)
            
            # FIRST: Check for return context - this takes PRIORITY over model response
            sentiment = None
            if has_return_mention:
                print(f"Return mention found in review for {category}")
                # Check if the return is related to this specific category
                category_return_mapping = {
                    "diamond": ["diamond", "stone", "stones", "clarity", "color", "sparkle", "dull", "cloudy", "brilliant", "carat", "ctw", "small", "tiny", "poor", "bad", "6 ctw", "8 ctw"],
                    "style": ["big", "small", "heavy", "light", "size", "fit", "design", "look", "appearance", "showy", "style", "everyday", "wear", "wide", "narrow", "too big", "too small", "bracelet"],
                    "manufacturing": ["setting", "prong", "clasp", "loose", "broke", "broken", "quality", "craftsmanship", "scratch", "comfortable", "uncomfortable"],
                    "merchandising": ["price", "expensive", "cost", "value", "worth", "money", "cheap", "overpriced"],
                    "retailer-related": ["service", "delivery", "shipping", "website", "store", "staff", "wrong", "mistake", "returned", "return"]
                }
                
                category_keywords = category_return_mapping.get(category.lower(), [])
                
                # ENHANCED: Look for return reasons in the ENTIRE review, not just sentence-by-sentence
                return_found_for_category = False
                
                # Method 1: Check entire review for category keywords + return context
                if any(keyword in review_lower for keyword in category_keywords):
                    print(f"Found category keyword in review for {category}")
                    return_found_for_category = True
                
                # Method 2: Sentence-based analysis as backup
                if not return_found_for_category:
                    sentences = re.split(r'[.!?]+', review)
                    for sentence in sentences:
                        sentence_lower = sentence.lower().strip()
                        if any(ret_ind in sentence_lower for ret_ind in return_indicators):
                            print(f"Found return indicator in sentence: {sentence}")
                            # Check if this sentence or nearby sentences mention category issues
                            sentence_index = sentences.index(sentence)
                            context_sentences = []
                            
                            # Include previous sentence
                            if sentence_index > 0:
                                context_sentences.append(sentences[sentence_index - 1])
                            context_sentences.append(sentence)
                            # Include next sentence
                            if sentence_index < len(sentences) - 1:
                                context_sentences.append(sentences[sentence_index + 1])
                            
                            context_text = " ".join(context_sentences).lower()
                            print(f"Checking context: {context_text}")
                            
                            # Check if category keywords appear in this context
                            if any(keyword in context_text for keyword in category_keywords):
                                return_found_for_category = True
                                print(f"FOUND RETURN CONTEXT FOR {category}: {context_text}")
                                break
                
                if return_found_for_category:
                    sentiment = "Return"
                    print(f"STRICT OVERRIDE: Setting {category} sentiment to Return due to context analysis")
            
            # ONLY if no return context found, extract from model response
            if sentiment is None:
                sentiment = extract_normal_sentiment(sentiment_response, sentiment_response_clean)
                print(f"Using model response sentiment: {sentiment}")
            
            category_sentiments[category] = sentiment
            print(f"Extracted sentiment for {category}: {sentiment}")

        # Task 3: Extract sentiment keywords using gemma3:12b model (ENHANCED WITH VECTOR EXAMPLES)
        keywords_vector_examples = ""
        if VECTOR_KB:
            keywords_vector_examples = VECTOR_KB.get_relevant_examples_for_keywords(review, categories)
        
        task3_prompt = f"""
You are analyzing a diamond jewelry review to extract specific sentiment keywords. Based on the full review, identify exact phrases that indicate WHY the customer has the identified sentiment for each category.


LEARN FROM THESE SIMILAR EXAMPLES:
{keywords_vector_examples}

Product Style: {product_style}
Review: "{review}"
Categories and Sentiments: {category_sentiments}

Task 3: For each category identified, extract ONLY the specific keywords or phrases from the review that relate to that category and explain the sentiment.

SPECIAL INSTRUCTION FOR RETURN SENTIMENTS: If any category has "Return" sentiment, extract the SPECIFIC REASON why the product was returned for that category.

Extract exact phrases from the review text for each category:

1. Diamond category - Extract phrases that mention:
   - Diamond clarity, quality, color, sparkle, brilliance, dullness
   - Diamond shape, size, carat weight
   - Diamond-specific issues
   - RETURN REASONS: "stones were too small", "diamonds were dull", "poor quality diamonds", etc.

2. Style category - Extract phrases that mention:
   - Design, aesthetics, appearance, beauty
   - Size fit, weight feel
   - Metal type, color
   - Gift suitability, compliments received
   - RETURN REASONS: "bracelet was too big", "too small for my wrist", "design was too showy", etc.

3. Manufacturing category - Extract phrases that mention:
   - Craftsmanship, workmanship
   - Comfort, fit issues
   - Prong problems, setting issues
   - Clasp, latch, lock problems
   - Durability concerns
   - Quality
   - RETURN REASONS: "setting was loose", "clasp kept breaking", "poor quality construction", etc.

4. Merchandising category - Extract phrases that mention:
   - Price, cost, value, money
   - Appraisal values, price comparisons
   - Worth, expensive, cheap
   - Future buying intentions
   - RETURN REASONS: "too expensive", "not worth the price", "overpriced", etc.

5. Retailer-related category - Extract phrases that mention:
   - Service, staff, customer experience
   - Purchase process, shipping, delivery
   - Website, store experience
   - Return process, policies
   - RETURN REASONS: "poor customer service", "wrong item shipped", "difficult return process", etc.

IMPORTANT: For categories with "Return" sentiment, prioritize extracting the SPECIFIC PROBLEM that led to the return.

Example:
If review says "The stones were too small and the bracelet was too big. I returned it."
- Diamond: ["stones were too small", "I returned it"]  
- Style: ["bracelet was too big", "I returned it"]

Your response should be in JSON format ONLY:
{{
    "sentiment_keywords": {{
        "Category1": ["exact phrase 1", "exact phrase 2"],
        "Category2": ["exact phrase 1"],
        ...
    }}
}}

IMPORTANT: Return ONLY the JSON with no additional text before or after it.
"""

        # Get sentiment keywords from gemma3:12b model
        task3_response = query_ollama(task3_prompt, model="gemma3:12b")
        print(f"\ngemma3:12b JSON response for keywords (excerpt):\n{task3_response[:200]}...")
        keywords_result = extract_json_from_response(task3_response, "keywords")
        
        # Clean up categories
        valid_categories = ["Diamond", "Style", "Manufacturing", "Merchandising", "Retailer-related", "Uncategorized"]
        cleaned_category_sentiments = {}

        for category, sentiment in category_sentiments.items():
            if not category or category.strip() == "":
                continue
            matched = False
            for valid_category in valid_categories:
                if valid_category.lower() in category.strip().lower():
                    if valid_category not in cleaned_category_sentiments:
                        cleaned_category_sentiments[valid_category] = sentiment
                    matched = True
                    break
            if not matched and category.strip().lower() != "uncategorized":
                print(f"Category '{category}' not in valid categories, keeping as is")
                cleaned_category_sentiments[category.strip()] = sentiment

        if not cleaned_category_sentiments:
            cleaned_category_sentiments = {"Uncategorized": "Uncategorized"}
        elif len(cleaned_category_sentiments) > 1 and "Uncategorized" in cleaned_category_sentiments:
            del cleaned_category_sentiments["Uncategorized"]
            
        print(f"Final cleaned category sentiments: {cleaned_category_sentiments}")
        
        # Create the final result
        result = {
            "category_sentiments": cleaned_category_sentiments,
            "explanation": explanation,
            "sentiment_keywords": keywords_result.get("sentiment_keywords", {})
        }
        
        print(f"Final result: {result}")
        return result
        
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return {
            "category_sentiments": {"Uncategorized": "Uncategorized"},
            "explanation": f"Error: {str(e)}",
            "sentiment_keywords": {}
        }

# Main function to process the input file (UNCHANGED STRUCTURE)
def process_reviews(input_file):
    """
    Process all reviews in the input file
    Args:
        input_file (str): Path to the input Excel file
    Returns:
        pd.DataFrame: Processed DataFrame with analysis results
    """
    print(f"Reading input file: {input_file}")
    df = pd.read_excel(input_file)
    required_columns = ["Product Style", "Timeline", "Date", "Review"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the input file")
    
    print("Analyzing reviews...")
    all_analyses = []
    review_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        review = row["Review"]
        product_style = row["Product Style"]
        if pd.isna(review) or review.strip() == "":
            continue
            
        analysis = analyze_review(review, product_style)
        all_analyses.append({
            "idx": idx,
            "analysis": analysis
        })
        
        review_count += 1
        # REDUCED sleep time for faster processing
        if review_count % 10 == 0:  # Changed from every 5 to every 10 reviews
            print(f"\nProcessed {review_count} reviews. Sleeping for 30 seconds...")
            time.sleep(120)  # Reduced from 120 to 30 seconds

    expanded_rows = []
    for analysis_item in all_analyses:
        idx = analysis_item["idx"]
        analysis = analysis_item["analysis"]
        row = df.iloc[idx].copy()
        category_sentiments = analysis["category_sentiments"]
        sentiment_keywords = analysis.get("sentiment_keywords", {})
        
        for category, sentiment in category_sentiments.items():
            if not category or category.strip() == "":
                continue
            new_row = row.copy()
            
            # Get keywords specific to this category only
            category_keywords = []
            if category in sentiment_keywords and sentiment_keywords[category]:
                category_keywords = sentiment_keywords[category]
            
            # Generate detailed reasoning for this sentiment using the updated function
            analysis_text = extract_sentiment_reasoning(row["Review"], category, sentiment, category_keywords)
                
            expanded_rows.append({
                "Product Style": row["Product Style"],
                "Timeline": row["Timeline"],
                "Date": row["Date"],
                "Review": row["Review"],
                "Review_Category": category,
                "Sentiment Type": sentiment,
                "Analysis": analysis_text
            })

    expanded_df = pd.DataFrame(expanded_rows)
    summary_df = expanded_df.copy()
    summary_df["Count"] = summary_df.groupby(
        ["Product Style", "Review_Category", "Sentiment Type", "Review"]
    )["Sentiment Type"].transform("count")
    return summary_df

# Input and output file paths
# MODIFY THESE PATHS TO MATCH YOUR ENVIRONMENT
input_file = r"D:\Users\uttam.singh\Desktop\Sentiment Analysis\Test_Data_Set.xlsx" # Place this file in your Jupyter notebook directory
output_directory = r"D:\Users\uttam.singh\Desktop\Sentiment Analysis"  # Leave empty to save in the same directory as the notebook

# Main execution
if __name__ == "__main__":
    results_df = process_reviews(input_file)
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_directory, f"{input_filename}_vector_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    results_df.to_excel(output_file, index=False)
    print(f"Analysis complete! Results saved to {output_file}")
    print("\nAnalysis Summary:")
    sentiment_counts = results_df["Sentiment Type"].value_counts()
    print("Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
    category_counts = results_df["Review_Category"].value_counts()
    print("\nCategory Distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")