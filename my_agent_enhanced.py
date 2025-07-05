import os
import asyncio
import json
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.exceptions import DocumentNotFoundException

# Load environment variables from .env file
load_dotenv()

# --- User and Session Configuration ---
USER_NAME = "Abhishek"
SESSION_ID = "session_002"
APP_NAME = "interactive_memory_agent"


# --- Couchbase Memory Class ---
class MemoryStorage:
    """
    Handles storing and retrieving memories from specific Couchbase documents.
    """
    def __init__(
        self,
        conn_str: str,
        username: str,
        password: str,
        bucket_name: str,
        scope_name: str = "agent_memories",
        collection_name: str = "storage",
    ):
        self.cluster = Cluster(
            conn_str, ClusterOptions(PasswordAuthenticator(username, password))
        )
        self.bucket = self.cluster.bucket(bucket_name)
        self.scope = self.bucket.scope(scope_name)
        self.collection = self.scope.collection(collection_name)
        print("[Memory System] Connected to Couchbase Capella")

    def add_memory(self, doc_id: str, memory_data: Union[str, Dict[str, str]]):
        """
        Adds a memory to the specified document.
        """
        try:
            doc = self.collection.get(doc_id).content_as[dict]
        except DocumentNotFoundException:
            doc = {"memories": []}

        # Prevent duplicate entries
        if memory_data not in doc.get("memories", []):
            doc.setdefault("memories", []).append(memory_data)
            self.collection.upsert(doc_id, doc)
            print(f"[Memory System] Saved memory to document '{doc_id}': '{memory_data}'")
        return True

    def retrieve_memories(self, doc_id: str) -> List[Any]:
        """
        Retrieves all memories from a specified document.
        """
        try:
            doc = self.collection.get(doc_id).content_as[dict]
            results = doc.get("memories", [])
            print(f"[Memory System] Retrieved {len(results)} items from document '{doc_id}'.")
            return results
        except DocumentNotFoundException:
            print(f"[Memory System] Document '{doc_id}' not found.")
            return []

# --- Couchbase Credentials and Instantiation ---
COUCHBASE_CONN_STR = os.getenv("COUCHBASE_CONN_STR")
COUCHBASE_USERNAME = os.getenv("COUCHBASE_USERNAME")
COUCHBASE_PASSWORD = os.getenv("COUCHBASE_PASSWORD")
COUCHBASE_BUCKET = os.getenv("COUCHBASE_BUCKET")

persistent_storage = MemoryStorage(
    conn_str=COUCHBASE_CONN_STR,
    username=COUCHBASE_USERNAME,
    password=COUCHBASE_PASSWORD,
    bucket_name=COUCHBASE_BUCKET,
)

# --- Memory Classification Tool ---
def classify_and_store_memory(information: str) -> Dict[str, Any]:
    """
    This tool will be called by the main agent to classify and store memories.
    The classification logic will be handled by the agent's AI reasoning.
    """
    # Get the classification result from the agent (this will be set by the main agent)
    classification_result = getattr(classify_and_store_memory, "classification_result", None)
    author_name = getattr(classify_and_store_memory, "user_name", "Unknown")
    
    if not classification_result:
        # Fallback: treat as single personal memory
        classification_result = [{"memory": information, "classification": "personal"}]
    
    stored_count = 0
    for item in classification_result:
        memory = item.get("memory", "")
        classification = item.get("classification", "personal")
        
        if memory:  # Only store if there's actual memory content
            if classification == "personal":
                persistent_storage.add_memory("abhishek", memory)
                stored_count += 1
            elif classification == "shared":
                shared_memory_obj = {"memory": memory, "author": author_name}
                persistent_storage.add_memory("shared", shared_memory_obj)
                stored_count += 1
    
    return {
        "status": "success",
        "message": f"Successfully stored {stored_count} memories.",
        "details": classification_result
    }

def retrieve_memories_by_source(source: str) -> Dict[str, Any]:
    """
    Retrieves memories from the specified source.
    """
    if source.lower() == 'personal':
        memories = persistent_storage.retrieve_memories("abhishek")
        return {"source": "personal", "memories": memories, "count": len(memories)}
    elif source.lower() == 'shared':
        memories = persistent_storage.retrieve_memories("shared")
        return {"source": "shared", "memories": memories, "count": len(memories)}
    else:
        return {"status": "error", "message": "Invalid source. Use 'personal' or 'shared'."}

# --- Main Interactive Agent ---
interactive_agent = Agent(
    name="memory_manager",
    model="gemini-1.5-flash",
    description="An intelligent memory manager that can decompose, classify, and store memories.",
    instruction="""
    You are an intelligent memory manager for Abhishek and his team.

    **When storing memories:**
    1. If the user wants to store information, analyze it carefully
    2. Break down the information into distinct memories
    3. Classify each memory as either "personal" or "shared":
       - Personal: Private information about the individual (appointments, family activities, personal plans)
       - Shared: Information relevant to the team (being on leave, work meetings, project updates)
    4. Before calling classify_and_store_memory, think through your classification
    5. Call classify_and_store_memory with the original information
    6. Provide a clear summary of what was stored

    **When retrieving memories:**
    1. Use retrieve_memories_by_source with "personal" or "shared" based on the query
    2. Present the results in a natural, conversational way
    3. For shared memories, always mention the author

    **Example classification thinking:**
    "I am taking my son to football tomorrow, which means I am on leave."
    - "I am taking my son to football tomorrow" → personal (family activity)
    - "I am on leave tomorrow" → shared (affects team)

    Be helpful and conversational in your responses.
    """,
    tools=[classify_and_store_memory, retrieve_memories_by_source],
)

# --- Session and Runner Setup ---
session_service = InMemorySessionService()
runner = Runner(
    agent=interactive_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# --- Enhanced Agent Call with Classification ---
async def call_agent_with_classification(query: str):
    """
    Enhanced agent call that handles memory classification within the agent's reasoning.
    """
    print(f"\n> You: {query}")
    
    # Set up the context for the tools
    setattr(classify_and_store_memory, "user_name", USER_NAME)
    
    # Check if this is a memory storage request
    if any(keyword in query.lower() for keyword in ["remember", "store", "save", "note"]):
        # For memory storage, we'll let the agent handle classification through its reasoning
        enhanced_query = f"""
        Please analyze this information and store it appropriately: "{query}"
        
        Break it down into distinct memories and classify each as personal or shared:
        - Personal: private information about the individual
        - Shared: information relevant to the team
        
        Then store the memories using the classify_and_store_memory tool.
        """
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=enhanced_query)])
    else:
        # For other queries, use as-is
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    
    async for event in runner.run_async(
        user_id=USER_NAME, session_id=SESSION_ID, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            print(f"< Agent: {final_response}")
            return final_response
    
    return "No response received."

async def interactive_chat():
    """
    Main interactive chat loop.
    """
    print("--- 🧠 Enhanced Memory Agent is ready ---")
    print("This agent can intelligently break down and classify your memories!")
    print()
    print("Try saying:")
    print("  'Remember I am taking my son to football tomorrow, which means I am on leave.'")
    print("  'What are the shared memories?'")
    print("  'What do you know about my personal plans?'")
    print()
    print("Type 'quit' to end the session.")
    print()

    while True:
        user_query = input("")
        if user_query.lower() in ["quit", "exit", "q"]:
            print("Ending session. Goodbye!")
            break
        
        if user_query.strip():  # Only process non-empty queries
            await call_agent_with_classification(user_query)

async def create_session():
    """
    Creates the agent session.
    """
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_NAME, session_id=SESSION_ID
    )

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("COUCHBASE_CONN_STR"):
        print("ERROR: Please set your GOOGLE_API_KEY and Couchbase credentials in the .env file.")
    else:
        asyncio.run(create_session())
        asyncio.run(interactive_chat()) 