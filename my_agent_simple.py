import os
import asyncio
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

# --- Simple Classification Function ---
def classify_memory_simple(information: str) -> List[Dict[str, str]]:
    """
    Simple rule-based classification of memories.
    """
    # Keywords that typically indicate personal memories
    personal_keywords = ["my", "i am", "i'm", "my son", "my daughter", "my family", "doctor", "appointment", "personal"]
    
    # Keywords that typically indicate shared memories
    shared_keywords = ["on leave", "out of office", "vacation", "meeting", "project", "team", "deadline", "work"]
    
    # Check if it's likely a shared memory
    info_lower = information.lower()
    is_shared = any(keyword in info_lower for keyword in shared_keywords)
    is_personal = any(keyword in info_lower for keyword in personal_keywords)
    
    # If both or neither, default to personal
    if is_shared and not is_personal:
        classification = "shared"
    else:
        classification = "personal"
    
    return [{"memory": information, "classification": classification}]

# --- Agent Tools ---
def store_information(information: str) -> Dict[str, str]:
    """
    Stores the user's information after simple classification.
    """
    author_name = getattr(store_information, "user_name", "Unknown")
    classified_memories = classify_memory_simple(information)

    if not classified_memories:
        return {"status": "failed", "message": "Could not understand the information provided."}

    for item in classified_memories:
        memory = item.get("memory")
        classification = item.get("classification")

        if classification == "personal":
            persistent_storage.add_memory("abhishek", memory)
        elif classification == "shared":
            # For shared memories, store it as an object with the author
            shared_memory_obj = {"memory": memory, "author": author_name}
            persistent_storage.add_memory("shared", shared_memory_obj)

    return {
        "status": "success",
        "message": f"I've stored {len(classified_memories)} new memories as {classified_memories[0]['classification']}.",
    }

def retrieve_memories(source: str) -> Dict[str, Any]:
    """
    Retrieves memories. Source should be 'personal' or 'shared'.
    """
    if source.lower() == 'personal':
        memories = persistent_storage.retrieve_memories("abhishek")
        return {"source": "personal", "memories": memories}
    elif source.lower() == 'shared':
        memories = persistent_storage.retrieve_memories("shared")
        return {"source": "shared", "memories": memories}
    else:
        return {"status": "failed", "message": "Invalid source. Please specify 'personal' or 'shared'."}

# --- Agent Definition ---
interactive_agent = Agent(
    name="interactive_memory_manager",
    model="gemini-1.5-flash",
    description="An agent that can store and retrieve personal and shared memories.",
    instruction="""
    You are an intelligent assistant for managing memories.

    - **To Store Information:** If the user provides information to remember, use the `store_information` tool.
    - **To Retrieve Information:** If the user asks what you know, use the `retrieve_memories` tool with 'personal' for Abhishek's private memories and 'shared' for team-accessible memories.
    - **Synthesize Answers:** When you retrieve memories, present them in a clear, natural way. For shared memories, always mention the author.
    - Be conversational and helpful.
    """,
    tools=[store_information, retrieve_memories],
)

# --- Runner and Session Setup ---
session_service = InMemorySessionService()
runner = Runner(
    agent=interactive_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# --- Main Execution Logic ---
async def call_agent_async(query: str):
    """Handles the async call to the agent."""
    print(f"\n> You: {query}")
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    # Set the user's name on the tool function so it's accessible as the "author"
    setattr(store_information, "user_name", USER_NAME)

    async for event in runner.run_async(
        user_id=USER_NAME, session_id=SESSION_ID, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            print(f"< Agent: {final_response}")
            return final_response
    return "No response received."

async def interactive_chat():
    """Main loop for the interactive chat."""
    print("--- 🧠 Interactive Memory Agent is ready ---")
    print("Try saying: 'Remember I am taking my son to football tomorrow, which means I am on leave.'")
    print("Then ask: 'What are the shared memories?' or 'What do you know about my personal plans?'")
    print("Type 'quit' to end the session.")

    while True:
        user_query = input("\n")
        if user_query.lower() in ["quit", "exit"]:
            print("Ending session. Goodbye!")
            break
        await call_agent_async(query=user_query)

async def create_session():
    """Creates the agent session before starting the chat."""
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_NAME, session_id=SESSION_ID
    )

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("COUCHBASE_CONN_STR"):
        print("ERROR: Please set your GOOGLE_API_KEY and Couchbase credentials in the .env file.")
    else:
        asyncio.run(create_session())
        asyncio.run(interactive_chat()) 