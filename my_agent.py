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

# --- User and Session Configuration ---cls

# This simulates the user interacting with the agent
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
        scope_name: str = "agent",
        collection_name_personal: str = "abhishek",
        collection_name_shared: str = "shared",
    ):
        self.cluster = Cluster(
            conn_str, ClusterOptions(PasswordAuthenticator(username, password))
        )
        self.bucket = self.cluster.bucket(bucket_name)
        self.scope = self.bucket.scope(scope_name)
        self.collection_personal = self.scope.collection(collection_name_personal)
        self.collection_shared = self.scope.collection(collection_name_shared)
        print("[Memory System] Connected to Couchbase Capella")

    def add_memory(self, doc_id: str, memory_data: Union[str, Dict[str, str]]):
        """
        Adds a memory to the specified document.
        For shared memories, memory_data should be a dict: {'memory': str, 'author': str}.
        For personal memories, memory_data is just the string.
        """
        try:
            doc = self.collection_personal.get(doc_id).content_as[dict]
        except DocumentNotFoundException:
            doc = {"memories": []}

        # Prevent duplicate entries
        if memory_data not in doc.get("memories", []):
            doc.setdefault("memories", []).append(memory_data)
            self.collection_personal.upsert(doc_id, doc)
            print(f"[Memory System] Saved memory to document '{doc_id}': '{memory_data}'")
        return True

    def retrieve_memories(self, doc_id: str) -> List[Any]:
        """
        Retrieves all memories from a specified document.
        """
        try:
            doc = self.collection_personal.get(doc_id).content_as[dict]
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

# --- AI-Powered Memory Processing ---
# Create a dedicated agent for memory classification
classification_agent = Agent(
    name="memory_classifier",
    model="gemini-1.5-flash",
    description="An agent that classifies information into personal or shared memories.",
    instruction="""
    Analyze the provided information and break it down into one or more distinct memories.
    For each memory, classify it as 'personal' or 'shared'.

    - 'personal' memories are about an individual's private plans, feelings, or information (e.g., taking a son to football, a doctor's appointment).
    - 'shared' memories are facts or plans relevant to a group or team (e.g., being on leave from work, project deadlines).

    Respond only with a valid Python list of dictionaries. Example:
    [{"memory": "I am taking my son to football tomorrow.", "classification": "personal"}, {"memory": "I will be on leave tomorrow.", "classification": "shared"}]
    """,
)

# Create a separate session service and runner for the classification agent
classification_session_service = InMemorySessionService()
classification_runner = Runner(
    agent=classification_agent,
    app_name="memory_classifier_app",
    session_service=classification_session_service,
)

async def break_and_classify_memory(information: str) -> List[Dict[str, str]]:
    """
    Breaks down information into memories and classifies them using the classification agent.
    """
    try:
        # Create a session for the classification agent
        await classification_session_service.create_session(
            app_name="memory_classifier_app", 
            user_id="classifier", 
            session_id="classification_session"
        )
        
        prompt = f"Analyze this information: {information}"
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
        
        async for event in classification_runner.run_async(
            user_id="classifier", 
            session_id="classification_session", 
            new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text
                try:
                    # Using eval is simple for this example, but for production, use a safer parser like json.loads
                    return eval(response_text)
                except Exception as e:
                    print(f"Error parsing classification response: {e}")
                    # Fallback: treat as single personal memory
                    return [{"memory": information, "classification": "personal"}]
        
        return []
    except Exception as e:
        print(f"Error in memory classification: {e}")
        # Fallback: treat as single personal memory
        return [{"memory": information, "classification": "personal"}]


# --- Agent Tools ---
async def process_and_store_information(information: str) -> Dict[str, str]:
    """
    Processes the user's information, classifies it, and stores it in Couchbase.
    """
    # This retrieves the author's name set in the main execution loop
    author_name = getattr(process_and_store_information, "user_name", "Unknown")
    classified_memories = await break_and_classify_memory(information)

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
        "message": f"I've stored {len(classified_memories)} new memories.",
    }

def retrieve_classified_memories(source: str) -> Dict[str, Any]:
    """
    Retrieves memories. Source should be 'personal' (for Abhishek) or 'shared' (for the team).
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

    - **To Store Information:** If the user provides a new piece of information to remember, use the `process_and_store_information` tool.
    - **To Retrieve Information:** If the user asks what you know, or asks about personal or shared topics, use the `retrieve_classified_memories` tool. Use 'personal' for Abhishek's private memories and 'shared' for team-accessible memories.
    - **Synthesize Answers:** When you retrieve memories, present them to the user in a clear, natural way. For shared memories, always mention the author.
    - Be conversational and helpful.
    """,
    tools=[process_and_store_information, retrieve_classified_memories],
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
    setattr(process_and_store_information, "user_name", USER_NAME)

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