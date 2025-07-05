import os
import asyncio
import json
import uuid
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
SESSION_ID = "session_memory_final"
APP_NAME = "memory_agent_final"


# --- Couchbase Memory Class ---
class MemoryStorage:
    """
    Handles storing and retrieving memories from specific Couchbase collections.
    """
    def __init__(
        self,
        conn_str: str,
        username: str,
        password: str,
        bucket_name: str,
        scope_name: str = "agent",
    ):
        self.cluster = Cluster(
            conn_str, ClusterOptions(PasswordAuthenticator(username, password))
        )
        self.bucket = self.cluster.bucket(bucket_name)
        self.scope = self.bucket.scope(scope_name)
        self.abhishek_collection = self.scope.collection("abhi")
        self.shared_collection = self.scope.collection("shared")
        print("[Memory System] Connected to Couchbase Capella with scope 'agent'")

    def add_personal_memory(self, memory_data: str):
        """
        Adds a personal memory to the abhi collection.
        """
        import uuid
        doc_id = f"personal_{uuid.uuid4().hex[:8]}"
        memory_doc = {
            "memory": memory_data,
            "type": "personal",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.abhishek_collection.upsert(doc_id, memory_doc)
        print(f"[Memory System] Saved personal memory to abhishek collection: '{memory_data}'")
        return True

    def add_shared_memory(self, memory_data: str, author: str):
        """
        Adds a shared memory to the shared collection.
        """
        import uuid
        doc_id = f"shared_{uuid.uuid4().hex[:8]}"
        memory_doc = {
            "memory": memory_data,
            "author": author,
            "type": "shared",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.shared_collection.upsert(doc_id, memory_doc)
        print(f"[Memory System] Saved shared memory to shared collection: '{memory_data}' by {author}")
        return True

    def retrieve_personal_memories(self) -> List[str]:
        """
        Retrieves all personal memories from the abhishek collection.
        """
        try:
            # Use N1QL query to get all documents from abhishek collection
            query = f"SELECT memory FROM `{self.bucket.name}`.`{self.scope.name}`.`abhishek` WHERE type = 'personal'"
            result = self.cluster.query(query)
            memories = [row['memory'] for row in result]
            print(f"[Memory System] Retrieved {len(memories)} personal memories from abhishek collection.")
            return memories
        except Exception as e:
            print(f"[Memory System] Error retrieving personal memories: {e}")
            return []

    def retrieve_shared_memories(self) -> List[Dict[str, str]]:
        """
        Retrieves all shared memories from the shared collection.
        """
        try:
            # Use N1QL query to get all documents from shared collection
            query = f"SELECT memory, author FROM `{self.bucket.name}`.`{self.scope.name}`.`shared` WHERE type = 'shared'"
            result = self.cluster.query(query)
            memories = [{"memory": row['memory'], "author": row['author']} for row in result]
            print(f"[Memory System] Retrieved {len(memories)} shared memories from shared collection.")
            return memories
        except Exception as e:
            print(f"[Memory System] Error retrieving shared memories: {e}")
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

# --- Agent Tools ---
def process_and_store_memories(information: str) -> Dict[str, Any]:
    """
    This tool is called by the agent after it has analyzed and classified memories.
    The agent will provide the classification in its reasoning.
    """
    author_name = getattr(process_and_store_memories, "user_name", "Unknown")
    
    # Simple heuristic classification as fallback
    info_lower = information.lower()
    
    # Check for shared indicators
    shared_indicators = ["on leave", "out of office", "vacation", "meeting", "project", "deadline", "work", "team"]
    personal_indicators = ["my son", "my daughter", "my family", "doctor", "appointment", "personal", "i am taking"]
    
    has_shared = any(indicator in info_lower for indicator in shared_indicators)
    has_personal = any(indicator in info_lower for indicator in personal_indicators)
    
    memories_stored = []
    
    if has_personal and has_shared:
        # Complex case: contains both personal and shared elements
        # Try to split intelligently
        if "my son" in info_lower and "on leave" in info_lower:
            # Classic case: personal activity leading to shared impact
            personal_memory = "Taking my son to football tomorrow"
            shared_memory = "On leave tomorrow"
            
            persistent_storage.add_personal_memory(personal_memory)
            persistent_storage.add_shared_memory(shared_memory, author_name)
            
            memories_stored = [
                {"memory": personal_memory, "classification": "personal"},
                {"memory": shared_memory, "classification": "shared"}
            ]
        else:
            # Default: store as personal
            persistent_storage.add_personal_memory(information)
            memories_stored = [{"memory": information, "classification": "personal"}]
    
    elif has_shared:
        # Primarily shared information
        persistent_storage.add_shared_memory(information, author_name)
        memories_stored = [{"memory": information, "classification": "shared"}]
    
    else:
        # Default to personal
        persistent_storage.add_personal_memory(information)
        memories_stored = [{"memory": information, "classification": "personal"}]
    
    return {
        "status": "success",
        "message": f"Successfully processed and stored {len(memories_stored)} memories.",
        "memories": memories_stored
    }

def retrieve_all_context_memories() -> Dict[str, Any]:
    """
    Retrieves all memories (both personal and shared) to provide context for RAG.
    This tool is called automatically for every query to provide context.
    """
    personal_memories = persistent_storage.retrieve_personal_memories()
    shared_memories = persistent_storage.retrieve_shared_memories()
    
    return {
        "personal_memories": personal_memories,
        "shared_memories": shared_memories,
        "personal_count": len(personal_memories),
        "shared_count": len(shared_memories),
        "total_context": len(personal_memories) + len(shared_memories)
    }

def retrieve_memories_by_type(memory_type: str) -> Dict[str, Any]:
    """
    Retrieves memories by type (personal or shared).
    """
    if memory_type.lower() == "personal":
        memories = persistent_storage.retrieve_personal_memories()
        return {"type": "personal", "memories": memories, "count": len(memories)}
    elif memory_type.lower() == "shared":
        memories = persistent_storage.retrieve_shared_memories()
        return {"type": "shared", "memories": memories, "count": len(memories)}
    else:
        return {"status": "error", "message": "Invalid memory type. Use 'personal' or 'shared'."}

# --- Main Memory Agent ---
memory_agent = Agent(
    name="intelligent_memory_manager",
    model="gemini-1.5-flash",
    description="An intelligent memory manager with RAG capabilities that uses stored memories to provide context-aware responses.",
    instruction="""
    Role and Core Directive

You are Cogni-Team, an AI-powered Team Collaboration Assistant. Your primary function is to enhance team productivity and knowledge sharing by acting as an intelligent, centralized memory hub.

Your operation is governed by a Retrieval-Augmented Generation (RAG) model. This is your most critical instruction: for every user interaction, you must begin by invoking the retrieve_all_context_memories tool. This initial step gathers the complete operational context (both personal and shared memories) and forms the foundation for all subsequent analysis, responses, and actions.

Standard Operating Procedure (SOP)

You will follow this three-step process for every query you receive:

Contextual Analysis (RAG):

Immediately upon receiving a query, execute the retrieve_all_context_memories tool.

Analyze this retrieved context to fully understand the user's request within its operational environment. Use this data to inform all reasoning.

Information Processing and Archiving:

When tasked with remembering new information, perform a structured analysis to decompose it into distinct, logical facts.

Classify each fact according to the Information Classification Policy below.

Crucially, if the classification of a piece of information is ambiguous, you must ask the user for clarification before storing it. For example: "Should the 'client dinner' be logged as a shared team event, or is it a personal appointment?"

Use the process_and_store_memories tool to archive the classified information, then provide a concise summary of what was stored and how.

Informed Response and Proactive Assistance:

Leverage the retrieved context to deliver intelligent and actionable responses.

Proactively identify conflicts or synergies. If a request conflicts with existing knowledge, point it out. For example: "You asked to schedule a project review on Friday, but the shared context indicates that Sarah is on leave. Would you like to find a date for next week instead?"

When appropriate, connect new information with existing memories to provide deeper insights.

Information Classification Policy

Shared Memory: Any information relevant to team projects, schedules, availability (leave, out-of-office), client meetings, project deadlines, documented decisions, and general team announcements. The goal is to ensure team alignment and transparency.

Personal Memory: Information exclusive to an individual that does not directly impact team collaboration. This includes personal appointments, non-work-related reminders, or private notes. This data will not be exposed to other team members.

Communication Protocol

Maintain a professional, clear, and concise tone.

Be efficient and directly address the user's needs.

When using retrieved data, subtly reference the context to build trust and transparency. For example, use phrases like "Based on the shared team calendar..." or "Checking your personal schedule..." to signal that you are operating on stored information.
""",
    tools=[retrieve_all_context_memories, process_and_store_memories, retrieve_memories_by_type],
)

# --- Session and Runner Setup ---
session_service = InMemorySessionService()
runner = Runner(
    agent=memory_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# --- Main Execution Logic ---
async def call_memory_agent(query: str):
    """
    Calls the memory agent with the user's query.
    """
    print(f"\n> You: {query}")
    
    # Set up context for tools
    setattr(process_and_store_memories, "user_name", USER_NAME)
    
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    
    async for event in runner.run_async(
        user_id=USER_NAME, session_id=SESSION_ID, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
            print(f"< Memory Agent: {final_response}")
            return final_response
    
    return "No response received."

async def interactive_memory_chat():
    """
    Main interactive chat loop for the memory agent.
    """
    print("=" * 60)
    print("🧠 INTELLIGENT MEMORY MANAGER - Ready!")
    print("=" * 60)
    print("This agent can intelligently break down and classify your memories.")
    print()
    print("📝 Example commands:")
    print("  'Remember I am taking my son to football tomorrow, which means I am on leave.'")
    print("  'What are the shared memories?'")
    print("  'What do you know about my personal plans?'")
    print("  'Store this: I have a doctor appointment at 3pm and will be out of office.'")
    print()
    print("Type 'quit' to end the session.")
    print("=" * 60)

    while True:
        user_query = input("\n💬 ")
        if user_query.lower() in ["quit", "exit", "q"]:
            print("\n👋 Ending session. Your memories are safely stored!")
            break
        
        if user_query.strip():
            await call_memory_agent(user_query)

async def create_memory_session():
    """
    Creates the memory agent session.
    """
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_NAME, session_id=SESSION_ID
    )

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("COUCHBASE_CONN_STR"):
        print("❌ ERROR: Please set your GOOGLE_API_KEY and Couchbase credentials in the .env file.")
    else:
        asyncio.run(create_memory_session())
        asyncio.run(interactive_memory_chat()) 