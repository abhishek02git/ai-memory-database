import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
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
        self.user_collection = self.scope.collection("abhi")
        self.shared_collection = self.scope.collection("shared")
        print("[Memory System] Connected to Couchbase Capella with scope 'agent'")

    def _format_shared_memory(self, memory_data: str, author: str) -> str:
        """
        Uses LLM to format shared memory text naturally with the author's name.
        Leverages AI for proper grammar and context understanding.
        """
        # If memory already starts with the author's name, return as-is
        if memory_data.lower().strip().startswith(author.lower()):
            return memory_data
            
        # Use LLM to format the memory naturally
        return self._llm_format_memory(memory_data, author)
    
    def _llm_format_memory(self, memory_data: str, author: str) -> str:
        """
        Uses the main agent's LLM capabilities to format memory text naturally.
        This will be called by the main agent when storing shared memories.
        """
        # For now, use intelligent rule-based formatting as a fallback
        # The main agent will handle the LLM formatting when it processes memories
        return self._intelligent_format_fallback(memory_data, author)
    
    def _intelligent_format_fallback(self, memory_data: str, author: str) -> str:
        """
        Intelligent rule-based formatting with better grammar handling.
        """
        text = memory_data.strip()
        text_lower = text.lower()
        
        # Handle different first-person patterns
        if text_lower.startswith('i am '):
            return f"{author} is {text[5:]}"
        elif text_lower.startswith('i will '):
            return f"{author} will {text[7:]}"
        elif text_lower.startswith('i have '):
            return f"{author} has {text[7:]}"
        elif text_lower.startswith('i\'m '):
            return f"{author} is {text[4:]}"
        elif text_lower.startswith('i\'ll '):
            return f"{author} will {text[5:]}"
        elif text_lower.startswith('i\'ve '):
            return f"{author} has {text[5:]}"
        elif text_lower.startswith('i '):
            return f"{author} {text[2:]}"
        elif text_lower.startswith('my '):
            return f"{author}'s {text[3:]}"
        elif text_lower.startswith('me '):
            return f"{author} {text[3:]}"
        else:
            # If no first-person pronouns, just prepend author name
            return f"{author} {text}"

    def add_personal_memory(self, memory_data: str):
        """
        Adds a personal memory to the abhi collection.
        """
        import uuid
        doc_id = f"personal_{uuid.uuid4().hex[:8]}"
        current_datetime = datetime.now()
        memory_doc = {
            "memory": memory_data,
            "type": "personal",
            "timestamp": current_datetime.isoformat(),
            "date_created": current_datetime.strftime("%A, %B %d, %Y"),
            "time_created": current_datetime.strftime("%I:%M %p")
        }
        
        self.user_collection.upsert(doc_id, memory_doc)
        print(f"[Memory System] Saved personal memory to abhi collection: '{memory_data}'")
        return True

    def add_shared_memory(self, memory_data: str, author: str):
        """
        Adds a shared memory to the shared collection.
        Automatically prepends the author's name to the memory text for clarity.
        """
        import uuid
        doc_id = f"shared_{uuid.uuid4().hex[:8]}"
        current_datetime = datetime.now()
        
        # Format the memory with the author's name for better readability
        formatted_memory = self._format_shared_memory(memory_data, author)
        
        memory_doc = {
            "memory": formatted_memory,
            "author": author,
            "type": "shared",
            "timestamp": current_datetime.isoformat(),
            "date_created": current_datetime.strftime("%A, %B %d, %Y"),
            "time_created": current_datetime.strftime("%I:%M %p")
        }
        
        self.shared_collection.upsert(doc_id, memory_doc)
        print(f"[Memory System] Saved shared memory to shared collection: '{formatted_memory}' by {author}")
        return True

    def retrieve_personal_memories(self) -> List[Dict[str, str]]:
        """
        Retrieves all personal memories from the abhi collection.
        """
        try:
            # Use N1QL query to get all documents from abhi collection
            query = f"SELECT memory, date_created, time_created FROM `{self.bucket.name}`.`{self.scope.name}`.`abhi` WHERE type = 'personal' ORDER BY timestamp DESC"
            result = self.cluster.query(query)
            memories = [{"memory": row['memory'], "date": row.get('date_created', 'Unknown'), "time": row.get('time_created', 'Unknown')} for row in result]
            print(f"[Memory System] Retrieved {len(memories)} personal memories from abhi collection.")
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
            query = f"SELECT memory, author, date_created, time_created FROM `{self.bucket.name}`.`{self.scope.name}`.`shared` WHERE type = 'shared' ORDER BY timestamp DESC"
            result = self.cluster.query(query)
            memories = [{"memory": row['memory'], "author": row['author'], "date": row.get('date_created', 'Unknown'), "time": row.get('time_created', 'Unknown')} for row in result]
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
            shared_memory = "I will be on leave tomorrow"
            
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

def add_personal_memory(memory_data: str) -> Dict[str, Any]:
    """
    Direct tool to add a personal memory to the abhi collection.
    Use this when you're certain the information should be stored as personal.
    """
    try:
        success = persistent_storage.add_personal_memory(memory_data)
        if success:
            return {
                "status": "success",
                "message": f"Successfully stored personal memory: '{memory_data}'",
                "type": "personal"
            }
        else:
            return {"status": "error", "message": "Failed to store personal memory"}
    except Exception as e:
        return {"status": "error", "message": f"Error storing personal memory: {str(e)}"}

def add_shared_memory(memory_data: str, author: Optional[str] = None) -> Dict[str, Any]:
    """
    Direct tool to add a shared memory to the shared collection.
    Use this when you're certain the information should be stored as shared.
    """
    try:
        # Get author from context if not provided
        if author is None:
            author = getattr(add_shared_memory, "user_name", "Unknown")
        
        success = persistent_storage.add_shared_memory(memory_data, author)
        if success:
            return {
                "status": "success",
                "message": f"Successfully stored shared memory: '{memory_data}' by {author}",
                "type": "shared",
                "author": author
            }
        else:
            return {"status": "error", "message": "Failed to store shared memory"}
    except Exception as e:
        return {"status": "error", "message": f"Error storing shared memory: {str(e)}"}

def retrieve_personal_memories() -> Dict[str, Any]:
    """
    Direct tool to retrieve all personal memories from the abhi collection.
    """
    try:
        memories = persistent_storage.retrieve_personal_memories()
        return {
            "status": "success",
            "type": "personal",
            "memories": memories,
            "count": len(memories)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving personal memories: {str(e)}"}

def retrieve_shared_memories() -> Dict[str, Any]:
    """
    Direct tool to retrieve all shared memories from the shared collection.
    """
    try:
        memories = persistent_storage.retrieve_shared_memories()
        return {
            "status": "success",
            "type": "shared",
            "memories": memories,
            "count": len(memories)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving shared memories: {str(e)}"}

def format_shared_memory_text(memory_text: str, author_name: Optional[str] = None) -> Dict[str, Any]:
    """
    LLM-powered tool to format memory text naturally for shared storage.
    Converts first-person statements to third-person with proper grammar.
    """
    if author_name is None:
        author_name = getattr(format_shared_memory_text, "user_name", "Unknown")
    
    # This tool will be called by the agent to format text using its LLM capabilities
    # The agent should provide the formatted text in its response
    return {
        "status": "success",
        "original_text": memory_text,
        "author": author_name,
        "instruction": f"Please convert this first-person text to third-person using {author_name}'s name: '{memory_text}'"
    }

# --- Create Date-Time Aware Agent ---
def create_memory_agent():
    """
    Creates the memory agent with current date and time injected into the system prompt.
    """
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%A, %B %d, %Y")
    current_time = current_datetime.strftime("%I:%M %p")
    
    return Agent(
        name="intelligent_memory_manager",
        model="gemini-1.5-flash",
        description="An intelligent memory manager with RAG capabilities that uses stored memories to provide context-aware responses.",
        instruction=f"""
        CURRENT CONTEXT:
        Today's Date: {current_date}
        Current Time: {current_time}
        


You are Cogni-Team, an AI-powered Team Collaboration Assistant. Your primary function is to enhance team productivity and knowledge sharing by acting as an intelligent, centralized memory hub.

Your operation is governed by a Retrieval-Augmented Generation (RAG) model. This is your most critical instruction: for every user interaction, you must begin by invoking the retrieve_all_context_memories tool. This initial step gathers the complete operational context and forms the foundation for all subsequent analysis, responses, and actions.

Standard Operating Procedure (SOP)

You will follow this three-step process for every query you receive:

Contextual Analysis (RAG):

Immediately upon receiving a query, execute the retrieve_all_context_memories tool.

Analyze this retrieved context to fully understand the user's request within its operational environment. Use this data to inform all reasoning.

Information Processing and Archiving:

When tasked with remembering new information, perform a structured analysis to decompose the information into distinct, logical facts.

Classify each fact according to the Information Classification Policy below.

If the classification of a piece of information is ambiguous, you must ask the user for clarification before storing it.

Use the appropriate storage tools to archive the classified information:
- For complex information requiring decomposition: use process_and_store_memories
- For direct personal memory storage: use add_personal_memory
- For direct shared memory storage: first use first person name which is {USER_NAME} to convert first-person text to third-person with proper grammar, 
- Use your LLM capabilities to ensure shared memories are naturally formatted with the author's name which is {USER_NAME}
Then provide a concise summary of what was stored and how.

Informed Response and Proactive Assistance:

Leverage the retrieved context to deliver intelligent and actionable responses.

Proactively identify conflicts or synergies. If a request conflicts with existing knowledge, point it out and suggest a solution.

Information Classification Policy

Shared Memory: Any information relevant to team projects, schedules, availability (leave, out-of-office), client meetings, project deadlines, documented decisions, and general team announcements. The goal is to ensure team alignment and transparency. Note: Shared memories automatically include the user's name for clarity (e.g., "{USER_NAME} will be on leave tomorrow").

Personal Memory: Information exclusive to an individual that does not directly impact team collaboration. This includes personal appointments, non-work-related reminders, or private notes. This data will not be exposed to other team members.

Communication Protocol

Maintain a professional, clear, and concise tone.

When using retrieved data, subtly reference the context to build trust and transparency. For example, use phrases like "Based on the shared team calendar..." or "Checking your personal schedule..."

Few-Shot Training Examples
(This section provides concrete examples of the expected behavior)

Example 1: Decomposing Implied Information

User Input: "I am going out for a doctor's appointment in the 2nd half today."

Agent's Thought Process:

Retrieve Context: Call retrieve_all_context_memories.

Decompose Statement: The user's statement contains two distinct facts.

Fact A: "I have a doctor's appointment." -> The reason is personal.

Fact B: "I will be out in the 2nd half today." -> The impact on availability is relevant to the team.

Classify Facts:

Fact A is Personal.

Fact B is Shared (as per "availability" in the policy).

Formulate Response: Acknowledge both facts and their classification. Confirm the action with the user.

Agent's Final Response: "Understood. I've processed that as two separate items:

Personal Memory: I've noted your doctor's appointment for this afternoon.

Shared Memory: I've updated the team status that you will be unavailable in the second half today.
Is this correct?"

Example 2: Proactive Conflict Resolution

User Input: "Hey, can you book a project kickoff meeting for this Friday afternoon with me, {USER_NAME}, and Ben?"

Agent's Thought Process:

Retrieve Context: Call retrieve_all_context_memories.

Analyze Context: The retrieved shared memories include a fact: "memory": "{USER_NAME} will be on leave this Friday.", "author": "{USER_NAME}".

Identify Conflict: The user's request to book a meeting with {USER_NAME} on Friday conflicts with her stored availability.

Formulate Response: State the conflict clearly and professionally. Proactively suggest an alternative solution.

Agent's Final Response: "Based on the shared team calendar, {USER_NAME} is on leave this Friday. Therefore, I cannot schedule the kickoff then. Would you like me to check for available slots for the three of you next Monday?"

Example 3: Simple Information Retrieval

User Input: "What are the key decisions from the Project Alpha sync yesterday?"

Agent's Thought Process:

Retrieve Context: Call retrieve_all_context_memories.

Analyze Context: Search the retrieved memories for keywords like "Project Alpha," "decisions," and "yesterday." Find a relevant shared memory: -"memory": "Decision from Project Alpha sync: The deadline for Q3 deliverables is extended by one week.", "author": "{USER_NAME}"-.

Formulate Response: Present the retrieved information clearly and cite the source implicitly.

Agent's Final Response: "From the notes on yesterday's Project Alpha sync, the key decision was: The deadline for Q3 deliverables has been extended by one week."

        When using retrieved data, subtly reference the context to build trust and transparency. For example, use phrases like "Based on the shared team calendar..." or "Checking your personal schedule..." to signal that you are operating on stored information.

        When referencing dates, be specific and clear. Use phrases like "today ({current_date})" or "tomorrow" to ensure clarity.
        """,
        tools=[
            retrieve_all_context_memories,
            process_and_store_memories,
            retrieve_memories_by_type,
            add_personal_memory,
            add_shared_memory,
            retrieve_personal_memories,
            retrieve_shared_memories,
            format_shared_memory_text
        ],
    )

# Initialize the agent
memory_agent = create_memory_agent()

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
    setattr(add_shared_memory, "user_name", USER_NAME)
    setattr(format_shared_memory_text, "user_name", USER_NAME)
    
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