import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from openai import pydantic_function_tool
from pydantic import BaseModel, Field

# Load environment variables from your .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"
model = "openai/gpt-oss-120b"

# Wrap the client so every call is traced
client = wrap_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
)


# ── Tools (same as workflow) ──────────────────────────────────────────────────


class get_album_by_title(BaseModel):
    """Tool to get album information by the album title"""

    title: str = Field(description="The title of the album")

    def exec(self):
        import sqlite3

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM music WHERE album = ?", (self.title,))
        rows = cursor.fetchall()
        conn.close()
        return rows


class get_album_by_artist(BaseModel):
    """Tool to get album information by the artist"""

    artist: str = Field(description="The name of the performing artist")

    def exec(self):
        import sqlite3

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM music WHERE artist = ?", (self.artist,))
        rows = cursor.fetchall()
        conn.close()
        return rows


class get_albums_by_year(BaseModel):
    """Tool to get album information by the year the record was released"""

    year: int = Field(description="The year the album was released")

    def exec(self):
        import sqlite3

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM music WHERE year = ?", (self.year,))
        rows = cursor.fetchall()
        conn.close()
        return rows


class get_albums_by_genre(BaseModel):
    """Tool to get album information by genre"""

    genre: str = Field(description="The name of the genre")

    def exec(self):
        import sqlite3

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM music WHERE genre = ?", (self.genre,))
        rows = cursor.fetchall()
        conn.close()
        return rows


tools = [
    pydantic_function_tool(get_album_by_title),
    pydantic_function_tool(get_album_by_artist),
    pydantic_function_tool(get_albums_by_year),
    pydantic_function_tool(get_albums_by_genre),
]

tool_lookup = {
    "get_album_by_title": get_album_by_title,
    "get_album_by_artist": get_album_by_artist,
    "get_albums_by_year": get_albums_by_year,
    "get_albums_by_genre": get_albums_by_genre,
}


# ── ReAct Agent ───────────────────────────────────────────────────────────────
#
# The key difference from the workflow:
#
#   Workflow  →  fixed steps: LLM → tool → LLM (always exactly 2 LLM calls)
#
#   ReAct     →  a loop:
#                   1. call LLM
#                   2. did it ask to use a tool?
#                      YES → execute the tool(s), add results to history, go to 1
#                      NO  → it produced a final answer, exit the loop
#
# This lets the agent decide how many reasoning steps it needs. It could call
# one tool, several tools across multiple iterations, or (for a simple question)
# no tools at all.

# The system prompt lives in the message history for the entire conversation.
# Every LLM call sees the same prompt plus everything that has happened so far.

# refactor this prompt to sound more user generated
message_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant at a record shop that has an inventory of records. \
         you will help answer customers' inquiries regarding records in this store. You have access \
         to tools and you can use one or more tool of your choice before answering. When you have enough details \
         to resopond to the customer's inquire, return a clear response.",
    }
]

user_input = input("User: ")
message_history.append({"role": "user", "content": user_input})

# ── The ReAct loop ────────────────────────────────────────────────────────────
while True:
    # Step 1: call the LLM with the full message history
    response = client.chat.completions.create(
        model=model,
        messages=message_history,
        tools=tools,
    )

    assistant_response = response.choices[0].message

    # Step 2: always add the assistant's response to the history so the agent
    # can reason over what it has already done on the next iteration.
    message_history.append(assistant_response)

    # Step 3: check whether the model wants to call a tool
    if not assistant_response.tool_calls:
        # No tool calls → the agent has reasoned its way to a final answer.
        # Exit the loop.
        print("\nAssistant:", assistant_response.content)
        break

    # Step 4: there are one or more tool calls — execute every one of them.
    # The model may request multiple tools in a single turn, so we loop.
    for tool_call in assistant_response.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        tool_id = tool_call.id

        print(f"[agent calling tool: {tool_name} with args {tool_args}]")

        # Execute the tool
        result = tool_lookup[tool_name].model_validate_json(tool_args).exec()

        # Add the tool result to the history so the agent can reason over it.
        # The tool_call_id links this result back to the specific tool call above.
        message_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result),
            }
        )

    # Loop back to Step 1 — the agent will now reason over the tool results
    # and decide whether it needs more tools or is ready to answer.
