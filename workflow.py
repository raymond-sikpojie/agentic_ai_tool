import os
import csv
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import pydantic_function_tool
from pydantic import BaseModel, Field

# Load environment variables from your .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.3-70b-versatile"

# Wrap the client so every call is traced
client = wrap_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
)


# create tool and specify the argument passed to it when called by the llm.
class get_album_by_title(BaseModel):
    """Tool to get album information by the album title"""

    title: str = Field(description="The title of the album")

    def exec(self):
        db_connection = sqlite3.connect("music.db")
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM music WHERE album = ?", (self.title,))
        rows = cursor.fetchall()

        db_connection.close()
        return rows


class get_albums_by_artist(BaseModel):
    """Tool to get album information by the artist"""

    artist: str = Field(description="The name of the performing artist")

    def exec(self):
        db_connection = sqlite3.connect("music.db")
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM music WHERE artist = ?", (self.artist,))
        rows = cursor.fetchall()
        db_connection.close()
        return rows


class get_albums_by_year(BaseModel):
    """Tool to get album information by the year the record was released"""

    year: int = Field(description="The year the album was released")

    def exec(self):
        db_connection = sqlite3.connect("music.db")
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM music WHERE year = ?", (self.year,))
        rows = cursor.fetchall()
        db_connection.close()
        return rows


class get_albums_by_genre(BaseModel):
    """Tool to get album information by the genre"""

    genre: str = Field(description="The name of the genre")

    def exec(self):
        db_connection = sqlite3.connect("music.db")
        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM music WHERE genre = ?", (self.genre,))
        rows = cursor.fetchall()
        db_connection.close()
        return rows


tools = [
    pydantic_function_tool(get_album_by_title),
    pydantic_function_tool(get_albums_by_artist),
    pydantic_function_tool(get_albums_by_year),
    pydantic_function_tool(get_albums_by_genre),
]

tool_lookup = {
    "get_album_by_title": get_album_by_title,
    "get_albums_by_artist": get_albums_by_artist,
    "get_albums_by_year": get_albums_by_year,
    "get_albums_by_genre": get_albums_by_genre,
}
csv_file = "enquiries.csv"

system_prompt = (
    "You are a helpful assistant at a record shop that has an inventory of records. "
    "You will help answer customers' inquiries regarding records in this store. "
    "You have access to tools and you can use any tool of your choice before returning a final response."
)


@traceable(name="workflow_agent")
def process_enquiry(enquiry):
    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history.append({"role": "user", "content": enquiry})

    # Step 1: call the LLM, let it pick a tool
    response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
        tools=tools,
    )

    # Step 2: add the assistant's message (including tool_calls) to history
    assistant_response = response.choices[0].message
    conversation_history.append(assistant_response)

    # Step 3: execute every tool the model requested and append the result to conversation history.
    for tool_call in assistant_response.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        tool_id = tool_call.id

        print(f"  [tool call: {tool_name} | args: {tool_args}]")

        result = tool_lookup[tool_name].model_validate_json(tool_args).exec()

        conversation_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result),
            }
        )

    # Step 4: second llm call to summarise the tool results into a final response.
    final_response = client.chat.completions.create(
        model=model,
        messages=conversation_history,
    )

    return final_response.choices[0].message.content


# Process CSV
with open(csv_file, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row["run"].strip().lower() == "yes":
        print(f"\nEnquiry: {row['email']}")
        row["response"] = process_enquiry(row["email"])  # method call
        print(f"Response: {row['response']}")

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nResponse has been written to enquiries.csv")
