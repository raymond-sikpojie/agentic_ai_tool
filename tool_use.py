import os
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
model = "openai/gpt-oss-120b"

# Wrap the client so every call is traced
client = wrap_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
)


# output structure
class Record(BaseModel):
    year: str
    album: str
    artist: str
    genre: str
    price: int


# create tool and specify the argument passed to it when called by the llm.
class GetAlbumByTitle(BaseModel):
    """Tool to get album information by the album title"""

    title: str = Field(description="The title of the album")

    def exec(self):
        import sqlite3

        conn = sqlite3.connect("music.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM music WHERE album = ?", (self.title,))
        rows = cursor.fetchall()

        # for row in rows:
        #     print(row)

        conn.close()
        return rows
        # row = cursor.fetchone()
        # conn.close()
        # return row


tool = pydantic_function_tool(GetAlbumByTitle)

# print(tool)

tool_lookup = {"GetAlbumByTitle": GetAlbumByTitle}

conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant at a record shop that has an inventory of records. \
         you will help answer customers' inquiries regarding records in this store. ",
    }
]

conversation_history.append(
    {"role": "user", "content": "Do you have the album Revolver?"}
)
# call the model use a tool
response = client.chat.completions.create(
    model=model,
    messages=conversation_history,
    tools=[tool],
)

# add the response to the conversation history
# assistant_response = response.choices[0].message.content
# conversation_history.append({"role": "assistant", "content": assistant_response})

conversation_history.append(response.choices[0].message)


function = response.choices[0].message.tool_calls[0].function
tool_id = response.choices[0].message.tool_calls[0].id

tool_name = function.name
tool_args = function.arguments

result = tool_lookup[tool_name].model_validate_json(tool_args).exec()

# Feed the tool result back into the conversation
conversation_history.append(
    {
        "role": "tool",
        "tool_call_id": tool_id,  # links result back to the tool call
        "content": str(result),
    }
)

# Call the LLM again with the tool call result in context
final_response = client.chat.completions.create(
    model=model,
    messages=conversation_history,
)

print(final_response.choices[0].message.content)

# print(result)
