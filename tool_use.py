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

# from pydantic import Field


class GetAlbumByTitle(BaseModel):
    """Tool to get album information by the album title"""

    year: str = Field(description="Album Year")
    album: str = Field(description="Album Title")
    artist: str = Field(description="Performing Artist")
    genre: str = Field(description="Album genre")
    price: int = Field(description="Price of the album")

    def exec(self):
        """Simple addition function."""
        return self.a + self.b


tool = pydantic_function_tool(Add)

# print(tool)

tool_lookup = {"Add": Add}

conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant at a record shop that has an inventory of records. \
         you will help answer customers' inquiries regarding records in this store. ",
    }
]

conversation_history.append(
    {"role": "user", "content": "Do you have any albums by The Beatles?"}
)
# call the model to get a response
response = client.chat.completions.create(
    model=model, messages=conversation_history, tools=[tool]
)

# add the response to the conversation history
assitant_response = response.choices[0].message.content
conversation_history.append({"role": "assistant", "content": assitant_response})

function = response.choices[0].message.tool_calls[0].function

tool_name = function.name
tool_args = function.arguments

result = tool_lookup[tool_name].model_validate_json(tool_args).exec()

print(result)
