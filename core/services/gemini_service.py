import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiService:
    def __init__(self, tools: list):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Convert MCP tools into Gemini function declarations
        self.tool_definitions = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.inputSchema,
            )
            for t in tools
        ]

        self.history = []
        self.tools = (
            types.Tool(function_declarations=self.tool_definitions)
            if self.tool_definitions
            else None
        )

    async def chat(self, message: str) -> tuple[str, list]:
        """Send a message, returns (text_response, list_of_tool_calls)"""
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=message)])
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=self.history,
            config=types.GenerateContentConfig(
                tools=[self.tools] if self.tools else []
            ),
        )

        candidate = response.candidates[0].content
        self.history.append(candidate)

        tool_calls = []
        text_parts = []

        for part in candidate.parts:
            if part.function_call:
                tool_calls.append(
                    {
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args),
                    }
                )
            elif part.text:
                text_parts.append(part.text)

        return " ".join(text_parts), tool_calls

    async def send_tool_results(self, tool_results: list) -> str:
        """Send tool results back to Gemini and get final response"""
        self.history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=r["name"], response=r["response"]
                        )
                    )
                    for r in tool_results
                ],
            )
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=self.history,
        )

        reply = response.text or ""
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=reply)])
        )
        return reply