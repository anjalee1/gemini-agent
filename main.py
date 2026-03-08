import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from mcp_client import MCPClient
from core.services.gemini_service import GeminiService
from core.chat import Chat

load_dotenv()

SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")


async def main():
    print("Starting Gemini CLI Chat with MCP tools...")
    print("Type 'exit' to quit\n")

    command, args = (
        ("uv", ["run", SERVER_PATH])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", [SERVER_PATH])
    )

    # extra_servers allows passing additional MCP servers via command line
    # e.g: python main.py weather_server.py calendar_server.py
    extra_servers = sys.argv[1:]
    clients = {}

    async with AsyncExitStack() as stack:
        # Always connect the main doc server
        main_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["main_client"] = main_client

        # Dynamically connect any extra servers passed as arguments
        for i, server_script in enumerate(extra_servers):
            client_id = f"client_{i}_{server_script}"
            extra_command, extra_args = (
                ("uv", ["run", server_script])
                if os.getenv("USE_UV", "0") == "1"
                else ("python", [server_script])
            )
            client = await stack.enter_async_context(
                MCPClient(command=extra_command, args=extra_args)
            )
            clients[client_id] = client

        # Collect tools from ALL connected servers
        all_tools = []
        for client in clients.values():
            tools = await client.list_tools()
            all_tools.extend(tools)

        print(f"Loaded {len(all_tools)} tools from {len(clients)} server(s): {[t.name for t in all_tools]}\n")

        gemini = GeminiService(tools=all_tools)
        chat = Chat(mcp_client=main_client, clients=clients, gemini=gemini)

        session = PromptSession(
            style=Style.from_dict({"prompt": "#aaaaaa"})
        )

        while True:
            try:
                user_input = await session.prompt_async("> ")
                if not user_input.strip():
                    continue
                if user_input.strip().lower() == "exit":
                    print("Goodbye!")
                    break

                response = await chat.run(user_input)
                print(f"\nGemini: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, ExceptionGroup):
        sys.exit(0)