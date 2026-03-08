from core.services.gemini_service import GeminiService
from mcp_client import MCPClient


class Chat:
    def __init__(
        self,
        mcp_client: MCPClient,
        clients: dict[str, MCPClient],
        gemini: GeminiService,
    ):
        self.main_client = mcp_client   # primary server
        self.clients = clients          # all servers including primary
        self.gemini = gemini

    async def _find_client_for_tool(self, tool_name: str) -> MCPClient | None:
        """Search all connected clients to find which one has the tool."""
        for client in self.clients.values():
            tools = await client.list_tools()
            if any(t.name == tool_name for t in tools):
                return client
        return None

    async def run(self, user_input: str) -> str:
        text, tool_calls = await self.gemini.chat(user_input)

        if tool_calls:
            tool_results = []
            for call in tool_calls:
                print(f"  [calling tool: {call['name']} with {call['args']}]")

                # Find whichever server owns this tool
                client = await self._find_client_for_tool(call["name"])

                if not client:
                    tool_results.append({
                        "name": call["name"],
                        "response": {"result": f"Tool '{call['name']}' not found in any connected server"},
                    })
                    continue

                result = await client.call_tool(call["name"], call["args"])
                content = result.content[0].text if result.content else "no result"
                tool_results.append({
                    "name": call["name"],
                    "response": {"result": content},
                })

            return await self.gemini.send_tool_results(tool_results)

        return text