class TicketTool:
    def query(self, question: str, ticket_id: str | None = None) -> dict:
        ticket_id = ticket_id or "T20260420001"
        return {
            "ticket_id": ticket_id,
            "status": "处理中",
            "owner": "IT服务台",
            "message": f"工单 {ticket_id} 当前状态为处理中，负责团队为IT服务台。"
        }


ticket_tool = TicketTool()
