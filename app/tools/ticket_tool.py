class TicketTool:
    def query(self, question: str) -> dict:
        return {
            "ticket_id": "T20260420001",
            "status": "处理中",
            "owner": "IT服务台",
            "message": f"根据你的问题“{question}”，当前工单状态为处理中。"
        }


ticket_tool = TicketTool()
