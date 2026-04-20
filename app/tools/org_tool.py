class OrgTool:
    def query(self, question: str) -> dict:
        return {
            "department": "研发中心",
            "leader": "张经理",
            "message": f"根据你的问题“{question}”，相关组织信息已查询。"
        }


org_tool = OrgTool()
