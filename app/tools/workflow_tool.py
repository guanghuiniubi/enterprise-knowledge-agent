class WorkflowTool:
    def query(self, question: str) -> dict:
        return {
            "workflow_type": "采购审批流程",
            "current_step": "部门负责人审批",
            "message": f"根据你的问题“{question}”，当前流程节点为部门负责人审批。"
        }


workflow_tool = WorkflowTool()
