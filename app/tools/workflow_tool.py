class WorkflowTool:
    def query(self, question: str, workflow_id: str | None = None) -> dict:
        workflow_id = workflow_id or "WF20260420001"
        return {
            "workflow_id": workflow_id,
            "workflow_type": "采购审批流程",
            "current_step": "部门负责人审批",
            "message": f"审批单 {workflow_id} 当前流程节点为部门负责人审批。"
        }


workflow_tool = WorkflowTool()
