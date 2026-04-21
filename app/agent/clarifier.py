from app.observability.tracing import traceable


class Clarifier:
    SLOT_QUESTION_MAP = {
        "ticket_id": "请补充一下你的工单号，我才能帮你查询处理进度。",
        "workflow_id": "请补充一下审批单号或业务单号，我才能帮你查询当前流程节点。",
        "department_name": "请补充一下具体部门名称，我才能帮你查询组织信息。",
    }

    @traceable(name="clarification_generate")
    def generate(self, missing_slots: list[str]) -> str:
        if not missing_slots:
            return "为了更准确地帮助你，请补充更多背景信息。"

        first_slot = missing_slots[0]
        return self.SLOT_QUESTION_MAP.get(
            first_slot,
            "为了继续处理你的问题，请补充关键信息。"
        )


clarifier = Clarifier()
