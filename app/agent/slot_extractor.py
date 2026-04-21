import re
from app.observability.tracing import traceable


class SlotExtractor:
    TICKET_ID_PATTERN = re.compile(r"\bT\d{8,}\b", re.IGNORECASE)
    WORKFLOW_ID_PATTERN = re.compile(r"\bWF\d{8,}\b", re.IGNORECASE)

    @traceable(name="slot_extract")
    def extract(self, text: str) -> dict:
        result = {}

        ticket_match = self.TICKET_ID_PATTERN.search(text)
        if ticket_match:
            result["ticket_id"] = ticket_match.group(0)

        workflow_match = self.WORKFLOW_ID_PATTERN.search(text)
        if workflow_match:
            result["workflow_id"] = workflow_match.group(0)

        return result


slot_extractor = SlotExtractor()
