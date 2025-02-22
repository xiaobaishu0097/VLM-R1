import tempfile

import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image


class PDF(FPDF):
    def __init__(self, header_content: str = ""):
        super().__init__()
        self.header_content = header_content
        self.add_font("DejaVuSans", "", "./utils/fonts/DejaVuSans.ttf")
        self.add_font("DejaVuSans", "B", "./utils/fonts/DejaVuSans-Bold.ttf")
        self.add_font("DejaVuSans", "I", "./utils/fonts/DejaVuSans-Oblique.ttf")
        self.set_font("DejaVuSans", size=14)

    def header(self):
        self.set_font("DejaVuSans", "B", 12)
        self.cell(
            w=0,
            h=10,
            text=self.header_content,
            border=0,
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVuSans", "I", 8)
        self.cell(
            w=0,
            h=10,
            text=f"Page {self.page_no()}",
            border=0,
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    def chapter_title(self, title):
        self.set_font("DejaVuSans", "B", 16)
        self.cell(
            w=0,
            h=10,
            text=title,
            border=0,
            align="L",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(10)

    def chapter_subtitle(self, title):
        self.set_font("DejaVuSans", "B", 12)
        self.cell(
            w=0,
            h=6,
            text=title,
            border=0,
            align="L",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(6)

    def chapter_body(self, body):
        self.set_font("DejaVuSans", "", 8)
        self.multi_cell(0, 3, body)
        self.ln()

    def add_image_from_array(self, image_array):
        # Convert np.array to image and save it as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            img = Image.fromarray(image_array)
            img.save(tmpfile.name)
            tmpfile.seek(0)
            self.image(tmpfile.name, x=10, y=self.get_y(), w=100)
        self.ln(110)  # Adjust line break to avoid overlaying the image

    def add_navigation_log(self, navigation_log: dict):
        self.chapter_title(navigation_log["title"])

        if "observation" in navigation_log:
            self.chapter_subtitle("Observation")
            self.add_image_from_array(np.asarray(navigation_log["observation"]))

        self.chapter_subtitle("Observation Description")
        self.chapter_body(navigation_log["observation_description"])

        self.chapter_subtitle("Prompt")
        self.chapter_body(navigation_log["prompt"])

        self.chapter_subtitle("Retrieval Results")
        self.chapter_body(navigation_log["retrieval_results"])

        self.chapter_subtitle("Agent Response")
        self.chapter_body(navigation_log["agent_response"])

        self.chapter_subtitle("Action Response")
        self.chapter_body(navigation_log["action_response"])

        self.chapter_subtitle("Selected Action")
        self.chapter_body(navigation_log["selected_action"])

        self.chapter_subtitle("Action Success")
        self.chapter_body(str(navigation_log["action_success"]))

        self.chapter_subtitle("Target Visibilty")
        self.chapter_body(navigation_log["target_visibility"])

        self.chapter_subtitle("Optimal Action")
        self.chapter_body(navigation_log["optimal_action"])

        self.add_page()


if __name__ == "__main__":
    pdf = PDF(header_content="Navigation Log")
    pdf.add_page()

    navigation_log = {
        "title": "Navigation Log",
        "observation_description": "This is the observation description",
        "prompt": "This is the prompt",
        "retrieval_results": "These are the retrieval results",
        "agent_response": "This is the agent response",
        "action_response": "This is the action response",
        "selected_action": "----------  —action",
        "action_success": True,
    }

    pdf.add_navigation_log(navigation_log)

    pdf.output("./work_dirs/debug/navigation_log.pdf")
