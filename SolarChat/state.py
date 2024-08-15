import re
import reflex as rx
from openai import OpenAI


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
    is_valid: bool


DEFAULT_CHATS = {
    "Hi, I'm Solar": [],
}


class ChatState(rx.State):
    """The app state."""

    api_key: str = ""
    prompt: str = "You are a friendly chatbot named 'Solar'. Respond in markdown."
    chat_model: str = "solar-1-mini-chat"

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat: str = "Hi, I'm Solar"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        if self.api_key == "":
            yield rx.window_alert("Please Input API KEY in setting tab")
            return

        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.solar_process_question

        async for value in model(question):
            yield value

    async def solar_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Add the question to the list of questions.
        qa = QA(question=question, answer="", is_valid=False)
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]
        for qa in self.chats[self.current_chat]:
            if qa.is_valid:
                messages.append({"role": "user", "content": qa.question})
                messages.append({"role": "assistant", "content": qa.answer})

        client = OpenAI(
            api_key=self.api_key, base_url="https://api.upstage.ai/v1/solar"
        )
        try:
            session = client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                stream=True,
            )
            for item in session:
                if hasattr(item.choices[0].delta, "content"):
                    answer_text = item.choices[0].delta.content
                    if answer_text is not None:
                        self.chats[self.current_chat][-1].answer += answer_text
                    yield
            self.chats[self.current_chat][-1].is_valid = True
        except Exception as e:
            answer_text = str(e)
            self.chats[self.current_chat][-1].answer += answer_text

        # Toggle the processing flag.
        self.processing = False
