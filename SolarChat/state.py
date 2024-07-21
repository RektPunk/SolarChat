import re
import reflex as rx
from openai import OpenAI


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


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

    def is_valid_setting(self) -> bool:
        return self.api_key != "" and self.prompt != ""

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
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        model = self.openai_process_question

        async for value in model(question):
            yield value

    async def openai_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
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
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
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
                    # Ensure answer_text is not None before concatenation
                    if answer_text is not None:
                        self.chats[self.current_chat][-1].answer += answer_text
                    else:
                        answer_text = ""
                        self.chats[self.current_chat][-1].answer += answer_text
                    self.chats = self.chats
                    yield
        except:
            if re.search(r"[ㄱ-ㅎ가-힣]", messages[-1]["content"]):
                answer_text = "연결에 문제가 있는 것 같습니다. 주된 이유는 API 키가 잘못되었기 때문입니다. 설정에서 올바른 API 키를 입력하고 다시 시도해 주세요."
            else:
                answer_text = "It seems that there is a connection issue. The main reason is that your API key is incorrect. Please enter the correct API key in the settings above and try again."
            self.chats[self.current_chat][-1].answer += answer_text

        # Toggle the processing flag.
        self.processing = False
