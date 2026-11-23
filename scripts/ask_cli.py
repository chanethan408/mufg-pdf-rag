import json
import sys

from app import rag

question = " ".join(sys.argv[1:]) or input("Question: ")
print(json.dumps(rag.answer_question(question), indent=2))
