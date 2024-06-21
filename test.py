from checkeval.checkeval import Checkeval
from dotenv import main
import os


main.load_dotenv()

c = Checkeval(api_key=os.environ.get("OPENAI_API_KEY"))

res = c.generate_checklist("This is a test", "This is a test")

print(res)